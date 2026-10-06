"""F4: cited answers like [Company A AR, p.47], validated against retrieved passages."""
import contextvars
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from docsage.config import settings
from docsage.generation.prompts import SYSTEM, build_user_prompt
from docsage.models import Hit
from docsage.observability import tracing

_CITE = re.compile(r"\[([^\[\]]+?),\s*p\.?\s*(\d+)\]")

# USD per million (input, output) tokens for models whose list price we know. Groq's
# llama models are not listed on its public pricing page; set LLM_PRICE_INPUT_PER_M /
# LLM_PRICE_OUTPUT_PER_M to price those.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    # Cloudflare Workers AI list prices (developers.cloudflare.com/workers-ai/platform/pricing).
    # Actual spend is $0 inside the free 10,000-neurons/day allowance.
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": (0.293, 2.253),
    "@cf/meta/llama-3.1-8b-instruct": (0.282, 0.827),
    "@cf/openai/gpt-oss-120b": (0.350, 0.750),
    "@cf/openai/gpt-oss-20b": (0.200, 0.300),
}

# DeepSeek list prices, USD per million tokens: (cache-hit input, cache-miss input, output),
# from api-docs.deepseek.com/quick_start/pricing. Peak = weekdays 01:00-04:00 and 06:00-10:00 UTC.
DEEPSEEK_PRICES = {
    "deepseek-flash": {"off": (0.003, 0.15, 0.60), "peak": (0.006, 0.30, 1.20)},
    "deepseek-v4-pro": {"off": (0.022, 0.66, 1.98), "peak": (0.044, 1.32, 3.96)},
}

PROVIDERS = {
    "deepseek": ("deepseek_base_url", "deepseek_api_key", "DEEPSEEK_API_KEY"),
    "groq": ("groq_base_url", "groq_api_key", "GROQ_API_KEY"),
    "cloudflare": ("", "cloudflare_api_token", "CLOUDFLARE_API_TOKEN"),  # URL needs the account id
    "openrouter": ("openrouter_base_url", "openrouter_api_key", "OPENROUTER_API_KEY"),
}
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
MAX_WAIT_S = 65  # Groq's per-minute token buckets reset within a minute

# Seconds this request spent sleeping on provider rate limits (read by pipeline.ask so
# latency can exclude it and report it separately).
throttle_seconds: contextvars.ContextVar[float] = contextvars.ContextVar("throttle_seconds", default=0.0)


_NORMALIZE = str.maketrans({"【": "[", "】": "]", " ": " ", " ": " "})


def normalize_answer(text: str) -> str:
    """Some models (e.g. gpt-oss) write citations as fullwidth brackets and use narrow
    no-break spaces. Map them to plain ASCII so citations parse and display cleanly."""
    return text.translate(_NORMALIZE)


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # billed (OpenRouter) or computed from tokens x price (Groq); 0 if unknown


def provider_config(provider: str | None = None) -> tuple[str, str, str]:
    """(base_url, api_key, env var name) for a provider (default: the configured one)."""
    provider = provider or settings.llm_provider
    try:
        url_attr, key_attr, env = PROVIDERS[provider]
    except KeyError:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}' (use: {', '.join(PROVIDERS)})"
        ) from None
    if provider == "cloudflare":
        if not settings.cloudflare_account_id:
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set (put it in .env)")
        base = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/v1"
        )
        return base, getattr(settings, key_attr), env
    return getattr(settings, url_attr), getattr(settings, key_attr), env


def post_chat(payload: dict, provider: str | None = None) -> dict:
    """POST a chat-completions payload to a provider (Groq, OpenRouter or DeepSeek); all
    speak the OpenAI chat format. Retries rate limits and transient server errors."""
    provider = provider or settings.llm_provider
    base_url, api_key, env = provider_config(provider)
    if not api_key:
        raise RuntimeError(f"{env} is not set (put it in .env)")
    if provider == "openrouter":
        payload = {**payload, "usage": {"include": True}}  # ask OpenRouter to return cost
    elif provider == "deepseek":
        # deepseek-flash thinks by default; reasoning tokens would eat max_tokens and add latency
        payload = {**payload, "thinking": {"type": "disabled"}}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
            wait = min(float(r.headers.get("retry-after") or 2**attempt), MAX_WAIT_S)
            throttle_seconds.set(throttle_seconds.get() + wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"{provider} error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("unreachable")


def is_deepseek_peak(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)  # noqa: UP017
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def usage_and_cost(data: dict, model: str, now: datetime | None = None) -> tuple[int, int, float]:
    """(input tokens, output tokens, cost in USD) from a chat-completions response."""
    u = data.get("usage") or {}
    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    if u.get("cost"):  # OpenRouter reports what it billed
        return tin, tout, float(u["cost"])
    if model in DEEPSEEK_PRICES and "prompt_cache_hit_tokens" in u:
        hit_p, miss_p, out_p = DEEPSEEK_PRICES[model]["peak" if is_deepseek_peak(now) else "off"]
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", max(tin - hit, 0))
        return tin, tout, (hit * hit_p + miss * miss_p + tout * out_p) / 1_000_000
    if settings.llm_price_input_per_m is not None and settings.llm_price_output_per_m is not None:
        pin, pout = settings.llm_price_input_per_m, settings.llm_price_output_per_m
    else:
        pin, pout = PRICES.get(model, (0.0, 0.0))
    return tin, tout, (tin * pin + tout * pout) / 1_000_000


def generate(question: str, hits: list[Hit], client=None) -> Generation:
    """`client` is a callable payload -> response dict (injectable for tests)."""
    if client is None:
        provider_config()  # fail fast on an unknown provider
        client = post_chat
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(question, hits)},
    ]
    params = {"max_tokens": 800, "temperature": 0}
    # Marked as a generation so Langfuse can group by model and compute usage/cost.
    with tracing.observation(
        "generate-answer",
        as_type="generation",
        model=settings.llm_model,
        model_parameters=params,
        input=messages,
    ) as gen:
        data = client({"model": settings.llm_model, **params, "messages": messages})
        message = data["choices"][0]["message"]
        text = normalize_answer(message["content"] or "").strip()
        # reasoning models (gpt-oss, DeepSeek) return their thinking separately; keep it on the
        # generation so a bad answer can be debugged (Langfuse best practice)
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        tin, tout, cost = usage_and_cost(data, settings.llm_model)
        result = Generation(text, tin, tout, cost)
        gen.update(
            output=text,
            usage_details={"input": tin, "output": tout},
            # explicit cost when known; omitted otherwise so Langfuse can fall back to its own
            # model pricing
            cost_details={"total": cost} if cost else None,
            metadata={
                "provider": settings.llm_provider,
                "finish_reason": data["choices"][0].get("finish_reason"),
                **({"reasoning": reasoning[:4000]} if reasoning else {}),
            },
        )
    return result


def extract_citations(text: str, hits: list[Hit]) -> list[dict]:
    """Map [DOC, p.N] markers to retrieved passages. Markers that don't match a
    retrieved passage are dropped; invalid_citations() reports them."""
    by_key = {(h.chunk.doc.lower(), h.chunk.page): h for h in hits}
    seen, out = set(), []
    for doc, page in _CITE.findall(text):
        key = (doc.strip().lower(), int(page))
        if key in seen or key not in by_key:
            continue
        seen.add(key)
        c = by_key[key].chunk
        out.append({"doc": c.doc, "page": c.page, "text": c.text})
    return out


def invalid_citations(text: str, hits: list[Hit]) -> list[str]:
    valid = {(h.chunk.doc.lower(), h.chunk.page) for h in hits}
    return [
        f"[{d}, p.{p}]" for d, p in _CITE.findall(text) if (d.strip().lower(), int(p)) not in valid
    ]
