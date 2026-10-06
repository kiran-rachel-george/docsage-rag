from datetime import UTC

import pytest
from fastapi.testclient import TestClient

from docsage import pipeline
from docsage.api.main import app
from docsage.models import Chunk, Hit


class FakeRetriever:
    def __init__(self, vscore):
        self.vscore = vscore

    def retrieve(self, query, **_):
        return [Hit(Chunk(0, "TCS AR", 47, "Attrition was 13.3%"), 0.03, vector_score=self.vscore)]


def fake_client(text):
    return lambda payload: {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.0012},
    }


def test_ask_returns_cited_answer(monkeypatch):
    real_generate = pipeline.generate
    fake = fake_client("Attrition was 13.3% [TCS AR, p.47].")
    monkeypatch.setattr(pipeline, "get_retriever", lambda s: FakeRetriever(0.8))
    monkeypatch.setattr(pipeline, "generate", lambda q, hits, client=None: real_generate(q, hits, fake))
    r = TestClient(app).post("/ask", json={"question": "What was TCS attrition?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"][0]["page"] == 47
    assert body["usage"]["input_tokens"] == 1000 and body["usage"]["cost_usd"] > 0


def test_low_score_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(pipeline, "get_retriever", lambda s: FakeRetriever(0.1))

    def boom(*a, **k):
        raise AssertionError("LLM must not be called on refusal")

    monkeypatch.setattr(pipeline, "generate", boom)
    r = TestClient(app).post("/ask", json={"question": "What is Company B's FY27 guidance?"})
    body = r.json()
    assert body["refused"] is True and body["citations"] == []
    assert body["answer"] == "Not in these documents."


def test_question_validation():
    assert TestClient(app).post("/ask", json={"question": ""}).status_code == 422


def test_post_chat_groq_request_shape_and_retry(monkeypatch):
    import httpx

    from docsage.config import settings
    from docsage.generation import answer

    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(answer.time, "sleep", lambda s: None)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers))
        if len(calls) == 1:  # first call is rate limited, then succeeds
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(answer.httpx, "post", fake_post)
    out = answer.post_chat({"model": "llama-3.3-70b-versatile", "messages": []})
    assert out["choices"][0]["message"]["content"] == "ok" and len(calls) == 2
    url, payload, headers = calls[-1]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer gsk_test"
    assert "usage" not in payload  # OpenRouter-only field must not be sent to Groq


def test_post_chat_missing_key_and_unknown_provider(monkeypatch):
    import pytest

    from docsage.config import settings
    from docsage.generation import answer

    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        answer.post_chat({})
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        answer.provider_config()


def test_cost_from_tokens_when_provider_reports_none(monkeypatch):
    from docsage.config import settings
    from docsage.generation.answer import usage_and_cost

    data = {"usage": {"prompt_tokens": 2000, "completion_tokens": 100}}
    assert usage_and_cost(data, "openai/gpt-oss-120b")[2] == pytest.approx(
        (2000 * 0.15 + 100 * 0.60) / 1e6
    )
    assert usage_and_cost(data, "llama-3.3-70b-versatile")[2] == 0.0  # unknown price
    monkeypatch.setattr(settings, "llm_price_input_per_m", 0.59)
    monkeypatch.setattr(settings, "llm_price_output_per_m", 0.79)
    assert usage_and_cost(data, "llama-3.3-70b-versatile")[2] == pytest.approx(
        (2000 * 0.59 + 100 * 0.79) / 1e6
    )
    assert usage_and_cost({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.5}}, "x")[2] == 0.5


def test_post_chat_deepseek_request_shape(monkeypatch):
    import httpx

    from docsage.config import settings
    from docsage.generation import answer

    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test-deepseek")
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, payload=json, headers=headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(answer.httpx, "post", fake_post)
    answer.post_chat({"model": "deepseek-flash", "messages": []})
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test-deepseek"
    assert seen["payload"]["thinking"] == {"type": "disabled"}
    assert "usage" not in seen["payload"]  # OpenRouter-only field

    # a per-call provider override (used by the eval judge) wins over LLM_PROVIDER
    monkeypatch.setattr(settings, "llm_provider", "groq")
    answer.post_chat({"model": "deepseek-flash", "messages": []}, provider="deepseek")
    assert seen["url"].startswith("https://api.deepseek.com")


def test_deepseek_cost_uses_cache_split_and_peak_hours():
    from datetime import datetime

    from docsage.generation.answer import is_deepseek_peak, usage_and_cost

    usage = {"usage": {"prompt_tokens": 3000, "completion_tokens": 100,
                       "prompt_cache_hit_tokens": 1000, "prompt_cache_miss_tokens": 2000}}
    mon_peak = datetime(2026, 10, 5, 7, 0, tzinfo=UTC)  # Monday 07:00 UTC
    mon_off = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    sat_morning = datetime(2026, 10, 3, 7, 0, tzinfo=UTC)  # weekend is never peak
    assert is_deepseek_peak(mon_peak) and not is_deepseek_peak(mon_off)
    assert not is_deepseek_peak(sat_morning)
    off = usage_and_cost(usage, "deepseek-flash", mon_off)[2]
    peak = usage_and_cost(usage, "deepseek-flash", mon_peak)[2]
    assert off == pytest.approx((1000 * 0.003 + 2000 * 0.15 + 100 * 0.60) / 1e6)
    assert peak == pytest.approx(2 * off)  # flash peak prices are exactly 2x off-peak


def test_cloudflare_endpoint_needs_account_id_and_prices_models(monkeypatch):
    from docsage.config import settings
    from docsage.generation import answer

    monkeypatch.setattr(settings, "cloudflare_api_token", "tok")
    monkeypatch.setattr(settings, "cloudflare_account_id", "")
    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        answer.provider_config("cloudflare")
    monkeypatch.setattr(settings, "cloudflare_account_id", "abc123")
    base, key, env = answer.provider_config("cloudflare")
    assert base == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"
    assert (key, env) == ("tok", "CLOUDFLARE_API_TOKEN")
    usage = {"usage": {"prompt_tokens": 2000, "completion_tokens": 100}}
    cost = answer.usage_and_cost(usage, "@cf/meta/llama-3.3-70b-instruct-fp8-fast")[2]
    assert cost == pytest.approx((2000 * 0.293 + 100 * 2.253) / 1e6)
