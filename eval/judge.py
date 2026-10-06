"""LLM judge with a pass/fail rubric and a written reason for every verdict."""
import json
import re

from docsage.config import settings
from docsage.generation.answer import post_chat, usage_and_cost

CORRECTNESS_PROMPT = """You are grading an answer to a question about company annual reports.

Question: {question}
Expected answer: {expected}
Candidate answer: {answer}

Rules:
- PASS if the candidate states the same key facts and figures as the expected answer. Wording,
  formatting (e.g. 3,28,594 vs 328,594) and extra correct detail do not matter.
- FAIL if a required figure is missing, wrong, or attributed to the wrong company, or if the
  candidate says the information is not available when the expected answer has it.
- For multi-company questions, every company's figure in the expected answer must be present.
Reply with JSON only: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by its source passages.

Answer: {answer}

Source passages:
{passages}

Rules:
- PASS only if every factual claim in the answer (each figure, name and comparison) is directly
  supported by the passages above.
- FAIL if any claim is unsupported, contradicted, or comes from outside knowledge.
Reply with JSON only: {{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""


def _ask(prompt: str) -> dict:
    model = settings.judge_model or settings.llm_model
    provider = settings.judge_provider or settings.llm_provider
    data = post_chat(
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 600,  # reasoning models think first; Groq counts this against TPM
            "messages": [{"role": "user", "content": prompt}],
            **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
        },
        provider=provider,
    )
    text = data["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        v = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        v = {}
    verdict = str(v.get("verdict", "")).lower()
    cost = usage_and_cost(data, model)[2]
    if verdict not in ("pass", "fail"):
        return {"pass": None, "reason": f"unparseable judge output: {text[:120]}", "cost": cost}
    return {"pass": verdict == "pass", "reason": str(v.get("reason", "")), "cost": cost}


def judge_correctness(question: str, expected: str, answer: str) -> dict:
    return _ask(CORRECTNESS_PROMPT.format(question=question, expected=expected, answer=answer))


def judge_faithfulness(answer: str, passages: list[dict]) -> dict:
    block = "\n\n".join(f"[{p['doc']}, p.{p['page']}]\n{p['text']}" for p in passages)
    return _ask(FAITHFULNESS_PROMPT.format(answer=answer, passages=block))
