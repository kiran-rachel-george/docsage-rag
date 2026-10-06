"""Question -> retrieve -> refuse-or-answer -> cited result. One Langfuse trace per call."""
import time
from dataclasses import dataclass, field
from functools import lru_cache

from docsage.config import settings
from docsage.generation.answer import (
    extract_citations,
    generate,
    invalid_citations,
    throttle_seconds,
)
from docsage.generation.refusal import REFUSAL_TEXT, should_refuse
from docsage.observability import tracing
from docsage.retrieval.hybrid import Retriever


@dataclass
class Answer:
    answer: str
    refused: bool
    citations: list[dict]
    hits: list[dict]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    invalid_citations: list[str] = field(default_factory=list)
    trace_id: str | None = None
    throttle_ms: float = 0.0  # time spent waiting on provider rate limits (excluded from latency_ms)


@lru_cache(maxsize=4)
def get_retriever(strategy: str) -> Retriever:
    return Retriever(strategy)


def _hit_view(h) -> dict:
    return {
        "doc": h.chunk.doc,
        "page": h.chunk.page,
        "score": round(h.score, 4),
        "vector_score": round(h.vector_score, 4),
        "bm25_score": round(h.bm25_score, 3),
        "text": h.chunk.text,
    }


def ask(
    question: str,
    strategy: str | None = None,
    client=None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    **retrieval_opts,
) -> Answer:
    """`client` is an injectable LLM callable (tests). `tags` mark the entry point
    (e.g. "api", "streamlit", "eval"); `session_id` groups a user's questions."""
    strategy = strategy or settings.chunking
    use_bm25 = retrieval_opts.get("use_bm25", settings.use_bm25)
    use_reranker = retrieval_opts.get("use_reranker", settings.use_reranker)
    t0 = time.perf_counter()
    throttle_seconds.set(0.0)

    # Config goes on every observation as filterable metadata (Langfuse wants string values).
    with tracing.propagate(
        session_id=session_id,
        tags=["docsage", *(tags or [])],
        metadata={
            "chunking": strategy,
            "hybrid": str(use_bm25),
            "reranker": str(use_reranker),
            "llm": settings.llm_model,
        },
    ), tracing.observation("answer-question", input=question) as root:
        with tracing.observation(
            "retrieve-context",
            as_type="retriever",
            input={"query": question, "top_k": settings.top_k},
        ) as span:
            hits = get_retriever(strategy).retrieve(question, **retrieval_opts)
            hit_view = [_hit_view(h) for h in hits]
            span.update(output=hit_view)

        best = max((h.vector_score for h in hits), default=0.0)
        with tracing.observation(
            "check-evidence",
            as_type="guardrail",
            input={"best_vector_cosine": round(best, 4), "threshold": settings.refusal_threshold},
        ) as span:
            refuse = should_refuse(hits)
            span.update(output={"refused": refuse})

        if refuse:
            result = Answer(REFUSAL_TEXT, True, [], hit_view, 0.0)
            reason = "low-evidence"
        else:
            gen = generate(question, hits, client=client)
            refused = gen.text.strip().rstrip(".").lower() == REFUSAL_TEXT.rstrip(".").lower()
            result = Answer(
                answer=gen.text,
                refused=refused,
                citations=[] if refused else extract_citations(gen.text, hits),
                hits=hit_view,
                latency_ms=0.0,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                cost_usd=gen.cost_usd,
                invalid_citations=invalid_citations(gen.text, hits),
            )
            reason = "llm-refused" if refused else "answered"

        result.throttle_ms = throttle_seconds.get() * 1000
        # latency excludes provider rate-limit sleeps; they are reported as throttle_ms
        result.latency_ms = (time.perf_counter() - t0) * 1000 - result.throttle_ms
        result.trace_id = tracing.current_trace_id()

        # Root input/output are what a reviewer sees at a glance: the question and the answer.
        root.update(
            output=result.answer,
            metadata={
                "outcome": reason,
                "citations": [f"{c['doc']} p.{c['page']}" for c in result.citations],
                "invalid_citations": result.invalid_citations,
                "latency_ms": round(result.latency_ms),
                "cost_usd": result.cost_usd,
            },
        )
        root.set_trace_io(input=question, output=result.answer)
        if not result.refused:
            # Deterministic integrity check; trend it on the dashboard.
            ok = bool(result.citations) and not result.invalid_citations
            tracing.score_current_trace(
                "citations_valid", 1.0 if ok else 0.0, comment=", ".join(result.invalid_citations) or None
            )
    return result
