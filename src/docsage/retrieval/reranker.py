"""F9: optional cross-encoder re-ranker behind a flag (default: jina-reranker-v1-turbo)."""
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from docsage.config import settings
from docsage.models import Hit


@lru_cache(maxsize=1)
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(settings.reranker_model)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    texts = [h.chunk.text[: settings.rerank_max_chars] for h in hits]
    scores = list(_model().rerank(query, texts))
    for h, s in zip(hits, scores, strict=True):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)
