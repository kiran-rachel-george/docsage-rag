"""Local ONNX embeddings (fastembed); no API key needed."""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from docsage.config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(settings.embedding_model)


def _norm(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _norm(np.array(list(_model().embed(texts, batch_size=batch_size)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    return _norm(np.array(list(_model().embed([QUERY_PREFIX + text])), dtype=np.float32))[0]
