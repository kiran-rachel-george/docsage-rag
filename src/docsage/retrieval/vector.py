"""Vector search: cosine similarity over normalised embeddings."""
import numpy as np


def vector_search(q: np.ndarray, emb: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = emb @ q
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
