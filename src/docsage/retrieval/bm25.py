"""BM25 keyword search."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
