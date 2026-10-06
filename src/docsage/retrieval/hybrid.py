"""F3: merge BM25 + vector results with reciprocal rank fusion, optional re-rank (F9)."""
import json

import numpy as np

from docsage.config import settings
from docsage.models import Chunk, Hit
from docsage.observability import tracing
from docsage.retrieval.bm25 import BM25Index
from docsage.retrieval.embedder import embed_query
from docsage.retrieval.vector import vector_search

RRF_K = 60
CANDIDATES = 50


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, strategy: str | None = None):
        self.strategy = strategy or settings.chunking
        path = settings.index_dir / self.strategy
        with (path / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]
        self.emb = np.load(path / "embeddings.npy")
        self.bm25 = BM25Index([c.text for c in self.chunks])

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        use_bm25: bool | None = None,
        use_reranker: bool | None = None,
    ) -> list[Hit]:
        k = k or settings.top_k
        use_bm25 = settings.use_bm25 if use_bm25 is None else use_bm25
        use_reranker = settings.use_reranker if use_reranker is None else use_reranker

        q = embed_query(query)
        n = CANDIDATES if (use_bm25 or use_reranker) else k
        vec = vector_search(q, self.emb, n)
        vscore = dict(vec)
        bm = self.bm25.search(query, n) if use_bm25 else []
        bscore = dict(bm)

        if use_bm25:
            fused = rrf([[i for i, _ in vec], [i for i, _ in bm]])
        else:
            fused = dict(vec)
        ranked = sorted(fused, key=fused.__getitem__, reverse=True)

        keep = settings.rerank_candidates if use_reranker else k
        hits = [
            Hit(
                self.chunks[i],
                fused[i],
                # BM25-only candidates weren't in the vector top-n; compute their cosine directly
                vector_score=vscore[i] if i in vscore else float(self.emb[i] @ q),
                bm25_score=bscore.get(i, 0.0),
            )
            for i in ranked[:keep]
        ]
        if use_reranker:
            from docsage.retrieval.reranker import rerank

            with tracing.observation(
                "rerank-passages",
                input={"query": query, "candidates": len(hits)},
                metadata={"model": settings.reranker_model},
            ) as span:
                hits = rerank(query, hits)
                span.update(output=[(h.chunk.doc, h.chunk.page, round(h.score, 3)) for h in hits[:k]])
        return hits[:k]
