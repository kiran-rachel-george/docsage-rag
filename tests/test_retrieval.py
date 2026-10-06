from docsage.generation.answer import extract_citations, invalid_citations
from docsage.generation.refusal import should_refuse
from docsage.models import Chunk, Hit
from docsage.retrieval.bm25 import BM25Index
from docsage.retrieval.hybrid import rrf


def hit(doc, page, vscore=0.8):
    return Hit(Chunk(0, doc, page, f"text of {doc} {page}"), 0.0, vector_score=vscore)


def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 2, 9]])
    assert scores[2] > scores[1] and scores[3] > scores[1] and scores[2] > scores[9]


def test_bm25_finds_exact_term():
    idx = BM25Index(["revenue grew", "attrition was 13.3 percent", "board of directors"])
    assert idx.search("attrition percent", 3)[0][0] == 1


def test_refusal_threshold():
    assert should_refuse([hit("TCS AR", 1, 0.3)], threshold=0.5)
    assert not should_refuse([hit("TCS AR", 1, 0.3), hit("TCS AR", 2, 0.7)], threshold=0.5)
    assert should_refuse([], threshold=0.5)


def test_citations_resolve_to_retrieved_passages():
    hits = [hit("TCS AR", 47), hit("Wipro AR", 12)]
    text = "Revenue was X [TCS AR, p.47]. Headcount rose [Wipro AR, p.12][TCS AR, p.47]."
    cites = extract_citations(text, hits)
    assert [(c["doc"], c["page"]) for c in cites] == [("TCS AR", 47), ("Wipro AR", 12)]
    assert invalid_citations(text, hits) == []


def test_invented_citation_is_flagged_not_returned():
    hits = [hit("TCS AR", 47)]
    text = "Claim [TCS AR, p.999]."
    assert extract_citations(text, hits) == []
    assert invalid_citations(text, hits) == ["[TCS AR, p.999]"]


def test_rerank_orders_by_cross_encoder_score_and_truncates(monkeypatch):
    from docsage.config import settings
    from docsage.retrieval import reranker

    seen = []

    class FakeCE:
        def rerank(self, query, texts):
            seen.extend(texts)
            return [0.1, 0.9, 0.5]

    monkeypatch.setattr(reranker, "_model", lambda: FakeCE())
    monkeypatch.setattr(settings, "rerank_max_chars", 10)
    hits = [Hit(Chunk(i, "TCS AR", i, "x" * 100), 0.0) for i in range(3)]
    out = reranker.rerank("q", hits)
    assert [h.chunk.id for h in out] == [1, 2, 0]
    assert all(len(t) == 10 for t in seen)


def test_normalize_answer_handles_fullwidth_citations():
    from docsage.generation.answer import normalize_answer

    raw = "Wipro has\u202f242,156\u202femployees\u3010Wipro AR, p.17\u3011."
    out = normalize_answer(raw)
    assert out == "Wipro has 242,156 employees[Wipro AR, p.17]."
    assert extract_citations(out, [hit("Wipro AR", 17)])[0]["page"] == 17
