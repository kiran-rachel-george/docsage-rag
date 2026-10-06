import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import aggregate, citation_coverage, doc_coverage, hit_at_k, percentile
from validate_golden import (
    GOLDEN,
    PDFS,
    ROOT,
    check_evidence,
    check_structure,
    load_golden,
)

SINGLE = {"type": "single", "sources": [{"doc": "TCS AR", "pages": [25, 55]}]}
CROSS = {
    "type": "cross",
    "sources": [{"doc": "TCS AR", "pages": [8]}, {"doc": "Wipro AR", "pages": [17, 20]}],
}
UNANS = {"type": "unanswerable", "sources": []}


def h(doc, page):
    return {"doc": doc, "page": page}


def test_hit_at_k_single_and_unanswerable():
    assert hit_at_k(SINGLE, [h("Wipro AR", 3), h("TCS AR", 55)]) is True
    assert hit_at_k(SINGLE, [h("TCS AR", 54), h("Wipro AR", 25)]) is False  # right doc, wrong page
    assert hit_at_k(UNANS, [h("TCS AR", 1)]) is None


def test_hit_at_k_cross_needs_every_document():
    assert hit_at_k(CROSS, [h("TCS AR", 8), h("Wipro AR", 20)]) is True
    assert hit_at_k(CROSS, [h("TCS AR", 8), h("TCS AR", 9)]) is False
    assert doc_coverage(CROSS, [h("TCS AR", 8)]) == 0.5


def test_citation_coverage():
    assert citation_coverage("Attrition was 13.7% [TCS AR, p.55].", False) == 1.0
    # citation placed after the full stop still counts
    assert citation_coverage("Wipro has 242,156 employees. [Wipro AR, p.17]", False) == 1.0
    two = "TCS has 584,519 employees [TCS AR, p.23]. Infosys has 328,594 employees."
    assert citation_coverage(two, False) == 0.5
    assert citation_coverage("Not in these documents.", True) is None


def test_percentile_nearest_rank():
    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95
    assert percentile([], 50) != percentile([], 50)  # nan


def _res(type_, hit, correct, refused, lat, cost, faithful=None, cov=1.0):
    return {"type": type_, "hit": hit, "correct": correct, "faithful": faithful, "refused": refused,
            "citation_coverage": cov, "latency_ms": lat, "cost_usd": cost}


def test_aggregate_refusals_and_rates():
    agg = aggregate([
        _res("single", True, True, False, 1000, 0.01, True),
        _res("single", False, False, True, 2000, 0.0),  # false refusal
        _res("cross", True, True, False, 3000, 0.02, False),
        _res("unanswerable", None, True, True, 100, 0.0, cov=None),
        _res("unanswerable", None, False, False, 1500, 0.01),  # answered instead of refusing
    ])
    assert agg["hit_at_5"] == pytest.approx(2 / 3)
    assert agg["correctness"] == pytest.approx(3 / 5)
    assert agg["faithfulness"] == pytest.approx(0.5)
    assert (agg["refused_unanswerable"], agg["n_unanswerable"], agg["false_refusals"]) == (1, 2, 1)
    assert agg["cost_per_1000"] == pytest.approx(1000 * 0.04 / 5)


def test_golden_set_structure():
    assert check_structure(load_golden()) == []


@pytest.mark.skipif(
    not all((ROOT / "data" / "raw" / f).exists() for f in PDFS.values()), reason="PDFs not present"
)
def test_golden_facts_appear_on_cited_pages():
    assert GOLDEN.exists()
    assert check_evidence(load_golden()) == []


def test_hit_text_counts_evidence_on_unlisted_page():
    from metrics import hit_text_at_k

    row = {"type": "single", "sources": [{"doc": "TCS AR", "pages": [8], "evidence": ["267,021"]}]}
    on_other_page = [{"doc": "TCS AR", "page": 58, "text": "Revenue from operations 267,021 100.0"}]
    assert hit_at_k(row, on_other_page) is False  # strict: page 58 is not listed
    assert hit_text_at_k(row, on_other_page) is True  # lenient: the figure is in the chunk
    wrong_doc = [{"doc": "Wipro AR", "page": 8, "text": "267,021"}]
    assert hit_text_at_k(row, wrong_doc) is False  # evidence must be in the right document
    assert hit_text_at_k({"type": "unanswerable", "sources": []}, on_other_page) is None


def test_hit_text_cross_needs_every_document():
    from metrics import hit_text_at_k

    row = {
        "type": "cross",
        "sources": [
            {"doc": "TCS AR", "pages": [8], "evidence": ["52,820"]},
            {"doc": "Wipro AR", "pages": [17], "evidence": ["132.0 bn"]},
        ],
    }
    only_tcs = [{"doc": "TCS AR", "page": 35, "text": "net income 52,820"}]
    both = [*only_tcs, {"doc": "Wipro AR", "page": 17, "text": "x"}]
    assert hit_text_at_k(row, only_tcs) is False and hit_text_at_k(row, both) is True
