"""Eval metrics: retrieval hit@5, citation coverage, refusal accuracy, latency, cost.
Pure functions with no I/O so they can be unit tested."""
import math
import re

_CITE = re.compile(r"\[[^\[\]]+?,\s*p\.?\s*\d+\]")
# move a citation that trails the full stop ("... 242,156. [Wipro AR, p.17]") inside the sentence
_TRAILING_CITE = re.compile(r"([.!?])\s*(\[[^\[\]]+?,\s*p\.?\s*\d+\])")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹\d\[])")


def hit_at_k(row: dict, hits: list[dict]) -> bool | None:
    """Single-fact: an expected page is among the retrieved chunks. Cross-document: every
    expected document has an expected page among them. None for unanswerable questions."""
    if not row["sources"]:
        return None
    retrieved = {(h["doc"], h["page"]) for h in hits}
    covered = [any((s["doc"], p) in retrieved for p in s["pages"]) for s in row["sources"]]
    return all(covered) if row["type"] == "cross" else any(covered)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def hit_text_at_k(row: dict, hits: list[dict]) -> bool | None:
    """Lenient variant of hit@5: a retrieved chunk of the right document counts if it is on an
    expected page OR contains an expected evidence string, since the same figure often appears
    on several pages. Same single/cross logic as hit_at_k."""
    if not row["sources"]:
        return None
    covered = []
    for s in row["sources"]:
        ok = False
        for h in hits:
            if h["doc"] != s["doc"]:
                continue
            if h["page"] in s["pages"] or any(ev in _norm(h.get("text", "")) for ev in s.get("evidence", [])):
                ok = True
                break
        covered.append(ok)
    return all(covered) if row["type"] == "cross" else any(covered)


def doc_coverage(row: dict, hits: list[dict]) -> float | None:
    """Fraction of expected documents with a correct page retrieved (partial credit view)."""
    if not row["sources"]:
        return None
    retrieved = {(h["doc"], h["page"]) for h in hits}
    ok = [any((s["doc"], p) in retrieved for p in s["pages"]) for s in row["sources"]]
    return sum(ok) / len(ok)


def citation_coverage(answer: str, refused: bool) -> float | None:
    """Share of sentences carrying a [DOC, p.N] citation. None when the answer is a refusal."""
    if refused:
        return None
    text = _TRAILING_CITE.sub(r" \2\1", answer.strip())
    sentences = [s for s in _SENT_SPLIT.split(text) if len(s.split()) >= 3]
    if not sentences:
        return 0.0
    return sum(bool(_CITE.search(s)) for s in sentences) / len(sentences)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)]


def _rate(flags: list[bool | None]) -> float | None:
    flags = [f for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else None


def aggregate(results: list[dict]) -> dict:
    """results: one dict per question with keys type, hit, cov, correct, faithful,
    refused, citation_coverage, latency_ms, cost_usd."""
    answerable = [r for r in results if r["type"] != "unanswerable"]
    unans = [r for r in results if r["type"] == "unanswerable"]
    lat = [r["latency_ms"] for r in results]
    cost = [r["cost_usd"] for r in results]
    cites = [r["citation_coverage"] for r in results if r["citation_coverage"] is not None]
    return {
        "n": len(results),
        "hit_at_5": _rate([r["hit"] for r in answerable]),
        "hit_text_at_5": _rate([r.get("hit_text") for r in answerable]),
        "hit_at_5_single": _rate([r["hit"] for r in answerable if r["type"] == "single"]),
        "hit_at_5_cross": _rate([r["hit"] for r in answerable if r["type"] == "cross"]),
        "correctness": _rate([r["correct"] for r in results]),
        "faithfulness": _rate([r["faithful"] for r in results]),
        "citation_coverage": sum(cites) / len(cites) if cites else None,
        "refused_unanswerable": sum(r["refused"] for r in unans),
        "n_unanswerable": len(unans),
        "false_refusals": sum(r["refused"] for r in answerable),
        "latency_p50_s": percentile(lat, 50) / 1000,
        "latency_p95_s": percentile(lat, 95) / 1000,
        "cost_per_1000": 1000 * sum(cost) / len(cost) if cost else 0.0,
    }
