"""Check eval/golden_set.jsonl: structure, counts, and that every expected fact really
appears on the cited PDF pages. Run: python eval/validate_golden.py"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
PDFS = {
    "TCS AR": "tcs_ar_fy26.pdf",
    "Infosys AR": "infosys_ar_fy26.pdf",
    "Wipro AR": "wipro_ar_fy26.pdf",
}
TYPES = {"single": 15, "cross": 10, "unanswerable": 5}
HOLDOUT = 5


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def check_structure(rows: list[dict]) -> list[str]:
    errs = []
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("duplicate ids")
    counts = {t: sum(r["type"] == t for r in rows) for t in TYPES}
    if counts != TYPES:
        errs.append(f"type counts {counts}, expected {TYPES}")
    if sum(r["split"] == "holdout" for r in rows) != HOLDOUT:
        errs.append(f"expected {HOLDOUT} holdout questions")
    for r in rows:
        if not r["question"].strip() or not r["expected_answer"].strip():
            errs.append(f"{r['id']}: empty question/answer")
        if r["type"] == "unanswerable" and r["sources"]:
            errs.append(f"{r['id']}: unanswerable must have no sources")
        if r["type"] != "unanswerable" and not r["sources"]:
            errs.append(f"{r['id']}: answerable needs sources")
        if r["type"] == "cross" and len({s["doc"] for s in r["sources"]}) < 2:
            errs.append(f"{r['id']}: cross-document needs >= 2 documents")
    if sum("table" in r["tags"] for r in rows) < 3:
        errs.append("need >= 3 table questions")
    if sum("topical" in r["tags"] for r in rows) < 2:
        errs.append("need >= 2 topical unanswerable questions")
    return errs


def check_evidence(rows: list[dict]) -> list[str]:
    """Each evidence string must appear on at least one of the listed pages."""
    errs, cache = [], {}
    for name, fname in PDFS.items():
        path = ROOT / "data" / "raw" / fname
        if not path.exists():
            return [f"missing PDF {path}"]
        with pymupdf.open(path) as pdf:
            cache[name] = [re.sub(r"\s+", " ", p.get_text()) for p in pdf]
    for r in rows:
        for s in r["sources"]:
            pages = cache[s["doc"]]
            for p in s["pages"]:
                if not 1 <= p <= len(pages):
                    errs.append(f"{r['id']}: {s['doc']} page {p} out of range")
            for ev in s["evidence"]:
                if not any(ev in pages[p - 1] for p in s["pages"] if 1 <= p <= len(pages)):
                    errs.append(f"{r['id']}: '{ev}' not found on {s['doc']} pages {s['pages']}")
    return errs


if __name__ == "__main__":
    rows = load_golden()
    errs = check_structure(rows) + check_evidence(rows)
    split = {k: sum(r["split"] == k for r in rows) for k in ("dev", "holdout")}
    print(f"{len(rows)} questions {split}")
    for e in errs:
        print("ERROR:", e)
    print("OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
