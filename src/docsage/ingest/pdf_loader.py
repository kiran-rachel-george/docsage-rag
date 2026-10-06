"""F1: parse PDFs with PyMuPDF; keep document name and page number on every page."""
import re
from collections import Counter
from pathlib import Path

import pymupdf

from docsage.models import Line, Page

DOC_NAMES = {"tcs": "TCS AR", "infosys": "Infosys AR", "wipro": "Wipro AR"}

MARGIN = 0.10  # top/bottom fraction of the page treated as header/footer area
MIN_REPEAT = 0.05  # a margin line on >= 5% of a document's pages is boilerplate


def doc_name(path: Path) -> str:
    key = re.match(r"[a-z]+", path.stem.lower())
    return DOC_NAMES.get(key.group() if key else "", path.stem)


def _in_margin(line: Line) -> bool:
    return line.y < MARGIN or line.y > 1 - MARGIN


def _has_words(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]", text)) >= 3


def _boilerplate_key(text: str) -> str:
    """Normalise so "WIPRO ... REPORT 2025-26 54" and "... 55" count as the same line."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def strip_boilerplate(pages: list[Page]) -> list[Page]:
    """Drop running headers/footers. A line is boilerplate if, across the document, it
    (a) sits in the top/bottom margin and repeats on >= 5% of pages, or (b) repeats on
    >= 5% of pages at the same vertical position (catches footers placed mid-page, as in
    Wipro). Page numbers are normalised, so "... 2025-26 54" and "... 55" match. This text
    matches almost every query term (company name, year) and adds noise to BM25 and
    embeddings."""
    margin_counts: Counter[str] = Counter()
    pos_counts: Counter[tuple[str, float]] = Counter()
    for p in pages:
        margin_counts.update({_boilerplate_key(ln.text) for ln in p.lines if _in_margin(ln)})
        pos_counts.update(
            {(_boilerplate_key(ln.text), round(ln.y, 2)) for ln in p.lines if _has_words(ln.text)}
        )
    threshold = max(3, MIN_REPEAT * len(pages))
    margin_rep = {k for k, n in margin_counts.items() if n >= threshold}
    pos_rep = {k for k, n in pos_counts.items() if n >= threshold}

    def is_boilerplate(ln: Line) -> bool:
        key = _boilerplate_key(ln.text)
        if _in_margin(ln) and (key in margin_rep or re.fullmatch(r"\d{1,4}", ln.text)):
            return True
        return _has_words(ln.text) and (key, round(ln.y, 2)) in pos_rep

    for p in pages:
        p.lines = [ln for ln in p.lines if not is_boilerplate(ln)]
    return pages


def load_pdf(path: Path) -> list[Page]:
    name = doc_name(path)
    pages = []
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            height = page.rect.height or 1.0
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    size = max(s["size"] for s in spans)
                    y = (line["bbox"][1] + line["bbox"][3]) / 2 / height
                    lines.append(Line(text, round(size, 1), round(y, 3)))
            pages.append(Page(name, i, lines))
    return strip_boilerplate(pages)
