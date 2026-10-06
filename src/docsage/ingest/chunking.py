"""F2: fixed-size and heading-aware chunkers. Chunks never span pages, so every
citation points at one exact page."""
import re
from collections import Counter

from docsage.models import Chunk, Page

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over whitespace-separated words, sized by token count."""
    words = text.split()
    out, start = [], 0
    while start < len(words):
        end, n = start, 0
        while end < len(words) and n + count_tokens(words[end]) <= size:
            n += count_tokens(words[end])
            end += 1
        end = max(end, start + 1)
        out.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back, m = end, 0
        while back > start + 1 and m < overlap:
            back -= 1
            m += count_tokens(words[back])
        start = max(back, start + 1)
    return out


def fixed_chunks(pages: list[Page], size: int = 512, overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in pages:
        for text in _split_tokens(p.text, size, overlap):
            chunks.append(Chunk(len(chunks), p.doc, p.page, text))
    return chunks


def body_font_size(pages: list[Page]) -> float:
    """Most common font size by character count = body text."""
    c: Counter[float] = Counter()
    for p in pages:
        for line in p.lines:
            c[line.size] += len(line.text)
    return c.most_common(1)[0][0] if c else 0.0


def _is_heading(text: str, size: float, body: float) -> bool:
    return (
        size >= body * 1.25
        and len(text) <= 120
        and any(ch.isalpha() for ch in text)
        and not text.rstrip().endswith(".")
    )


MIN_CHUNK_TOKENS = 40  # sections shorter than this are merged into a neighbour


def _merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fold tiny sections (a heading plus a line or two) into the next one, or into the
    previous one if they come last. Keeps everything on the same page."""
    out: list[tuple[str, str]] = []
    pending = ""
    for head, text in sections:
        label = f"{head}\n{text}" if head else text
        if pending:
            label = f"{pending}\n{label}"
            pending = ""
        if count_tokens(text) < MIN_CHUNK_TOKENS:
            pending = label
            continue
        out.append((head, label))
    if pending:
        if out:
            out[-1] = (out[-1][0], "\n".join([out[-1][1], pending]))
        else:
            out.append((sections[0][0], pending))
    return out


def heading_chunks(pages: list[Page], size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split each page at headings (large-font lines); long sections fall back to windows,
    tiny ones are merged into a neighbour. The current heading carries across pages."""
    chunks: list[Chunk] = []
    docs: dict[str, list[Page]] = {}
    for p in pages:
        docs.setdefault(p.doc, []).append(p)

    for doc_pages in docs.values():
        body = body_font_size(doc_pages)
        heading = ""
        for p in doc_pages:
            sections: list[tuple[str, list[str]]] = [(heading, [])]
            for line in p.lines:
                if _is_heading(line.text, line.size, body):
                    head, lines = sections[-1]
                    if head and not lines and len(sections) > 1:
                        # consecutive heading lines (wrapped titles) join into one heading
                        sections[-1] = (head + " " + line.text, lines)
                    else:
                        sections.append((line.text, []))
                    heading = sections[-1][0]
                else:
                    sections[-1][1].append(line.text)
            texts = [(h, " ".join(ls).strip()) for h, ls in sections]
            texts = [(h, t) for h, t in texts if t]
            for head, text in _merge_small(texts):
                for piece in _split_tokens(text, size, overlap):
                    chunks.append(Chunk(len(chunks), p.doc, p.page, piece, head))
    return chunks


CHUNKERS = {"fixed": fixed_chunks, "heading": heading_chunks}
