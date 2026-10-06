"""Shared data types."""
from dataclasses import asdict, dataclass


@dataclass
class Line:
    text: str
    size: float
    y: float = 0.5  # vertical centre as a fraction of page height (0 = top)


@dataclass
class Page:
    doc: str  # short name, e.g. "TCS AR"
    page: int  # 1-based PDF page index
    lines: list[Line]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    text: str
    heading: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # final ranking score (RRF, or reranker)
    vector_score: float = 0.0  # cosine similarity; used for the refusal check
    bm25_score: float = 0.0
