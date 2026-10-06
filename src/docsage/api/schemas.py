"""Request/response models for /ask."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    doc: str
    page: int
    text: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    usage: Usage
    invalid_citations: list[str] = []
    trace_id: str | None = None  # Langfuse trace id (null when tracing is off)
