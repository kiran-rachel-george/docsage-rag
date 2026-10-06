"""F7: FastAPI app exposing POST /ask."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from docsage import pipeline
from docsage.api.schemas import AskRequest, AskResponse, Usage
from docsage.observability import tracing


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    tracing.shutdown()  # flush buffered traces before the process exits


app = FastAPI(title="DocSage", description="Cited Q&A over annual-report PDFs", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        a = pipeline.ask(req.question, tags=["api"])
    except FileNotFoundError as e:
        raise HTTPException(503, "Index not built; run `make index`") from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    return AskResponse(
        answer=a.answer,
        refused=a.refused,
        citations=a.citations,
        latency_ms=round(a.latency_ms, 1),
        usage=Usage(
            input_tokens=a.input_tokens, output_tokens=a.output_tokens, cost_usd=a.cost_usd
        ),
        invalid_citations=a.invalid_citations,
        trace_id=a.trace_id,
    )
