"""F8: Streamlit UI - question box, answer, clickable citations showing the source text.

Runs locally (`make ui`) and on Streamlit Community Cloud. Hosted settings come from the
app's Secrets, which are copied into environment variables below so the same settings code
works in both places. Set DEMO_MODE=1 in the Secrets to hide developer controls.
"""
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

# Streamlit Cloud keeps secrets in st.secrets; the app's settings read environment variables.
# This must run before any docsage import, because settings are read at import time.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str | int | float | bool):
            os.environ.setdefault(str(_key).upper(), str(_value))
except FileNotFoundError:
    pass  # local run without .streamlit/secrets.toml: settings come from .env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsage import pipeline
from docsage.config import settings
from docsage.retrieval.embedder import embed_query

HOSTED = os.environ.get("DEMO_MODE", "") == "1"
MAX_QUESTIONS = int(os.environ.get("DEMO_MAX_QUESTIONS", "20")) if HOSTED else 10**9

st.set_page_config(page_title="DocSage", page_icon="📄", layout="centered")

EXAMPLES = [
    "What was TCS's employee attrition rate in FY26?",
    "How many employees does Wipro have in total?",
    "What share of Infosys's workforce were women in fiscal 2026?",
    "What is Wipro's revenue guidance for FY27?",
]


def available_strategies() -> list[str]:
    found = [s for s in ("heading", "fixed") if (settings.index_dir / s / "chunks.jsonl").exists()]
    return found or ["heading"]


@st.cache_resource(show_spinner="Loading the index and search model (first visit only)...")
def load(strat: str):
    retriever = pipeline.get_retriever(strat)
    embed_query("warm up")  # downloads and loads the embedding model once, not on the first question
    return retriever


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if any(code in msg for code in ("error 429", "error 402", "error 503")):
        return "The language model service is busy or over its free quota. Please try again in a minute."
    if "is not set" in msg:
        return "This demo is not configured with an API key yet."
    return "Something went wrong while answering. Please try again." if HOSTED else msg


st.title("📄 DocSage")
st.caption(
    "Ask questions about the FY26 annual reports of TCS, Infosys and Wipro. "
    "Every claim cites a page; if the answer isn't in the documents, DocSage says so."
)

strategies = available_strategies()
strategy, use_bm25, use_reranker = strategies[0], True, False
with st.sidebar:
    st.header("About")
    st.write(
        "Hybrid search (keywords + embeddings) over about 1,300 report pages, "
        "then an LLM answers using only the retrieved passages."
    )
    if not HOSTED:
        strategy = st.radio("Chunking", strategies, index=0, horizontal=True)
        use_bm25 = st.toggle("Hybrid search (BM25 + vector)", value=True)
        use_reranker = st.toggle("Re-ranker (slow on CPU)", value=False)
    st.caption(f"Model: `{settings.llm_model}`")

if "question" not in st.session_state:
    st.session_state.question = ""
if "session_id" not in st.session_state:
    # groups one visitor's questions in Langfuse
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:12]}"
if "asked" not in st.session_state:
    st.session_state.asked = 0

st.write("Try one:")
for ex in EXAMPLES:
    if st.button(ex, use_container_width=True):
        st.session_state.question = ex

question = st.text_input(
    "Your question", key="question", placeholder="e.g. What was TCS's FY26 revenue?", max_chars=300
)
ask = st.button("Ask", type="primary", use_container_width=True)

if ask and question.strip():
    if st.session_state.asked >= MAX_QUESTIONS:
        st.info(
            f"This public demo allows {MAX_QUESTIONS} questions per visit to protect its free "
            "quota. Reload the page to continue."
        )
        st.stop()
    try:
        load(strategy)
        with st.spinner("Searching the reports..."):
            a = pipeline.ask(
                question.strip(),
                strategy=strategy,
                session_id=st.session_state.session_id,
                tags=["streamlit", "hosted" if HOSTED else "local"],
                use_bm25=use_bm25,
                use_reranker=use_reranker,
            )
        st.session_state.asked += 1
    except FileNotFoundError:
        st.error("The search index is missing from this deployment.")
        st.stop()
    except RuntimeError as e:
        st.error(friendly_error(e))
        st.stop()

    if a.refused:
        st.warning(a.answer)
    else:
        st.markdown(a.answer)
    if a.invalid_citations:
        st.error(f"The model cited passages it was not given: {', '.join(a.invalid_citations)}")

    if a.citations:
        st.subheader("Sources")
        for c in a.citations:
            with st.expander(f"{c['doc']}, p.{c['page']}"):
                st.write(c["text"])

    st.caption(
        f"{a.latency_ms / 1000:.1f} s · {a.input_tokens} in / {a.output_tokens} out tokens"
        + ("" if HOSTED else f" · ${a.cost_usd:.4f}")
    )
    with st.expander("Retrieved passages (debug)"):
        for h in a.hits:
            st.markdown(
                f"**{h['doc']}, p.{h['page']}**  · cosine {h['vector_score']} · "
                f"score {h['score']} · bm25 {h['bm25_score']}"
            )
            st.text(h["text"][:400])
