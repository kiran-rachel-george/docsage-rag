"""F8: Streamlit UI - question box, answer, clickable citations showing the source text."""
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docsage import pipeline
from docsage.config import settings

st.set_page_config(page_title="DocSage", page_icon="📄", layout="centered")

EXAMPLES = [
    "What was TCS's employee attrition rate in FY26?",
    "How many employees does Wipro have in total?",
    "Compare Infosys and Wipro revenue growth in FY26.",
    "What is Wipro's FY27 revenue guidance?",
]

st.title("📄 DocSage")
st.caption(
    "Ask questions about the FY26 annual reports of TCS, Infosys and Wipro. "
    "Every claim cites a page; if the answer isn't in the documents, DocSage says so."
)

with st.sidebar:
    st.header("Settings")
    strategy = st.radio("Chunking", ["heading", "fixed"], index=0, horizontal=True)
    use_bm25 = st.toggle("Hybrid search (BM25 + vector)", value=True)
    use_reranker = st.toggle("Re-ranker (slow on CPU)", value=False)
    st.caption(f"Model: `{settings.llm_model}`")

if "question" not in st.session_state:
    st.session_state.question = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:12]}"  # groups one visitor's questions in Langfuse

st.write("Try one:")
for ex in EXAMPLES:
    if st.button(ex, use_container_width=True):
        st.session_state.question = ex

question = st.text_input("Your question", key="question", placeholder="e.g. What was TCS's FY26 revenue?")
ask = st.button("Ask", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Loading index...")
def load(strat: str):
    return pipeline.get_retriever(strat)


if ask and question.strip():
    try:
        load(strategy)
        with st.spinner("Searching the reports..."):
            a = pipeline.ask(
                question.strip(),
                strategy=strategy,
                session_id=st.session_state.session_id,
                tags=["streamlit"],
                use_bm25=use_bm25,
                use_reranker=use_reranker,
            )
    except FileNotFoundError:
        st.error("Index not built. Run `make index` first.")
        st.stop()
    except RuntimeError as e:
        st.error(str(e))
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
        f"{a.latency_ms / 1000:.1f} s · {a.input_tokens} in / {a.output_tokens} out tokens · "
        f"${a.cost_usd:.4f}"
    )
    with st.expander("Retrieved passages (debug)"):
        for h in a.hits:
            st.markdown(
                f"**{h['doc']}, p.{h['page']}**  · cosine {h['vector_score']} · "
                f"score {h['score']} · bm25 {h['bm25_score']}"
            )
            st.text(h["text"][:400])
