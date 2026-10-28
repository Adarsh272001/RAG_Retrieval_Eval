"""Streamlit app: ask questions over a document folder and inspect what was retrieved.

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.chunking import CHUNKERS, make_chunks
from src.generate import answer_extractive, answer_with_llm, llm_available
from src.loader import load_corpus
from src.pipeline import build_retrievers

st.set_page_config(page_title="RAG Retrieval Lab", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Building index...")
def get_index(corpus_dir: str, chunker: str, size: int, overlap: float, dense_model: str):
    docs = load_corpus(corpus_dir)
    chunks = make_chunks(docs, chunker, size, overlap)
    retrievers = build_retrievers(chunks, None if dense_model == "none" else dense_model)
    return docs, chunks, retrievers


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
corpus_dir = st.sidebar.text_input("Corpus folder", "data/corpus")
chunker = st.sidebar.selectbox("Chunking strategy", list(CHUNKERS), index=2)
size = st.sidebar.slider("Chunk size (words)", 30, 300, 100, step=10)
overlap = st.sidebar.slider("Overlap (fixed / long sections)", 0.0, 0.5, 0.2, step=0.1)
retriever_kind = st.sidebar.selectbox("Retriever", ["hybrid", "dense", "bm25"])
dense_model = st.sidebar.selectbox(
    "Embedding model", ["minilm", "bge-small", "lsa"],
    help="'lsa' is an offline TF-IDF+SVD baseline; the others download a sentence-transformers model.",
    disabled=(retriever_kind == "bm25"),
)
top_k = st.sidebar.slider("Passages to retrieve (k)", 1, 10, 3)

if retriever_kind == "bm25":
    dense_model, label = "none", "bm25"
else:
    label = f"{retriever_kind}[{dense_model}]"

try:
    docs, chunks, retrievers = get_index(corpus_dir, chunker, size, overlap, dense_model)
except ImportError as exc:
    st.error(f"{exc}\n\nTip: choose the offline 'lsa' embedding model or the 'bm25' retriever.")
    st.stop()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

retriever = retrievers[label]
st.sidebar.caption(f"{len(docs)} documents -> {len(chunks)} chunks")

# ------------------------------------------------------------------ main
st.title("🔎 RAG Retrieval Lab")
st.caption("Ask a question, see which passages each retrieval strategy returns, and compare strategies in the Evaluation tab.")

tab_ask, tab_eval = st.tabs(["Ask", "Evaluation results"])

with tab_ask:
    question = st.text_input("Your question", placeholder="e.g. What is the late fee for tuition payment?")
    use_llm = False
    if llm_available():
        use_llm = st.checkbox("Generate an answer with Claude (uses ANTHROPIC_API_KEY)", value=False)
    else:
        st.caption("Set ANTHROPIC_API_KEY and `pip install anthropic` to enable LLM answers; showing the best passage instead.")

    if question.strip():
        hits = retriever.search(question, top_k)
        top_chunks = [chunks[h.index] for h in hits]

        st.subheader("Answer")
        if use_llm:
            with st.spinner("Asking Claude..."):
                try:
                    st.write(answer_with_llm(question, top_chunks))
                except Exception as exc:  # network / auth / quota problems
                    st.error(f"LLM call failed: {exc}")
        else:
            st.info(answer_extractive(top_chunks))

        st.subheader(f"Top {len(hits)} retrieved passages  ·  {label}")
        for rank, (hit, chunk) in enumerate(zip(hits, top_chunks), start=1):
            title = f"#{rank} · {chunk.doc_id}" + (f" › {chunk.heading}" if chunk.heading else "")
            with st.expander(f"{title}   (score {hit.score:.3f})", expanded=(rank == 1)):
                st.write(chunk.text)

with tab_eval:
    summary_path = Path("results/summary.csv")
    if not summary_path.exists():
        st.warning("No results yet. Run `python -m src.evaluate` in a terminal, then refresh this page.")
    else:
        summary = pd.read_csv(summary_path).sort_values("mrr_at_10", ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        best = summary.iloc[0]
        st.success(
            f"Best configuration: **{best.chunker} / {int(best['size'])} words / {best.retriever}** "
            f"(MRR@10 = {best.mrr_at_10:.3f}, Hit@1 = {best.hit_at_1:.3f})"
        )
        for img in ["mrr_by_config.png", "hit3_by_question_type.png"]:
            if Path("results", img).exists():
                st.image(str(Path("results", img)))
