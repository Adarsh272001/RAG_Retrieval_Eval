"""Glue code shared by the evaluation script and the Streamlit app."""
from __future__ import annotations

from .chunking import Chunk
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, get_embedder

RETRIEVER_KINDS = ("bm25", "dense", "hybrid")


def build_retrievers(chunks: list[Chunk], dense_model: str | None) -> dict:
    """Return {label: retriever} for BM25 and, if ``dense_model`` is given, dense + hybrid.

    Labels look like ``bm25``, ``dense[minilm]`` and ``hybrid[minilm]``.
    """
    bm25 = BM25Retriever(chunks)
    out: dict = {"bm25": bm25}
    if dense_model:
        dense = DenseRetriever(chunks, get_embedder(dense_model), name=f"dense[{dense_model}]")
        out[f"dense[{dense_model}]"] = dense
        out[f"hybrid[{dense_model}]"] = HybridRetriever([bm25, dense], name=f"hybrid[{dense_model}]")
    return out
