"""Answer generation on top of retrieved chunks.

Two modes:
* extractive (default, offline): return the best chunk as the answer.
* LLM (optional): if the ``anthropic`` package is installed and ANTHROPIC_API_KEY is set,
  ask Claude to answer *only* from the retrieved context and say so when the context is
  insufficient. The model name can be overridden with the ANTHROPIC_MODEL env var.
"""
from __future__ import annotations

import os

from .chunking import Chunk

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = (
    "You answer questions about institute policy documents. Use ONLY the numbered context "
    "passages provided. Cite the passage numbers you used like [1]. If the passages do not "
    "contain the answer, say that you could not find it instead of guessing."
)


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] (source: {c.doc_id}) {c.text}" for i, c in enumerate(chunks, start=1))


def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def answer_extractive(chunks: list[Chunk]) -> str:
    return chunks[0].text if chunks else "No relevant passage found."


def answer_with_llm(question: str, chunks: list[Chunk], model: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
