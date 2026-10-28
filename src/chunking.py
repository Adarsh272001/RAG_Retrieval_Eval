"""Chunking strategies.

All strategies measure ``size`` in *words* so that they can be compared fairly.

* ``fixed``    - sliding window of ``size`` words with ``overlap`` fraction overlap.
* ``sentence`` - consecutive whole sentences packed up to ``size`` words (no overlap).
* ``heading``  - one chunk per Markdown section (prefixed with its heading); sections
                 longer than ``size`` words are split with the fixed-window strategy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading: str = ""

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _flatten_markdown(text: str) -> str:
    """Turn '## Title' lines into 'Title.' so headings behave like sentences."""
    out = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        out.append(m.group(2).rstrip(".") + "." if m else line)
    return "\n".join(out)


def _windows(words: list[str], size: int, overlap: float) -> list[list[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, size - int(size * overlap))
    n, i, out = len(words), 0, []
    while i < n:
        out.append(words[i : i + size])
        if i + size >= n:
            break
        i += step
    return out


def chunk_fixed(doc: Document, size: int, overlap: float = 0.2) -> list[Chunk]:
    words = _flatten_markdown(doc.text).split()
    return [
        Chunk(f"{doc.doc_id}::f{j}", doc.doc_id, " ".join(w))
        for j, w in enumerate(_windows(words, size, overlap))
    ]


def chunk_sentence(doc: Document, size: int, overlap: float = 0.0) -> list[Chunk]:
    flat = _flatten_markdown(doc.text)
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(" ".join(flat.split())) if s.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > size:
            chunks.append(Chunk(f"{doc.doc_id}::s{len(chunks)}", doc.doc_id, " ".join(current)))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(Chunk(f"{doc.doc_id}::s{len(chunks)}", doc.doc_id, " ".join(current)))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] using Markdown headings; text before the first heading gets ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, " ".join(" ".join(b).split())) for h, b in sections if " ".join(b).strip()]


def chunk_heading(doc: Document, size: int, overlap: float = 0.2) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(doc.text):
        words = body.split()
        parts = [words] if len(words) <= size else _windows(words, size, overlap)
        for part in parts:
            text = " ".join(part)
            if heading:
                text = f"{heading}. {text}"
            chunks.append(Chunk(f"{doc.doc_id}::h{len(chunks)}", doc.doc_id, text, heading))
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentence, "heading": chunk_heading}


def make_chunks(docs: list[Document], chunker: str, size: int, overlap: float = 0.2) -> list[Chunk]:
    if chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{chunker}'. Choose from {sorted(CHUNKERS)}")
    out: list[Chunk] = []
    for doc in docs:
        out.extend(CHUNKERS[chunker](doc, size, overlap))
    return out
