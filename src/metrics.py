"""Retrieval metrics.

A retrieved chunk is *relevant* to a question if it contains the question's gold
evidence phrase (case- and whitespace-insensitive). Because relevance is defined by
the text itself rather than by chunk IDs, the same question set can be used to compare
different chunk sizes and chunking strategies fairly.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def is_relevant(chunk_text: str, evidence: str) -> bool:
    return normalise(evidence) in normalise(chunk_text)


def first_relevant_rank(flags: Sequence[bool]) -> int | None:
    """1-based rank of the first True in ``flags`` (ranked list), else None."""
    for i, f in enumerate(flags, start=1):
        if f:
            return i
    return None


def hit_at_k(ranks: Iterable[int | None], k: int) -> float:
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: Iterable[int | None], k: int = 10) -> float:
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)
