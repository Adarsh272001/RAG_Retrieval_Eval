"""Retrievers: BM25 (sparse), dense embeddings, and hybrid (Reciprocal Rank Fusion)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it me my of on or "
    "the to was what when which who will with you your am been could should would there their "
    "this that these those into than then so such not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    index: int  # index into the chunk list
    score: float


class Retriever(Protocol):
    def scores(self, query: str) -> np.ndarray: ...
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def _top_k(scores: np.ndarray, k: int) -> list[Hit]:
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(int(i), float(scores[i])) for i in order]


# --------------------------------------------------------------------------- BM25
class BM25Retriever:
    """Okapi BM25 implemented from scratch (no external dependency)."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                    out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.scores(query), k)


# --------------------------------------------------------------------------- Dense
class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str, query_prefix: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for dense retrieval: "
                "pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False)
        )[0]


class LSAEmbedder:
    """TF-IDF + truncated SVD (Latent Semantic Analysis).

    A classical, fully offline 'dense' baseline. It is fitted on the chunk texts, so it
    needs no model download - useful for quick runs and for CI.
    """

    model_name = "lsa"

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self._vec = None
        self._svd = None

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self._vec.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=42).fit(X)
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        Z = self._svd.transform(self._vec.transform(texts))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(norms == 0, 1, norms)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


EMBEDDER_REGISTRY = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "e5-small": ("intfloat/e5-small-v2", "query: "),
}


def get_embedder(name: str):
    """Create an embedder from a short name ('minilm', 'bge-small', 'lsa') or a HF model id."""
    if name == "lsa":
        return LSAEmbedder()
    model_id, prefix = EMBEDDER_REGISTRY.get(name, (name, ""))
    return SentenceTransformerEmbedder(model_id, prefix)


class DenseRetriever:
    """Cosine-similarity search over normalised embeddings (exact, in NumPy)."""

    def __init__(self, chunks: list[Chunk], embedder, name: str = "dense"):
        self.name = name
        self.embedder = embedder
        texts = [c.text for c in chunks]
        if hasattr(embedder, "fit"):
            embedder.fit(texts)
        self.matrix = embedder.encode_documents(texts)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.encode_query(query)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.scores(query), k)


# --------------------------------------------------------------------------- Hybrid
class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers: score(d) = sum_r 1 / (rrf_k + rank_r(d))."""

    def __init__(self, retrievers: list, rrf_k: int = 60, depth: int = 50, name: str = "hybrid"):
        self.retrievers, self.rrf_k, self.depth, self.name = retrievers, rrf_k, depth, name

    def scores(self, query: str) -> np.ndarray:
        n = len(self.retrievers[0].scores(query))
        fused = np.zeros(n)
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, self.depth), start=1):
                fused[hit.index] += 1.0 / (self.rrf_k + rank)
        return fused

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.scores(query), k)
