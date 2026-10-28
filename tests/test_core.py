"""Run with:  python -m unittest discover -s tests -v   (no pytest needed)"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.chunking import Chunk, make_chunks
from src.loader import Document, load_corpus
from src.metrics import first_relevant_rank, hit_at_k, is_relevant, mrr_at_k, normalise
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, LSAEmbedder

ROOT = Path(__file__).resolve().parents[1]
DOC = Document(
    "demo.md",
    "# Demo\n\n## Alpha\n\nThe quick brown fox jumps over the lazy dog. Foxes are fast animals. "
    "They live in forests.\n\n## Beta\n\nPython is a programming language. It is widely used in data science.\n",
)


class FakeEmbedder:
    """Deterministic bag-of-letters embedder so dense logic can be tested offline."""

    def _vec(self, text):
        v = np.zeros(26)
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


class TestChunking(unittest.TestCase):
    def test_fixed_respects_size_and_covers_all_words(self):
        chunks = make_chunks([DOC], "fixed", 10, 0.2)
        self.assertTrue(all(c.n_words <= 10 for c in chunks))
        joined = " ".join(c.text for c in chunks)
        for w in ["quick", "forests.", "science."]:
            self.assertIn(w, joined)

    def test_fixed_overlap_repeats_words(self):
        chunks = make_chunks([DOC], "fixed", 10, 0.5)
        self.assertEqual(chunks[0].text.split()[5:], chunks[1].text.split()[:5])

    def test_sentence_never_splits_a_sentence(self):
        chunks = make_chunks([DOC], "sentence", 12)
        for c in chunks:
            self.assertTrue(c.text.rstrip().endswith((".", "!", "?")))

    def test_heading_one_chunk_per_section_when_large(self):
        chunks = make_chunks([DOC], "heading", 200)
        self.assertEqual([c.heading for c in chunks], ["Alpha", "Beta"])
        self.assertTrue(chunks[0].text.startswith("Alpha."))

    def test_heading_splits_long_sections(self):
        self.assertGreater(len(make_chunks([DOC], "heading", 8)), 2)

    def test_unknown_chunker_raises(self):
        with self.assertRaises(ValueError):
            make_chunks([DOC], "nope", 10)

    def test_no_empty_chunks(self):
        for name in ["fixed", "sentence", "heading"]:
            self.assertTrue(all(c.text.strip() for c in make_chunks([DOC], name, 9)))


class TestMetrics(unittest.TestCase):
    def test_is_relevant_ignores_case_and_whitespace(self):
        self.assertTrue(is_relevant("Late  fee of\nRs 100", "late fee of rs 100"))
        self.assertFalse(is_relevant("nothing here", "late fee"))

    def test_rank_and_aggregates(self):
        self.assertEqual(first_relevant_rank([False, False, True]), 3)
        self.assertIsNone(first_relevant_rank([False, False]))
        ranks = [1, 2, None, 4]
        self.assertAlmostEqual(hit_at_k(ranks, 1), 0.25)
        self.assertAlmostEqual(hit_at_k(ranks, 3), 0.5)
        self.assertAlmostEqual(mrr_at_k(ranks, 10), (1 + 0.5 + 0 + 0.25) / 4)
        self.assertAlmostEqual(mrr_at_k([4], 3), 0.0)  # outside the cut-off

    def test_normalise(self):
        self.assertEqual(normalise("  A\n B  "), "a b")


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("0", "d", "Hostel gates close at 10:30 PM for first-year students"),
            Chunk("1", "d", "Library fines are Rs 2 per day for overdue books"),
            Chunk("2", "d", "Tuition fee must be paid within 15 days of semester start"),
        ]

    def test_bm25_finds_obvious_match(self):
        r = BM25Retriever(self.chunks)
        self.assertEqual(r.search("overdue library book fine", 1)[0].index, 1)
        self.assertEqual(r.search("when do hostel gates close", 1)[0].index, 0)

    def test_bm25_unknown_terms_score_zero(self):
        self.assertEqual(BM25Retriever(self.chunks).scores("zzzz qqqq").sum(), 0)

    def test_dense_with_fake_embedder_returns_self_match(self):
        r = DenseRetriever(self.chunks, FakeEmbedder())
        for i, c in enumerate(self.chunks):
            self.assertEqual(r.search(c.text, 1)[0].index, i)

    def test_hybrid_rrf_combines_rankings(self):
        bm25 = BM25Retriever(self.chunks)
        dense = DenseRetriever(self.chunks, FakeEmbedder())
        hybrid = HybridRetriever([bm25, dense])
        q = "tuition fee paid within days"
        self.assertEqual(hybrid.search(q, 1)[0].index, 2)
        self.assertEqual(len(hybrid.search(q, 10)), 3)

    def test_lsa_embedder_runs_offline(self):
        r = DenseRetriever(self.chunks, LSAEmbedder(n_components=5))
        self.assertEqual(len(r.search("library fines", 2)), 2)


class TestSampleData(unittest.TestCase):
    """Guards the sample corpus / question set against typos."""

    def test_every_evidence_phrase_exists_in_its_gold_document(self):
        docs = {d.doc_id: normalise(d.text) for d in load_corpus(ROOT / "data/corpus")}
        qs = pd.read_csv(ROOT / "data/eval/questions.csv")
        self.assertGreaterEqual(len(qs), 30)
        self.assertEqual(qs.id.nunique(), len(qs))
        for row in qs.itertuples():
            self.assertIn(row.source_doc, docs, f"{row.id}: unknown document")
            self.assertIn(normalise(row.evidence), docs[row.source_doc], f"{row.id}: evidence not found")

    def test_evidence_is_unique_within_corpus(self):
        """Evidence should identify one place only, otherwise 'relevant' is ambiguous."""
        docs = load_corpus(ROOT / "data/corpus")
        qs = pd.read_csv(ROOT / "data/eval/questions.csv")
        for row in qs.itertuples():
            n = sum(normalise(d.text).count(normalise(row.evidence)) for d in docs)
            self.assertEqual(n, 1, f"{row.id}: evidence appears {n} times")


if __name__ == "__main__":
    unittest.main()
