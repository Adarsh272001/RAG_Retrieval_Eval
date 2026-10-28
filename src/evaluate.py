"""Run the retrieval experiment grid and write results to ``results/``.

Example
-------
    python -m src.evaluate                                  # BM25 + MiniLM dense + hybrid
    python -m src.evaluate --dense-models minilm,bge-small  # compare two embedding models
    python -m src.evaluate --dense-models lsa               # fully offline run (no downloads)
    python -m src.evaluate --dense-models none              # BM25 only
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .chunking import make_chunks
from .loader import load_corpus
from .metrics import first_relevant_rank, hit_at_k, is_relevant, mrr_at_k
from .pipeline import build_retrievers

K_MAX = 10


def run_experiments(
    corpus_dir: str,
    questions_csv: str,
    chunkers: list[str],
    sizes: list[int],
    dense_models: list[str],
    overlap: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = load_corpus(corpus_dir)
    questions = pd.read_csv(questions_csv)
    required = {"id", "question", "evidence", "source_doc"}
    if not required.issubset(questions.columns):
        raise ValueError(f"questions CSV needs columns {sorted(required)}")
    if "qtype" not in questions.columns:
        questions["qtype"] = "all"

    per_question: list[dict] = []
    summary: list[dict] = []

    # BM25 needs no model; each dense model is added as extra columns of the grid.
    for chunker in chunkers:
        for size in sizes:
            chunks = make_chunks(docs, chunker, size, overlap)
            retrievers = build_retrievers(chunks, None)
            for dm in dense_models:
                t0 = time.time()
                extra = build_retrievers(chunks, dm)
                extra.pop("bm25")
                retrievers.update(extra)
                print(f"  built {dm} index for {chunker}/{size} ({time.time() - t0:.1f}s)")

            for label, retriever in retrievers.items():
                ranks_ev, ranks_doc = [], []
                for q in questions.itertuples():
                    hits = retriever.search(q.question, K_MAX)
                    ev_flags = [is_relevant(chunks[h.index].text, q.evidence) for h in hits]
                    doc_flags = [chunks[h.index].doc_id == q.source_doc for h in hits]
                    r_ev, r_doc = first_relevant_rank(ev_flags), first_relevant_rank(doc_flags)
                    ranks_ev.append(r_ev)
                    ranks_doc.append(r_doc)
                    per_question.append(
                        dict(chunker=chunker, size=size, retriever=label, qid=q.id, qtype=q.qtype,
                             rank_evidence=r_ev, rank_doc=r_doc)
                    )
                n_words = [c.n_words for c in chunks]
                summary.append(
                    dict(
                        chunker=chunker, size=size, retriever=label, n_chunks=len(chunks),
                        avg_words=round(sum(n_words) / len(n_words), 1),
                        hit_at_1=round(hit_at_k(ranks_ev, 1), 4),
                        hit_at_3=round(hit_at_k(ranks_ev, 3), 4),
                        hit_at_5=round(hit_at_k(ranks_ev, 5), 4),
                        mrr_at_10=round(mrr_at_k(ranks_ev, 10), 4),
                        doc_hit_at_3=round(hit_at_k(ranks_doc, 3), 4),
                    )
                )
            print(f"done {chunker}/{size}")
    return pd.DataFrame(summary), pd.DataFrame(per_question)


def make_plots(summary: pd.DataFrame, per_q: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) MRR@10 for every configuration
    summary = summary.assign(config=summary.chunker + "-" + summary["size"].astype(str))
    config_order = list(dict.fromkeys(summary.config))  # keep numeric order of the grid
    pivot = summary.pivot(index="config", columns="retriever", values="mrr_at_10").loc[config_order]
    ax = pivot.plot.bar(figsize=(10, 4.8), width=0.8)
    ax.set_ylabel("MRR@10")
    ax.set_xlabel("chunker - chunk size (words)")
    ax.set_title("Retrieval quality by chunking strategy, chunk size and retriever")
    ax.set_ylim(0, 1.05)
    ax.legend(title="retriever", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "mrr_by_config.png", dpi=160)
    plt.close()

    # 2) Hit@3 split by question type (lexical vs paraphrased), averaged over configs
    if per_q["qtype"].nunique() > 1:
        per_q = per_q.assign(hit3=(per_q.rank_evidence.fillna(99) <= 3).astype(float))
        by_type = per_q.groupby(["retriever", "qtype"]).hit3.mean().unstack()
        ax = by_type.plot.bar(figsize=(7, 4.2))
        ax.set_ylabel("Hit@3 (mean over all configs)")
        ax.set_title("Lexical vs paraphrased questions")
        ax.set_ylim(0, 1.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "hit3_by_question_type.png", dpi=160)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--questions", default="data/eval/questions.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--chunkers", default="fixed,sentence,heading")
    p.add_argument("--sizes", default="50,100,200")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--dense-models", default="minilm", help="comma list, or 'none' for BM25 only")
    args = p.parse_args()

    dense = [] if args.dense_models == "none" else [m.strip() for m in args.dense_models.split(",") if m.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, per_q = run_experiments(
        args.corpus, args.questions,
        [c.strip() for c in args.chunkers.split(",")],
        [int(s) for s in args.sizes.split(",")],
        dense, args.overlap,
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_q.to_csv(out_dir / "per_question.csv", index=False)
    make_plots(summary, per_q, out_dir)

    best = summary.sort_values(["mrr_at_10", "hit_at_1"], ascending=False).head(5)
    print("\nTop 5 configurations by MRR@10:")
    print(best.to_string(index=False))
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
