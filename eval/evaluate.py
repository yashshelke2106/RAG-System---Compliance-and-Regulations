"""Retrieval + refusal evaluation.

Metrics:
  * Hit@k   : fraction of in-corpus questions whose gold section appears in the
              top-k reranked passages.
  * MRR     : mean reciprocal rank of the gold section.
  * Refusal : fraction of out-of-corpus questions correctly refused, and
              fraction of in-corpus questions NOT wrongly refused.

    python -m eval.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.hybrid_retriever import HybridRetriever   # noqa: E402
from src.rag.pipeline import RAGPipeline                # noqa: E402

EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text())


def retrieval_metrics(retriever: HybridRetriever, k: int = 5) -> dict:
    hits, rr = 0, 0.0
    n = len(EVAL["in_corpus"])
    for item in EVAL["in_corpus"]:
        scored = retriever.retrieve(item["question"], top_k=k)
        rank = None
        for i, sc in enumerate(scored, start=1):
            if sc.chunk.doc_id == item["gold_doc"] and sc.chunk.section == item["gold_section"]:
                rank = i
                break
        if rank:
            hits += 1
            rr += 1.0 / rank
    return {"hit@%d" % k: hits / n, "mrr": rr / n, "n": n}


def refusal_metrics(pipeline: RAGPipeline) -> dict:
    correct_refuse = sum(
        1 for it in EVAL["out_of_corpus"] if pipeline.answer(it["question"]).refused
    )
    wrong_refuse = sum(
        1 for it in EVAL["in_corpus"] if pipeline.answer(it["question"]).refused
    )
    return {
        "out_of_corpus_refused": correct_refuse / len(EVAL["out_of_corpus"]),
        "in_corpus_wrongly_refused": wrong_refuse / len(EVAL["in_corpus"]),
    }


def main() -> None:
    retriever = HybridRetriever()
    print("Retrieval:", retrieval_metrics(retriever))
    pipeline = RAGPipeline(retriever=retriever)
    print("Refusal:", refusal_metrics(pipeline))


if __name__ == "__main__":
    main()
