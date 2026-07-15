"""End-to-end RAG orchestration with a two-gate refusal policy.

Gate 1 (retrieval): if the best cross-encoder score < REFUSAL_THRESHOLD, the
corpus almost certainly does not contain the answer -> refuse without calling
the generator.

Gate 2 (generation): if the model itself emits INSUFFICIENT_CONTEXT, we refuse
even though retrieval passed. Both gates must be clear to return an answer.

This is the core value proposition for regulated users: the system would rather
say "not in the corpus" than fabricate an unsupported compliance statement.
"""
from __future__ import annotations

from .config import settings
from .citations import build_citations, filter_to_used
from .hybrid_retriever import HybridRetriever
from .llm import INSUFFICIENT, generate
from .schemas import Answer

REFUSAL_MESSAGE = (
    "I can't answer that from the provided corpus. No document section in the "
    "knowledge base sufficiently addresses this question, so I won't guess. "
    "Please add a relevant source or rephrase the question."
)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        scored = self.retriever.retrieve(question, top_k=top_k)
        top_score = scored[0].score if scored else 0.0

        # Gate 1: retrieval confidence.
        if not scored or top_score < settings.refusal_threshold:
            return Answer(
                question=question,
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                top_score=top_score,
            )

        # Generate grounded answer.
        raw = generate(question, scored)

        # Gate 2: model-declared insufficiency.
        if INSUFFICIENT in raw or not raw.strip():
            return Answer(
                question=question,
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                top_score=top_score,
            )

        citations = filter_to_used(raw, build_citations(scored))
        return Answer(
            question=question,
            answer=raw,
            refused=False,
            citations=citations,
            top_score=top_score,
        )
