"""Cross-encoder reranker. A bi-encoder (embeddings) is fast but coarse; a
cross-encoder jointly reads (query, passage) and yields a far sharper relevance
score. Crucially, that calibrated score is what powers the refusal decision:
if even the best passage scores low, the answer isn't in the corpus."""
from __future__ import annotations

from .config import settings
from .schemas import Chunk, ScoredChunk

_model = None


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._ce = None

    def _get(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self.model_name)
        return self._ce

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        ce = self._get()
        pairs = [(query, c.text) for c in chunks]
        raw_scores = ce.predict(pairs)
        scored = [
            ScoredChunk(chunk=c, score=_sigmoid(float(s)), debug={"raw": float(s)})
            for c, s in zip(chunks, raw_scores)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]
