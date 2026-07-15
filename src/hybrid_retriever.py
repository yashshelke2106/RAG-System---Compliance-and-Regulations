"""Hybrid retrieval = dense (FAISS) + lexical (BM25) fused with Reciprocal Rank
Fusion (RRF), then re-ranked with a cross-encoder.

RRF is used instead of raw-score blending because dense cosine scores and BM25
scores live on different scales; RRF only needs the *rank* from each retriever,
which makes fusion robust and parameter-light.
"""
from __future__ import annotations

from .config import settings
from .bm25_store import BM25Store
from .vector_store import FaissStore
from .reranker import CrossEncoderReranker
from .schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    dense: list[tuple[Chunk, float]],
    lexical: list[tuple[Chunk, float]],
    k: int,
    top_n: int,
) -> list[Chunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    dense_rank: dict[str, int] = {}
    lex_rank: dict[str, int] = {}

    for rank, (chunk, _) in enumerate(dense):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk
        dense_rank[chunk.chunk_id] = rank
    for rank, (chunk, _) in enumerate(lexical):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk
        lex_rank[chunk.chunk_id] = rank

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_n]
    return [by_id[cid] for cid in ranked_ids]


class HybridRetriever:
    def __init__(
        self,
        faiss_store: FaissStore | None = None,
        bm25_store: BM25Store | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.faiss = faiss_store or FaissStore.load()
        self.bm25 = bm25_store or BM25Store.load()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.rerank_top_k

        dense = self.faiss.search(query, settings.dense_top_k)
        lexical = self.bm25.search(query, settings.bm25_top_k)

        fused = reciprocal_rank_fusion(
            dense, lexical, k=settings.rrf_k, top_n=settings.fusion_top_k
        )
        if not fused:
            return []

        # Cross-encoder rerank the fused candidate pool.
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        return reranked
