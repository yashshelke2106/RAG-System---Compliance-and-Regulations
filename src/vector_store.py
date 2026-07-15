"""FAISS dense index + parallel chunk metadata store, persisted to disk.

We use an inner-product index on L2-normalized vectors (== cosine similarity).
Metadata (the Chunk objects) is pickled alongside the index so the same integer
ids returned by FAISS map back to source passages for citation.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import settings
from .embeddings import embed_texts, embed_query
from .schemas import Chunk

_INDEX_FILE = "faiss.index"
_META_FILE = "chunks.pkl"


class FaissStore:
    def __init__(self) -> None:
        self.index = None
        self.chunks: list[Chunk] = []

    # ---- build / persist ------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        import faiss

        self.chunks = chunks
        vectors = embed_texts([c.text for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self.index = index

    def save(self, index_dir: Path | None = None) -> None:
        import faiss

        index_dir = Path(index_dir) if index_dir else settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / _INDEX_FILE))
        with open(index_dir / _META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "FaissStore":
        import faiss

        index_dir = Path(index_dir) if index_dir else settings.index_dir
        idx_path = index_dir / _INDEX_FILE
        meta_path = index_dir / _META_FILE
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run `python -m scripts.build_index` first."
            )
        store = cls()
        store.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            store.chunks = pickle.load(f)
        return store

    # ---- query ----------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        q = embed_query(query).reshape(1, -1)
        scores, ids = self.index.search(q, min(top_k, len(self.chunks)))
        results: list[tuple[Chunk, float]] = []
        for cid, score in zip(ids[0], scores[0]):
            if cid == -1:
                continue
            results.append((self.chunks[int(cid)], float(score)))
        return results
