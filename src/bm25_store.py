"""Lexical retrieval with BM25 (rank-bm25). Complements dense retrieval by
catching exact terminology — statute numbers, defined terms, acronyms — that
embeddings often blur."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from .config import settings
from .schemas import Chunk

_BM25_FILE = "bm25.pkl"
_TOKEN_RE = re.compile(r"[A-Za-z0-9§\.]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self) -> None:
        self.bm25 = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def save(self, index_dir: Path | None = None) -> None:
        index_dir = Path(index_dir) if index_dir else settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / _BM25_FILE, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "BM25Store":
        index_dir = Path(index_dir) if index_dir else settings.index_dir
        path = index_dir / _BM25_FILE
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index in {index_dir}.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.bm25 = data["bm25"]
        store.chunks = data["chunks"]
        return store

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            out.append((self.chunks[i], float(scores[i])))
        return out
