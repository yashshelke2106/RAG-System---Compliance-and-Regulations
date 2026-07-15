"""Central configuration. All values overridable via environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


# Project root = two levels up from this file (src/rag/config.py -> project root)
ROOT = Path(__file__).resolve().parents[2]
_load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # Models (Hugging Face IDs; downloaded & cached on first use)
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    generator_model: str = os.environ.get("GENERATOR_MODEL", "google/flan-t5-base")

    # Chunking
    chunk_size: int = _int("CHUNK_SIZE", 900)      # characters
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)

    # Retrieval
    dense_top_k: int = _int("DENSE_TOP_K", 20)
    bm25_top_k: int = _int("BM25_TOP_K", 20)
    fusion_top_k: int = _int("FUSION_TOP_K", 20)
    rerank_top_k: int = _int("RERANK_TOP_K", 5)
    rrf_k: int = _int("RRF_K", 60)                 # Reciprocal Rank Fusion constant

    # Refusal: if the best reranker score is below this, we DO NOT answer.
    refusal_threshold: float = _float("REFUSAL_THRESHOLD", 0.15)

    # Paths
    data_dir: Path = field(default_factory=lambda: ROOT / os.environ.get("DATA_DIR", "data/raw"))
    index_dir: Path = field(default_factory=lambda: ROOT / os.environ.get("INDEX_DIR", "storage"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.index_dir = Path(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
