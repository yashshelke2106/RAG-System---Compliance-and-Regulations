"""Convenience wrapper: build indexes from the default data dir.

    python -m scripts.build_index
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingest import ingest_dir            # noqa: E402
from src.rag.vector_store import FaissStore      # noqa: E402
from src.rag.bm25_store import BM25Store         # noqa: E402


def main() -> None:
    chunks = ingest_dir()
    print(f"Ingested {len(chunks)} chunks")

    fs = FaissStore()
    fs.build(chunks)
    fs.save()

    bs = BM25Store()
    bs.build(chunks)
    bs.save()
    print("Indexes built.")


if __name__ == "__main__":
    main()
