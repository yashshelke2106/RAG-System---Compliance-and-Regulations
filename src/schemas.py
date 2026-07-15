"""Shared data structures used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance for citation."""
    chunk_id: str
    text: str
    doc_id: str          # source filename (e.g., "AML-Policy.pdf")
    doc_title: str       # human title (e.g., "Anti-Money-Laundering Policy")
    section: str         # e.g., "§4.2" or "4.2 Customer Due Diligence"
    page: int            # 1-based page number in the source PDF

    def citation(self) -> str:
        """Human-readable source tag, e.g. 'AML Policy §4.2, page 3'."""
        sec = self.section if self.section.startswith("§") else f"§{self.section}"
        return f"{self.doc_title} {sec}, page {self.page}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Optional per-stage diagnostics (dense rank, bm25 rank, rerank score)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    marker: str          # e.g., "[1]"
    source: str          # e.g., "AML Policy §4.2, page 3"
    doc_id: str
    section: str
    page: int
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    top_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "top_score": self.top_score,
            "citations": [asdict(c) for c in self.citations],
        }
