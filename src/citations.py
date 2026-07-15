"""Attach citation markers to retrieved passages and extract which markers the
generated answer actually used."""
from __future__ import annotations

import re

from .schemas import Citation, ScoredChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")


def build_citations(scored: list[ScoredChunk]) -> list[Citation]:
    cites: list[Citation] = []
    for i, sc in enumerate(scored, start=1):
        c = sc.chunk
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        cites.append(
            Citation(
                marker=f"[{i}]",
                source=c.citation(),
                doc_id=c.doc_id,
                section=c.section,
                page=c.page,
                snippet=snippet,
            )
        )
    return cites


def used_markers(answer: str) -> set[int]:
    return {int(m) for m in _MARKER_RE.findall(answer)}


def filter_to_used(answer: str, citations: list[Citation]) -> list[Citation]:
    """Keep only the citations the answer actually references; if the model cited
    nothing, fall back to returning all provided sources (still grounded)."""
    used = used_markers(answer)
    if not used:
        return citations
    return [c for c in citations if int(c.marker.strip("[]")) in used]
