"""Section-aware chunking.

Compliance documents are hierarchical (§1, §1.1, §4.2 ...). We keep chunks aligned
to section boundaries where possible so that every chunk carries a precise
"section + page" citation. Long sections are split with character overlap so no
single chunk exceeds the embedding model's comfortable context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Matches headings like "4.2 Customer Due Diligence" or "§4.2 ..." at line start.
SECTION_RE = re.compile(r"^\s*(?:§\s*)?(\d+(?:\.\d+)*)\s+(.+?)\s*$")


@dataclass
class PageText:
    page: int
    text: str


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of ~`size` chars, breaking on sentence/space
    boundaries when possible, keeping `overlap` chars of context between windows."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # try to break on a sentence end, else a space
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Turn a list of page texts into section-tagged chunks.

    Returns a list of dicts: {text, section, page}. The caller adds doc-level
    metadata and stable chunk ids.
    """
    # Walk pages line by line, tracking the current section heading and page.
    current_section = "Preamble"
    # Accumulate (section, page, buffer) segments.
    segments: list[tuple[str, int, str]] = []
    buf: list[str] = []
    buf_section = current_section
    buf_page = pages[0].page if pages else 1

    def flush():
        nonlocal buf, buf_section, buf_page
        joined = " ".join(buf).strip()
        if joined:
            segments.append((buf_section, buf_page, joined))
        buf = []

    for pg in pages:
        for line in pg.text.splitlines():
            m = SECTION_RE.match(line)
            if m and len(m.group(1)) <= 12:
                # New section boundary -> flush previous buffer.
                flush()
                current_section = f"{m.group(1)} {m.group(2)}".strip()
                buf_section = current_section
                buf_page = pg.page
                buf.append(line.strip())
            else:
                if not buf:
                    buf_section = current_section
                    buf_page = pg.page
                buf.append(line.strip())
    flush()

    out: list[dict] = []
    for section, page, seg_text in segments:
        for piece in _split_with_overlap(seg_text, chunk_size, chunk_overlap):
            # Store a compact section label: "§4.2" if numeric, else raw.
            num = section.split(" ", 1)[0]
            label = f"§{num}" if re.match(r"^\d+(\.\d+)*$", num) else section
            out.append({"text": piece, "section": label, "page": page})
    return out
