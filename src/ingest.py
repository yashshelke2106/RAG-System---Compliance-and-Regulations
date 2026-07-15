"""PDF ingestion: read PDFs -> per-page text -> section-aware chunks with metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .chunking import PageText, chunk_pages
from .config import settings
from .schemas import Chunk


def _doc_title(doc_id: str) -> str:
    """Turn 'AML-Policy.pdf' into 'AML Policy'."""
    stem = Path(doc_id).stem
    return stem.replace("-", " ").replace("_", " ").strip()


def read_pdf_pages(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page=i, text=text))
    return pages


def chunk_id_for(doc_id: str, page: int, section: str, idx: int) -> str:
    raw = f"{doc_id}|{page}|{section}|{idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def ingest_pdf(path: Path) -> list[Chunk]:
    doc_id = path.name
    title = _doc_title(doc_id)
    pages = read_pdf_pages(path)
    raw_chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    chunks: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(doc_id, rc["page"], rc["section"], idx),
                text=rc["text"],
                doc_id=doc_id,
                doc_title=title,
                section=rc["section"],
                page=rc["page"],
            )
        )
    return chunks


def ingest_dir(data_dir: Path | None = None) -> list[Chunk]:
    data_dir = Path(data_dir) if data_dir else settings.data_dir
    all_chunks: list[Chunk] = []
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    for pdf in pdfs:
        all_chunks.extend(ingest_pdf(pdf))
    return all_chunks
