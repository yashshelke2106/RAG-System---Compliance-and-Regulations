"""Command-line interface: ingest/build the index and query it.

    python -m src.rag.cli build
    python -m src.rag.cli ask "What triggers enhanced due diligence?"
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .config import settings

app = typer.Typer(add_completion=False, help="Compliance RAG assistant CLI.")
console = Console()


@app.command()
def build(data_dir: str = typer.Option(None, help="Folder of PDFs to ingest.")) -> None:
    """Ingest PDFs and build the FAISS + BM25 indexes."""
    from pathlib import Path
    from .ingest import ingest_dir
    from .vector_store import FaissStore
    from .bm25_store import BM25Store

    dd = Path(data_dir) if data_dir else settings.data_dir
    console.print(f"[bold]Ingesting[/bold] PDFs from {dd} ...")
    chunks = ingest_dir(dd)
    console.print(f"  -> {len(chunks)} chunks")

    console.print("Building FAISS dense index ...")
    fs = FaissStore()
    fs.build(chunks)
    fs.save()

    console.print("Building BM25 lexical index ...")
    bs = BM25Store()
    bs.build(chunks)
    bs.save()

    console.print(f"[green]Done.[/green] Indexes saved to {settings.index_dir}")


@app.command()
def ask(question: str, top_k: int = typer.Option(None, help="Passages to rerank.")) -> None:
    """Ask a question against the built corpus."""
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    ans = pipeline.answer(question, top_k=top_k)

    style = "yellow" if ans.refused else "green"
    console.print(Panel(ans.answer, title=f"Answer (top_score={ans.top_score:.3f})", border_style=style))
    if ans.citations:
        console.print("[bold]Sources:[/bold]")
        for c in ans.citations:
            console.print(f"  {c.marker} {c.source}")
            console.print(f"      [dim]{c.snippet}[/dim]")


if __name__ == "__main__":
    app()
