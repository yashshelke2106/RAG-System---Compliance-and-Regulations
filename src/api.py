"""FastAPI service exposing the RAG pipeline. Run:

    uvicorn src.rag.api:app --reload

Then open http://127.0.0.1:8000/ for the dashboard, or /docs for the API.

The pipeline (and its models) is loaded once at startup and reused across
requests. If no index exists, /query returns a clear 503 telling you to build.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

# Path to the single-file dashboard (project_root/web/index.html)
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="Compliance & Regulatory Q&A Assistant",
    version="0.1.0",
    description="Grounded, citation-first RAG over a compliance corpus. Refuses "
    "when the answer is not in the documents.",
)

# Convenience CORS fallback so the dashboard also works if opened from a
# different origin. Same-origin (served at /) needs no CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What triggers enhanced due diligence?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    marker: str
    source: str
    doc_id: str
    section: str
    page: int
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    top_score: float
    citations: list[CitationOut]


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _pipeline


@app.on_event("startup")
def _warmup() -> None:
    try:
        get_pipeline()
    except HTTPException:
        # Index not built yet; surfaced per-request instead of crashing startup.
        pass


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the Q&A dashboard at the API root."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse(
        {"detail": "Dashboard not found. Expected web/index.html.", "docs": "/docs"},
        status_code=404,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    ans = pipeline.answer(req.question, top_k=req.top_k)
    return QueryResponse(**ans.to_dict())
