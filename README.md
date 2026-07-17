# Compliance & Regulatory Q&A Assistant

> A grounded, **citation-first** Retrieval-Augmented Generation (RAG) system for regulated domains. It answers questions over a corpus of compliance/regulatory PDFs with **cited passages** — *"per §4.2, page 2"* — and **refuses to answer when the information isn't in the corpus** instead of hallucinating.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Retrieval](https://img.shields.io/badge/retrieval-FAISS%20%2B%20BM25%20hybrid-4f7cff)
![Tests](https://img.shields.io/badge/tests-12%20passing-2ecc9b)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Runs **fully local — no API keys**: sentence-transformers embeddings, a FAISS dense index, BM25 lexical search, a cross-encoder reranker, and a Flan-T5 generator.

---

## Why this project

Regulated industries (fintech, banking, consulting) live and die on *"show me the source."* The differentiator here isn't retrieval alone — it's **grounded citations plus a hard no-answer path**. The system would rather say *"not in the corpus"* than fabricate an unsupported compliance statement.

It demonstrates hands-on handling of the pieces that matter to reviewers: document **chunking**, a **vector store (FAISS)**, **hybrid retrieval**, **reranking**, and **retrieval evaluation**.

---

## How it works

```
PDFs ─► ingest ─► section-aware chunking ─┐
                                          ├─► FAISS dense index ─┐
                                          └─► BM25 lexical index ─┤
                                                                  ▼
query ─────────────────────────────► Reciprocal Rank Fusion (RRF)
                                                                  ▼
                                              cross-encoder reranker
                                                                  ▼
                              ┌─── top score < threshold ? ──► REFUSE (gate 1)
                              ▼
                    Flan-T5 grounded generation
                              ▼
                    emits INSUFFICIENT_CONTEXT ? ──► REFUSE (gate 2)
                              ▼
                    answer + [n] citations → source §/page
```

**Two-gate refusal.** Gate 1 is retrieval confidence (the calibrated cross-encoder score); Gate 2 lets the generator itself declare the context insufficient. Both must pass to return an answer.

**Why hybrid + RRF.** Dense embeddings catch paraphrase; BM25 catches exact terminology (statute numbers, defined terms, acronyms). They're fused with Reciprocal Rank Fusion, which needs only each retriever's *rank* — robust to the fact that cosine and BM25 scores live on different scales.

> A full walkthrough of every module lives in [`Compliance-RAG-Codebase-Explained.pdf`](./Compliance-RAG-Codebase-Explained.pdf).

---

## Quickstart

```bash
# 1. Install (CPU-only; first run also downloads ~500MB of models)
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the FAISS + BM25 indexes from the sample corpus in data/raw
python -m src.rag.cli build

# 3a. Ask from the CLI
python -m src.rag.cli ask "What triggers enhanced due diligence?"

# 3b. Or launch the API + web dashboard
uvicorn src.rag.api:app --reload
#     → open http://127.0.0.1:8000/   (dashboard)
#     → http://127.0.0.1:8000/docs    (interactive API)
```

### Example — grounded answer

```
╭─ Answer (top_score=0.94) ─────────────────────────────────────╮
│ Enhanced due diligence is required where the customer is a    │
│ Politically Exposed Person, is domiciled in a high-risk       │
│ jurisdiction, or conducts unusually large transactions [1].   │
╰───────────────────────────────────────────────────────────────╯
Sources:
  [1] AML Policy §4.1, page 2
```

### Example — refusal (out of corpus)

```
$ python -m src.rag.cli ask "What is the corporate tax rate in Ireland?"
╭─ Answer (top_score=0.02) ─────────────────────────────────────╮
│ I can't answer that from the provided corpus. ...             │
╰───────────────────────────────────────────────────────────────╯
```

---

## Web dashboard

A dependency-free single page (`web/index.html`) served by the API at its root. Start the server and open **http://127.0.0.1:8000/**. Type a question (or click an example), and see the grounded/refused badge, the retrieval confidence score, and each cited source with its §section and page.

---

## Repository structure

```
compliance-rag/
├── data/raw/                     # sample compliance PDFs (generated, original content)
├── src/rag/
│   ├── config.py                 # env-overridable settings (models, top-k, refusal threshold)
│   ├── schemas.py                # Chunk / ScoredChunk / Citation / Answer
│   ├── chunking.py               # section-aware chunking (§ + page metadata)
│   ├── ingest.py                 # PDF → chunks
│   ├── embeddings.py             # sentence-transformers (local)
│   ├── vector_store.py           # FAISS index + metadata (persisted)
│   ├── bm25_store.py             # BM25 lexical index
│   ├── reranker.py               # cross-encoder reranker
│   ├── hybrid_retriever.py       # dense + BM25 + RRF + rerank
│   ├── llm.py                    # Flan-T5 grounded generation
│   ├── citations.py              # citation markers & filtering
│   ├── pipeline.py               # orchestration + two-gate refusal
│   ├── api.py                    # FastAPI service (serves dashboard + /query)
│   └── cli.py                    # command-line interface
├── web/index.html                # single-file dashboard (served at /)
├── scripts/
│   ├── generate_sample_pdfs.py   # (re)build the demo corpus
│   └── build_index.py            # ingest + build indexes
├── eval/
│   ├── eval_set.json             # in-corpus + out-of-corpus questions
│   └── evaluate.py               # Hit@k, MRR, refusal accuracy
├── tests/                        # pytest: chunking, fusion, citations, refusal
├── requirements.txt
├── .env.example
└── README.md
```

> **Not committed** (see `.gitignore`): `.venv/`, `storage/` (rebuildable indexes), `__pycache__/`, and `.env`.

---

## Evaluation

```bash
python -m eval.evaluate
```

Reports **Hit@5** and **MRR** for retrieval against gold `(document, section)` labels, plus refusal accuracy on a held-out set of out-of-corpus questions. Extend `eval/eval_set.json` as you add documents.

## Tests

```bash
pytest -q
```

Covers section-aware chunking, RRF fusion, citation extraction, and both refusal gates — using fakes, so tests run **without downloading models**.

---

## Configuration

All settings live in `src/rag/config.py`, overridable via `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | dense embeddings |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | cross-encoder rerank |
| `GENERATOR_MODEL` | `flan-t5-base` | grounded answer generation |
| `REFUSAL_THRESHOLD` | `0.15` | min rerank score to answer (gate 1) |
| `RERANK_TOP_K` | `5` | passages fed to the generator |

Tune `REFUSAL_THRESHOLD` up for stricter grounding (more refusals, fewer wrong answers) or down for higher recall.

---

## Bring your own corpus

Drop any `.pdf` files into `data/raw/` and re-run `python -m src.rag.cli build`. Section citations work best with numbered headings (`4.2 Title`); otherwise chunks are cited by document and page.

The included corpus (`AML-Policy.pdf`, `KYC-Onboarding-Standard.pdf`, `Data-Retention-Privacy-Policy.pdf`) is **original illustrative content** modeled on the *structure* of real regulatory guidance — not copies of any real regulation — so citations resolve exactly for the demo.

---

## Production notes (swapping components)

- **Vector DB:** FAISS is used for a zero-infra local demo. `vector_store.py` is a thin adapter — swap for pgvector or a managed store by implementing `build` / `save` / `load` / `search`.
- **LLM:** replace `llm.generate()` with an API call (OpenAI/Anthropic) while keeping the strict grounding prompt and the `INSUFFICIENT_CONTEXT` sentinel.
- **Embeddings:** any sentence-transformers model, or an API embedder, via `embeddings.embed_texts()`.

---
