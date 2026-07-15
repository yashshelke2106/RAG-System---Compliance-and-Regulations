"""Compliance & Regulatory Q&A Assistant — a grounded, citation-first RAG system.

Pipeline: ingest -> chunk -> hybrid retrieve (BM25 + dense/FAISS)
          -> cross-encoder rerank -> grounded generate OR refuse.
"""

__version__ = "0.1.0"
