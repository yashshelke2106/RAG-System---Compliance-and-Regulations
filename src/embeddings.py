"""Local dense embeddings via sentence-transformers (lazy-loaded, cached)."""
from __future__ import annotations

import numpy as np

from .config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings so inner-product == cosine similarity."""
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return int(_get_model().get_sentence_embedding_dimension())
