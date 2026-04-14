"""
Query embedder:
Uses the same sentence-transformers model as the ingestion pipeline
to embed user queries into the shared vector space.

The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load (== once) and cache the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' ...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string into a vector.

    Returns a list of floats (the embedding).
    """
    model = _get_model()
    embedding = model.encode(query, show_progress_bar=False)
    return embedding.tolist()
