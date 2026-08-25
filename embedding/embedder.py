"""
Shared embedding layer.

This is the ONLY place in the codebase that loads the sentence-transformers
model. Both the write path (api/state.py, ingesting documents) and the read
path (retrieval/retriever.py, embedding queries) import from here.

Why this matters:
    Query vectors and chunk vectors must live in the same vector space.
    If indexing and retrieval used separately-configured models -- different
    checkpoints, different normalisation, different instruction prefixes --
    cosine similarity between them would be meaningless, and retrieval would
    degrade silently with no error anywhere. One model object, one code path,
    one place to change.

The model is lazy-loaded and cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

# How many distinct query embeddings to keep in memory. Queries repeat more
# than you would expect (re-asks, refreshes, eval harness runs), and embedding
# is pure -- same string always yields the same vector.
_QUERY_CACHE_SIZE = 256


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' ...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts (document chunks). Used by the ingestion path.

    Returns one vector per input text, in the same order.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


@lru_cache(maxsize=_QUERY_CACHE_SIZE)
def _embed_query_cached(query: str) -> tuple[float, ...]:
    """
    Cached inner function. Returns a tuple because lru_cache hands back the
    same object on every hit -- an immutable one cannot be mutated by a caller
    and corrupt the cache for everyone else.
    """
    model = get_model()
    vector = model.encode(query, show_progress_bar=False)
    return tuple(vector.tolist())


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string. Used by the retrieval path.

    Repeated queries are served from an in-memory cache.
    """
    return list(_embed_query_cached(query))


def clear_query_cache() -> None:
    """Drop cached query vectors. Call this if the embedding model changes."""
    _embed_query_cached.cache_clear()
