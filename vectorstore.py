"""
Shared ChromaDB client.

One PersistentClient for the whole process, pointed at settings.chroma_persist_dir.
Both the state manager (writes) and the retriever (reads) get their client from
here rather than constructing their own.

Also centralises the collection configuration. Every collection in this project
must use cosine distance, because retriever.py converts Chroma's distance to a
similarity with `1 - distance`, which is only correct for cosine.
"""

from __future__ import annotations

import logging

import chromadb

from config import settings

logger = logging.getLogger(__name__)

# Every collection is created with cosine space. Note that get_or_create_collection
# IGNORES this metadata if the collection already exists -- a collection created
# before this setting was introduced stays on its original metric forever, and
# nothing will warn you. See verify_collection_space() below.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """Lazy-load and cache the ChromaDB client."""
    global _client
    if _client is None:
        logger.info("Opening ChromaDB at '%s'", settings.chroma_persist_dir)
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_or_create(name: str):
    """Get a collection, creating it with cosine space if it does not exist."""
    return get_client().get_or_create_collection(
        name=name, metadata=COLLECTION_METADATA,
    )


def verify_collection_spaces() -> list[str]:
    """
    Check every existing collection actually uses cosine distance.

    Returns a list of collection names that do NOT. Any collection in that list
    was created before cosine was configured and will produce wrong similarity
    scores -- it needs to be deleted and re-ingested.
    """
    bad = []
    for col in get_client().list_collections():
        meta = col.metadata or {}
        if meta.get("hnsw:space") != "cosine":
            bad.append(col.name)
            logger.warning(
                "Collection '%s' uses space=%s, expected cosine. "
                "Similarity scores from it are wrong -- re-ingest this collection.",
                col.name, meta.get("hnsw:space", "default (l2)"),
            )
    return bad
