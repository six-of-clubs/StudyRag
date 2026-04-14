"""
Retriever:

Queries the ChromaDB vector store for the most relevant chunks given a
user query. Applies the similarity threshold to filter out weak matches.

ChromaDB returns cosine distances (0 = identical, 2 = opposite).
We convert to similarity (1 - distance) so higher = better, and the
threshold in config works intuitively: 0.3 means "at least 30% similar".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chromadb

from config import settings
from retrieval.embedder import embed_query

logger = logging.getLogger(__name__)

COLLECTION_NAME = "studyrag"


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with its relevance score."""

    text: str
    metadata: dict = field(default_factory=dict)
    similarity: float = 0.0  # higher is better


def _get_collection() -> chromadb.Collection:
    """Open the persisted ChromaDB collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_collection(
        name=COLLECTION_NAME,
    )


def retrieve(query: str) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Steps:
        1. Embed the query
        2. Search ChromaDB for nearest neighbours
        3. Convert distances to similarities
        4. Filter by similarity threshold
        5. Return sorted by relevance (best first)

    Returns an empty list if nothing passes the threshold — this is the
    signal for the orchestrator to decline answering. ---> no source!
    """
    try:
        collection = _get_collection()
    except Exception as e:
        logger.error("Could not open vector store: %s", e)
        return []

    if collection.count() == 0:
        logger.warning("Vector store is empty. Ingest documents first.")
        return []

    # Embed and search
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(settings.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack results (ChromaDB returns lists-of-lists)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Convert cosine distance → similarity and filter
    chunks = []
    for text, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - dist  # cosine distance → cosine similarity  (absolute distance needed?)

        if similarity < settings.similarity_threshold:
            logger.debug(
                "Filtered out chunk (sim=%.3f < threshold=%.3f): %s p.%s",
                similarity, settings.similarity_threshold,
                meta.get("source_file", "?"), meta.get("page_number", "?"),
            )
            continue

        chunks.append(RetrievedChunk(
            text=text,
            metadata=meta,
            similarity=similarity,
        ))

    # Sort best first
    chunks.sort(key=lambda c: c.similarity, reverse=True)

    logger.info(
        "Retrieved %d chunk(s) above threshold (%.2f) for query: '%s'",
        len(chunks), settings.similarity_threshold, query[:80],
    )
    return chunks
