"""
Queries specific ChromaDB collections for the most relevant chunks given
a user query. Only searches collections explicitly passed in — there is
no global fallback. This prevents cross-contamination between subjects.

ChromaDB returns cosine *distances* (0 = identical, 2 = opposite).
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


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with its relevance score."""

    text: str
    metadata: dict = field(default_factory=dict)
    similarity: float = 0.0  # 0..1, higher is better


def retrieve(query: str, collection_names: list[str]) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query, searching ONLY
    the specified collections.

    Args:
        query: the user's question
        collection_names: ChromaDB collection names to search. If empty,
                          returns nothing (the orchestrator will decline).

    Steps:
        1. Embed the query once
        2. Search each collection for nearest neighbours
        3. Merge, convert distances to similarities
        4. Filter by similarity threshold
        5. Deduplicate, sort by relevance, take top-k

    Returns an empty list if nothing passes the threshold — this is the
    signal for the orchestrator to decline answering.
    """
    if not collection_names:
        logger.warning("No collections to search — retriever has nothing to look at.")
        return []

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    query_embedding = embed_query(query)

    all_chunks: list[RetrievedChunk] = []

    for col_name in collection_names:
        try:
            collection = client.get_collection(col_name)
        except Exception:
            logger.warning("Collection '%s' not found — skipping.", col_name)
            continue

        if collection.count() == 0:
            logger.debug("Collection '%s' is empty — skipping.", col_name)
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(settings.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - dist

            if similarity < settings.similarity_threshold:
                logger.debug(
                    "Filtered out chunk (sim=%.3f < threshold=%.3f): %s p.%s",
                    similarity, settings.similarity_threshold,
                    meta.get("source_file", "?"), meta.get("page_number", "?"),
                )
                continue

            all_chunks.append(RetrievedChunk(
                text=text,
                metadata=meta,
                similarity=similarity,
            ))

    # Deduplicate by text content (same chunk could theoretically appear
    # in both a folder and a chat-temp collection)
    seen = set()
    unique: list[RetrievedChunk] = []
    for chunk in all_chunks:
        h = hash(chunk.text[:200])
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    # Sort best first, take top-k
    unique.sort(key=lambda c: c.similarity, reverse=True)
    result = unique[: settings.top_k]

    logger.info(
        "Retrieved %d chunk(s) above threshold (%.2f) from %d collection(s) for: '%s'",
        len(result), settings.similarity_threshold, len(collection_names), query[:80],
    )
    return result
