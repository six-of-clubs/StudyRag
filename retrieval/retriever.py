"""
Vector retrieval.

Queries specific ChromaDB collections for candidate chunks. Only searches
collections explicitly passed in. This prevents cross-contamination.

TWO-STAGE RETRIEVAL
    This module is stage one: fast, approximate, and deliberately generous.
    It returns `settings.retrieval_candidates` chunks (default 30), not the
    final `settings.top_k` (default 5).

    Stage two is the cross-encoder in reranker.py, which re-scores those
    candidates accurately and cuts down to top_k. A reranker can only reorder
    what it is handed.

    Consequence: settings.similarity_threshold should be somewhat permissive, here (0.1).
    This acts only as a floor against obvious irrelevance. The real relevance decision
    belongs to the reranker, which is the better judge.

ChromaDB returns cosine *distances* (0 = identical, 2 = opposite). We convert
to similarity (1 - distance) so higher = better. This conversion is only valid
because every collection is created with hnsw:space=cosine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import vectorstore
from config import settings
from embedding import embed_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with its relevance score."""

    text: str
    metadata: dict = field(default_factory=dict)
    similarity: float = 0.0  # 0..1, higher is better
    # Set by the reranker. None means this chunk has not been reranked.
    rerank_score: float | None = None


def retrieve(query: str, collection_names: list[str]) -> list[RetrievedChunk]:
    """
    Args:
        query: the user's question
        collection_names: ChromaDB collection names to search. If empty,
                          returns nothing (the orchestrator will decline).

    Steps:
        1. Embed the query once (shared embedder, cached)
        2. Search each collection for nearest neighbours
        3. Convert distances to similarities, apply the permissive floor
        4. Merge, deduplicate, sort, take top `retrieval_candidates`

    Returns a candidate pool for the reranker.
    Returns an empty list if nothing passes the floor.
    """
    if not collection_names:
        logger.warning("No collections to search -- retriever has nothing to look at.")
        return []

    client = vectorstore.get_client()
    query_embedding = embed_query(query)
    n_candidates = settings.retrieval_candidates

    all_chunks: list[RetrievedChunk] = []

    for col_name in collection_names:
        try:
            collection = client.get_collection(col_name)
        except Exception:
            logger.warning("Collection '%s' not found -- skipping.", col_name)
            continue

        count = collection.count()
        if count == 0:
            logger.debug("Collection '%s' is empty -- skipping.", col_name)
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_candidates, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - dist

            if similarity < settings.similarity_threshold:
                logger.debug(
                    "Below floor (sim=%.3f < %.3f): %s p.%s",
                    similarity, settings.similarity_threshold,
                    meta.get("source_file", "?"), meta.get("page_number", "?"),
                )
                continue

            all_chunks.append(RetrievedChunk(
                text=text,
                metadata=meta,
                similarity=similarity,
            ))

    # Deduplicate by text content (the same chunk can appear in both a folder
    # collection and a chat-temp collection)
    seen: set[int] = set()
    unique: list[RetrievedChunk] = []
    for chunk in all_chunks:
        h = hash(chunk.text[:200])
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    unique.sort(key=lambda c: c.similarity, reverse=True)
    candidates = unique[:n_candidates]

    logger.info(
        "Retrieved %d candidate(s) from %d collection(s) [floor=%.2f, pool=%d] for: '%s'",
        len(candidates), len(collection_names),
        settings.similarity_threshold, n_candidates, query[:80],
    )
    return candidates
