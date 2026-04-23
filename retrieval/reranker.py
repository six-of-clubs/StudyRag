"""
Reranker for StudyRAG.

Uses a cross-encoder model to re-score retrieved chunks against the
original query. This is far more accurate than the initial embedding
similarity because the cross-encoder sees the query and chunk *together*
rather than comparing independent embeddings.

Flow:
    1. Receive top-k chunks from the retriever (loosely filtered)
    2. Score each (query, chunk) pair with the cross-encoder
    3. Sort by cross-encoder score (higher = more relevant)
    4. Filter out chunks below the reranker threshold
    5. Return the re-ranked list

The cross-encoder is ~90MB and runs on CPU in milliseconds per pair,
so reranking 5-10 chunks adds negligible latency.
"""

from __future__ import annotations

import logging
from sentence_transformers import CrossEncoder

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Cross-encoder model — loaded once, cached for the process lifetime
_reranker: CrossEncoder | None = None

# Threshold for the cross-encoder score. Scores are roughly -10 to +10,
# with positive meaning relevant. 0.0 is a reasonable starting point;
# tune based on your documents.
RERANKER_THRESHOLD = -2.0

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model '%s' ...", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded.")
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Re-score and re-sort chunks using a cross-encoder model.

    Args:
        query: the user's original question
        chunks: chunks from the retriever, already filtered by vector
                similarity threshold

    Returns:
        Re-ranked list of chunks, sorted by cross-encoder score,
        with low-scoring chunks removed.
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        # No point reranking a single chunk, but still score it
        model = _get_reranker()
        score = model.predict([(query, chunks[0].text)])[0]
        logger.debug("Single chunk reranker score: %.3f", score)
        if score < RERANKER_THRESHOLD:
            logger.info("Single chunk below reranker threshold (%.3f < %.3f)",
                        score, RERANKER_THRESHOLD)
            return []
        return chunks

    model = _get_reranker()

    # Build (query, chunk_text) pairs
    pairs = [(query, chunk.text) for chunk in chunks]

    # Score all pairs in one batch
    scores = model.predict(pairs)

    # Attach scores and sort
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Log all scores for debugging
    for chunk, score in scored:
        logger.debug(
            "  reranker: %.3f | %s p.%s | sim=%.3f",
            score,
            chunk.metadata.get("source_file", "?"),
            chunk.metadata.get("page_number", "?"),
            chunk.similarity,
        )

    # Filter by threshold
    reranked = []
    for chunk, score in scored:
        if score >= RERANKER_THRESHOLD:
            # Update the similarity to reflect the reranker's opinion
            # Normalize cross-encoder score to 0..1 range for display
            # ms-marco scores are roughly -10 to +10, sigmoid maps to 0..1
            import math
            normalized = 1.0 / (1.0 + math.exp(-score))
            chunk.similarity = normalized
            reranked.append(chunk)
        else:
            logger.debug(
                "  filtered by reranker: %.3f < %.3f | %s p.%s",
                score, RERANKER_THRESHOLD,
                chunk.metadata.get("source_file", "?"),
                chunk.metadata.get("page_number", "?"),
            )

    logger.info(
        "Reranked %d → %d chunks (threshold=%.1f) for: '%s'",
        len(chunks), len(reranked), RERANKER_THRESHOLD, query[:60],
    )
    return reranked
