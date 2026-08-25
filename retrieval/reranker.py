"""
Cross-encoder reranking:

A bi-encoder (the embedding model) turns the query and each chunk into vectors
independently, then compares them. A cross-encoder feeds the query and chunk
through the transformer together, so every query token can attend to every
chunk token. That is far more accurate, and far too slow to run over a whole
collection, which is why it runs only over the candidate pool that vector
search narrowed down.

The threshold here, not the retriever's similarity floor, is the real relevance
gate. If nothing clears it, the orchestrator declines rather than feeding the
LLM weak context and hoping the citation policy catches it.
"""

from __future__ import annotations

import logging
import math

from sentence_transformers import CrossEncoder

from config import settings
from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model '%s' ...", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("Reranker loaded.")
    return _reranker


def _to_similarity(score: float) -> float:
    """
    Map a raw cross-encoder logit (roughly -10..+10) into 0..1 for display.

    The UI shows a similarity bar and users expect a
    percentage. Filtering uses the raw score, not this.
    """
    return 1.0 / (1.0 + math.exp(-score))


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Re-score, re-sort, filter, and truncate the candidate pool.

    Args:
        query: the user's original question
        chunks: candidates from the retriever

    Returns:
        At most settings.top_k chunks, best first. Empty if none clear the
        threshold -- the signal for the orchestrator to decline.
    """
    if not chunks:
        return []

    model = _get_reranker()
    threshold = settings.reranker_threshold

    scores = model.predict([(query, chunk.text) for chunk in chunks])

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    kept: list[RetrievedChunk] = []
    for chunk in ranked:
        if chunk.rerank_score < threshold:
            continue
        # Overwrite the vector similarity with the cross-encoder's opinion,
        # since that is what the citation display should reflect.
        chunk.similarity = _to_similarity(chunk.rerank_score)
        kept.append(chunk)
        if len(kept) >= settings.top_k:
            break

    if logger.isEnabledFor(logging.DEBUG):
        for chunk in ranked:
            logger.debug(
                "  rerank %+.3f %s | %s p.%s",
                chunk.rerank_score,
                "keep" if chunk.rerank_score >= threshold else "drop",
                chunk.metadata.get("source_file", "?"),
                chunk.metadata.get("page_number", "?"),
            )

    above = sum(1 for c in ranked if c.rerank_score >= threshold)
    logger.info(
        "Reranked %d candidate(s) -> %d above threshold %.1f -> kept top %d for: '%s'",
        len(chunks), above, threshold, len(kept), query[:60],
    )

    if not kept and ranked:
        logger.info(
            "Best candidate scored %+.3f, below threshold %.1f -- declining.",
            ranked[0].rerank_score, threshold,
        )

    return kept
