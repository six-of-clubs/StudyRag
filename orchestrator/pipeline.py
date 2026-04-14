"""
Orchestrator pipeline for StudyRAG.

This is the single entry point for answering a question. It runs
the full chain:

    query → retrieve → (rerank) → relevance gate → prompt → LLM → cite

The cite-or-decline policy is enforced at two levels:
    1. PRE-LLM:  if no chunks pass the similarity threshold, decline
                  immediately without wasting an LLM call.
    2. POST-LLM: if the model's response contains no citations,
                  treat it as a decline (the model hallucinated).
"""

from __future__ import annotations

import logging

from retrieval.retriever import retrieve
from retrieval.reranker import rerank
from generation.llm import generate
from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from generation.citation import CitedResponse, parse_citations

logger = logging.getLogger(__name__)


def ask(query: str) -> CitedResponse:
    """
    Answer a question using the RAG pipeline.

    Args:
        query: the user's question

    Returns:
        A CitedResponse containing the answer (or a decline message),
        the list of source references, and a declined flag.
    """
    query = query.strip()
    if not query:
        return CitedResponse(
            answer="Please provide a question.",
            sources=[],
            declined=True,
        )

    # --- Step 1: Retrieve ---
    chunks = retrieve(query)

    # --- Step 2: Rerank (no-op until Phase 2) ---
    chunks = rerank(query, chunks)

    # --- Step 3: Relevance gate (pre-LLM) ---
    if not chunks:
        logger.info("No relevant chunks found — declining to answer.")
        return CitedResponse(
            answer=(
                "I cannot answer this question based on the provided documents. "
                "No sufficiently relevant passages were found."
            ),
            sources=[],
            declined=True,
        )

    # --- Step 4: Build prompt and call LLM ---
    user_prompt = build_user_prompt(query, chunks)
    response_text = generate(SYSTEM_PROMPT, user_prompt)

    # --- Step 5: Parse citations (post-LLM gate) ---
    result = parse_citations(response_text, chunks)

    logger.info(
        "Pipeline complete — declined=%s, sources=%d",
        result.declined, len(result.sources),
    )
    return result
