"""
Orchestrator pipeline for StudyRAG.

This is the single entry point for answering a question. It runs
the full chain:

    query → resolve collections → retrieve → (rerank) → relevance gate → prompt → LLM → cite

Scoping rules:
    - folder_id determines which academic subject's collection is searched.
    - chat_id optionally adds the chat's temporary document collection.
    - If neither is provided, there is nothing to search → automatic decline.
    - The retriever NEVER sees collections outside the specified scope,
      so there is zero cross-contamination between subjects.

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


def ask(
    query: str,
    folder_id: str | None = None,
    chat_id: str | None = None,
) -> CitedResponse:
    """
    Answer a question using the RAG pipeline, scoped to a specific
    folder and/or chat.

    Args:
        query: the user's question
        folder_id: source folder to search (the academic subject)
        chat_id: chat whose temporary documents should also be searched

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

    # --- Step 1: Resolve scope to collection names ---
    from api.state import state

    collection_names = state.get_collection_names(folder_id, chat_id)

    if not collection_names:
        logger.info("No collections in scope — declining to answer.")
        return CitedResponse(
            answer=(
                "I cannot answer this question because no source folder is selected. "
                "Please choose a source folder before asking a question."
            ),
            sources=[],
            declined=True,
        )

    logger.info(
        "Searching %d collection(s): %s",
        len(collection_names), collection_names,
    )

    # --- Step 2: Retrieve (scoped) ---
    chunks = retrieve(query, collection_names)

    # --- Step 3: Rerank (no-op until Phase 2) ---
    chunks = rerank(query, chunks)

    # --- Step 4: Relevance gate (pre-LLM) ---
    if not chunks:
        logger.info("No relevant chunks found — declining to answer.")
        return CitedResponse(
            answer=(
                "I cannot answer this question based on the provided documents. "
                "No sufficiently relevant passages were found in the selected sources."
            ),
            sources=[],
            declined=True,
        )

    # --- Step 5: Build prompt and call LLM ---
    user_prompt = build_user_prompt(query, chunks)
    response_text = generate(SYSTEM_PROMPT, user_prompt)

    # --- Step 6: Parse citations (post-LLM gate) ---
    result = parse_citations(response_text, chunks)

    logger.info(
        "Pipeline complete — declined=%s, sources=%d, collections=%s",
        result.declined, len(result.sources), collection_names,
    )
    return result
