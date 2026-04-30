"""
Orchestrator pipeline:

This is the single entry point for answering a question. It runs
the full chain:

    query → resolve collections → retrieve → rerank → relevance gate
         → build prompt (with domain context) → LLM → cite

Domain context:
    The folder name (e.g. "Foundation Models", "Linear Algebra") is
    injected into the prompt so the LLM interprets terms within the
    correct academic field.

Model modes:
    - "fast"     → mistral:7b      (quick answers)
    - "thinking" → deepseek-r1:8b  (step-by-step reasoning)
    - "math"     → phi4-mini       (proofs, equations, calculations)
"""

from __future__ import annotations

import logging

from config import MODEL_PRESETS, DEFAULT_MODE
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
    mode: str | None = None,
) -> CitedResponse:
    """
    Answer a question using the RAG pipeline, scoped to a specific
    folder and/or chat.

    Args:
        query: the user's question
        folder_id: source folder to search (the academic subject)
        chat_id: chat whose temporary documents should also be searched
        mode: model mode — "fast", "thinking", or "math"

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

    # --- Step 1: Resolve model ---
    mode = mode or DEFAULT_MODE
    model_name = MODEL_PRESETS.get(mode, MODEL_PRESETS[DEFAULT_MODE])
    logger.info("Using mode '%s' → model '%s'", mode, model_name)

    # --- Step 2: Resolve scope and domain context ---
    from api.state import state

    collection_names = state.get_collection_names(folder_id, chat_id)

    # Resolve folder name for domain context
    subject = None
    if folder_id:
        folder = state.get_folder(folder_id)
        if folder:
            subject = folder.name

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
        "Searching %d collection(s): %s (subject: %s)",
        len(collection_names), collection_names, subject or "none",
    )

    # --- Step 3: Retrieve (scoped) ---
    chunks = retrieve(query, collection_names)

    # --- Step 4: Rerank ---
    chunks = rerank(query, chunks)

    # --- Step 5: Relevance gate (pre-LLM) ---
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

    # --- Step 6: Build prompt with domain context and call LLM ---
    user_prompt = build_user_prompt(query, chunks, subject=subject)
    response_text = generate(SYSTEM_PROMPT, user_prompt, model_name=model_name)

    # --- Step 7: Parse citations (post-LLM gate) ---
    result = parse_citations(response_text, chunks)

    logger.info(
        "Pipeline complete — mode=%s, subject=%s, declined=%s, sources=%d",
        mode, subject or "none", result.declined, len(result.sources),
    )
    return result
