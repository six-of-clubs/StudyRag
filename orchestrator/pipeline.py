"""
Orchestrator pipeline:

Single entry point for answering a question. Runs the full chain:

    query → resolve collections → retrieve → rerank
         → relevance gate → prompt → LLM → citation gate → [repair] → cite

THREE PLACES CAN DECLINE
    1. No collections in scope        -- no folder selected
    2. Nothing survives the reranker  -- question unrelated to the documents
    3. No usable citation             -- see the repair step below

THE REPAIR
    A 7B model regularly writes a correct, well-grounded answer and forgets
    the citation markers. Discarding the answer can waste a good generation 
    and shows the student a decline they did not deserve. So when citation 
    parsing comes back with reason="uncited", the pipeline makes one corrective 
    call asking for the same answer with markers added. If that also comes back 
    uncited, it declines for real.

    Set CITATION_REPAIR=false in .env to disable and fail fast instead.
"""

from __future__ import annotations

import logging


from config import MODEL_PRESETS, DEFAULT_MODE, settings
from retrieval.retriever import retrieve
from retrieval.reranker import rerank
from generation.llm import generate
from generation.prompts import SYSTEM_PROMPT, build_user_prompt, build_repair_prompt
from generation.citation import CitedResponse, parse_citations

logger = logging.getLogger(__name__)

CITATION_REPAIR = settings.citation_repair

_NO_FOLDER = (
    "I cannot answer this question because no source folder is selected. "
    "Please choose a source folder before asking a question."
)
_NO_CHUNKS = (
    "I cannot answer this question based on the provided documents. "
    "No sufficiently relevant passages were found in the selected sources."
)
_NO_CITATION = (
    "I cannot answer this question based on the provided documents. "
    "(An answer was generated but could not be traced to a specific source.)"
)


def ask(
    query: str,
    folder_id: str | None = None,
    chat_id: str | None = None,
    mode: str | None = None,
) -> CitedResponse:
    """
    Answer a question using the RAG pipeline, scoped to a folder and/or chat.

    Args:
        query: the user's question
        folder_id: source folder to search (the academic subject)
        chat_id: chat whose temporary documents should also be searched
        mode: model mode -- "fast", "thinking", or "math"

    Returns:
        A CitedResponse with the answer or a decline message, the resolved
        sources, and a declined flag.
    """
    query = query.strip()
    if not query:
        return CitedResponse(
            answer="Please provide a question.",
            sources=[], declined=True, reason="no_sources",
        )

    # --- Step 1: Resolve model ---
    mode = mode or DEFAULT_MODE
    model_name = MODEL_PRESETS.get(mode, MODEL_PRESETS[DEFAULT_MODE])
    logger.info("Using mode '%s' → model '%s'", mode, model_name)

    # --- Step 2: Resolve scope and domain context ---
    from api.state import state

    collection_names = state.get_collection_names(folder_id, chat_id)

    subject = None
    if folder_id:
        folder = state.get_folder(folder_id)
        if folder:
            subject = folder.name

    if not collection_names:
        logger.info("No collections in scope — declining to answer.")
        return CitedResponse(
            answer=_NO_FOLDER, sources=[], declined=True, reason="no_sources",
        )

    logger.info(
        "Searching %d collection(s): %s (subject: %s)",
        len(collection_names), collection_names, subject or "none",
    )

    # --- Step 3: Retrieve a wide candidate pool ---
    candidates = retrieve(query, collection_names)

    # --- Step 4: Rerank down to top_k ---
    chunks = rerank(query, candidates)

    # --- Step 5: Relevance gate (pre-LLM) ---
    if not chunks:
        logger.info("No relevant chunks found — declining to answer.")
        return CitedResponse(
            answer=_NO_CHUNKS, sources=[], declined=True, reason="no_sources",
        )

    # --- Step 6: Generate ---
    user_prompt = build_user_prompt(query, chunks, subject=subject)
    response_text = generate(SYSTEM_PROMPT, user_prompt, model_name=model_name)

    # --- Step 7: Citation gate (post-LLM) ---
    result = parse_citations(response_text, chunks)

    # --- Step 8: Repair, if the only problem was missing markers ---
    if result.reason == "uncited" and CITATION_REPAIR:
        logger.info("Answer was uncited — attempting one citation repair.")
        repair_prompt = build_repair_prompt(query, chunks, result.answer)
        try:
            repaired_text = generate(
                SYSTEM_PROMPT, repair_prompt, model_name=model_name,
            )
            repaired = parse_citations(repaired_text, chunks)
        except ConnectionError:
            logger.warning("Ollama unreachable during repair — keeping decline.")
            repaired = result

        if repaired.reason == "ok":
            logger.info("Citation repair succeeded (%d source(s)).",
                        len(repaired.sources))
            result = repaired
        else:
            logger.info("Citation repair failed (reason=%s) — declining.",
                        repaired.reason)
            result = CitedResponse(
                answer=_NO_CITATION, sources=[], declined=True, reason="uncited",
            )
    elif result.reason == "uncited":
        result = CitedResponse(
            answer=_NO_CITATION, sources=[], declined=True, reason="uncited",
        )

    logger.info(
        "Pipeline complete — mode=%s, subject=%s, reason=%s, sources=%d",
        mode, subject or "none", result.reason, len(result.sources),
    )
    return result
