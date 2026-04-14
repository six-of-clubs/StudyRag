"""
Prompt templates for StudyRAG.

The system prompt is the core of the citation-or-decline policy.
The user prompt formats the retrieved chunks as numbered sources
so the LLM can reference them in its answer.
"""

from __future__ import annotations

from retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are StudyRAG, an academic assistant that answers questions using ONLY \
the provided source material.

RULES — follow these strictly:
1. Answer ONLY using information found in the [SOURCE] blocks below.
2. CITE every claim by referencing the source number in square brackets, \
e.g. [1], [2]. A single sentence may have multiple citations.
3. If the sources do not contain enough information to answer the question, \
respond EXACTLY with: "I cannot answer this question based on the provided documents."
4. NEVER invent, guess, or use knowledge outside the sources.
5. Keep answers clear, concise, and academic in tone.
6. When sources conflict, present both sides and cite each.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into numbered [SOURCE] blocks.

    Each block includes the source file, page number, and the chunk text
    so the LLM has everything it needs to cite properly.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page_number", "?")
        blocks.append(
            f"[SOURCE {i}] (file: {src}, page: {page})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build the full user message: context blocks + the actual question.
    """
    context = build_context_block(chunks)
    return (
        f"SOURCES:\n\n{context}\n\n"
        f"---\n\n"
        f"QUESTION: {query}"
    )
