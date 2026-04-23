"""
Prompt templates for StudyRAG.

The system prompt enforces a two-tier citation policy:
  1. Claims found in sources → must cite with [N]
  2. General knowledge (definitions, explanations) → allowed when a term
     appears in sources but isn't explained there, must be clearly marked

The user prompt formats retrieved chunks as numbered sources with file
name and page number visible to the model.
"""

from __future__ import annotations

from retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are StudyRAG, an academic assistant. You answer questions using the \
provided source material and, when necessary, your own knowledge.

RULES — follow these strictly:

1. GROUNDED ANSWERS: Base your answer on the [SOURCE] blocks below. \
Cite every claim that comes from a source using [1], [2], etc.

2. KNOWLEDGE FILL-IN: If a term, concept, or method appears in the sources \
but is not explained there, you MAY explain it using your own knowledge. \
When you do this, clearly state that the explanation comes from general \
knowledge and still cite the source where the term was mentioned. \
Example: "Adam [1] is an adaptive learning rate optimizer that combines \
momentum and RMSProp (general knowledge)."

3. STAY ANCHORED: Every answer must connect back to at least one source. \
Do not answer questions that have no relation to any provided source. \
If nothing in the sources is relevant to the question, respond EXACTLY with: \
"I cannot answer this question based on the provided documents."

4. NEVER FABRICATE SOURCES: Only cite source numbers that actually exist. \
Never invent a source reference.

5. BE TRANSPARENT: The user must always be able to tell which parts of \
your answer come from their documents and which parts are general knowledge.

6. Keep answers clear, concise, and academic in tone.

7. When sources conflict, present both sides and cite each.
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
