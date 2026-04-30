"""
Design principles:
  - ANSWER FIRST: Lead with the explanation, cite at the end.
  - AUTHORITATIVE: When a known concept appears in sources,
    explain it confidently using general knowledge.
  - DOMAIN-AWARE: The folder name is injected as subject context
    so the model interprets terms within the right field.
  - GROUNDED: Every answer must connect to at least one source.
"""

from __future__ import annotations

from retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are StudyRAG, an expert academic tutor. You help students understand \
their course material by answering questions about their lecture slides, \
textbooks, and exercises.

HOW TO ANSWER:

1. LEAD WITH THE ANSWER. Start by directly explaining the concept the \
student is asking about. Never open with disclaimers about what the \
sources do or do not contain.

2. BE AUTHORITATIVE. When a well-known academic concept appears in the \
sources, explain it confidently and correctly using your knowledge. \
You are an expert tutor — students expect clear, correct explanations, \
not hedging or vague statements.

3. USE DOMAIN CONTEXT. The sources come from a specific academic subject \
indicated at the top of the prompt. Interpret every term in the context \
of that subject. A term can mean very different things across fields — \
always choose the interpretation that fits the subject.

4. CITE YOUR SOURCES. After explaining, reference where the concept \
appears in the student's materials using [1], [2], etc. This helps them \
locate the relevant section in their own documents.

5. DISTINGUISH SOURCE vs KNOWLEDGE. If your explanation goes beyond \
what the sources literally say, note it briefly with "(general knowledge)" \
so the student knows which parts come from their materials and which \
parts you are adding as a tutor.

6. DECLINE ONLY WHEN TRULY UNRELATED. If the question has absolutely \
no connection to any provided source, respond EXACTLY with: \
"I cannot answer this question based on the provided documents." \
But if the sources mention the concept — even in passing — answer it \
fully.

7. NEVER FABRICATE SOURCE NUMBERS. Only cite [N] values that exist.

8. Keep answers clear, well-structured, and at an academic level \
appropriate for the student's course material.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into numbered [SOURCE] blocks.
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


def build_user_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    subject: str | None = None,
) -> str:
    """
    Build the full user message: subject context + source blocks + question.

    Args:
        query: the student's question
        chunks: retrieved source chunks
        subject: the folder name (e.g. "Foundation Models", "Linear Algebra")
                 injected as domain context so the model interprets terms
                 within the right academic field.
    """
    parts = []

    if subject:
        parts.append(f"ACADEMIC SUBJECT: {subject}\n")

    context = build_context_block(chunks)
    parts.append(f"SOURCES:\n\n{context}")
    parts.append(f"---\n\nQUESTION: {query}")

    return "\n\n".join(parts)
