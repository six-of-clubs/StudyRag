"""
Prompt construction.

Design principles:
  - ANSWER FIRST: Lead with the explanation, cite at the end.
  - AUTHORITATIVE: When a known concept appears in sources, explain it
    confidently using general knowledge.
  - DOMAIN-AWARE: The folder name is injected as subject context so the model
    interprets terms within the right field.
  - GROUNDED: Every answer must connect to at least one source.

TWO FORMATTING DECISIONS WORTH KNOWING

1. Source blocks are headed `=== SOURCE 1 ===`, not `[SOURCE 1]`.
   The old format handed the model a bracketed example on every single line
   of context, and 7B models imitate what they see. Responses came back
   citing `[SOURCE 1]`, which the citation regex could not match, and a
   perfectly good answer was discarded as uncited. The context format and
   the citation format must not look alike.

2. The citation rule appears twice -- once in the system prompt and once as
   the last line of the user prompt. Small models weight the end of the
   prompt heavily, and a rule sitting fourth out of eight in a long system
   message is the one they drop first.
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

2. CITE EVERY ANSWER. You are given numbered sources. Every answer must \
reference at least one of them using square brackets around the number \
alone: [1], [2], [3]. Place the marker at the end of the sentence it \
supports.
   Correct:   The determinant is zero exactly when the matrix is singular [2].
   Correct:   Both proofs rely on the same lemma [1][3].
   Incorrect: [SOURCE 2], (Source 2), [Source 2], "see source 2"
Cite only numbers that were actually given to you. Never invent one.

3. BE AUTHORITATIVE. When a well-known academic concept appears in the \
sources, explain it confidently and correctly using your knowledge. \
Students expect clear, correct explanations, not hedging.

4. USE DOMAIN CONTEXT. The sources come from a specific academic subject \
named at the top of the prompt. Interpret every term in the context of \
that subject -- a term can mean very different things across fields.

5. DISTINGUISH SOURCE FROM KNOWLEDGE. If your explanation goes beyond what \
the sources literally say, mark that part with "(general knowledge)" so \
the student knows which parts come from their materials.

6. DECLINE ONLY WHEN TRULY UNRELATED. If the question has no connection to \
any provided source, reply with exactly this sentence and nothing else:
"I cannot answer this question based on the provided documents."
Do not soften it, do not add "alone" or "but", do not follow it with an \
answer anyway. If the sources mention the concept at all -- even in \
passing -- answer the question instead of declining.

7. Keep answers clear, well-structured, and at a level appropriate to the \
student's course material.
"""

# Appended to every user prompt. Last thing the model reads before answering.
_CITATION_REMINDER = (
    "Remember: end the sentences you take from the sources with [1], [2], "
    "and so on. An answer with no bracketed source number cannot be shown "
    "to the student."
)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into numbered source blocks.

    The `=== SOURCE N ===` header is deliberately unlike the `[N]` citation
    format -- see the module docstring.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page_number", "?")
        blocks.append(
            f"=== SOURCE {i} ===\n"
            f"file: {src}, page: {page}\n\n"
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
        chunks: retrieved source chunks, in citation order
        subject: the folder name, injected as domain context
    """
    parts = []

    if subject:
        parts.append(f"ACADEMIC SUBJECT: {subject}")

    parts.append(f"SOURCES:\n\n{build_context_block(chunks)}")
    parts.append(f"---\n\nQUESTION: {query}")
    parts.append(_CITATION_REMINDER)

    return "\n\n".join(parts)


def build_repair_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    previous_answer: str,
) -> str:
    """
    Ask the model to add citation markers to an answer it already produced.

    Used when parse_citations returns reason="uncited": the content was fine,
    only the markers were missing. Rewriting is explicitly forbidden so the
    repair cannot quietly change the substance of the answer.
    """
    return (
        f"You previously answered this question:\n\n"
        f"QUESTION: {query}\n\n"
        f"YOUR ANSWER:\n{previous_answer}\n\n"
        f"---\n\nSOURCES:\n\n{build_context_block(chunks)}\n\n"
        f"---\n\n"
        f"Your answer is missing its source citations. Return the SAME answer, "
        f"word for word, with [1], [2] etc. added at the end of the sentences "
        f"the sources support. Change nothing else -- do not rewrite, do not "
        f"add new information, do not add commentary. If none of the sources "
        f"actually support your answer, reply with exactly:\n"
        f"\"I cannot answer this question based on the provided documents.\""
    )
