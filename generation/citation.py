"""
Citation extraction and validation.

Parses the LLM's response for citation markers and maps them back to the
retrieved chunks. This is the post-LLM half of the cite-or-decline guarantee:
an answer that cannot be traced to a source does not reach the student.

WHY THIS FILE IS FUSSY
    The pre-LLM gates (retrieval floor, reranker threshold) are permissive by
    design -- they let borderline chunks through so the model gets a fair shot.
    That makes this file the last line of defence, and it has to distinguish
    three very different situations that all used to look identical:

        model_declined : the model correctly said it could not answer
        uncited        : the model answered, but produced no usable citation
        ok             : the model answered and cited

    Collapsing `uncited` into `declined` was throwing away correct answers.
    A 7B model routinely writes a good explanation and simply forgets rule 4
    of an 8-rule system prompt. That is a formatting failure, not a knowledge
    failure, and the orchestrator can repair it with one corrective call.

FORMAT TOLERANCE
    The model is asked for [1] but will produce [1, 2], [Source 3], [1][2],
    and other variants. All are accepted. What is NOT accepted is a citation
    number outside the range of sources actually supplied -- see the note on
    matrix notation in _extract_citations.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Reasoning models (deepseek-r1) wrap their scratchpad in <think> tags.
# Citations inside that block are not part of the answer and must not count.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Matches a bracketed run of numbers, with an optional "source"/"sources" word:
#   [1]  [1, 2]  [1;3]  [1 and 2]  [Source 4]  [Sources 1, 2]
_CITATION_BLOCK = re.compile(
    r"\[\s*(?:sources?\s*)?((?:\d+\s*(?:,|;|&|and)?\s*)+)\]",
    re.IGNORECASE,
)

_DIGITS = re.compile(r"\d+")

DECLINE_PHRASE = "I cannot answer this question based on the provided documents"

# If the decline phrase is followed by one of these, the model is qualifying
# rather than refusing -- "...based on the provided documents ALONE, but the
# slides do mention..." is an answer, not a decline.
_QUALIFIERS = (
    "alone", "but", "however", "though", "although", "unless",
    "entirely", "fully", "completely", "only", "in full", "by itself",
)


@dataclass
class SourceReference:
    """A resolved citation pointing back to a specific document location."""

    source_number: int      # the [N] used in the response
    source_file: str
    page_number: int | str
    similarity: float


@dataclass
class CitedResponse:
    """The final output of the orchestrator."""

    answer: str
    sources: list[SourceReference]
    declined: bool
    # Why this response looks the way it does. "ok" | "model_declined"
    # | "uncited" | "no_sources". The orchestrator branches on this.
    reason: str = "ok"


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> scratchpad blocks from a model response."""
    return _THINK_BLOCK.sub("", text).strip()


def _is_decline(text: str) -> bool:
    """
    Decide whether the model refused to answer.

    Anchored to the START of the response, not a substring search anywhere in
    it. A model that answers fully and then adds a hedging sentence containing
    the decline wording is answering, and its answer should not be discarded.
    """
    stripped = text.strip().lstrip("*#> \t").strip()
    low = stripped.lower()
    phrase = DECLINE_PHRASE.lower()

    if not low.startswith(phrase):
        return False

    tail = low[len(phrase):].lstrip(" ,.;:-—")
    if tail.startswith(_QUALIFIERS):
        logger.debug("Decline phrase found but qualified -- treating as an answer.")
        return False

    return True


def _extract_citations(text: str, n_sources: int) -> list[int]:
    """
    Pull citation numbers out of the response.

    A bracketed block is discarded WHOLE if any number in it falls outside
    1..n_sources. Two reasons:

      1. Hallucinated source numbers should not be silently half-accepted.
      2. This is a maths tutor. Answers contain vectors and matrices written
         as [1, 0] or [2 3]. Those are notation, not citations. Requiring
         every number in the block to be a valid source index rejects most
         of them, since a stray 0 or an out-of-range index gives it away.
    """
    found: set[int] = set()

    for match in _CITATION_BLOCK.finditer(text):
        numbers = [int(d) for d in _DIGITS.findall(match.group(1))]
        if not numbers:
            continue
        if any(n < 1 or n > n_sources for n in numbers):
            logger.debug(
                "Ignoring bracketed block %r -- out of range for %d source(s).",
                match.group(0), n_sources,
            )
            continue
        found.update(numbers)

    return sorted(found)


def parse_citations(
    response_text: str,
    chunks: list[RetrievedChunk],
) -> CitedResponse:
    """
    Parse the LLM response and resolve citation markers to sources.

    Args:
        response_text: raw text from the LLM
        chunks: the retrieved chunks in the order they were presented,
                i.e. SOURCE 1 == chunks[0]

    Returns:
        A CitedResponse. Check `.reason` to distinguish a genuine decline
        from a formatting failure the orchestrator may want to repair.
    """
    answer = strip_reasoning(response_text)

    if not answer:
        logger.warning("Model returned an empty response.")
        return CitedResponse(
            answer=(
                "I cannot answer this question based on the provided documents."
            ),
            sources=[], declined=True, reason="uncited",
        )

    if _is_decline(answer):
        return CitedResponse(
            answer=answer, sources=[], declined=True, reason="model_declined",
        )

    cited_numbers = _extract_citations(answer, len(chunks))

    if not cited_numbers:
        # The model answered but gave nothing to trace. Do NOT rewrite the
        # answer here -- the orchestrator gets the original text so it can
        # attempt a citation repair before falling back to a decline.
        logger.info(
            "Answer contains no valid citations (%d source(s) were supplied).",
            len(chunks),
        )
        return CitedResponse(
            answer=answer, sources=[], declined=True, reason="uncited",
        )

    sources = [
        SourceReference(
            source_number=num,
            source_file=chunks[num - 1].metadata.get("source_file", "unknown"),
            page_number=chunks[num - 1].metadata.get("page_number", "?"),
            similarity=chunks[num - 1].similarity,
        )
        for num in cited_numbers
    ]

    return CitedResponse(answer=answer, sources=sources, declined=False, reason="ok")
