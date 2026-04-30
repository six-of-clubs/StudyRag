"""
Citation extraction and validation_

Parses the LLM's response for [N] citation markers and maps them
back to the retrieved chunks, producing structured source references.
Also detects when the model declined to answer.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Matches [1], [2], [1, 3], [1][2], etc.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

DECLINE_PHRASE = "I cannot answer this question based on the provided documents"


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
    declined: bool  # True if the model refused to answer


def parse_citations(
    response_text: str,
    chunks: list[RetrievedChunk],
) -> CitedResponse:
    """
    Parse the LLM response and resolve citation markers to sources.

    Args:
        response_text: raw text from the LLM
        chunks: the retrieved chunks (in the same order they were
                presented to the LLM, i.e. [SOURCE 1] = chunks[0])

    Returns:
        A CitedResponse with the answer, resolved sources, and
        whether the model declined.
    """
    # Check if the model declined
    if DECLINE_PHRASE.lower() in response_text.lower():
        return CitedResponse(
            answer=response_text.strip(),
            sources=[],
            declined=True,
        )

    # Extract all cited source numbers
    cited_numbers = sorted(set(int(m) for m in _CITATION_PATTERN.findall(response_text)))

    # Resolve to source metadata
    sources = []
    for num in cited_numbers:
        idx = num - 1  # [SOURCE 1] → chunks[0]
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            sources.append(SourceReference(
                source_number=num,
                source_file=chunk.metadata.get("source_file", "unknown"),
                page_number=chunk.metadata.get("page_number", "?"),
                similarity=chunk.similarity,
            ))
        else:
            logger.warning(
                "LLM cited [%d] but only %d sources were provided — ignoring.",
                num, len(chunks),
            )

    # If the model didn't cite anything and didn't decline, flag it
    if not sources:
        logger.warning(
            "LLM produced an answer with no citations. "
            "Treating as a declined response for safety."
        )
        return CitedResponse(
            answer=(
                "I cannot answer this question based on the provided documents. "
                "(The model generated a response but did not cite any sources.)"
            ),
            sources=[],
            declined=True,
        )

    return CitedResponse(
        answer=response_text.strip(),
        sources=sources,
        declined=False,
    )
