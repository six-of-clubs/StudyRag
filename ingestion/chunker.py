"""
Text chunker:

Takes a list of Document objects from the loader and splits them into
smaller, overlapping chunks suitable for embedding. Each chunk inherits
its parent's metadata and gets an additional `chunk_index`.

METADATA INJECTION: Each chunk's text is prefixed with its source file
and page number so that the embedding model can match queries like
"explain slide 12". Without this, metadata is invisible to vector search.

Strategy:
    Fixed-size character windows with overlap. Splits prefer sentence
    boundaries when possible so chunks don't cut mid-sentence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config import settings
from ingestion.loader import Document

logger = logging.getLogger(__name__)

# Sentence-ending pattern: split on . ! ? followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A text fragment ready for embedding, with full provenance."""

    text: str
    metadata: dict = field(default_factory=dict)
    # metadata inherits from Document + adds: chunk_index


def _build_metadata_prefix(doc: Document) -> str:
    """
    Build a human-readable prefix from document metadata.

    This gets prepended to every chunk so the embedding model can
    match queries that reference filenames, slide/page numbers, or
    file types.
    """
    source = doc.metadata.get("source_file", "unknown")
    page = doc.metadata.get("page_number", "?")
    file_type = doc.metadata.get("file_type", "")

    # Use "Slide" for presentations, "Page" for everything else
    page_label = "Slide" if file_type == "pptx" else "Page"

    return f"[{source} | {page_label} {page}]"


def _find_split_point(text: str, target: int) -> int:
    """
    Find the best position to split `text` at approximately `target` chars.

    Prefers splitting at a sentence boundary. Falls back to whitespace.
    Falls back to the hard target if neither is found.
    """
    if target >= len(text):
        return len(text)

    # Look for the last sentence boundary before the target
    best = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() <= target:
            best = match.end()
        else:
            break

    if best > 0:
        return best

    # Fall back: last whitespace before target
    space = text.rfind(" ", 0, target)
    if space > 0:
        return space + 1

    return target


def chunk_document(doc: Document) -> list[Chunk]:
    """
    Split a single Document into overlapping Chunks.

    Uses `settings.chunk_size` and `settings.chunk_overlap` from config.
    Each chunk is prefixed with metadata (filename + page number).
    """
    text = doc.text.strip()
    if not text:
        return []

    prefix = _build_metadata_prefix(doc)
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            end = _find_split_point(text, end)

        chunk_text = text[start:end].strip()
        if chunk_text:
            # Prepend metadata so it's part of the embedded text
            enriched_text = f"{prefix} {chunk_text}"

            chunks.append(Chunk(
                text=enriched_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": index,
                },
            ))
            index += 1

        step = max(end - start - overlap, 1)
        start += step

    logger.debug(
        "Chunked %s page %s → %d chunk(s)",
        doc.metadata.get("source_file", "?"),
        doc.metadata.get("page_number", "?"),
        len(chunks),
    )
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Split a list of Documents into Chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    logger.info("Created %d chunk(s) from %d document(s)", len(all_chunks), len(docs))
    return all_chunks
