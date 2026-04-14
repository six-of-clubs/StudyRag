"""
Reranker.

Currently a pass-through. Will be tried to replace with a cross-encoder
reranker.
"""

from __future__ import annotations

from retrieval.retriever import RetrievedChunk


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    'placeholder'
    return chunks
