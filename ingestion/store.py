"""
Vector store for StudyRAG.

Embeds text chunks using sentence-transformers, stores them in ChromaDB
with their metadata, and provides a CLI entry point for batch ingestion.

Usage:
    python -m ingestion.store --source ./documents/
    python -m ingestion.store --source ./documents/lecture3.pdf
    python -m ingestion.store --reset   # wipe the store and re-ingest
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from ingestion.chunker import Chunk, chunk_documents
from ingestion.loader import load_directory, load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "studyrag"


def _get_chroma_client() -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the main collection with cosine similarity."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: Chunk) -> str:
    """
    Deterministic ID for a chunk based on its source and position.

    This means re-ingesting the same file won't create duplicates.
    """
    key = (
        f"{chunk.metadata.get('source_file', '')}"
        f":p{chunk.metadata.get('page_number', 0)}"
        f":c{chunk.metadata.get('chunk_index', 0)}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ingest_chunks(chunks: list[Chunk]) -> int:
    """
    Embed and store a list of chunks in ChromaDB.

    Returns the number of chunks added (skips existing ones).
    """
    if not chunks:
        logger.warning("No chunks to ingest.")
        return 0

    # Load embedding model
    logger.info("Loading embedding model '%s' ...", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)

    # Prepare IDs and check for existing
    client = _get_chroma_client()
    collection = _get_or_create_collection(client)

    ids = [_chunk_id(c) for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Filter out already-stored chunks
    existing = set()
    try:
        result = collection.get(ids=ids)
        existing = set(result["ids"])
    except Exception:
        pass  # collection might be empty

    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]
    if not new_indices:
        logger.info("All %d chunk(s) already in store. Nothing to add.", len(chunks))
        return 0

    new_ids = [ids[i] for i in new_indices]
    new_texts = [texts[i] for i in new_indices]
    new_metadatas = [metadatas[i] for i in new_indices]

    # Embed
    logger.info("Embedding %d new chunk(s) ...", len(new_ids))
    start = time.time()
    embeddings = model.encode(new_texts, show_progress_bar=True).tolist()
    elapsed = time.time() - start
    logger.info("Embedded in %.1fs (%.0f chunks/sec)", elapsed, len(new_ids) / elapsed)

    # Store
    collection.add(
        ids=new_ids,
        documents=new_texts,
        embeddings=embeddings,
        metadatas=new_metadatas,
    )
    logger.info(
        "Stored %d chunk(s). Collection now has %d total.",
        len(new_ids), collection.count(),
    )
    return len(new_ids)


def reset_store():
    """Delete all data from the vector store."""
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Collection '%s' deleted.", COLLECTION_NAME)
    except Exception:
        logger.info("Nothing to reset — collection doesn't exist.")


def ingest_path(source: str | Path, reset: bool = False) -> int:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Args:
        source: path to a file or directory
        reset: if True, wipe the store before ingesting

    Returns:
        number of new chunks added
    """
    if reset:
        reset_store()

    source = Path(source)
    if source.is_dir():
        docs = load_directory(source)
    elif source.is_file():
        docs = load_file(source)
    else:
        raise FileNotFoundError(f"Path not found: {source}")

    if not docs:
        logger.warning("No documents loaded from %s", source)
        return 0

    chunks = chunk_documents(docs)
    return ingest_chunks(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="StudyRAG — Ingest documents into the vector store",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./documents",
        help="Path to a file or directory to ingest (default: ./documents/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the vector store before ingesting",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  StudyRAG — Document Ingestion")
    print("=" * 60)
    print()

    start = time.time()
    added = ingest_path(args.source, reset=args.reset)
    elapsed = time.time() - start

    print()
    print(f"  Done in {elapsed:.1f}s — {added} new chunk(s) added.")
    print("=" * 60)


if __name__ == "__main__":
    main()
