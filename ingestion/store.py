"""
CLI ingestion for StudyRAG.

Ingests documents into a specific source folder's ChromaDB collection
via the state manager. There is no global collection — every document
belongs to exactly one folder.

Usage:
    python -m ingestion.store --folder "Linear Algebra" --source ./documents/linalg/
    python -m ingestion.store --folder "Calculus" --source ./documents/calc101.pdf
    python -m ingestion.store --list                       # show all folders
    python -m ingestion.store --folder "Linear Algebra" --reset  # wipe a folder

If the folder doesn't exist yet, it will be created automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_state():
    """Late import to avoid circular imports at module level."""
    from api.state import state
    return state


def list_folders():
    """Print all existing source folders."""
    state = _get_state()
    folders = state.list_folders()

    if not folders:
        print("  No source folders yet.")
        print("  Create one with: python -m ingestion.store --folder \"Name\" --source ./path/")
        return

    print(f"  {'Folder':<30} {'Documents':<12} {'ID'}")
    print("  " + "-" * 60)
    for f in folders:
        doc_count = len(f.documents)
        print(f"  {f.name:<30} {doc_count:<12} {f.id}")


def resolve_folder(folder_name: str) -> str:
    """
    Find a folder by name, or create it if it doesn't exist.
    Returns the folder ID.
    """
    state = _get_state()

    # Try to find existing folder (case-insensitive match)
    for f in state.list_folders():
        if f.name.lower() == folder_name.lower():
            logger.info("Found existing folder '%s' (id=%s)", f.name, f.id)
            return f.id

    # Create new folder
    folder = state.create_folder(folder_name)
    logger.info("Created new folder '%s' (id=%s)", folder.name, folder.id)
    return folder.id


def reset_folder(folder_name: str):
    """Delete a folder and all its documents, then recreate it empty."""
    state = _get_state()

    for f in state.list_folders():
        if f.name.lower() == folder_name.lower():
            doc_count = len(f.documents)
            state.delete_folder(f.id)
            logger.info(
                "Deleted folder '%s' (%d document(s) removed).", f.name, doc_count
            )
            # Recreate empty
            new = state.create_folder(f.name)
            logger.info("Recreated empty folder '%s' (id=%s)", new.name, new.id)
            return

    logger.warning("Folder '%s' not found — nothing to reset.", folder_name)


def ingest_to_folder(folder_id: str, source: Path) -> int:
    """
    Ingest a file or directory into a folder's collection.

    Returns total number of chunks added across all files.
    """
    state = _get_state()
    total = 0

    if source.is_file():
        files = [source]
    elif source.is_dir():
        from ingestion.loader import SUPPORTED_EXTENSIONS
        files = sorted(
            f for f in source.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            logger.warning("No supported files found in %s", source)
            return 0
        logger.info("Found %d file(s) in %s", len(files), source)
    else:
        raise FileNotFoundError(f"Path not found: {source}")

    for file_path in files:
        try:
            count = state.ingest_to_folder(folder_id, file_path)
            total += count
            logger.info("  %s → %d chunk(s)", file_path.name, count)
        except Exception as e:
            logger.error("  Failed to ingest %s: %s", file_path.name, e)

    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="StudyRAG — Ingest documents into a source folder",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Name of the source folder to ingest into (created if it doesn't exist)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to a file or directory to ingest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing source folders",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the specified folder and recreate it empty (requires --folder)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  StudyRAG — Document Ingestion")
    print("=" * 60)
    print()

    # List mode
    if args.list:
        list_folders()
        print()
        return

    # Validation
    if not args.folder:
        print("  Error: --folder is required.")
        print("  Use --list to see existing folders.")
        print()
        print("  Examples:")
        print('    python -m ingestion.store --folder "Linear Algebra" --source ./documents/linalg/')
        print('    python -m ingestion.store --folder "Calculus" --source ./documents/calc.pdf')
        print("    python -m ingestion.store --list")
        sys.exit(1)

    # Reset mode
    if args.reset:
        reset_folder(args.folder)
        if not args.source:
            print()
            print("  Folder reset complete.")
            print("=" * 60)
            return

    # Ingest mode
    if not args.source:
        print("  Error: --source is required for ingestion.")
        print('  Example: python -m ingestion.store --folder "Linear Algebra" --source ./documents/linalg/')
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"  Error: path not found: {source}")
        sys.exit(1)

    folder_id = resolve_folder(args.folder)

    start = time.time()
    added = ingest_to_folder(folder_id, source)
    elapsed = time.time() - start

    print()
    print(f"  Done in {elapsed:.1f}s — {added} new chunk(s) added to '{args.folder}'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
