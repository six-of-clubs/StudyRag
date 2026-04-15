"""
Interactive CLI for testing the StudyRAG pipeline.

Usage:
    python cli.py

Lists available source folders, asks you to pick one, then enters a
question loop scoped to that folder. Type 'switch' to change folders,
'quit' to exit.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from api.state import state
from orchestrator.pipeline import ask


def pick_folder() -> str | None:
    """Display folders and let the user pick one. Returns folder_id or None."""
    folders = state.list_folders()

    if not folders:
        print("  No source folders found.")
        print("  Create folders and upload documents via the web UI first,")
        print("  or use the API: POST /api/folders")
        return None

    print()
    print("  Available source folders:")
    print("  " + "-" * 40)
    for i, f in enumerate(folders, 1):
        doc_count = len(f.documents)
        print(f"  {i}. {f.name}  ({doc_count} doc(s))")
    print()

    while True:
        try:
            choice = input("  Select folder number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice.lower() in ("quit", "exit", "q"):
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                selected = folders[idx]
                print(f"  → Using: {selected.name}")
                return selected.id
            else:
                print(f"  Enter a number between 1 and {len(folders)}")
        except ValueError:
            print("  Enter a number.")


def main():
    print("=" * 60)
    print("  StudyRAG — Interactive Query")
    print("=" * 60)

    folder_id = pick_folder()
    if folder_id is None:
        print("\nBye!")
        sys.exit(0)

    print()
    print("  Type your questions. Commands: 'switch', 'quit'")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if query.lower() == "switch":
            folder_id = pick_folder()
            if folder_id is None:
                print("Bye!")
                break
            print()
            continue

        if not query:
            continue

        result = ask(query, folder_id=folder_id)

        print()
        print("📝 Answer:")
        print(result.answer)
        print()

        if result.sources:
            print("📚 Sources:")
            for src in result.sources:
                print(
                    f"   [{src.source_number}] {src.source_file} "
                    f"p.{src.page_number} (similarity: {src.similarity:.2f})"
                )
        elif result.declined:
            print("⚠️  Declined — not enough evidence in the selected sources.")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
