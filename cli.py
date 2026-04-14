"""
Interactive CLI for testing the StudyRAG pipeline.

Usage:
    python -m orchestrator.pipeline

Type questions, see cited answers. Type 'quit' to exit.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from orchestrator.pipeline import ask


def main():
    print("=" * 60)
    print("  StudyRAG — Interactive Query (type 'quit' to exit)")
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

        if not query:
            continue

        result = ask(query)

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
            print("⚠️  Declined — not enough evidence in your documents.")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
