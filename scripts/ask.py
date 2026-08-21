"""Ask the indexed documents a question from the terminal.

    python -m scripts.ask "How many days of annual leave do I get?"
    python -m scripts.ask                       # interactive loop
    python -m scripts.ask "..." --show-context  # print the retrieved chunks too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docassist.config import load_settings  # noqa: E402
from docassist.logging_utils import setup_logging  # noqa: E402
from docassist.pipeline import RAGPipeline, RAGResponse  # noqa: E402

RULE = "─" * 78


def render(response: RAGResponse, show_context: bool) -> None:
    if response.failed:
        print(f"\n{RULE}\n⚠  {response.answer}\n{RULE}\n", file=sys.stderr)
        return

    print(f"\n{RULE}\n{response.answer}\n")

    if response.sources:
        print("Sources")
        for source in response.sources:
            print(f"  {source.label}   (relevance {source.score:.3f})")
            print(f"      {source.snippet[:150]}")
    elif not response.abstained:
        print("Sources: none cited — treat this answer with suspicion.")

    if response.invalid_citations:
        print(f"\n⚠ Answer cited non-existent passages: {response.invalid_citations}")

    if show_context and response.retrieved:
        print(f"\n{RULE}\nRetrieved context")
        for index, chunk in enumerate(response.retrieved, start=1):
            print(f"\n[{index}] {chunk.citation}  score={chunk.score:.3f}")
            print(f"    {chunk.snippet(500)}")

    meta = [f"model={response.model}", f"{response.latency_ms} ms"]
    if response.usage:
        meta.append(
            f"tokens in/out={response.usage.get('input_tokens', 0)}/"
            f"{response.usage.get('output_tokens', 0)}"
        )
    if not response.abstained:
        meta.append(f"citation coverage={response.citation_coverage:.0%}")
    print(f"\n{RULE}\n{'  |  '.join(meta)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the indexed documents.")
    parser.add_argument("question", nargs="*", help="question (omit for interactive mode)")
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    settings = load_settings()
    pipeline = RAGPipeline(settings)

    stats = pipeline.stats()
    if stats.chunk_count == 0:
        print(
            "The index is empty. Build it first:\n"
            "    python -m scripts.generate_test_pdfs\n"
            "    python -m scripts.ingest --rebuild",
            file=sys.stderr,
        )
        return 1

    if args.question:
        render(pipeline.safe_answer(" ".join(args.question), args.top_k), args.show_context)
        return 0

    print(f"Enterprise Document Assistant — {stats.summary()}")
    print(f"{settings.describe()}\nAsk a question, or Ctrl-C to quit.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        render(pipeline.safe_answer(question, args.top_k), args.show_context)


if __name__ == "__main__":
    raise SystemExit(main())
