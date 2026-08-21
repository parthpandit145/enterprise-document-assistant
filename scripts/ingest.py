"""Build or refresh the vector index.

    python -m scripts.ingest              # upsert: only new/changed content costs work
    python -m scripts.ingest --rebuild    # wipe the index and start clean
    python -m scripts.ingest --stats      # just report what is currently indexed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docassist.config import load_settings  # noqa: E402
from docassist.embeddings import build_embeddings  # noqa: E402
from docassist.ingest import build_index  # noqa: E402
from docassist.logging_utils import setup_logging  # noqa: E402
from docassist.vectorstore import get_stats, open_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index enterprise documents into Chroma.")
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the existing index first"
    )
    parser.add_argument(
        "--stats", action="store_true", help="show what is indexed and exit"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    settings = load_settings()

    if args.stats:
        stats = get_stats(open_store(settings, build_embeddings(settings)))
        print(f"\nIndex: {settings.persist_dir}")
        print(f"  {stats.summary()}")
        for source, count in stats.sources.items():
            print(f"    {source:<50} {count:>5} chunks")
        return 0

    print(f"\nConfig: {settings.describe()}")
    print(f"Source folder: {settings.pdf_dir}\n")

    try:
        report = build_index(settings, rebuild=args.rebuild)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\n{report.summary()}")
    print(f"Index stored at: {settings.persist_dir}")
    print('\nNext:  python -m scripts.ask "What is the annual leave entitlement?"')
    print("  or:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
