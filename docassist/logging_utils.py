"""Console logging for the CLI entry points."""

from __future__ import annotations

import logging
import os


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries log a lot at INFO and none of it is about our pipeline.
    for noisy in ("httpx", "chromadb", "sentence_transformers", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Silence the tokenizers fork warning that sentence-transformers triggers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
