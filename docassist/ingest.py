"""Index building — stages 1 through 4 in one call.

Kept separate from the query path so the CLI, the tests and the Streamlit app
all build the index the same way.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from docassist.chunker import chunk_documents
from docassist.config import Settings
from docassist.embeddings import build_embeddings
from docassist.loader import load_documents
from docassist.vectorstore import IndexStats, get_stats, index_chunks, reset_index

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    pages: int
    chunks: int
    stats: IndexStats
    elapsed_s: float

    def summary(self) -> str:
        return (
            f"Indexed {self.chunks} chunk(s) from {self.pages} page(s) "
            f"across {self.stats.source_count} document(s) in {self.elapsed_s:.1f}s"
        )


def build_index(settings: Settings, rebuild: bool = False) -> IngestReport:
    """Load -> chunk -> embed -> store.

    With `rebuild=False` this upserts: unchanged chunks keep their id and are
    overwritten in place, so adding one new PDF costs one PDF's worth of work
    rather than re-embedding the whole corpus.
    """
    started = time.perf_counter()

    if rebuild:
        reset_index(settings)

    documents = load_documents(settings)
    chunks = chunk_documents(documents, settings)

    embeddings = build_embeddings(settings)
    store = index_chunks(chunks, settings, embeddings)

    report = IngestReport(
        pages=len(documents),
        chunks=len(chunks),
        stats=get_stats(store),
        elapsed_s=time.perf_counter() - started,
    )
    logger.debug(report.summary())
    return report
