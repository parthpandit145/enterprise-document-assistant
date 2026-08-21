"""Stage 2 — chunking.

A whole page is the wrong unit for retrieval: too much unrelated text rides
along with whatever actually matched, and the embedding gets averaged into
mush. Splitting into ~1000-character chunks keeps each vector about one idea.

The 200-character overlap exists so a fact that straddles a chunk boundary
still appears whole in at least one chunk.
"""

from __future__ import annotations

import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docassist.config import Settings

logger = logging.getLogger(__name__)


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Tried in order: split on paragraphs first, then lines, then sentences,
        # and only fall back to mid-word cuts if a single "word" is enormous.
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
        add_start_index=True,
    )


def chunk_id(doc: Document) -> str:
    """A stable id for one chunk.

    Derived from content + origin rather than a counter, so re-ingesting an
    unchanged corpus upserts the same rows instead of duplicating them.
    """
    payload = "|".join(
        [
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("page", "")),
            str(doc.metadata.get("start_index", "")),
            doc.page_content,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def chunk_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split page-level Documents into retrieval-sized chunks."""
    splitter = build_splitter(settings)
    chunks = splitter.split_documents(documents)

    # Drop fragments too short to carry meaning — usually a stray heading that
    # got orphaned at the end of a page.
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 40]

    per_source: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        per_source[source] = per_source.get(source, 0) + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = chunk_id(chunk)

    if chunks:
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        logger.info(
            "Chunked %d page(s) into %d chunk(s) — avg %.0f chars "
            "(size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            avg,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        for source, count in sorted(per_source.items()):
            logger.info("  %-45s %4d chunk(s)", source, count)

    return chunks
