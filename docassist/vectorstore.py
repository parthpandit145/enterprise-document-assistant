"""Stage 4 — vector storage.

Chroma holds three things per chunk: the text, its embedding, and its
metadata (source file, page). Persisted to disk so indexing is a one-off cost
rather than something every query pays for.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from docassist.config import Settings

logger = logging.getLogger(__name__)

# Chroma only accepts scalar metadata values.
_ALLOWED_META = (str, int, float, bool)


@dataclass
class IndexStats:
    chunk_count: int
    sources: dict[str, int]

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def summary(self) -> str:
        if self.chunk_count == 0:
            return "index is empty"
        return f"{self.chunk_count} chunks from {self.source_count} document(s)"


def _sanitise(doc: Document) -> Document:
    doc.metadata = {
        key: value
        for key, value in doc.metadata.items()
        if isinstance(value, _ALLOWED_META) and value is not None
    }
    return doc


def open_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    """Open (or lazily create) the persistent collection.

    The index is built in cosine space rather than Chroma's default L2. With
    normalised embeddings the two rank identically, but cosine makes the
    relevance score LangChain hands back literally the cosine similarity —
    so SCORE_THRESHOLD is a number you can reason about instead of an
    artefact of the distance metric.
    """
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_index(settings: Settings) -> None:
    """Delete the on-disk index. Used by `ingest --rebuild`."""
    if settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)
        logger.info("Removed existing index at %s", settings.persist_dir)


def index_chunks(
    chunks: list[Document],
    settings: Settings,
    embeddings: Embeddings,
    batch_size: int = 128,
) -> Chroma:
    """Embed and store chunks, upserting on the stable chunk id.

    Re-running ingestion over an unchanged corpus is therefore a no-op rather
    than a way to accumulate duplicate copies of every chunk.
    """
    store = open_store(settings, embeddings)
    if not chunks:
        return store

    prepared = [_sanitise(chunk) for chunk in chunks]
    ids = [chunk.metadata["chunk_id"] for chunk in prepared]

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        store.add_documents(documents=batch, ids=ids[start : start + batch_size])
        logger.info(
            "Embedded %d/%d chunks", min(start + batch_size, len(prepared)), len(prepared)
        )

    return store


def get_stats(store: Chroma) -> IndexStats:
    """Chunk count and per-source breakdown of what is currently indexed."""
    try:
        raw = store.get(include=["metadatas"])
    except Exception as exc:  # collection may not exist yet
        logger.debug("Could not read index stats: %s", exc)
        return IndexStats(chunk_count=0, sources={})

    metadatas = raw.get("metadatas") or []
    sources: dict[str, int] = {}
    for meta in metadatas:
        name = (meta or {}).get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1

    return IndexStats(chunk_count=len(metadatas), sources=dict(sorted(sources.items())))
