"""Stage 5 — retrieval.

The question is embedded with the same model used for indexing, compared
against every stored chunk, and the best few come back. Only those go to the
LLM — not the whole corpus. That is what keeps the prompt small, the cost low
and the answer focused.

Chunks below the relevance threshold are dropped. If that leaves nothing, the
pipeline abstains rather than handing the LLM irrelevant text and hoping.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from docassist.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One chunk that came back from the vector search."""

    document: Document
    score: float

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def page(self) -> int:
        return int(self.document.metadata.get("page", 0))

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def citation(self) -> str:
        return f"{self.source} (page {self.page})"

    def snippet(self, limit: int = 320) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= limit else body[:limit].rstrip() + "…"


def retrieve(
    store: Chroma,
    question: str,
    settings: Settings,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks for `question`, filtered by relevance and de-duplicated."""
    k = top_k or settings.top_k

    if settings.search_type == "mmr":
        # MMR re-ranks a wider candidate pool to reduce redundancy — worth it
        # for "summarise this" questions, where plain similarity tends to
        # return five near-identical chunks from the same section.
        docs = store.max_marginal_relevance_search(question, k=k, fetch_k=max(k * 4, 20))
        scored = [(doc, 1.0) for doc in docs]
    else:
        with warnings.catch_warnings():
            # For a truly off-topic question the best cosine similarity can go
            # slightly negative, and LangChain warns that scores left its
            # expected 0..1 range. That is the correct answer here, not a
            # problem — clamp below and move on.
            warnings.filterwarnings("ignore", message=".*[Rr]elevance scores.*")
            scored = store.similarity_search_with_relevance_scores(question, k=k)
        scored = [(doc, max(0.0, min(1.0, score))) for doc, score in scored]

    results: list[RetrievedChunk] = []
    seen: set[str] = set()

    for doc, score in scored:
        if score < settings.score_threshold:
            continue
        # Overlapping chunks mean the same passage can surface twice; sending
        # it to the LLM twice just wastes context.
        key = doc.metadata.get("chunk_id") or doc.page_content[:200]
        if key in seen:
            continue
        seen.add(key)
        results.append(RetrievedChunk(document=doc, score=float(score)))

    logger.info(
        "Retrieved %d/%d chunk(s) above threshold %.2f for: %r",
        len(results),
        len(scored),
        settings.score_threshold,
        question[:80],
    )
    return results


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Format retrieved chunks into the numbered block the prompt expects.

    Numbering is what makes citation possible: the model is told to write
    [1] / [2], and those indices map back to real files and pages.
    """
    parts: list[str] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[{index}] Source: {chunk.source} | Page: {chunk.page}\n"
            f"{chunk.text.strip()}"
        )
        if used + len(block) > max_chars and parts:
            logger.info("Context truncated at %d chars (%d chunks used)", used, index - 1)
            break
        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)
