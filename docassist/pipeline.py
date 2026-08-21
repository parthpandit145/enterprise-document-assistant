"""The RAG pipeline — the whole thing wired together.

    question -> embed -> search Chroma -> filter by relevance
             -> build numbered context -> LLM under the grounding prompt
             -> verify citations -> answer + sources

Two things here go beyond a textbook RAG loop, and both exist to make the
"reduces hallucination" claim actually true rather than aspirational:

  * Abstention. If nothing clears the relevance threshold, the LLM is never
    called. A model handed irrelevant context will still try to be helpful,
    and that is exactly where fabrication comes from.
  * Citation verification. Every [n] the model writes is checked against the
    passages it was actually given. A citation pointing at a passage that
    doesn't exist is a fabrication that would otherwise look authoritative.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from docassist.config import Settings, load_settings
from docassist.embeddings import build_embeddings
from docassist.llm import LLM, LLMError, build_llm
from docassist.prompts import NO_ANSWER_MARKER, SYSTEM_PROMPT, build_user_prompt
from docassist.retriever import RetrievedChunk, build_context, retrieve
from docassist.vectorstore import IndexStats, get_stats, open_store

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    """One passage the answer actually cited."""

    index: int
    source: str
    page: int
    score: float
    snippet: str

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.source} — page {self.page}"


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[Source]
    retrieved: list[RetrievedChunk]
    abstained: bool
    model: str
    latency_ms: int
    # True when the pipeline could not run at all (no credentials, provider
    # down). Distinct from `abstained`, which means it ran and found nothing —
    # conflating the two tells the user their documents are lacking when the
    # real problem is their config.
    failed: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    # Fraction of the answer's factual lines that carry a citation (0..1).
    citation_coverage: float = 0.0
    # Citation numbers the model emitted that point at no real passage.
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True when the answer is either a clean abstention or fully cited."""
        return self.abstained or (
            not self.invalid_citations and self.citation_coverage > 0.0
        )


def _score_citations(answer: str, n_passages: int) -> tuple[float, list[int], set[int]]:
    """Check the answer's [n] markers against the passages it was given."""
    cited = {int(m) for m in _CITATION.findall(answer)}
    invalid = sorted(n for n in cited if not 1 <= n <= n_passages)
    valid = {n for n in cited if 1 <= n <= n_passages}

    # A "claim line" is any non-trivial line of prose. Headings and bullets
    # that are just labels don't need a citation of their own.
    lines = [line.strip() for line in answer.splitlines() if len(line.strip()) > 25]
    if not lines:
        return (1.0 if valid else 0.0), invalid, valid

    with_citation = sum(1 for line in lines if _CITATION.search(line))
    return with_citation / len(lines), invalid, valid


class RAGPipeline:
    """Query-time entry point. Build once, call `answer()` many times."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: Chroma | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or open_store(self.settings, build_embeddings(self.settings))
        self._llm = llm
        self._llm_failed: str | None = None

    @property
    def llm(self) -> LLM:
        """Built lazily so the app can show the index before touching the API."""
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    def stats(self) -> IndexStats:
        return get_stats(self.store)

    def answer(self, question: str, top_k: int | None = None) -> RAGResponse:
        started = time.perf_counter()
        question = question.strip()

        if not question:
            raise ValueError("Question is empty.")

        chunks = retrieve(self.store, question, self.settings, top_k=top_k)

        if not chunks:
            # Nothing relevant was found. Calling the LLM here would be asking
            # it to answer from memory — which is the failure mode this whole
            # system exists to prevent.
            logger.info("No chunk cleared the threshold — abstaining without an LLM call.")
            return RAGResponse(
                question=question,
                answer=NO_ANSWER_MARKER,
                sources=[],
                retrieved=[],
                abstained=True,
                model="none (no relevant context retrieved)",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        context = build_context(chunks, self.settings.max_context_chars)
        result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(context, question))

        answer = result.text.strip()
        abstained = NO_ANSWER_MARKER.lower() in answer.lower()
        coverage, invalid, valid = _score_citations(answer, len(chunks))

        if invalid:
            logger.warning(
                "Answer cited passages that were never provided: %s", invalid
            )

        sources = [
            Source(
                index=index,
                source=chunk.source,
                page=chunk.page,
                score=chunk.score,
                snippet=chunk.snippet(),
            )
            for index, chunk in enumerate(chunks, start=1)
            if index in valid
        ]

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieved=chunks,
            abstained=abstained,
            model=result.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=result.usage,
            citation_coverage=0.0 if abstained else coverage,
            invalid_citations=invalid,
        )

    def safe_answer(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Like `answer()`, but turns provider failures into a visible message
        instead of a traceback. Used by the UI."""
        try:
            return self.answer(question, top_k=top_k)
        except (LLMError, ImportError) as exc:
            return RAGResponse(
                question=question,
                answer=str(exc),
                sources=[],
                retrieved=[],
                abstained=False,
                model="unavailable",
                latency_ms=0,
                failed=True,
            )
