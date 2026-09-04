"""Tests for the parts that keep answers honest.

These use a stub LLM rather than a real provider, so they run offline and
deterministically. What is being tested is the pipeline's behaviour around
the model — abstention, citation verification, source attribution — not the
model's own writing.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from docassist.config import load_settings
from docassist.llm import ExtractiveLLM, LLMResult
from docassist.pipeline import RAGPipeline, _score_citations
from docassist.prompts import NO_ANSWER_MARKER, SYSTEM_PROMPT, build_user_prompt
from docassist.retriever import RetrievedChunk, build_context


class StubLLM:
    """Returns a canned answer and records what it was asked."""

    name = "stub"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.calls = 0

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls += 1
        self.last_system, self.last_user = system, user
        return LLMResult(text=self.reply, model="stub")


def _chunk(text: str, source: str = "handbook.pdf", page: int = 3, score: float = 0.6):
    return RetrievedChunk(
        document=Document(
            page_content=text,
            metadata={"source": source, "page": page, "chunk_id": text[:16]},
        ),
        score=score,
    )


def _pipeline(llm, store) -> RAGPipeline:
    return RAGPipeline(settings=load_settings(), store=store, llm=llm)


def _store_returning(pairs):
    store = MagicMock()
    store.similarity_search_with_relevance_scores.return_value = pairs
    return store


# --- context construction --------------------------------------------------


def test_context_is_numbered_and_labelled_for_citation():
    context = build_context([_chunk("Leave is 30 days."), _chunk("MFA is required.")], 5000)

    assert "[1] Source: handbook.pdf | Page: 3" in context
    assert "[2] Source: handbook.pdf | Page: 3" in context


def test_context_respects_the_character_budget():
    chunks = [_chunk("x" * 500) for _ in range(10)]
    context = build_context(chunks, max_chars=1200)

    # Budget is enforced per whole chunk, so allow one chunk of slack.
    assert len(context) < 1200 + 600


# --- abstention ------------------------------------------------------------


def test_no_relevant_context_means_the_llm_is_never_called():
    """The cheapest hallucination to prevent is the one you don't generate."""
    llm = StubLLM("The CEO is Jane Doe.")
    pipeline = _pipeline(llm, _store_returning([]))

    response = pipeline.answer("Who is the CEO?")

    assert llm.calls == 0
    assert response.abstained
    assert response.answer == NO_ANSWER_MARKER
    assert response.sources == []
    assert response.grounded


def test_low_scoring_chunks_are_filtered_out():
    settings = load_settings()
    below = settings.score_threshold / 2
    store = _store_returning([(_chunk("Unrelated text.").document, below)])
    llm = StubLLM("Something made up.")

    response = _pipeline(llm, store).answer("Who won the World Cup?")

    assert llm.calls == 0
    assert response.abstained


def test_provider_failure_is_not_reported_as_abstention():
    """A missing API key is a config problem, not a gap in the documents."""

    class BrokenLLM:
        name = "broken"

        def generate(self, system, user):
            from docassist.llm import LLMError

            raise LLMError("No Anthropic credentials found.")

    store = _store_returning([(_chunk("Annual leave is 30 days.").document, 0.6)])
    response = _pipeline(BrokenLLM(), store).safe_answer("How much leave?")

    assert response.failed
    assert not response.abstained
    assert not response.grounded
    assert "credentials" in response.answer


def test_answer_with_a_tacked_on_refusal_is_not_scored_as_a_refusal():
    """Smaller models often answer, then append the refusal line anyway.

    Treating that as a refusal loses a correct answer and inflates the
    benchmark's refusal rate.
    """
    store = _store_returning([(_chunk("Annual leave is 30 days.").document, 0.6)])
    reply = (
        "Full-time employees are entitled to 30 days of annual leave "
        "per calendar year, in addition to public holidays. [1]\n"
        f"{NO_ANSWER_MARKER}"
    )
    response = _pipeline(StubLLM(reply), store).answer("How much leave?")

    assert not response.abstained
    assert "30 days" in response.answer
    assert NO_ANSWER_MARKER not in response.answer  # contradictory line dropped
    assert response.citation_coverage == 1.0
    assert response.grounded


def test_model_refusal_is_reported_as_abstention():
    store = _store_returning([(_chunk("Leave policy text.").document, 0.5)])
    response = _pipeline(StubLLM(NO_ANSWER_MARKER), store).answer("Parental leave pay?")

    assert response.abstained
    assert response.citation_coverage == 0.0


# --- citation verification -------------------------------------------------


def test_only_cited_chunks_are_reported_as_sources():
    store = _store_returning(
        [
            (_chunk("Annual leave is 30 days.", page=1).document, 0.7),
            (_chunk("Overtime is time off in lieu.", page=2).document, 0.4),
        ]
    )
    response = _pipeline(StubLLM("Employees get 30 days [1]."), store).answer("Leave?")

    # Passage 2 was retrieved but not used — listing it would imply the answer
    # rests on it, which it does not.
    assert [s.index for s in response.sources] == [1]
    assert response.sources[0].source == "handbook.pdf"
    assert response.sources[0].page == 1


def test_citation_pointing_at_a_nonexistent_passage_is_flagged():
    store = _store_returning([(_chunk("Only one passage.").document, 0.6)])
    response = _pipeline(StubLLM("As stated [4], the policy is X."), store).answer("X?")

    assert response.invalid_citations == [4]
    assert not response.grounded


def test_uncited_answer_is_not_considered_grounded():
    store = _store_returning([(_chunk("Some relevant policy text here.").document, 0.6)])
    response = _pipeline(
        StubLLM("The company provides a generous benefits package to all staff."), store
    ).answer("Benefits?")

    assert response.citation_coverage == 0.0
    assert not response.grounded


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Employees get 30 days of annual leave each calendar year [1].", 1.0),
        ("Employees get 30 days [1].\nOvertime is compensated in lieu of pay [2].", 1.0),
        ("Employees get 30 days [1].\nThe company also has a great culture overall.", 0.5),
    ],
)
def test_citation_coverage_scoring(answer, expected):
    coverage, invalid, _ = _score_citations(answer, n_passages=2)
    assert coverage == expected
    assert invalid == []


# --- the prompt contract ---------------------------------------------------


def test_the_model_receives_the_grounding_rules_and_the_context():
    llm = StubLLM("Answer [1].")
    store = _store_returning([(_chunk("Annual leave is 30 days.").document, 0.6)])

    _pipeline(llm, store).answer("How much leave?")

    assert "ONLY the numbered context passages" in llm.last_system
    assert NO_ANSWER_MARKER in llm.last_system
    assert "Annual leave is 30 days." in llm.last_user
    assert "How much leave?" in llm.last_user


# --- extractive provider ---------------------------------------------------


def test_extractive_provider_quotes_the_context_and_cites_it():
    context = build_context(
        [_chunk("Full-time employees are entitled to 30 days of annual leave per year.")],
        5000,
    )
    result = ExtractiveLLM().generate(
        SYSTEM_PROMPT, build_user_prompt(context, "How many days of annual leave?")
    )

    assert "30 days of annual leave" in result.text
    assert "[1]" in result.text


def test_extractive_provider_abstains_when_nothing_matches():
    context = build_context([_chunk("Fuel represented 11.3 per cent of operating cost.")], 5000)
    result = ExtractiveLLM().generate(
        SYSTEM_PROMPT, build_user_prompt(context, "parental leave entitlement?")
    )

    assert result.text == NO_ANSWER_MARKER
