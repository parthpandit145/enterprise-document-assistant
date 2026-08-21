"""Stage 6 — answer synthesis.

Every provider hides behind one method, `generate(system, user) -> LLMResult`,
so the pipeline never knows or cares which model produced the answer.

Providers:
    anthropic   Claude via the Anthropic Messages API — the default.
    openai      OpenAI chat models, via langchain-openai.
    ollama      A local model served by Ollama — no data leaves the machine.
    extractive  No model at all. Stitches an answer out of the sentences in the
                retrieved context. It exists so the retrieval half of the
                pipeline can be exercised and tested with zero credentials,
                and as a hard floor on hallucination: it is physically
                incapable of emitting a word that isn't in the documents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from docassist.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce an answer. Carries a fixable message."""


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> LLMResult: ...


# ---------------------------------------------------------------------------
# Anthropic (default)
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """Claude via the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError("pip install anthropic") from exc

        self._anthropic = anthropic
        # Zero-arg constructor: the SDK resolves ANTHROPIC_API_KEY, then
        # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile on disk.
        self._client = anthropic.Anthropic()
        self.model = settings.llm_model
        self.effort = settings.llm_effort
        self.max_tokens = settings.llm_max_tokens
        self.name = f"anthropic:{self.model}"

    def generate(self, system: str, user: str) -> LLMResult:
        anthropic = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                # Grounded extraction doesn't need deep reasoning; medium effort
                # keeps latency and cost down without hurting answer quality.
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY in .env, "
                "or switch to LLM_PROVIDER=extractive to run without any key."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Unknown model id: {self.model!r}") from exc
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after", "60")
            raise LLMError(f"Rate limited by Anthropic. Retry in {retry_after}s.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API — check the network.") from exc
        except TypeError as exc:
            # With no credentials at all, the SDK raises a bare TypeError from
            # header construction rather than AuthenticationError — a raw
            # traceback is the first thing a new user would otherwise see.
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError(
                "No Anthropic credentials found.\n"
                "  Either put a key in .env:      ANTHROPIC_API_KEY=sk-ant-...\n"
                "  or run the pipeline with no key at all:  LLM_PROVIDER=extractive"
            ) from exc

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            reason = getattr(detail, "explanation", None) or "no explanation given"
            raise LLMError(f"The model declined to answer: {reason}")

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        return LLMResult(
            text=text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


# ---------------------------------------------------------------------------
# Optional providers
# ---------------------------------------------------------------------------


class OpenAILLM:
    def __init__(self, settings: Settings) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError("pip install langchain-openai") from exc

        model = settings.llm_model
        if model.startswith("claude"):
            model = "gpt-4o-mini"  # the default is a Claude id; don't send it to OpenAI
        self.model = model
        self.name = f"openai:{model}"
        self._chat = ChatOpenAI(model=model, temperature=0, max_tokens=settings.llm_max_tokens)

    def generate(self, system: str, user: str) -> LLMResult:
        try:
            response = self._chat.invoke(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        usage = getattr(response, "usage_metadata", None) or {}
        return LLMResult(
            text=str(response.content).strip(),
            model=self.model,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )


class OllamaLLM:
    def __init__(self, settings: Settings) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError("pip install langchain-ollama") from exc

        model = settings.llm_model
        if model.startswith("claude") or model.startswith("gpt"):
            model = "llama3.1"
        self.model = model
        self.name = f"ollama:{model}"
        self._chat = ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0)

    def generate(self, system: str, user: str) -> LLMResult:
        try:
            response = self._chat.invoke(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception as exc:
            raise LLMError(
                f"Ollama call failed: {exc}\nIs `ollama serve` running and `{self.model}` pulled?"
            ) from exc
        return LLMResult(text=str(response.content).strip(), model=self.model)


# ---------------------------------------------------------------------------
# Extractive fallback — no model, no keys, no network
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    """a an the and or of to in on for with is are was were be been at by from as
    that this these those it its what which who whom how when where why does do did
    can could should would will shall may might must not no if then than there here
    about into over under our your their his her they we you i me my""".split()
)


class ExtractiveLLM:
    """Answers by quoting the retrieved context, never by generating new text.

    Scores each sentence by keyword overlap with the question and returns the
    best few, with their citation numbers. Not as fluent as a real model, but
    it makes the retrieval quality visible on its own and gives the test suite
    something deterministic to assert against.
    """

    name = "extractive:none"
    model = "extractive"

    def __init__(self, settings: Settings | None = None, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    @staticmethod
    def _keywords(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) > 2 and w not in _STOPWORDS}

    def generate(self, system: str, user: str) -> LLMResult:
        from docassist.prompts import NO_ANSWER_MARKER

        # The user prompt is the rendered template; recover its two halves.
        context, _, tail = user.partition("\n\n---\n\nQuestion: ")
        question = tail.split("\n")[0] if tail else ""
        context = context.replace("Context passages:\n\n", "", 1)

        wanted = self._keywords(question)
        scored: list[tuple[float, int, str]] = []
        # Chunks overlap by design, so the same sentence can arrive in two
        # passages — and a chunk boundary may have clipped one copy short.
        # Keep only the first (highest-ranked) appearance of each.
        seen: list[str] = []

        for block in context.split("\n\n---\n\n"):
            header, _, body = block.partition("\n")
            match = re.match(r"\[(\d+)\]", header.strip())
            if not match:
                continue
            index = int(match.group(1))
            for sentence in _SENTENCE_SPLIT.split(body):
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                key = " ".join(sentence.lower().split()).rstrip(".!?;:")
                if any(key.startswith(prev) or prev.startswith(key) for prev in seen):
                    continue
                seen.append(key)
                overlap = wanted & self._keywords(sentence)
                if not overlap:
                    continue
                # Normalise by length so a long paragraph doesn't win purely on
                # having more words to match against.
                score = len(overlap) / (1 + len(sentence) / 200)
                scored.append((score, index, sentence))

        if not scored:
            return LLMResult(text=NO_ANSWER_MARKER, model=self.model)

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = scored[: self.max_sentences]
        # Restore document order so the answer reads sensibly.
        chosen.sort(key=lambda item: item[1])

        lines = [f"- {sentence} [{index}]" for _, index, sentence in chosen]
        return LLMResult(text="\n".join(lines), model=self.model)


# ---------------------------------------------------------------------------


def build_llm(settings: Settings) -> LLM:
    builders = {
        "anthropic": AnthropicLLM,
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
        "extractive": ExtractiveLLM,
    }
    try:
        builder = builders[settings.llm_provider]
    except KeyError:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}") from None

    llm = builder(settings)
    logger.info("Answer synthesis via %s", llm.name)
    return llm
