"""Central configuration.

Everything the pipeline can be tuned with lives here, is read from the
environment (via a `.env` file), and has a default that works out of the box.
No module below this one reads `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_PROVIDERS = ("huggingface", "openai")
LLM_PROVIDERS = ("anthropic", "openai", "ollama", "extractive")
SEARCH_TYPES = ("similarity", "mmr")


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = _env(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one run of the pipeline."""

    # -- paths ---------------------------------------------------------------
    pdf_dir: Path
    persist_dir: Path
    collection_name: str

    # -- chunking ------------------------------------------------------------
    chunk_size: int
    chunk_overlap: int

    # -- embeddings ----------------------------------------------------------
    embedding_provider: str
    embedding_model: str

    # -- retrieval -----------------------------------------------------------
    top_k: int
    search_type: str
    score_threshold: float
    max_context_chars: int

    # -- generation ----------------------------------------------------------
    llm_provider: str
    llm_model: str
    llm_effort: str
    llm_max_tokens: int
    ollama_base_url: str

    # Extensions the loader will pick up alongside .pdf.
    text_extensions: tuple[str, ...] = field(default=(".txt", ".md"))

    def validate(self) -> None:
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {EMBEDDING_PROVIDERS}, "
                f"got {self.embedding_provider!r}"
            )
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {LLM_PROVIDERS}, got {self.llm_provider!r}"
            )
        if self.search_type not in SEARCH_TYPES:
            raise ValueError(
                f"SEARCH_TYPE must be one of {SEARCH_TYPES}, got {self.search_type!r}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size}) — otherwise chunks never advance."
            )
        if self.top_k < 1:
            raise ValueError(f"TOP_K must be >= 1, got {self.top_k}")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                f"SCORE_THRESHOLD must be between 0 and 1, got {self.score_threshold}"
            )

    def describe(self) -> str:
        # The extractive provider ignores llm_model entirely; printing it would
        # suggest a model is being called when none is.
        llm = (
            self.llm_provider
            if self.llm_provider == "extractive"
            else f"{self.llm_provider}:{self.llm_model}"
        )
        return (
            f"embeddings={self.embedding_provider}:{self.embedding_model} | "
            f"llm={llm} | "
            f"chunk={self.chunk_size}/{self.chunk_overlap} | "
            f"top_k={self.top_k} ({self.search_type}) | "
            f"threshold={self.score_threshold}"
        )


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Read `.env` (if present) plus the process environment into a Settings."""
    load_dotenv(dotenv_path=env_file or (PROJECT_ROOT / ".env"), override=False)

    settings = Settings(
        pdf_dir=_resolve(_env("PDF_DIR", "data/pdfs")),
        persist_dir=_resolve(_env("PERSIST_DIR", "storage/chroma")),
        collection_name=_env("COLLECTION_NAME", "enterprise_docs"),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        embedding_provider=_env("EMBEDDING_PROVIDER", "huggingface").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=_env_int("TOP_K", 5),
        search_type=_env("SEARCH_TYPE", "similarity").lower(),
        score_threshold=_env_float("SCORE_THRESHOLD", 0.15),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 12000),
        llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
        llm_model=_env("LLM_MODEL", "claude-opus-5"),
        llm_effort=_env("LLM_EFFORT", "medium").lower(),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    settings.validate()
    return settings
