"""Stage 3 — embeddings.

An embedding model maps a piece of text to a vector such that semantically
similar text lands nearby. That is what lets a question about "paid time off"
retrieve a chunk that only ever says "annual leave" — keyword search cannot
do that.

The default is a local sentence-transformers model: free, offline after the
first download, and good enough that the retrieval quality is not the
bottleneck. OpenAI embeddings are available behind the same interface.
"""

from __future__ import annotations

import functools
import logging

from langchain_core.embeddings import Embeddings

from docassist.config import Settings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _huggingface(model_name: str) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - install-time guidance
        raise ImportError(
            "HuggingFace embeddings need `langchain-huggingface` and "
            "`sentence-transformers`:\n    pip install langchain-huggingface sentence-transformers"
        ) from exc

    logger.info("Loading local embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # Normalised vectors make cosine similarity a plain dot product, which
        # is what Chroma's relevance scoring assumes.
        encode_kwargs={"normalize_embeddings": True},
    )


def _openai(model_name: str) -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - install-time guidance
        raise ImportError(
            "OpenAI embeddings need `langchain-openai`:\n    pip install langchain-openai"
        ) from exc

    if model_name.startswith("sentence-transformers/"):
        # The default is a HuggingFace model id; swap in a sane OpenAI one
        # rather than failing on a model name their API has never heard of.
        model_name = "text-embedding-3-small"

    logger.info("Using OpenAI embedding model: %s", model_name)
    return OpenAIEmbeddings(model=model_name)


def build_embeddings(settings: Settings) -> Embeddings:
    """Return the embedding model named by the settings.

    The same model must be used for indexing and for querying — vectors from
    two different models are not comparable, and the search silently returns
    nonsense rather than erroring.
    """
    if settings.embedding_provider == "huggingface":
        return _huggingface(settings.embedding_model)
    if settings.embedding_provider == "openai":
        return _openai(settings.embedding_model)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
