"""Provider factory: creates embedding and LLM providers from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybrid_recsys.providers.embeddings.sentence_tf import SentenceTransformerProvider
from hybrid_recsys.providers.llm.mock import MockLLMProvider

if TYPE_CHECKING:
    from hybrid_recsys.config import Settings
    from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
    from hybrid_recsys.providers.llm.base import LLMProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create the configured embedding provider."""
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerProvider(settings.embedding_model)
    if settings.embedding_provider == "openai":
        from hybrid_recsys.providers.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
        )
    msg = f"Unknown embedding provider: {settings.embedding_provider}"
    raise ValueError(msg)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the configured LLM provider."""
    if settings.llm_provider == "mock":
        return MockLLMProvider()
    if settings.llm_provider == "openai":
        from hybrid_recsys.providers.llm.openai import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    msg = f"Unknown LLM provider: {settings.llm_provider}"
    raise ValueError(msg)
