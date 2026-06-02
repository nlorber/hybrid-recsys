"""Provider factory: creates embedding and LLM providers from settings.

Selection is config-driven and vendor-agnostic — the pipeline and API depend only on
the ``EmbeddingProvider`` / ``LLMProvider`` ABCs, never on a concrete vendor. Switching
provider is pure configuration; adding a vendor means implementing the ABC and adding
one dispatch branch here. Optional vendor SDKs are imported lazily.
"""

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
    provider = settings.embedding_provider
    if provider == "sentence-transformers":
        return SentenceTransformerProvider(settings.embedding_model)
    if provider == "openai":
        from hybrid_recsys.providers.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
        )
    msg = f"Unknown embedding provider: {provider!r}"
    raise ValueError(msg)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the configured LLM re-ranking provider."""
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLMProvider()
    if provider == "openai":
        from hybrid_recsys.providers.llm.openai import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gpt-4o-mini",
            base_url=settings.llm_base_url,
        )
    if provider == "anthropic":
        from hybrid_recsys.providers.llm.anthropic import AnthropicLLMProvider

        return AnthropicLLMProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "claude-haiku-4-5-20251001",
            base_url=settings.llm_base_url,
        )
    msg = f"Unknown LLM provider: {provider!r}"
    raise ValueError(msg)
