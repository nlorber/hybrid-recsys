"""Tests for the provider factory dispatch logic.

Each branch is patched at the provider-class level so no vendor SDK or network is
required — the point is to verify the factory selects and wires the correct class
from settings, and raises on unknown provider names.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hybrid_recsys import factory
from hybrid_recsys.config import Settings
from hybrid_recsys.providers.llm.mock import MockLLMProvider


class TestCreateEmbeddingProvider:
    def test_sentence_transformers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)
        monkeypatch.setattr(factory, "SentenceTransformerProvider", ctor)
        result = factory.create_embedding_provider(
            Settings(embedding_provider="sentence-transformers", embedding_model="m")
        )
        assert result is sentinel
        ctor.assert_called_once_with("m")

    def test_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)
        monkeypatch.setattr(
            "hybrid_recsys.providers.embeddings.openai.OpenAIEmbeddingProvider", ctor
        )
        result = factory.create_embedding_provider(
            Settings(
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                embedding_api_key="k",
            )
        )
        assert result is sentinel
        ctor.assert_called_once()

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            factory.create_embedding_provider(Settings(embedding_provider="bogus"))


class TestCreateLLMProvider:
    def test_mock(self) -> None:
        result = factory.create_llm_provider(Settings(llm_provider="mock"))
        assert isinstance(result, MockLLMProvider)

    def test_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)
        monkeypatch.setattr("hybrid_recsys.providers.llm.openai.OpenAILLMProvider", ctor)
        result = factory.create_llm_provider(Settings(llm_provider="openai", llm_api_key="k"))
        assert result is sentinel
        ctor.assert_called_once()

    def test_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)
        monkeypatch.setattr("hybrid_recsys.providers.llm.anthropic.AnthropicLLMProvider", ctor)
        result = factory.create_llm_provider(Settings(llm_provider="anthropic", llm_api_key="k"))
        assert result is sentinel
        ctor.assert_called_once()

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            factory.create_llm_provider(Settings(llm_provider="bogus"))
