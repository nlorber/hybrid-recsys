"""Tests for OpenAI embedding provider with mocked API calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hybrid_recsys.providers.embeddings.openai import OpenAIEmbeddingProvider


@pytest.fixture()
def mock_openai() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def provider(mock_openai: MagicMock) -> OpenAIEmbeddingProvider:
    with patch(
        "hybrid_recsys.providers.embeddings.openai.OpenAI", return_value=mock_openai
    ):
        p = OpenAIEmbeddingProvider(api_key="test-key", model="text-embedding-3-small")
    return p


class TestEmbedBatch:
    def test_sorts_by_index_when_api_returns_out_of_order(
        self, provider: OpenAIEmbeddingProvider, mock_openai: MagicMock
    ) -> None:
        """The OpenAI API may return embeddings in arbitrary order.
        The provider must re-sort by the .index field so output[i]
        corresponds to input[i]."""
        mock_openai.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.9, 0.8], index=2),
                SimpleNamespace(embedding=[0.1, 0.2], index=0),
                SimpleNamespace(embedding=[0.5, 0.6], index=1),
            ]
        )

        result = provider.embed_batch(["zero", "one", "two"])

        assert result == [[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]]

    def test_passes_model_and_input_to_api(
        self, provider: OpenAIEmbeddingProvider, mock_openai: MagicMock
    ) -> None:
        mock_openai.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0], index=0)]
        )

        provider.embed_batch(["hello"])

        mock_openai.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-small"
        )

    def test_respects_custom_base_url(self) -> None:
        """Verify base_url is forwarded to the OpenAI client (Azure scenario)."""
        with patch(
            "hybrid_recsys.providers.embeddings.openai.OpenAI"
        ) as mock_cls:
            OpenAIEmbeddingProvider(
                api_key="k", model="m", base_url="https://my-azure.openai.azure.com"
            )
            mock_cls.assert_called_once_with(
                api_key="k", base_url="https://my-azure.openai.azure.com"
            )
