"""Tests for OpenAI LLM provider with mocked API calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hybrid_recsys.providers.llm.openai import OpenAILLMProvider
from hybrid_recsys.retrieval.reranker import RERANK_RESPONSE_SCHEMA


@pytest.fixture()
def mock_openai() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def provider(mock_openai: MagicMock) -> OpenAILLMProvider:
    with patch("hybrid_recsys.providers.llm.openai.OpenAI", return_value=mock_openai):
        p = OpenAILLMProvider(api_key="test-key", model="gpt-4o-mini")
    return p


SAMPLE_CANDIDATES = [
    {"program_id": "prog_1", "description": "A show about AI"},
    {"program_id": "prog_2", "description": "A show about cooking"},
    {"program_id": "prog_3", "description": "A show about history"},
]


def _completion(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestRerank:
    def test_constructs_prompt_with_query_and_candidates(
        self, provider: OpenAILLMProvider, mock_openai: MagicMock
    ) -> None:
        """Verify the provider builds a prompt containing the query,
        candidate descriptions, and program IDs."""
        mock_openai.chat.completions.create.return_value = _completion(
            '{"program_ids": ["prog_1"]}'
        )

        provider.rerank("machine learning", SAMPLE_CANDIDATES, size=1, lang="en")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "machine learning" in prompt
        assert "prog_1" in prompt
        assert "A show about AI" in prompt
        assert "prog_3" in prompt

    def test_uses_language_specific_prompt_template(
        self, provider: OpenAILLMProvider, mock_openai: MagicMock
    ) -> None:
        """French lang should produce a French prompt."""
        mock_openai.chat.completions.create.return_value = _completion(
            '{"program_ids": ["prog_1"]}'
        )

        provider.rerank("query", SAMPLE_CANDIDATES, size=1, lang="fr")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "Sélectionne" in prompt or "programme" in prompt

    def test_sends_temperature_zero(
        self, provider: OpenAILLMProvider, mock_openai: MagicMock
    ) -> None:
        mock_openai.chat.completions.create.return_value = _completion(
            '{"program_ids": ["prog_1"]}'
        )

        provider.rerank("q", SAMPLE_CANDIDATES, size=1, lang="en")

        assert mock_openai.chat.completions.create.call_args[1]["temperature"] == 0.0

    def test_requests_strict_json_schema_output(
        self, provider: OpenAILLMProvider, mock_openai: MagicMock
    ) -> None:
        mock_openai.chat.completions.create.return_value = _completion(
            '{"program_ids": ["prog_2", "prog_1"]}'
        )

        result = provider.rerank("q", SAMPLE_CANDIDATES, size=2, lang="en")

        response_format = mock_openai.chat.completions.create.call_args[1]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == RERANK_RESPONSE_SCHEMA
        assert result == ["prog_2", "prog_1"]

    def test_none_content_falls_back_to_empty_list(
        self, provider: OpenAILLMProvider, mock_openai: MagicMock
    ) -> None:
        """When the API returns content=None (e.g. a refusal), the provider returns []."""
        mock_openai.chat.completions.create.return_value = _completion(None)

        result = provider.rerank("q", SAMPLE_CANDIDATES, size=1, lang="en")

        assert result == []

    def test_respects_custom_base_url(self) -> None:
        """Verify base_url is forwarded to the OpenAI client (Azure scenario)."""
        with patch("hybrid_recsys.providers.llm.openai.OpenAI") as mock_cls:
            OpenAILLMProvider(api_key="k", model="m", base_url="https://azure.openai.com")
            mock_cls.assert_called_once_with(api_key="k", base_url="https://azure.openai.com")
