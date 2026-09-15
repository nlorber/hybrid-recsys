"""Tests for the Anthropic LLM provider with a mocked SDK.

The ``anthropic`` package is an optional dependency and is not installed in the
default dev environment, so a fake module is injected into ``sys.modules`` before
the provider performs its lazy import.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from hybrid_recsys.retrieval.reranker import RERANK_RESPONSE_SCHEMA

SAMPLE_CANDIDATES = [
    {"program_id": "prog_1", "description": "A show about AI"},
    {"program_id": "prog_2", "description": "A show about cooking"},
    {"program_id": "prog_3", "description": "A show about history"},
]


class _FakeMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.reply = '{"program_ids": ["prog_1"]}'

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        block = types.SimpleNamespace(type="text", text=self.reply)
        return types.SimpleNamespace(content=[block])


class _FakeAnthropic:
    last_init: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init = kwargs
        self.messages = _FakeMessages()


@pytest.fixture()
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAnthropic]:
    module = types.ModuleType("anthropic")
    module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return _FakeAnthropic


def _make_provider(**kwargs: Any):  # noqa: ANN202
    from hybrid_recsys.providers.llm.anthropic import AnthropicLLMProvider

    return AnthropicLLMProvider(api_key="test-key", **kwargs)


class TestRerank:
    def test_builds_prompt_with_query_and_candidates(self, fake_sdk: Any) -> None:
        provider = _make_provider()
        provider.rerank("machine learning", SAMPLE_CANDIDATES, size=1, lang="en")
        prompt = provider._client.messages.last_kwargs["messages"][0]["content"]
        assert "machine learning" in prompt
        assert "prog_1" in prompt
        assert "A show about AI" in prompt

    def test_parses_program_ids_from_text_block(self, fake_sdk: Any) -> None:
        provider = _make_provider()
        provider._client.messages.reply = '{"program_ids": ["prog_2", "prog_1"]}'
        result = provider.rerank("q", SAMPLE_CANDIDATES, size=2, lang="en")
        assert result == ["prog_2", "prog_1"]

    def test_requests_json_schema_output(self, fake_sdk: Any) -> None:
        provider = _make_provider()
        provider.rerank("q", SAMPLE_CANDIDATES, size=1, lang="en")
        output_config = provider._client.messages.last_kwargs["output_config"]
        assert output_config == {
            "format": {"type": "json_schema", "schema": RERANK_RESPONSE_SCHEMA}
        }

    def test_sends_temperature_zero(self, fake_sdk: Any) -> None:
        provider = _make_provider()
        provider.rerank("q", SAMPLE_CANDIDATES, size=1, lang="en")
        assert provider._client.messages.last_kwargs["temperature"] == 0.0

    def test_unparseable_reply_returns_empty(self, fake_sdk: Any) -> None:
        provider = _make_provider()
        provider._client.messages.reply = "not a list at all"
        assert provider.rerank("q", SAMPLE_CANDIDATES, size=1, lang="en") == []

    def test_forwards_api_key_and_base_url(self, fake_sdk: type[_FakeAnthropic]) -> None:
        _make_provider(base_url="https://proxy.example.com")
        assert fake_sdk.last_init.get("api_key") == "test-key"
        assert fake_sdk.last_init.get("base_url") == "https://proxy.example.com"
