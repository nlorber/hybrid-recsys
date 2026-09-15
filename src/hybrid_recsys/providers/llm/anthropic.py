"""Anthropic (Claude) LLM provider for re-ranking.

Mirrors the OpenAI provider: builds the same rerank prompt, calls the Messages API with
the shared JSON-schema output constraint, and parses the returned program-id list. The
``anthropic`` SDK is an optional dependency, imported lazily so the package installs
without it.
"""

import logging

from hybrid_recsys.providers.llm.base import LLMProvider
from hybrid_recsys.retrieval.reranker import (
    RERANK_RESPONSE_SCHEMA,
    build_rerank_prompt,
    parse_rerank_response,
)

logger = logging.getLogger(__name__)


class AnthropicLLMProvider(LLMProvider):
    """LLM re-ranker using the Anthropic Messages API (Claude) with structured output.

    The model must support structured outputs (``output_config.format``).

    Args:
        api_key: Anthropic API key (falls back to the ANTHROPIC_API_KEY env var).
        model: Claude model name.
        base_url: Optional custom base URL.
        max_tokens: Maximum tokens for the re-rank completion.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        base_url: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        from anthropic import Anthropic

        # api_key=None → SDK falls back to ANTHROPIC_API_KEY; base_url=None → default endpoint.
        self._client = Anthropic(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        """Re-rank candidates using Claude."""
        prompt = build_rerank_prompt(query, candidates, size, lang)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": RERANK_RESPONSE_SCHEMA}},
        )
        content = "".join(str(getattr(block, "text", "")) for block in response.content)
        result = parse_rerank_response(content)
        if not result:
            logger.warning("Claude returned an unparseable rerank response")
        return result
