"""OpenAI LLM provider for re-ranking."""

import logging

from openai import OpenAI

from hybrid_recsys.providers.llm.base import LLMProvider
from hybrid_recsys.retrieval.reranker import build_rerank_prompt, parse_rerank_response

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """LLM re-ranker using OpenAI ChatCompletion.

    Compatible with both OpenAI and Azure OpenAI via base_url.

    Args:
        api_key: OpenAI API key.
        model: Chat model name (e.g., 'gpt-4o-mini').
        base_url: Optional custom base URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        """Re-rank candidates using OpenAI ChatCompletion."""
        prompt = build_rerank_prompt(query, candidates, size, lang)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        content = response.choices[0].message.content or "[]"
        result = parse_rerank_response(content)

        if not result:
            logger.warning("LLM returned unparseable response, returning empty list")
        return result
