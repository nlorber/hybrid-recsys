"""Mock LLM provider that returns candidates in input order."""

from hybrid_recsys.providers.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """No-op re-ranker: returns candidate IDs in input order.

    Used for testing and demo mode without an LLM API.
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        """Return candidate program_ids in input order, truncated to size."""
        return [c["program_id"] for c in candidates][:size]
