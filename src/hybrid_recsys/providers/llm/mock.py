"""Mock LLM provider that re-ranks by keyword overlap with the query."""

from hybrid_recsys.providers.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Re-ranks candidates by keyword overlap with the query.

    Computes a simple score: number of lowercased query words found in each
    candidate's description. Ties are broken by input order.
    """

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        """Return candidate program_ids ranked by keyword overlap, truncated to size."""
        query_words = set(query.lower().split())

        def _overlap(candidate: dict[str, str]) -> int:
            desc_words = set(candidate.get("description", "").lower().split())
            return len(query_words & desc_words)

        ranked = sorted(candidates, key=_overlap, reverse=True)
        return [c["program_id"] for c in ranked][:size]
