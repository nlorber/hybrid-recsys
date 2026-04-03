"""Abstract base class for LLM re-ranking providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for LLM-based re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        """Re-rank candidates using an LLM.

        Args:
            query: User's search query.
            candidates: List of dicts with 'program_id' and 'description'.
            size: Number of results to return.
            lang: Language code (fr/en/de) for prompt generation.

        Returns:
            List of program_ids sorted by relevance, length == size.
        """
