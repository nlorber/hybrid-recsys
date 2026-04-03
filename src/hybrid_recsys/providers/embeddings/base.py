"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors (same order as input).
        """
