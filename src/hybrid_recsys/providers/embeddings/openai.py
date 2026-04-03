"""OpenAI embedding provider (works with both OpenAI and Azure OpenAI)."""

from openai import OpenAI

from hybrid_recsys.providers.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the OpenAI API.

    Compatible with both OpenAI and Azure OpenAI via base_url configuration.

    Args:
        api_key: OpenAI API key.
        model: Model name (e.g., 'text-embedding-3-small').
        base_url: Optional custom base URL (for Azure OpenAI or proxies).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string via OpenAI API."""
        response = self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI API."""
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
