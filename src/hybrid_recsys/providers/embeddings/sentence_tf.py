"""Sentence-transformers embedding provider (local, no API key needed)."""

from sentence_transformers import SentenceTransformer

from hybrid_recsys.providers.embeddings.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Default model: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, trained
    on 50+ languages including French and German).
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]
