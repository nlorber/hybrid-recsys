"""Shared test fixtures for hybrid-recsys."""

import hashlib

import pytest

from hybrid_recsys.models import CatalogItem, MediaItem
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider


class FixedEmbedder(EmbeddingProvider):
    """Returns distinct deterministic embeddings per batch position."""

    def embed(self, text: str) -> list[float]:
        return [0.1] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Returns deterministic, text-sensitive embeddings via SHA-256 hash.

    Different input texts produce different unit vectors, giving the retrieval
    pipeline a real signal to discriminate between queries and catalog items.
    Keep ``embed_batch`` consistent with ``embed`` so index building and query
    embedding use the same underlying transform.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def _hash_embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[: self._dim]]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec

    def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]


@pytest.fixture()
def small_catalog() -> list[CatalogItem]:
    """Two-program catalog for lightweight tests."""
    return [
        CatalogItem(
            program_id="p1",
            title="Tech show",
            description="Technology and artificial intelligence",
            lang="en",
            media=[MediaItem(media_id="m1", episode=1, duration=600, title="Ep1")],
        ),
        CatalogItem(
            program_id="p2",
            title="History show",
            description="History of ancient Rome",
            lang="en",
            media=[
                MediaItem(media_id="m2", episode=1, duration=900, title="Ep1"),
                MediaItem(media_id="m3", episode=2, duration=1200, title="Ep2"),
            ],
        ),
    ]
