"""Shared test fixtures for hybrid-recsys."""

import pytest

from hybrid_recsys.models import CatalogItem, MediaItem
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider


class FixedEmbedder(EmbeddingProvider):
    """Returns distinct deterministic embeddings per batch position."""

    def embed(self, text: str) -> list[float]:
        return [0.1] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]


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
