"""Smoke tests for the offline index-building orchestrator."""

from pathlib import Path

import pytest

from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore
from hybrid_recsys.indexing.vectorizer import Vectorizer
from hybrid_recsys.models import CatalogItem, MediaItem
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
from hybrid_recsys.providers.nlp.spacy import SpacyNLP


class FixedEmbedder(EmbeddingProvider):
    """Returns distinct deterministic embeddings per batch position."""

    def embed(self, text: str) -> list[float]:
        return [0.1] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * 8 for i in range(len(texts))]


@pytest.fixture
def small_catalog():
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


class TestVectorizerBuild:
    def test_build_creates_loadable_index(self, small_catalog, tmp_path) -> None:
        class TestSettings(Settings):
            @property
            def index_dir(self) -> Path:
                return tmp_path

        v = Vectorizer(FixedEmbedder(), SpacyNLP(), TestSettings())
        v.build(small_catalog)

        index = IndexStore().load("en", tmp_path)
        assert set(index.program_ids) == {"p1", "p2"}

    def test_build_preserves_media_data(self, small_catalog, tmp_path) -> None:
        class TestSettings(Settings):
            @property
            def index_dir(self) -> Path:
                return tmp_path

        v = Vectorizer(FixedEmbedder(), SpacyNLP(), TestSettings())
        v.build(small_catalog)

        index = IndexStore().load("en", tmp_path)
        assert len(index.media_data["p2"]) == 2

    def test_build_groups_by_language(self, tmp_path) -> None:
        mixed_catalog = [
            CatalogItem(
                program_id="en1",
                title="English show",
                description="English technology content",
                lang="en",
                media=[
                    MediaItem(media_id="m1", episode=1, duration=600, title="E1")
                ],
            ),
            CatalogItem(
                program_id="fr1",
                title="French show",
                description="Technologie française contenu",
                lang="fr",
                media=[
                    MediaItem(media_id="m2", episode=1, duration=600, title="E1")
                ],
            ),
        ]

        class TestSettings(Settings):
            @property
            def index_dir(self) -> Path:
                return tmp_path

        v = Vectorizer(FixedEmbedder(), SpacyNLP(), TestSettings())
        v.build(mixed_catalog)

        en_index = IndexStore().load("en", tmp_path)
        fr_index = IndexStore().load("fr", tmp_path)
        assert en_index.program_ids == ["en1"]
        assert fr_index.program_ids == ["fr1"]
