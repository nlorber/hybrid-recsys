"""Tests for the recommendation pipeline with mock providers."""

from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.models import RecoRequest
from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.ann_search import build_annoy_index
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline


class FakeEmbeddingProvider(EmbeddingProvider):
    """Returns deterministic embeddings based on text hash."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def _hash_embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[: self._dim]]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec

    def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]


@pytest.fixture
def pipeline_env(tmp_path: Path):
    """Set up a pipeline with fake data and mock providers."""
    dim = 8
    embedder = FakeEmbeddingProvider(dim=dim)

    descriptions = {
        "p1": "Technology and artificial intelligence advances",
        "p2": "History of ancient Roman civilization",
        "p3": "Space exploration and astronomy discoveries",
        "p4": "Machine learning and neural networks",
        "p5": "Medieval European history and culture",
    }

    program_ids = list(descriptions.keys())
    desc_list = list(descriptions.values())

    emb_vectors = embedder.embed_batch(desc_list)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(desc_list)
    tfidf_vectors = [
        matrix[i].toarray().flatten().tolist() for i in range(len(desc_list))
    ]  # noqa: E501
    tfidf_dim = len(tfidf_vectors[0])

    ann_emb = build_annoy_index(emb_vectors, metric="angular", n_trees=10)
    ann_tfidf = build_annoy_index(tfidf_vectors, metric="angular", n_trees=10)

    media_data = {
        "p1": [
            {"media_id": "m1", "episode": 1, "duration": 600, "title": "Ep1"},
            {"media_id": "m2", "episode": 2, "duration": 1200, "title": "Ep2"},
        ],
        "p2": [{"media_id": "m3", "episode": 1, "duration": 900, "title": "Ep1"}],
        "p3": [{"media_id": "m4", "episode": 1, "duration": 1800, "title": "Ep1"}],
        "p4": [{"media_id": "m5", "episode": 1, "duration": 500, "title": "Ep1"}],
        "p5": [{"media_id": "m6", "episode": 1, "duration": 700, "title": "Ep1"}],
    }

    index = LanguageIndex(
        program_ids=program_ids,
        program_descriptions=descriptions,
        media_data=media_data,
        ann_embedding=ann_emb,
        ann_tfidf=ann_tfidf,
        tfidf_vectorizer=vectorizer,
        embedding_dim=dim,
        tfidf_dim=tfidf_dim,
    )

    store = IndexStore()
    store.save("en", index, tmp_path)

    class TestSettings(Settings):
        @property
        def index_dir(self) -> Path:
            return tmp_path

    pipeline = RecommendationPipeline(
        embedding_provider=embedder,
        llm_provider=MockLLMProvider(),
        settings=TestSettings(),
    )

    return pipeline


class TestRecommendationPipeline:
    def test_returns_correct_response_shape(self, pipeline_env) -> None:
        request = RecoRequest(query="artificial intelligence", lang="en", size=3)
        response = pipeline_env.recommend(request)
        assert len(response.programs) <= 3
        assert len(response.medias) <= 3
        assert all(isinstance(p, str) for p in response.programs)
        assert all(isinstance(m, str) for m in response.medias)

    def test_returns_valid_program_ids(self, pipeline_env) -> None:
        request = RecoRequest(query="technology", lang="en", size=3)
        response = pipeline_env.recommend(request)
        valid_ids = {"p1", "p2", "p3", "p4", "p5"}
        for pid in response.programs:
            assert pid in valid_ids

    def test_returns_valid_media_ids(self, pipeline_env) -> None:
        request = RecoRequest(query="history", lang="en", size=3)
        response = pipeline_env.recommend(request)
        valid_ids = {"m1", "m2", "m3", "m4", "m5", "m6"}
        for mid in response.medias:
            assert mid in valid_ids

    def test_respects_size_parameter(self, pipeline_env) -> None:
        for size in [1, 2, 5]:
            request = RecoRequest(query="science", lang="en", size=size)
            response = pipeline_env.recommend(request)
            assert len(response.programs) <= size
            assert len(response.medias) <= size

    def test_uses_default_duration_when_not_specified(self, pipeline_env) -> None:
        request = RecoRequest(query="technology", lang="en", size=3)
        response = pipeline_env.recommend(request)
        assert response.medias

    def test_uses_explicit_duration(self, pipeline_env) -> None:
        request = RecoRequest(query="technology", lang="en", size=3, duration=1800)
        response = pipeline_env.recommend(request)
        assert response.medias

    def test_media_are_earliest_episodes(self, pipeline_env) -> None:
        request = RecoRequest(query="technology", lang="en", size=5)
        response = pipeline_env.recommend(request)
        # m1 is episode 1 of p1, m2 is episode 2. Only m1 should appear.
        if "m1" in response.medias or "m2" in response.medias:
            assert "m1" in response.medias
            assert "m2" not in response.medias
