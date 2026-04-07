"""Integration tests for the FastAPI app."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer

from hybrid_recsys.api import app
from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.ann_search import build_annoy_index
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline
from tests.conftest import FixedEmbedder


@pytest.fixture
def test_index_dir(tmp_path: Path):
    """Create a temporary index with test data."""
    dim = 8
    tfidf_dim = 3

    emb_vecs = [[0.1 * (i + 1)] * dim for i in range(3)]
    tfidf_vecs = [[0.1 * (i + 1)] * tfidf_dim for i in range(3)]

    ann_emb = build_annoy_index(emb_vecs)
    ann_tfidf = build_annoy_index(tfidf_vecs)

    # Three single-word documents → vocabulary of exactly 3 terms → tfidf_dim=3
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["alpha", "beta", "gamma"])

    index = LanguageIndex(
        program_ids=["p1", "p2", "p3"],
        program_descriptions={
            "p1": "Tech AI",
            "p2": "History Rome",
            "p3": "Science space",
        },
        media_data={
            "p1": [
                {"media_id": "m1", "episode": 1, "duration": 600, "title": "Ep1"},
            ],
            "p2": [
                {"media_id": "m2", "episode": 1, "duration": 900, "title": "Ep1"},
            ],
            "p3": [
                {"media_id": "m3", "episode": 1, "duration": 1200, "title": "Ep1"},
            ],
        },
        ann_embedding=ann_emb,
        ann_tfidf=ann_tfidf,
        tfidf_vectorizer=vectorizer,
        embedding_dim=dim,
        tfidf_dim=tfidf_dim,
    )

    store = IndexStore()
    store.save("en", index, tmp_path)
    return tmp_path


@pytest.fixture
def client(test_index_dir):
    """Create a test client with mock providers and test data."""

    class TestSettings(Settings):
        @property
        def index_dir(self) -> Path:
            return test_index_dir

    pipeline = RecommendationPipeline(
        embedding_provider=FixedEmbedder(),
        llm_provider=MockLLMProvider(),
        settings=TestSettings(),
    )

    with TestClient(app) as c:
        app.state.pipeline = pipeline
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "technology",
                "lang": "en",
                "size": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "programs" in data
        assert "medias" in data

    def test_recommend_respects_size(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "history",
                "lang": "en",
                "size": 1,
            },
        )
        data = response.json()
        assert len(data["programs"]) <= 1
        assert len(data["medias"]) <= 1

    def test_recommend_with_duration(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "science",
                "lang": "en",
                "size": 2,
                "duration": 900,
            },
        )
        assert response.status_code == 200

    def test_recommend_validates_lang(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "test",
                "lang": "xx",
                "size": 1,
            },
        )
        assert response.status_code == 422

    def test_recommend_validates_size_bounds(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "test",
                "lang": "en",
                "size": 0,
            },
        )
        assert response.status_code == 422

        response = client.post(
            "/recommend",
            json={
                "query": "test",
                "lang": "en",
                "size": 11,
            },
        )
        assert response.status_code == 422

    def test_recommend_validates_query_length(self, client) -> None:
        response = client.post(
            "/recommend",
            json={
                "query": "x" * 301,
                "lang": "en",
                "size": 1,
            },
        )
        assert response.status_code == 422
