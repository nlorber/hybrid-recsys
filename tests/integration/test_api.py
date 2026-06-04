"""Integration tests for the FastAPI app."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer

from hybrid_recsys.api import app
from hybrid_recsys.config import Settings
from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.ann_search import build_ann_index
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline
from tests.conftest import FakeEmbeddingProvider


@pytest.fixture
def test_index_dir(tmp_path: Path):
    """Create a temporary index with test data."""
    dim = 8
    tfidf_dim = 3

    # Build dense vectors from the program descriptions with the same text-sensitive
    # embedder used for queries, so the dense retrieval space is coherent and an
    # exact-match query lands on its program (gives the tests real retrieval signal).
    descriptions = ["Tech AI", "History Rome", "Science space"]
    embedder = FakeEmbeddingProvider(dim=dim)
    emb_vecs = embedder.embed_batch(descriptions)
    tfidf_vecs = [[0.1 * (i + 1)] * tfidf_dim for i in range(3)]

    ann_emb = build_ann_index(emb_vecs)
    ann_tfidf = build_ann_index(tfidf_vecs)

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
        program_titles={"p1": "Tech show", "p2": "Rome", "p3": "Space"},
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
        ann_metric="cosine",
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
        embedding_provider=FakeEmbeddingProvider(dim=8),
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

    def test_recommend_discriminates_by_query(self, client) -> None:
        """Different queries retrieve different top programs.

        Guards against a degenerate/constant query embedder: querying a
        program's own description must rank that program first, and two
        distinct queries must not collapse to the same top result.
        """
        r1 = client.post("/recommend/explain", json={"query": "Tech AI", "lang": "en", "size": 3})
        r2 = client.post(
            "/recommend/explain", json={"query": "History Rome", "lang": "en", "size": 3}
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        top1 = r1.json()["programs"][0]["program_id"]
        top2 = r2.json()["programs"][0]["program_id"]
        assert top1 == "p1"
        assert top2 == "p2"

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


class TestRecommendExplainEndpoint:
    def test_explain_returns_enriched_programs(self, client) -> None:
        response = client.post(
            "/recommend/explain",
            json={"query": "technology", "lang": "en", "size": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "technology"
        assert data["lang"] == "en"
        assert len(data["programs"]) <= 2
        if data["programs"]:
            program = data["programs"][0]
            assert {
                "rank",
                "program_id",
                "title",
                "description",
                "lang",
                "rrf_score",
                "sources",
                "reranked",
            }.issubset(program)
            assert program["rank"] == 1
            assert all(src in {"dense", "sparse"} for src in program["sources"])

    def test_explain_validates_lang(self, client) -> None:
        response = client.post(
            "/recommend/explain",
            json={"query": "test", "lang": "xx", "size": 1},
        )
        assert response.status_code == 422


class TestDemoEndpoint:
    def test_root_serves_demo_html(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "hybrid" in response.text.lower()

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

    def test_recommend_returns_400_for_missing_index(self, client) -> None:
        """A valid lang with no built index returns 400, not 500."""
        response = client.post(
            "/recommend",
            json={
                "query": "histoire",
                "lang": "fr",
                "size": 2,
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "fr" in detail
        assert "index" in detail.lower()
