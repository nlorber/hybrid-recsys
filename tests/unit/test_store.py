"""Tests for IndexStore save/load round-trip."""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from hybrid_recsys.indexing.store import IndexStore, LanguageIndex
from hybrid_recsys.retrieval.ann_search import build_annoy_index


@pytest.fixture
def sample_index():
    """Build a minimal LanguageIndex for round-trip testing."""
    dim = 4
    tfidf_dim = 3
    emb_vecs = [[0.1 * (i + 1)] * dim for i in range(3)]
    tfidf_vecs = [[0.3, 0.6, 0.0], [0.0, 0.5, 0.5], [0.7, 0.1, 0.2]]

    ann_emb = build_annoy_index(emb_vecs)
    ann_tfidf = build_annoy_index(tfidf_vecs)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(["alpha beta", "beta gamma", "alpha gamma"])

    return LanguageIndex(
        program_ids=["p1", "p2", "p3"],
        program_descriptions={"p1": "first", "p2": "second", "p3": "third"},
        media_data={
            "p1": [{"media_id": "m1", "episode": 1, "duration": 600, "title": "E1"}],
            "p2": [],
        },
        ann_embedding=ann_emb,
        ann_tfidf=ann_tfidf,
        tfidf_vectorizer=vectorizer,
        embedding_dim=dim,
        tfidf_dim=tfidf_dim,
        ann_metric="angular",
    )


class TestIndexStoreRoundTrip:
    def test_program_ids_preserved(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        assert loaded.program_ids == ["p1", "p2", "p3"]

    def test_descriptions_preserved(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        assert loaded.program_descriptions == {
            "p1": "first",
            "p2": "second",
            "p3": "third",
        }

    def test_media_data_preserved(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        assert loaded.media_data["p1"] == [
            {"media_id": "m1", "episode": 1, "duration": 600, "title": "E1"}
        ]
        assert loaded.media_data["p2"] == []

    def test_dimensions_preserved(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        assert loaded.embedding_dim == 4
        assert loaded.tfidf_dim == 3

    def test_ann_index_queryable_after_load(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        results = loaded.ann_embedding.get_nns_by_vector([0.1, 0.1, 0.1, 0.1], 2)
        assert len(results) > 0

    def test_vectorizer_usable_after_load(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        vec = loaded.tfidf_vectorizer.transform(["alpha"])
        assert vec.shape[1] == 3

    def test_load_missing_lang_raises_file_not_found(self, tmp_path) -> None:
        store = IndexStore()
        with pytest.raises(FileNotFoundError, match="No index found for language 'xx'"):
            store.load("xx", tmp_path)

    def test_ann_metric_preserved(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        loaded = store.load("en", tmp_path)
        assert loaded.ann_metric == "angular"

    def test_multiple_languages_do_not_interfere(self, sample_index, tmp_path) -> None:
        store = IndexStore()
        store.save("en", sample_index, tmp_path)
        store.save("fr", sample_index, tmp_path)
        loaded_en = store.load("en", tmp_path)
        loaded_fr = store.load("fr", tmp_path)
        assert loaded_en.program_ids == loaded_fr.program_ids
