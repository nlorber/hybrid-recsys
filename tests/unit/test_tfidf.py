"""Tests for TF-IDF pipeline."""

from hybrid_recsys.indexing.tfidf import TfidfPipeline
from hybrid_recsys.providers.nlp.spacy import SpacyNLP


class TestTfidfPipeline:
    def setup_method(self) -> None:
        self.nlp = SpacyNLP()
        self.pipeline = TfidfPipeline(self.nlp)

    def test_fit_transform_returns_vectorizer_and_vectors(self) -> None:
        program_ids = ["p1", "p2", "p3"]
        descriptions = [
            "Technology and artificial intelligence",
            "History of ancient civilizations",
            "Science and space exploration",
        ]
        vectorizer, vectors = self.pipeline.fit_transform(
            program_ids,
            descriptions,
            "en",
        )
        assert len(vectors) == 3
        assert set(vectors.keys()) == {"p1", "p2", "p3"}
        dims = {len(v) for v in vectors.values()}
        assert len(dims) == 1
        assert dims.pop() > 0

    def test_transform_query_uses_fitted_vocab(self) -> None:
        program_ids = ["p1", "p2"]
        descriptions = ["Technology and computers", "History and archaeology"]
        vectorizer, _ = self.pipeline.fit_transform(program_ids, descriptions, "en")
        query_vec = self.pipeline.transform_query("technology", vectorizer, "en")
        assert isinstance(query_vec, list)
        assert len(query_vec) > 0
        assert any(v != 0.0 for v in query_vec)

    def test_query_unknown_terms_produce_zero_vector(self) -> None:
        program_ids = ["p1"]
        descriptions = ["Technology and computers"]
        vectorizer, _ = self.pipeline.fit_transform(program_ids, descriptions, "en")
        query_vec = self.pipeline.transform_query("xyzzyplugh", vectorizer, "en")
        assert all(v == 0.0 for v in query_vec)

    def test_french_preprocessing(self) -> None:
        program_ids = ["p1", "p2"]
        descriptions = [
            "Les nouvelles technologies informatiques",
            "L'histoire des civilisations anciennes",
        ]
        vectorizer, vectors = self.pipeline.fit_transform(
            program_ids,
            descriptions,
            "fr",
        )
        assert len(vectors) == 2
        assert any(v != 0.0 for v in vectors["p1"])

    def test_german_preprocessing(self) -> None:
        program_ids = ["p1", "p2"]
        descriptions = [
            "Technologie und künstliche Intelligenz",
            "Geschichte der alten Zivilisationen",
        ]
        vectorizer, vectors = self.pipeline.fit_transform(
            program_ids,
            descriptions,
            "de",
        )
        assert len(vectors) == 2
        assert any(v != 0.0 for v in vectors["p1"])

    def test_vectors_have_correct_dimensionality(self) -> None:
        program_ids = ["p1", "p2", "p3"]
        descriptions = [
            "Technology and artificial intelligence",
            "History of ancient civilizations",
            "Science and space exploration",
        ]
        vectorizer, vectors = self.pipeline.fit_transform(
            program_ids,
            descriptions,
            "en",
        )
        vocab_size = len(vectorizer.get_feature_names_out())
        for vec in vectors.values():
            assert len(vec) == vocab_size
