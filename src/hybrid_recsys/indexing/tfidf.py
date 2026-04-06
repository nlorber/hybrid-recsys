"""TF-IDF vectorization pipeline with multilingual preprocessing."""

from typing import cast

from sklearn.feature_extraction.text import TfidfVectorizer

from hybrid_recsys.providers.nlp.spacy import SpacyNLP

# Caps vocabulary size to control memory usage and noise from rare terms
MAX_TFIDF_FEATURES = 10_000


class TfidfPipeline:
    """Fit TF-IDF on a corpus and transform queries.

    Uses SpacyNLP for language-specific lemmatization and stopword removal
    before vectorization.
    """

    def __init__(self, nlp: SpacyNLP) -> None:
        self._nlp = nlp

    def fit_transform(
        self,
        program_ids: list[str],
        descriptions: list[str],
        lang: str,
    ) -> tuple[TfidfVectorizer, dict[str, list[float]]]:
        """Preprocess descriptions, fit TF-IDF, and return vectors.

        Args:
            program_ids: Program identifiers (same order as descriptions).
            descriptions: Raw program descriptions.
            lang: Language code (fr/en/de).

        Returns:
            Tuple of (fitted vectorizer, dict mapping program_id to TF-IDF vector).
        """
        processed = [self._nlp.preprocess(desc, lang) for desc in descriptions]
        vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
        matrix = vectorizer.fit_transform(processed)

        vectors: dict[str, list[float]] = {}
        for i, pid in enumerate(program_ids):
            vectors[pid] = matrix[i].toarray().flatten().tolist()

        return vectorizer, vectors

    def transform_query(
        self,
        query: str,
        vectorizer: TfidfVectorizer,
        lang: str,
    ) -> list[float]:
        """Preprocess and vectorize a query using a fitted vectorizer.

        Args:
            query: Raw query text.
            vectorizer: Previously fitted TfidfVectorizer.
            lang: Language code (fr/en/de).

        Returns:
            TF-IDF vector as a list of floats.
        """
        processed = self._nlp.preprocess(query, lang)
        vec = vectorizer.transform([processed])
        return cast("list[float]", vec.toarray().flatten().tolist())
