"""Multilingual NLP via spaCy with NLTK stopwords."""

import re
from functools import lru_cache

import spacy
from nltk.corpus import stopwords
from spacy.language import Language

LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    "fr": ("french", "fr_core_news_sm"),
    "en": ("english", "en_core_web_sm"),
    "de": ("german", "de_core_news_sm"),
}


class SpacyNLP:
    """Multilingual text preprocessing using spaCy and NLTK stopwords.

    Models and stopwords are cached per language on first load.
    """

    @staticmethod
    @lru_cache(maxsize=3)
    def _load(lang: str) -> tuple[Language, frozenset[str]]:
        """Load spaCy model and stopwords for a language."""
        if lang not in LANGUAGE_MAP:
            msg = f"Unsupported language: {lang}. Must be one of {list(LANGUAGE_MAP)}"
            raise ValueError(msg)
        nltk_name, spacy_model = LANGUAGE_MAP[lang]
        nlp = spacy.load(spacy_model)
        stop_words = frozenset(stopwords.words(nltk_name))
        return nlp, stop_words

    def preprocess(self, text: str, lang: str) -> str:
        """Preprocess text for TF-IDF vectorization.

        Pipeline: lowercase -> remove non-word chars -> spaCy tokenize ->
        lemmatize -> remove stopwords and punctuation -> join.

        Args:
            text: Raw input text.
            lang: Language code (fr/en/de).

        Returns:
            Preprocessed string of space-separated lemmas.
        """
        nlp, stop_words = self._load(lang)
        text = re.sub(r"\W+", " ", text.lower())
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in stop_words and not token.is_punct
        ]
        return " ".join(tokens)
