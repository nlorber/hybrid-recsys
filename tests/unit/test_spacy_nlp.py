"""Tests for SpacyNLP preprocessing."""

import pytest

from hybrid_recsys.providers.nlp.spacy import SpacyNLP


@pytest.fixture
def nlp() -> SpacyNLP:
    return SpacyNLP()


class TestSpacyNLPPreprocess:
    def test_empty_string_returns_empty(self, nlp: SpacyNLP) -> None:
        assert nlp.preprocess("", "en") == ""

    def test_single_char_input(self, nlp: SpacyNLP) -> None:
        result = nlp.preprocess("x", "en")
        assert isinstance(result, str)

    def test_english_preprocessing(self, nlp: SpacyNLP) -> None:
        result = nlp.preprocess("The dogs are running quickly", "en")
        assert "the" not in result.split()
        assert "are" not in result.split()

    def test_french_preprocessing(self, nlp: SpacyNLP) -> None:
        result = nlp.preprocess("Les chiens courent rapidement", "fr")
        assert "les" not in result.split()
        assert isinstance(result, str)

    def test_german_preprocessing(self, nlp: SpacyNLP) -> None:
        result = nlp.preprocess("Die Hunde laufen schnell", "de")
        assert isinstance(result, str)

    def test_unsupported_language_raises(self, nlp: SpacyNLP) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            nlp.preprocess("hello", "ja")

    def test_strips_non_word_characters(self, nlp: SpacyNLP) -> None:
        result = nlp.preprocess("hello!!! world???", "en")
        assert "!" not in result
        assert "?" not in result
