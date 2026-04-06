"""Tests for prompt building, LLM response parsing, and rerank_programs behavior."""

from hybrid_recsys.providers.llm.base import LLMProvider
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.reranker import (
    build_rerank_prompt,
    parse_rerank_response,
    rerank_programs,
)


class TestBuildRerankPrompt:
    def test_includes_query(self) -> None:
        candidates = [{"program_id": "p1", "description": "Tech podcast"}]
        prompt = build_rerank_prompt("artificial intelligence", candidates, 1, "en")
        assert "artificial intelligence" in prompt

    def test_includes_all_candidate_ids(self) -> None:
        candidates = [
            {"program_id": "p1", "description": "Tech"},
            {"program_id": "p2", "description": "History"},
        ]
        prompt = build_rerank_prompt("query", candidates, 2, "en")
        assert "p1" in prompt
        assert "p2" in prompt

    def test_includes_size(self) -> None:
        candidates = [{"program_id": "p1", "description": "Tech"}]
        prompt = build_rerank_prompt("query", candidates, 3, "en")
        assert "3" in prompt

    def test_french_template(self) -> None:
        candidates = [{"program_id": "p1", "description": "Tech"}]
        prompt = build_rerank_prompt("intelligence artificielle", candidates, 1, "fr")
        assert "Sélectionne" in prompt
        assert "intelligence artificielle" in prompt

    def test_german_template(self) -> None:
        candidates = [{"program_id": "p1", "description": "Tech"}]
        prompt = build_rerank_prompt("Technologie", candidates, 1, "de")
        assert "Wählen Sie" in prompt

    def test_unknown_lang_falls_back_to_english(self) -> None:
        candidates = [{"program_id": "p1", "description": "Tech"}]
        prompt = build_rerank_prompt("query", candidates, 1, "es")
        assert "Select exactly" in prompt

    def test_includes_descriptions(self) -> None:
        candidates = [{"program_id": "p1", "description": "Deep space astronomy show"}]
        prompt = build_rerank_prompt("space", candidates, 1, "en")
        assert "Deep space astronomy show" in prompt


class TestParseRerankResponse:
    def test_valid_python_list(self) -> None:
        result = parse_rerank_response("['p1', 'p2', 'p3']")
        assert result == ["p1", "p2", "p3"]

    def test_valid_json_style_list(self) -> None:
        result = parse_rerank_response('["p1", "p2"]')
        assert result == ["p1", "p2"]

    def test_empty_list(self) -> None:
        result = parse_rerank_response("[]")
        assert result == []

    def test_empty_string_returns_empty(self) -> None:
        result = parse_rerank_response("")
        assert result == []

    def test_not_a_list_returns_empty(self) -> None:
        result = parse_rerank_response("not a list at all")
        assert result == []

    def test_extra_text_before_list_returns_empty(self) -> None:
        result = parse_rerank_response("Here are my picks: ['p1', 'p2']")
        assert result == []

    def test_non_string_elements_returns_empty(self) -> None:
        result = parse_rerank_response("[1, 2, 3]")
        assert result == []

    def test_mixed_types_returns_empty(self) -> None:
        result = parse_rerank_response("['p1', 2, 'p3']")
        assert result == []

    def test_dict_returns_empty(self) -> None:
        result = parse_rerank_response("{'key': 'value'}")
        assert result == []


class FailingLLMProvider(LLMProvider):
    """LLM provider that always raises, for testing fallback behavior."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        raise RuntimeError("LLM service unavailable")


class UnderSelectingLLMProvider(LLMProvider):
    """LLM provider that returns fewer results than requested."""

    def __init__(self, n: int = 1) -> None:
        self._n = n

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, str]],
        size: int,
        lang: str,
    ) -> list[str]:
        return [c["program_id"] for c in candidates][: self._n]


DESCRIPTIONS = {
    "p1": "Tech podcast",
    "p2": "History show",
    "p3": "Science series",
    "p4": "Sports talk",
    "p5": "Music review",
}
RRF_RANKING = ["p1", "p2", "p3", "p4", "p5"]


class TestRerankPrograms:
    def test_skips_llm_when_rrf_within_size(self) -> None:
        result = rerank_programs(
            llm=FailingLLMProvider(),
            query="tech",
            rrf_ranking=["p1", "p2"],
            descriptions=DESCRIPTIONS,
            size=3,
            lang="en",
        )
        assert result == ["p1", "p2"]

    def test_fallback_to_rrf_on_llm_failure(self) -> None:
        result = rerank_programs(
            llm=FailingLLMProvider(),
            query="tech",
            rrf_ranking=RRF_RANKING,
            descriptions=DESCRIPTIONS,
            size=3,
            lang="en",
        )
        assert result == ["p1", "p2", "p3"]

    def test_pads_from_rrf_when_llm_underselects(self) -> None:
        result = rerank_programs(
            llm=UnderSelectingLLMProvider(n=1),
            query="tech",
            rrf_ranking=RRF_RANKING,
            descriptions=DESCRIPTIONS,
            size=3,
            lang="en",
        )
        assert len(result) == 3
        assert result[0] == "p1"
        # Remaining slots filled from RRF order, skipping p1
        assert result[1] == "p2"
        assert result[2] == "p3"

    def test_padding_avoids_duplicates(self) -> None:
        result = rerank_programs(
            llm=UnderSelectingLLMProvider(n=2),
            query="tech",
            rrf_ranking=RRF_RANKING,
            descriptions=DESCRIPTIONS,
            size=3,
            lang="en",
        )
        assert len(result) == len(set(result))

    def test_normal_llm_rerank(self) -> None:
        result = rerank_programs(
            llm=MockLLMProvider(),
            query="tech",
            rrf_ranking=RRF_RANKING,
            descriptions=DESCRIPTIONS,
            size=3,
            lang="en",
        )
        assert len(result) == 3
        assert all(pid in DESCRIPTIONS for pid in result)
