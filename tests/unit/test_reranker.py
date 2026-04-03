"""Tests for prompt building and LLM response parsing."""

from hybrid_recsys.retrieval.reranker import build_rerank_prompt, parse_rerank_response


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
