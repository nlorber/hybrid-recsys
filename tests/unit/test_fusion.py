"""Tests for Reciprocal Rank Fusion."""

from hybrid_recsys.retrieval.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        result = reciprocal_rank_fusion(
            ranked_lists=[["a", "b", "c"]],
            k=5,
            output_size=3,
        )
        assert result == ["a", "b", "c"]

    def test_two_lists_with_equal_weights(self) -> None:
        result = reciprocal_rank_fusion(
            ranked_lists=[["a", "b", "c"], ["b", "a", "d"]],
            k=5,
            output_size=4,
        )
        assert set(result[:2]) == {"a", "b"}
        assert set(result[2:]) == {"c", "d"}

    def test_weighted_lists(self) -> None:
        # "a": 3/(1+5) + 2/(2+5) = 0.5 + 0.2857 = 0.7857
        # "b": 3/(2+5) + 2/(1+5) = 0.4286 + 0.3333 = 0.7619
        result = reciprocal_rank_fusion(
            ranked_lists=[["a", "b"], ["b", "a"]],
            weights=[3.0, 2.0],
            k=5,
            output_size=2,
        )
        assert result == ["a", "b"]

    def test_overlap_importance_high_k(self) -> None:
        result = reciprocal_rank_fusion(
            ranked_lists=[["x", "shared"], ["y", "shared"]],
            k=100,
            output_size=3,
        )
        assert result[0] == "shared"

    def test_overlap_importance_low_k(self) -> None:
        result = reciprocal_rank_fusion(
            ranked_lists=[["x", "shared"], ["y", "shared"]],
            k=1,
            output_size=3,
        )
        assert result[0] == "shared"

    def test_output_size_truncates(self) -> None:
        result = reciprocal_rank_fusion(
            ranked_lists=[["a", "b", "c", "d"]],
            k=5,
            output_size=2,
        )
        assert len(result) == 2
        assert result == ["a", "b"]

    def test_empty_list(self) -> None:
        result = reciprocal_rank_fusion(ranked_lists=[[]], k=5, output_size=5)
        assert result == []

    def test_no_lists(self) -> None:
        result = reciprocal_rank_fusion(ranked_lists=[], k=5, output_size=5)
        assert result == []

    def test_default_weights_are_uniform(self) -> None:
        result_no_weights = reciprocal_rank_fusion(
            ranked_lists=[["a", "b"], ["b", "a"]],
            k=5,
            output_size=2,
        )
        result_equal = reciprocal_rank_fusion(
            ranked_lists=[["a", "b"], ["b", "a"]],
            weights=[1.0, 1.0],
            k=5,
            output_size=2,
        )
        assert result_no_weights == result_equal

    def test_exact_scores_program_config(self) -> None:
        """Test with actual program-level config: weights=[3,2], k=5.

        Scores:
          p1: 3/(1+5) + 2/(3+5) = 0.5000 + 0.2500 = 0.7500
          p2: 3/(2+5) + 2/(1+5) = 0.4286 + 0.3333 = 0.7619  <- highest
          p3: 3/(3+5) + 2/(2+5) = 0.3750 + 0.2857 = 0.6607
        """
        result = reciprocal_rank_fusion(
            ranked_lists=[["p1", "p2", "p3"], ["p2", "p3", "p1"]],
            weights=[3.0, 2.0],
            k=5,
            output_size=3,
        )
        assert result == ["p2", "p1", "p3"]
