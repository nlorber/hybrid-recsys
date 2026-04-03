"""Tests for duration scoring."""

import math

from hybrid_recsys.retrieval.scorer import duration_score


class TestDurationScore:
    def test_zero_delta(self) -> None:
        assert duration_score(0.0) == 1.0

    def test_positive_delta_media_shorter(self) -> None:
        result = duration_score(10.0)
        assert result == 1.0 / (10.0**2 + 1)

    def test_negative_delta_media_longer(self) -> None:
        result = duration_score(-10.0)
        expected = 1.0 / (10.0**2 + 1) + (-1.0)
        assert result == expected

    def test_large_positive_delta(self) -> None:
        result = duration_score(1000.0)
        assert result > 0.0
        assert result < 0.01

    def test_large_negative_delta(self) -> None:
        result = duration_score(-1000.0)
        assert result < 0.0
        assert math.isclose(result, -1.0, abs_tol=0.01)

    def test_custom_penalty(self) -> None:
        result = duration_score(-10.0, penalty=-0.5)
        expected = 1.0 / (10.0**2 + 1) + (-0.5)
        assert result == expected

    def test_positive_is_always_better_than_negative(self) -> None:
        pos = duration_score(5.0)
        neg = duration_score(-5.0)
        assert pos > neg

    def test_smaller_delta_scores_higher(self) -> None:
        close = duration_score(2.0)
        far = duration_score(20.0)
        assert close > far

    def test_monotonically_decreasing_for_positive(self) -> None:
        scores = [duration_score(float(d)) for d in range(0, 100, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
