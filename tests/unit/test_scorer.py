"""Tests for duration scoring.

Delta is expressed in seconds; the scorer normalises to minutes internally.
One minute of delta -> base ≈ 0.5; ten minutes -> base ≈ 0.01.
"""

import math

from hybrid_recsys.retrieval.scorer import duration_score

_MIN = 60.0  # seconds per minute


class TestDurationScore:
    def test_zero_delta(self) -> None:
        assert duration_score(0.0) == 1.0

    def test_positive_delta_media_shorter(self) -> None:
        # 60 s = 1 min -> 1/(1^2+1) = 0.5
        result = duration_score(60.0)
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    def test_negative_delta_media_longer(self) -> None:
        # -60 s = -1 min -> base=0.5, score = 0.5 + (-1.0) = -0.5
        result = duration_score(-60.0)
        assert math.isclose(result, -0.5, rel_tol=1e-9)

    def test_large_positive_delta(self) -> None:
        # 600 s = 10 min -> 1/(100+1) ≈ 0.0099
        result = duration_score(600.0)
        assert result > 0.0
        assert result < 0.02

    def test_large_negative_delta(self) -> None:
        # -3600 s = -60 min -> base ≈ 0, score ≈ -1.0
        result = duration_score(-3600.0)
        assert result < 0.0
        assert math.isclose(result, -1.0, abs_tol=0.01)

    def test_custom_penalty(self) -> None:
        # -60 s = -1 min -> base=0.5, score = 0.5 + (-0.5) = 0.0
        result = duration_score(-60.0, penalty=-0.5)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_positive_is_always_better_than_negative(self) -> None:
        pos = duration_score(300.0)  # 5 min short
        neg = duration_score(-300.0)  # 5 min long
        assert pos > neg

    def test_smaller_delta_scores_higher(self) -> None:
        close = duration_score(120.0)  # 2 min
        far = duration_score(1200.0)  # 20 min
        assert close > far

    def test_monotonically_decreasing_for_positive(self) -> None:
        # Steps of 60 s (1 min) from 0 to 9 min
        scores = [duration_score(float(d) * 60.0) for d in range(0, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
