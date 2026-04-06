"""Tests for offline evaluation metrics."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from evaluate import dcg_at_k, ndcg_at_k, precision_at_k, recall_at_k


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_precision(self) -> None:
        assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3) == 0.0

    def test_partial_precision(self) -> None:
        result = precision_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3)
        assert result == pytest.approx(2 / 3)

    def test_k_smaller_than_list(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0

    def test_empty_results(self) -> None:
        assert precision_at_k([], {"a", "b"}, k=3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_recall(self) -> None:
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b", "c"}, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self) -> None:
        assert ndcg_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_ndcg_rewards_top_ranking(self) -> None:
        score_top = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        score_bottom = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        assert score_top > score_bottom

    def test_dcg_formula(self) -> None:
        # Manual DCG: rel=1 at pos 1, rel=0 at pos 2, rel=1 at pos 3
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
        assert dcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(
            1.0 / math.log2(2) + 1.0 / math.log2(4)
        )
