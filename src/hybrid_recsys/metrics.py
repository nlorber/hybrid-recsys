"""Information retrieval metrics: precision, recall, DCG, nDCG."""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / len(top_k)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k results."""
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    score = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG at k. Returns 0 if no relevant items exist."""
    ideal = sorted(ranked[:k], key=lambda x: x in relevant, reverse=True)
    idcg = dcg_at_k(ideal, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg
