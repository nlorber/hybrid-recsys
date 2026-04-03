"""Reciprocal Rank Fusion for combining multiple ranked lists."""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 5,
    output_size: int = 10,
) -> list[str]:
    """Fuse multiple ranked lists into one using Reciprocal Rank Fusion.

    Score for item i = sum over lists L of: weight_L / (rank_of_i_in_L + k)

    Args:
        ranked_lists: List of ranked item ID lists (most relevant first).
        weights: Relative weight per list. Defaults to uniform weights.
        k: Overlap importance parameter. Higher values favor items appearing
           in multiple lists; lower values favor items ranked high in any list.
        output_size: Maximum number of items in the fused output.

    Returns:
        Fused list of item IDs sorted by descending RRF score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: defaultdict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(ranked_list, start=1):
            scores[item] += weight / (rank + k)

    fused = sorted(scores, key=lambda item: scores[item], reverse=True)
    return fused[:output_size]
