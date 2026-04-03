"""Annoy-based Approximate Nearest Neighbor search."""

from typing import Literal

from annoy import AnnoyIndex

AnnoyMetric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


def build_annoy_index(
    vectors: list[list[float]],
    metric: AnnoyMetric = "angular",
    n_trees: int = 10,
) -> AnnoyIndex:
    """Build an Annoy index from a list of vectors.

    Args:
        vectors: List of vectors to index. All must have same dimensionality.
        metric: Distance metric. 'angular' for cosine similarity.
        n_trees: Number of trees. More trees = better precision, slower build.

    Returns:
        Built AnnoyIndex ready for querying.
    """
    dim = len(vectors[0])
    index = AnnoyIndex(dim, metric)
    for i, vec in enumerate(vectors):
        index.add_item(i, vec)
    index.build(n_trees)
    return index


def query_annoy_index(
    index: AnnoyIndex,
    query_vector: list[float],
    k: int,
) -> list[int]:
    """Query an Annoy index for nearest neighbors.

    Args:
        index: Built AnnoyIndex.
        query_vector: Query vector (same dimension as indexed vectors).
        k: Number of neighbors to return.

    Returns:
        List of indices of nearest neighbors, closest first.
    """
    return index.get_nns_by_vector(query_vector, k, include_distances=False)
