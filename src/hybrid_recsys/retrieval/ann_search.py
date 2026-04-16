"""HNSW-based Approximate Nearest Neighbor search via Voyager."""

from typing import Literal

from voyager import Index, Space

IndexMetric = Literal["cosine", "euclidean", "dot"]

_SPACE_BY_METRIC: dict[IndexMetric, Space] = {
    "cosine": Space.Cosine,
    "euclidean": Space.Euclidean,
    "dot": Space.InnerProduct,
}


def build_ann_index(
    vectors: list[list[float]],
    metric: IndexMetric = "cosine",
    m: int = 16,
    ef_construction: int = 200,
) -> Index:
    """Build a Voyager HNSW index from a list of vectors.

    Args:
        vectors: List of vectors to index. All must have same dimensionality.
        metric: Distance metric. 'cosine' for normalized similarity.
        m: Graph degree parameter. Higher = better recall, more memory.
        ef_construction: Build-time candidate list size. Higher = better recall,
            slower build.

    Returns:
        Built Voyager Index ready for querying.
    """
    dim = len(vectors[0])
    index = Index(
        _SPACE_BY_METRIC[metric], num_dimensions=dim, M=m, ef_construction=ef_construction
    )
    index.add_items(vectors)
    return index


def query_ann_index(
    index: Index,
    query_vector: list[float],
    k: int,
) -> list[int]:
    """Query a Voyager index for nearest neighbors.

    Args:
        index: Built Voyager Index.
        query_vector: Query vector (same dimension as indexed vectors).
        k: Number of neighbors to return.

    Returns:
        List of indices of nearest neighbors, closest first.
    """
    neighbors, _ = index.query(query_vector, k=k)
    return [int(i) for i in neighbors]
