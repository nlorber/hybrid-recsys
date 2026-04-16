"""Application settings via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from hybrid_recsys.retrieval.ann_search import IndexMetric


class Settings(BaseSettings):
    """Recommendation engine configuration.

    All settings can be overridden via environment variables prefixed with RECSYS_.
    Example: RECSYS_EMBEDDING_PROVIDER=openai
    """

    model_config = SettingsConfigDict(env_prefix="RECSYS_")

    # Provider selection
    embedding_provider: str = "sentence-transformers"
    llm_provider: str = "mock"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Duration scoring
    default_duration: int = 600
    duration_penalty: float = -1.0

    # RRF parameters — program level
    rrf_program_weights: list[float] = [3.0, 2.0]
    rrf_program_k: int = 5

    # RRF parameters — media level
    rrf_media_weights: list[float] = [3.0, 2.0, 3.0]
    rrf_media_k: int = 8

    # ANN parameters (Voyager HNSW)
    ann_metric: IndexMetric = "cosine"
    ann_m: int = 16
    ann_ef_construction: int = 200
    ann_query_k: int = 20

    # Paths
    data_dir: Path = Path("data")

    @property
    def index_dir(self) -> Path:
        """Directory for built indexes."""
        return self.data_dir / "index"

    @property
    def catalog_path(self) -> Path:
        """Path to the catalog JSON file."""
        return self.data_dir / "catalog.json"
