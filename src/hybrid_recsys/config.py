"""Application settings via pydantic-settings."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hybrid_recsys.retrieval.ann_search import IndexMetric


class Settings(BaseSettings):
    """Recommendation engine configuration.

    All settings can be overridden via environment variables prefixed with RECSYS_.
    Example: RECSYS_EMBEDDING_PROVIDER=openai
    """

    model_config = SettingsConfigDict(env_prefix="RECSYS_", env_file=".env", extra="ignore")

    # Embedding provider (vendor-neutral; selected by name, configured generically)
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None

    # LLM re-ranking provider (vendor-neutral; selected by name, configured generically)
    llm_provider: str = "mock"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None

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

    # LLM re-ranking
    llm_rerank_timeout: float = 5.0

    # Paths
    data_dir: Path = Path("data")

    @field_validator(
        "embedding_api_key",
        "embedding_base_url",
        "llm_model",
        "llm_api_key",
        "llm_base_url",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat blank env vars (e.g. ``RECSYS_LLM_BASE_URL=``) as unset.

        pydantic reads an empty env var as ``""``; passing that as a provider
        ``base_url``/``api_key`` breaks the SDK clients, so coerce blanks to None.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @property
    def index_dir(self) -> Path:
        """Directory for built indexes."""
        return self.data_dir / "index"

    @property
    def catalog_path(self) -> Path:
        """Path to the catalog JSON file."""
        return self.data_dir / "catalog.json"
