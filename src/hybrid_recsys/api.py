"""FastAPI application for the recommendation engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request

from hybrid_recsys.config import Settings
from hybrid_recsys.models import RecoRequest, RecoResponse
from hybrid_recsys.providers.embeddings.sentence_tf import SentenceTransformerProvider
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hybrid_recsys.providers.embeddings.base import EmbeddingProvider
    from hybrid_recsys.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def _create_pipeline(settings: Settings) -> RecommendationPipeline:
    """Create the pipeline with the configured providers."""
    embedder: EmbeddingProvider
    if settings.embedding_provider == "sentence-transformers":
        embedder = SentenceTransformerProvider(settings.embedding_model)
    elif settings.embedding_provider == "openai":
        from hybrid_recsys.providers.embeddings.openai import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
        )
    else:
        msg = f"Unknown embedding provider: {settings.embedding_provider}"
        raise ValueError(msg)

    llm: LLMProvider
    if settings.llm_provider == "mock":
        llm = MockLLMProvider()
    elif settings.llm_provider == "openai":
        from hybrid_recsys.providers.llm.openai import OpenAILLMProvider

        llm = OpenAILLMProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    else:
        msg = f"Unknown LLM provider: {settings.llm_provider}"
        raise ValueError(msg)

    return RecommendationPipeline(embedder, llm, settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize pipeline on startup."""
    settings = Settings()
    logger.info("Initializing recommendation pipeline")
    app.state.pipeline = _create_pipeline(settings)
    yield
    del app.state.pipeline


def get_pipeline(request: Request) -> RecommendationPipeline:
    """FastAPI dependency: resolve the recommendation pipeline from app state."""
    pipeline: RecommendationPipeline | None = getattr(
        request.app.state, "pipeline", None
    )
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


app = FastAPI(
    title="hybrid-recsys",
    description="Multilingual hybrid recommendation engine",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecoResponse)
def recommend(
    request: RecoRequest,
    pipeline: RecommendationPipeline = Depends(get_pipeline),  # noqa: B008
) -> RecoResponse:
    """Generate recommendations for a query."""
    return pipeline.recommend(request)
