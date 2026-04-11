"""FastAPI application for the recommendation engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request

from hybrid_recsys.config import Settings
from hybrid_recsys.factory import create_embedding_provider, create_llm_provider
from hybrid_recsys.models import RecoRequest, RecoResponse
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize pipeline on startup."""
    settings = Settings()
    logger.info("Initializing recommendation pipeline")
    embedder = create_embedding_provider(settings)
    llm = create_llm_provider(settings)
    app.state.pipeline = RecommendationPipeline(embedder, llm, settings)
    yield
    del app.state.pipeline


def get_pipeline(request: Request) -> RecommendationPipeline:
    """FastAPI dependency: resolve the recommendation pipeline from app state."""
    pipeline: RecommendationPipeline | None = getattr(request.app.state, "pipeline", None)
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
