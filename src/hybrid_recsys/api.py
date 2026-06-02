"""FastAPI application for the recommendation engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from hybrid_recsys.config import Settings
from hybrid_recsys.factory import create_embedding_provider, create_llm_provider
from hybrid_recsys.models import RecoExplainResponse, RecoRequest, RecoResponse
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


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
    version=version("hybrid-recsys"),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def demo() -> HTMLResponse:
    """Serve the self-contained interactive demo UI."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


_SUPPORTED_LANGUAGES = ("en", "fr", "de")


@app.post("/recommend", response_model=RecoResponse)
def recommend(
    request: RecoRequest,
    pipeline: RecommendationPipeline = Depends(get_pipeline),  # noqa: B008
) -> RecoResponse:
    """Generate recommendations for a query."""
    try:
        return pipeline.recommend(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No index found for language '{request.lang}'. "
                f"Supported languages with built indexes: {list(_SUPPORTED_LANGUAGES)}. "
                "Run 'hybrid-recsys index' to build missing indexes."
            ),
        ) from exc


@app.post("/recommend/explain", response_model=RecoExplainResponse)
def recommend_explain(
    request: RecoRequest,
    pipeline: RecommendationPipeline = Depends(get_pipeline),  # noqa: B008
) -> RecoExplainResponse:
    """Recommendations enriched with hybrid ranking signals (powers the demo UI)."""
    try:
        programs = pipeline.recommend_explained(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No index found for language '{request.lang}'. "
                f"Supported languages with built indexes: {list(_SUPPORTED_LANGUAGES)}. "
                "Run 'hybrid-recsys index' to build missing indexes."
            ),
        ) from exc
    return RecoExplainResponse(query=request.query, lang=request.lang, programs=programs)
