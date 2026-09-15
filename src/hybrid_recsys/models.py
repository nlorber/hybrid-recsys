"""Domain models for the recommendation engine."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class MediaItem(BaseModel):
    """A single episode within a program."""

    media_id: str
    episode: int
    duration: int  # seconds
    title: str


class CatalogItem(BaseModel):
    """A podcast program with its episodes."""

    program_id: str
    title: str
    description: str
    lang: str
    media: list[MediaItem]


class RecoRequest(BaseModel):
    """Recommendation request."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    lang: str = Field(pattern=r"^(fr|en|de)$")
    size: int = Field(default=3, ge=1, le=10)
    duration: int | None = Field(default=None, gt=0)


class RecoResponse(BaseModel):
    """Recommendation response with ranked program and media IDs."""

    programs: list[str]
    medias: list[str]


class ExplainedProgram(BaseModel):
    """A ranked program enriched with retrieval explainability metadata."""

    rank: int
    program_id: str
    title: str
    description: str
    lang: str
    rrf_score: float
    sources: list[str]  # retrievers that surfaced it: "dense" and/or "sparse"
    reranked: bool  # whether the LLM re-ranker moved it from its RRF position


class RecoExplainResponse(BaseModel):
    """Enriched recommendation response exposing the hybrid ranking signals."""

    query: str
    lang: str
    programs: list[ExplainedProgram]
