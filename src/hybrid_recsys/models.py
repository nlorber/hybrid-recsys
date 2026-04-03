"""Domain models for the recommendation engine."""

from pydantic import BaseModel, Field


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

    query: str = Field(max_length=300)
    lang: str = Field(pattern=r"^(fr|en|de)$")
    size: int = Field(default=3, ge=1, le=10)
    duration: int | None = Field(default=None, gt=0)


class RecoResponse(BaseModel):
    """Recommendation response with ranked program and media IDs."""

    programs: list[str]
    medias: list[str]
