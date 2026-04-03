"""CLI for hybrid-recsys: index, serve, demo."""

import json
import logging

import typer

from hybrid_recsys.config import Settings

app = typer.Typer(help="Multilingual hybrid recommendation engine")


@app.command()
def index() -> None:
    """Build per-language indexes from the catalog."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings()

    catalog_path = settings.catalog_path
    if not catalog_path.exists():
        typer.echo(
            f"Catalog not found at {catalog_path}. Run generate_catalog.py first."
        )
        raise typer.Exit(1)

    from hybrid_recsys.indexing.vectorizer import Vectorizer
    from hybrid_recsys.models import CatalogItem
    from hybrid_recsys.providers.embeddings.sentence_tf import (
        SentenceTransformerProvider,
    )
    from hybrid_recsys.providers.nlp.spacy import SpacyNLP

    with open(catalog_path) as f:
        data = json.load(f)
    catalog = [CatalogItem(**p) for p in data["programs"]]
    typer.echo(f"Loaded {len(catalog)} programs from {catalog_path}")

    embedder = SentenceTransformerProvider(settings.embedding_model)
    nlp = SpacyNLP()
    vectorizer = Vectorizer(embedder, nlp, settings)
    vectorizer.build(catalog)
    typer.echo(f"Indexes built and saved to {settings.index_dir}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("hybrid_recsys.api:app", host=host, port=port, reload=False)


@app.command()
def demo(
    query: str,
    lang: str = "en",
    size: int = 3,
    duration: int | None = None,
) -> None:
    """Index the catalog and run a single recommendation query."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings()

    # Build indexes if not present
    if not settings.index_dir.exists() or not any(settings.index_dir.iterdir()):
        typer.echo("Indexes not found. Building...")
        index()

    from hybrid_recsys.models import RecoRequest
    from hybrid_recsys.providers.embeddings.sentence_tf import (
        SentenceTransformerProvider,
    )
    from hybrid_recsys.providers.llm.mock import MockLLMProvider
    from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

    embedder = SentenceTransformerProvider(settings.embedding_model)
    llm = MockLLMProvider()
    pipeline = RecommendationPipeline(embedder, llm, settings)

    request = RecoRequest(query=query, lang=lang, size=size, duration=duration)
    response = pipeline.recommend(request)

    typer.echo(f"\nQuery: '{query}' (lang={lang}, size={size}, duration={duration})")
    typer.echo(f"\nRecommended programs: {response.programs}")
    typer.echo(f"Recommended media:    {response.medias}")


if __name__ == "__main__":
    app()
