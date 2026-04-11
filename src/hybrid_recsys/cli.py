"""CLI for hybrid-recsys: index, serve, demo."""

import json
import logging
import platform
import sys

import typer

from hybrid_recsys.config import Settings

app = typer.Typer(help="Multilingual hybrid recommendation engine")


def _warn_macos_arm64() -> None:
    """Warn about Annoy 1.17.3 single-result bug on macOS arm64 + Python 3.12."""
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and sys.version_info[:2] == (3, 12)
    ):
        logging.getLogger(__name__).warning(
            "macOS arm64 + Python 3.12 detected. "
            "Annoy 1.17.3 may return only 1 result per query. "
            "Use Python 3.11 or run on Linux to avoid this issue."
        )


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """Check platform before running any command."""
    _warn_macos_arm64()
    if ctx.invoked_subcommand is None:
        raise typer.Exit()


@app.command()
def index() -> None:
    """Build per-language indexes from the catalog."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings()

    catalog_path = settings.catalog_path
    if not catalog_path.exists():
        typer.echo(f"Catalog not found at {catalog_path}. Run generate_catalog.py first.")
        raise typer.Exit(1)

    from hybrid_recsys.factory import create_embedding_provider
    from hybrid_recsys.indexing.vectorizer import Vectorizer
    from hybrid_recsys.models import CatalogItem
    from hybrid_recsys.providers.nlp.spacy import SpacyNLP

    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)
    catalog = [CatalogItem(**p) for p in data["programs"]]
    typer.echo(f"Loaded {len(catalog)} programs from {catalog_path}")

    embedder = create_embedding_provider(settings)
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

    # Generate catalog if not present
    if not settings.catalog_path.exists():
        typer.echo("Catalog not found. Generating synthetic catalog...")
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        script = root / "scripts" / "generate_catalog.py"
        import subprocess

        subprocess.run([sys.executable, str(script)], check=True)
        typer.echo(f"Catalog written to {settings.catalog_path}")

    # Build indexes if not present
    if not settings.index_dir.exists() or not any(settings.index_dir.iterdir()):
        typer.echo("Indexes not found. Building...")
        index()

    from hybrid_recsys.factory import create_embedding_provider, create_llm_provider
    from hybrid_recsys.models import RecoRequest
    from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

    embedder = create_embedding_provider(settings)
    llm = create_llm_provider(settings)
    pipeline = RecommendationPipeline(embedder, llm, settings)

    request = RecoRequest(query=query, lang=lang, size=size, duration=duration)
    response = pipeline.recommend(request)

    typer.echo(f"\nQuery: '{query}' (lang={lang}, size={size}, duration={duration})")
    typer.echo(f"\nRecommended programs: {response.programs}")
    typer.echo(f"Recommended media:    {response.medias}")


if __name__ == "__main__":
    app()
