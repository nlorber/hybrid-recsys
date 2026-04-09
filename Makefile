.PHONY: setup index test lint typecheck format serve demo docker clean

## Install all dependencies and download NLP models
setup:
	uv sync --extra dev --extra openai
	uv run python -m spacy download fr_core_news_sm
	uv run python -m spacy download en_core_web_sm
	uv run python -m spacy download de_core_news_sm
	uv run python -m nltk.downloader stopwords
	uv run pre-commit install

## Build search indexes from catalog
index:
	uv run python scripts/generate_catalog.py && uv run hybrid-recsys index

## Run the full test suite with coverage
test:
	uv run pytest

## Run ruff linter
lint:
	uv run ruff check src/ tests/

## Run mypy strict type checking
typecheck:
	uv run mypy --strict src/

## Apply ruff formatter
format:
	uv run ruff format src/ tests/

## Start the API server
serve:
	uv run hybrid-recsys serve

## Run a demo recommendation query
demo:
	uv run hybrid-recsys demo "true crime podcast" --lang en --size 3

## Build Docker image
docker:
	docker compose up --build

## Remove build artifacts, caches, and compiled files
clean:
	rm -rf dist/ build/ .eggs/ *.egg-info/ .venv/
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.json htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
