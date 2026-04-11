# ---- builder ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

ENV HF_HOME=/app/.hf_cache

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev

ENV NLTK_DATA=/app/nltk_data

RUN uv run python -m spacy download fr_core_news_sm && \
    uv run python -m spacy download en_core_web_sm && \
    uv run python -m spacy download de_core_news_sm && \
    uv run python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')"

RUN uv run python scripts/generate_catalog.py && \
    uv run hybrid-recsys index

# ---- runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/.hf_cache .hf_cache
COPY --from=builder /app/nltk_data nltk_data
COPY --from=builder /app/src src
COPY --from=builder /app/data data

ENV PATH="/app/.venv/bin:$PATH"
ENV NLTK_DATA="/app/nltk_data"
ENV HF_HOME="/app/.hf_cache"

EXPOSE 8000

ENTRYPOINT ["hybrid-recsys", "serve"]
