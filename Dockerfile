# ---- builder ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

COPY src/ src/
COPY scripts/ scripts/
RUN poetry install --without dev

RUN poetry run python -m spacy download fr_core_news_sm && \
    poetry run python -m spacy download en_core_web_sm && \
    poetry run python -m spacy download de_core_news_sm && \
    poetry run python -c "import nltk; nltk.download('stopwords')"

RUN poetry run python scripts/generate_catalog.py && \
    poetry run hybrid-recsys index

# ---- runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY --from=builder /app/data data

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["hybrid-recsys", "serve"]
