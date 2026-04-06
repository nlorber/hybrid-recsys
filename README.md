# hybrid-recsys

![CI](https://github.com/nlorber/hybrid-recsys/actions/workflows/test.yml/badge.svg)
<!-- Coverage badge: manually maintained — update after significant test changes -->
![Coverage](https://img.shields.io/badge/coverage-83%25-yellowgreen)

Multilingual content recommendation engine combining dual retrieval, Reciprocal
Rank Fusion, and LLM re-ranking.

> **Note:** This project is an anonymized and rewritten version of a system originally built in a professional context. All proprietary code, data, company references, and internal infrastructure details have been removed. The synthetic podcast catalog replaces real content data.

---

## Architecture

```
query
  │
  ├─── embed (dense) ──────────────────┐
  │                                    │
  └─── TF-IDF vectorise ───────────────┤
                                       │
                              ANN search (Annoy)
                                       │
                                  RRF fusion
                                  (programs)
                                       │
                                  LLM rerank
                                (with fallback)
                                       │
                                    programs
                                       │
              ┌────────────────────────┼──────────────────────┐
              ▼                        ▼                      ▼
       emb media list         tfidf media list        duration scoring
              │                        │                      │
              └────────────────────────┼──────────────────────┘
                                       │
                                  RRF fusion
                                   (media)
                                       │
                                     media
```

**Pipeline in plain English:**

1. The query is embedded (dense vector) and TF-IDF vectorised (sparse) in
   parallel.
2. Both vectors are searched against per-language Annoy indexes, producing two
   ranked lists of programs.
3. The lists are merged via Reciprocal Rank Fusion.
4. An optional LLM re-ranks the fused list (falls back to RRF on failure).
5. For each top program the earliest episode is selected; those episodes form three
   ranked lists (embedding order, TF-IDF order, duration proximity score).
6. A second RRF pass over the media lists yields the final media ranking.

---

## Quick Start

> **Note:** First run downloads ~560 MB of models (3 spaCy + sentence-transformers). Allow 5-10 minutes on a typical connection.

```bash
# 1. Clone & install (one command installs all dependencies + language models)
git clone https://github.com/nlorber/hybrid-recsys.git
cd hybrid-recsys
make setup

# 2. Generate a synthetic catalog and build indexes
poetry run python scripts/generate_catalog.py
poetry run hybrid-recsys index

# 3. Run a demo query
poetry run hybrid-recsys demo "true crime podcast" --lang en --size 3
```

<details>
<summary>Manual install (without Make)</summary>

```bash
poetry install
poetry run python -m spacy download en_core_web_sm
poetry run python -m spacy download fr_core_news_sm
poetry run python -m spacy download de_core_news_sm
poetry run python -m nltk.downloader stopwords
```
</details>

---

## API

Start the FastAPI server:

```bash
poetry run hybrid-recsys serve
# Listening on http://0.0.0.0:8000
```

POST a recommendation request:

```bash
curl -s -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"query": "science for kids", "lang": "en", "size": 3}' \
  | jq .
```

Example response:

```json
{
  "programs": ["prg_0042", "prg_0017", "prg_0091"],
  "medias":   ["med_00224", "med_00089", "med_00490"]
}
```

Interactive API docs: <http://localhost:8000/docs>

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Dual retrieval** (embedding + TF-IDF) | Embeddings capture semantics; TF-IDF captures exact keywords. Combining both handles a wider range of queries. |
| **Reciprocal Rank Fusion** | No labeled training data required; robust to score-scale differences between embedding and TF-IDF results. |
| **LLM re-ranking with fallback** | LLMs add semantic understanding but are unreliable; falling back to RRF keeps the pipeline stable. |
| **Annoy for ANN search** | Zero extra dependencies, native cosine similarity (angular metric), sufficient for catalogs up to ~1 M items. |
| **Per-language indexes** | Keeps TF-IDF vocabularies clean and query-time routing trivial. |
| **Configurable duration penalty** | Asymmetric: media shorter than requested incurs only natural decay; media longer than requested gets an additional penalty. |

See [docs/DESIGN.md](docs/DESIGN.md) for the full rationale and formulas.

---

## Configuration

All settings use the `RECSYS_` prefix and can be set via environment variables or
a `.env` file:

```bash
RECSYS_EMBEDDING_PROVIDER=openai
RECSYS_EMBEDDING_MODEL=text-embedding-3-small
RECSYS_OPENAI_API_KEY=sk-...
RECSYS_LLM_PROVIDER=openai
```

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for the full list of variables and
available providers.

---

## Architecture Details

See [docs/DESIGN.md](docs/DESIGN.md) for:

- Full RRF formula with parameter explanation
- Duration scoring formula and asymmetric penalty rationale
- Annoy vs. FAISS / ScaNN trade-off analysis
- Indexing strategy and provider abstraction design

---

## Tests

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit

# Integration tests only
poetry run pytest tests/integration
```

---

## Known Issues

### Annoy 1.17.3 on macOS arm64 + Python 3.12

`get_nns_by_vector` returns only 1 result regardless of the `n` argument. This is
a known upstream bug in Annoy 1.17.3 on macOS arm64 combined with Python 3.12.

**Workaround:** Use Python 3.11 on macOS, or run on Linux where the issue does not
occur. The project works correctly on Linux / CI.

---

## Tech Stack

| Layer | Library |
|---|---|
| Embeddings | sentence-transformers / OpenAI |
| Sparse retrieval | scikit-learn TF-IDF |
| ANN index | Annoy |
| NLP tokenisation | spaCy |
| Re-ranking | OpenAI (optional) / Mock |
| API server | FastAPI + Uvicorn |
| CLI | Typer |
| Config | pydantic-settings |
