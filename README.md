# hybrid-recsys

![CI](https://github.com/nlorber/hybrid-recsys/actions/workflows/test.yml/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/nlorber/658807b3d9251dbce468b6c738ccd10d/raw/coverage-hybrid-recsys.json)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![mypy](https://img.shields.io/badge/type_check-mypy_strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Multilingual content recommendation engine combining dual retrieval, Reciprocal
Rank Fusion, and optional LLM re-ranking.

---

## Architecture

```mermaid
flowchart TD
    Q[Query] --> EMB[Dense Embedding]
    Q --> TFIDF[TF-IDF Vectorise]

    EMB --> ANN_E[Voyager HNSW<br/>embedding index]
    TFIDF --> ANN_T[Voyager HNSW<br/>TF-IDF index]

    ANN_E --> RRF1[RRF Fusion<br/>programs]
    ANN_T --> RRF1
    RRF1 --> LLM[LLM Re-rank<br/>with fallback]

    LLM --> EMB_M[Embedding<br/>media list]
    LLM --> TFIDF_M[TF-IDF<br/>media list]
    LLM --> DUR[Duration<br/>scoring]

    EMB_M --> RRF2[RRF Fusion<br/>media]
    TFIDF_M --> RRF2
    DUR --> RRF2

    RRF2 --> OUT[Final media ranking]
```

**Pipeline in plain English:**

1. The query is embedded (dense vector) and TF-IDF vectorised (sparse) in
   parallel.
2. Both vectors are searched against per-language Voyager HNSW indexes, producing two
   ranked lists of programs.
3. The lists are merged via Reciprocal Rank Fusion.
4. An optional LLM re-ranks the fused list (falls back to RRF on failure).
5. For each top program the earliest episode is selected; those episodes form three
   ranked lists (embedding order, TF-IDF order, duration proximity score).
6. A second RRF pass over the media lists yields the final media ranking.

---

## Results

Measured on the 200-program synthetic catalog (1075 media items across `en`, `fr`, `de`).

### Latency

| Metric | Value |
|--------|-------|
| p50    | 10.5 ms |
| p95    | 15.8 ms |
| Max    | 35.3 ms |

Benchmarked over 200 queries, single-threaded, no LLM re-ranking, on macOS arm64 with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Latency is dominated by query embedding and HNSW ANN search.

### Retrieval Quality

Evaluated on 20 topic-based queries across `en`, `fr`, `de` (5 trilingual topics +
5 English-only topics), with relevance judged by whether a returned program's
topic matches the query topic.

| Metric    | @3    | @5    |
|-----------|-------|-------|
| Precision | 0.783 | 0.670 |
| Recall    | 0.653 | 0.842 |
| nDCG      | 0.920 | 0.934 |

nDCG near `1.0` indicates that relevant programs consistently rank at the top of
the returned list. Precision drops from `@3` to `@5` as extra slots fill with
off-topic items once the topic's relevant pool is exhausted; recall rises
correspondingly.

### Retrieval Ablation: Dense vs Sparse vs Hybrid

Same 20-query benchmark, mock LLM (no re-ranking). Measures the contribution of
each retrieval signal in isolation.

| Mode    | @k | Precision | Recall | nDCG  |
|---------|----|-----------:|-------:|------:|
| dense   | @3 | 0.833     | 0.675  | 0.959 |
| dense   | @5 | 0.680     | 0.833  | 0.941 |
| sparse  | @3 | 0.717     | 0.615  | 0.850 |
| sparse  | @5 | 0.590     | 0.773  | 0.852 |
| hybrid  | @3 | 0.783     | 0.653  | 0.920 |
| hybrid  | @5 | 0.670     | 0.842  | 0.934 |

Dense retrieval alone leads on precision@3 and nDCG@3; hybrid closes the gap at
@5 by recovering additional relevant programs through TF-IDF's lexical matching.
Sparse-only consistently underperforms both, confirming that semantic embeddings
carry the primary signal for this catalog. Reproduce with:
`uv run python scripts/eval_ablation.py`

---

## Why This Design

- **Dual retrieval (dense + sparse) with RRF fusion** — runs both embedding ANN and TF-IDF, merges with Reciprocal Rank Fusion. _Dense embeddings miss exact keyword matches; TF-IDF misses semantic similarity. The ablation above shows the trade-off rather than blanket dominance: dense alone leads on precision@3, while the hybrid recovers relevant items through lexical matching for a recall@5 gain (0.842 vs 0.833). Hybrid favors recall and robustness over peak top-3 precision._
- **LLM re-ranking with automatic fallback** — optional re-ranker behind a vendor-neutral `LLMProvider` ABC (OpenAI or Anthropic/Claude, selected by config); falls back to RRF order on timeout or parse failure. _Network latency and API errors are real in production. The system must return results even when the LLM is unavailable._
- **Voyager (HNSW) over FAISS** — single static file, no server process, pip-installable wheel. _Scales to ~10M items with minimal operational overhead. FAISS becomes relevant at 100M+ or when GPU acceleration is needed._
- **Per-language indexes** — separate HNSW + TF-IDF indexes per language. _Multilingual embedding models underperform monolingual ones on non-English content; per-language indexing avoids cross-lingual noise in retrieval._
- **Duration-aware scoring with asymmetric penalty** — penalizes results longer than requested more than shorter ones. _A product requirement: prioritize shorter-than-requested content over longer._

---

## Quick Start

> **Note:** First run downloads ~560 MB of models (3 spaCy + sentence-transformers). Allow 5-10 minutes on a typical connection.

```bash
# 1. Clone & install (one command installs all dependencies + language models)
git clone https://github.com/nlorber/hybrid-recsys.git
cd hybrid-recsys
make setup

# 2. Generate a synthetic catalog and build indexes
uv run python scripts/generate_catalog.py
uv run hybrid-recsys index

# 3. Run a demo query
uv run hybrid-recsys demo "true crime podcast" --lang en --size 3
```

<details>
<summary>Manual install (without Make)</summary>

```bash
uv sync
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download fr_core_news_sm
uv run python -m spacy download de_core_news_sm
uv run python -m nltk.downloader stopwords
```
</details>

---

## API

Start the FastAPI server:

```bash
uv run hybrid-recsys serve
# Listening on http://0.0.0.0:8000
```

Then open <http://localhost:8000> for the interactive demo UI, or call the API directly.

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

For the ranking signals behind each result — the RRF score, which retriever(s)
surfaced it (dense / sparse), and whether the LLM re-ranker moved it — call
`POST /recommend/explain` (same request body). This powers the demo UI.

```bash
curl -s -X POST http://localhost:8000/recommend/explain \
     -H "Content-Type: application/json" \
     -d '{"query": "science for kids", "lang": "en", "size": 3}' \
  | jq .
```

Liveness check: `GET /health` returns `{"status": "ok"}`.

Interactive API docs: <http://localhost:8000/docs>

---

## Configuration

All settings use the `RECSYS_` prefix and can be set via environment variables or
a `.env` file:

```bash
RECSYS_EMBEDDING_PROVIDER=sentence-transformers   # local, no key; or: openai
RECSYS_LLM_PROVIDER=anthropic                      # or: mock | openai
RECSYS_LLM_MODEL=claude-haiku-4-5-20251001         # provider default if unset
RECSYS_LLM_API_KEY=sk-ant-...                      # falls back to ANTHROPIC_API_KEY
```

Provider selection is vendor-neutral and config-driven — embeddings and LLM
re-ranking are chosen independently by name, so the two roles can use different
vendors (e.g. local embeddings + a Claude re-ranker).

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for the full list of variables and
available providers.

---

## Architecture Details

See [docs/DESIGN.md](docs/DESIGN.md) for:

- Full RRF formula with parameter explanation
- Duration scoring formula and asymmetric penalty rationale
- Voyager (HNSW) vs. FAISS / ScaNN trade-off analysis
- Indexing strategy and provider abstraction design

---

## Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit

# Integration tests only
uv run pytest tests/integration
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Embeddings | sentence-transformers / OpenAI |
| Sparse retrieval | scikit-learn TF-IDF |
| ANN index | Voyager (HNSW) |
| NLP tokenisation | spaCy |
| Re-ranking | OpenAI / Anthropic (optional) / Mock |
| API server | FastAPI + Uvicorn |
| CLI | Typer |
| Config | pydantic-settings |
