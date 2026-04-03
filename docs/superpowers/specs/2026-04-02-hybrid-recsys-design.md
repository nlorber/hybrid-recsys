# hybrid-recsys — Design Spec

A multilingual content recommendation engine combining dual retrieval (embeddings + TF-IDF), Approximate Nearest Neighbor search, Reciprocal Rank Fusion, and LLM re-ranking. Extracted from a production system into a clean, standalone, runnable portfolio demo.

## Architecture

```
hybrid-recsys/
├── src/hybrid_recsys/
│   ├── config.py                  # Settings (pydantic-settings), provider selection
│   ├── providers/
│   │   ├── embeddings/
│   │   │   ├── base.py            # ABC: embed(text), embed_batch(texts)
│   │   │   ├── openai.py          # OpenAI / Azure OpenAI
│   │   │   └── sentence_tf.py     # sentence-transformers (default)
│   │   ├── llm/
│   │   │   ├── base.py            # ABC: rerank(query, candidates, size)
│   │   │   ├── openai.py          # OpenAI ChatCompletion
│   │   │   └── mock.py            # Returns candidates as-is
│   │   └── nlp/
│   │       └── spacy.py           # Multilingual spaCy + stopwords (cached)
│   ├── indexing/
│   │   ├── vectorizer.py          # Orchestrator: per-language embedding + TF-IDF generation
│   │   ├── tfidf.py               # TF-IDF pipeline (preprocessing + vectorization)
│   │   └── store.py               # Save/load per-language artifacts
│   ├── retrieval/
│   │   ├── pipeline.py            # RecommendationPipeline: full orchestration
│   │   ├── ann_search.py          # Annoy index build + query
│   │   ├── fusion.py              # Reciprocal Rank Fusion
│   │   ├── reranker.py            # LLM re-ranking with prompt generation + fallback
│   │   └── scorer.py              # Duration scoring
│   ├── models.py                  # Pydantic models
│   ├── api.py                     # FastAPI app
│   └── cli.py                     # CLI: index, serve, demo
├── data/
│   ├── catalog.json               # Synthetic podcast catalog
│   └── index/                     # Built indexes (gitignored)
├── scripts/
│   └── generate_catalog.py        # Synthetic multilingual catalog generator
├── tests/
│   ├── unit/
│   │   ├── test_fusion.py
│   │   ├── test_scorer.py
│   │   ├── test_tfidf.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_api.py
├── docs/
│   ├── DESIGN.md
│   └── PROVIDERS.md
├── pyproject.toml
└── README.md
```

## Algorithm

### Dual Retrieval

Two independent similarity signals for the user's query:

1. **Embedding similarity** — query embedded via sentence-transformers (or OpenAI), searched against program embedding vectors via Annoy ANN index
2. **TF-IDF similarity** — query preprocessed (lemmatized, stopwords removed) and vectorized with the same fitted TF-IDF vectorizer used on the catalog, searched against program TF-IDF vectors via a second Annoy ANN index

Both indexes use `angular` metric (cosine distance) with 10 trees. Queries over-fetch `k=20` results to allow headroom after any post-processing.

### Text Preprocessing

Applied to both catalog descriptions (at index time) and queries (at query time):

```python
text = re.sub(r"\W+", " ", text.lower())
doc = spacy_model(text)
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
result = " ".join(tokens)
```

Language-specific spaCy models:
- FR: `fr_core_news_sm`
- EN: `en_core_web_sm`
- DE: `de_core_news_sm`

Stopwords from NLTK: french, english, german.

### Reciprocal Rank Fusion (RRF)

Fuses multiple ranked lists into one combined ranking.

```
score(item) = sum over lists L of: weight_L / (rank_of_item_in_L + k)
```

- `rank`: 1-based position in each list
- `k`: overlap importance parameter. Higher k favors items appearing in multiple lists. Lower k favors items ranked high in any single list.
- `weights`: relative importance of each list

Parameters:
- Program-level: `weights=[3, 2]`, `k=5` — embeddings weighted higher than TF-IDF
- Media-level: `weights=[3, 2, 3]`, `k=8` — embeddings, TF-IDF, duration

### Duration Scoring

Scores media by proximity to the user's preferred duration.

```
delta = requested_duration - media_duration

if delta >= 0 (media shorter than requested):
    score = 1 / (delta^2 + 1)
if delta < 0 (media longer than requested):
    score = 1 / (delta^2 + 1) + penalty
```

- `penalty`: configurable, default `-1.0`. Longer media is penalized more heavily than shorter.
- `default_duration`: `600` seconds (used when request doesn't specify a duration)

### LLM Re-ranking

After RRF produces a program ranking:

1. **Skip condition**: if RRF result count <= requested `size`, return RRF list directly (no LLM needed)
2. **Prompt generation**: build a language-specific prompt containing program descriptions of the RRF candidates, ask the LLM to select exactly `size` program IDs sorted by relevance
3. **Response parsing**: parse LLM response as a Python list literal
4. **Fallback**: on parse failure, API error, or underselection — pad with RRF-ranked programs. On overselection — truncate.

### Media Selection

For each ranked program:
1. Filter media belonging to that program
2. Take the earliest episode (min episode number) — recommends series starters
3. Score each media item by duration proximity

Three media lists are built:
- Media from embedding-ranked programs (ordered by program rank)
- Media from TF-IDF-ranked programs (ordered by program rank)
- Media scored by duration (ordered by duration score)

These three lists are fused via RRF with `weights=[3, 2, 3]`, `k=8`.

## Indexing Strategy

### Offline Phase (`hybrid-recsys index`)

Per language (fr/en/de):
1. Filter catalog programs by language
2. Preprocess descriptions via spaCy
3. Generate embeddings via configured provider
4. Fit TF-IDF vectorizer on preprocessed descriptions
5. Build 2 Annoy indexes (embedding + TF-IDF), 10 trees each
6. Save to `data/index/{lang}/`:
   - `ann_embedding.ann` — Annoy embedding index
   - `ann_tfidf.ann` — Annoy TF-IDF index
   - `tfidf_vectorizer.pkl` — fitted sklearn TfidfVectorizer
   - `metadata.json` — program_id to index mapping, descriptions, media data
   - `embedding_dim.json` — embedding dimensionality (needed to load Annoy index)

### Online Phase (per request)

1. Load cached index artifacts for `request.lang`
2. Embed query
3. Preprocess + vectorize query via saved TF-IDF vectorizer
4. Query both Annoy indexes (k=20)
5. Map Annoy result indices back to program IDs via metadata
6. RRF fuse program lists
7. Optionally LLM rerank
8. Build media lists, score durations, RRF fuse
9. Return response

### Caching

- spaCy models + stopwords: cached per language (loaded once on first use)
- Index artifacts: cached per language (loaded from disk once on first request)

## Data Models

### CatalogItem (input data)
```python
class MediaItem(BaseModel):
    media_id: str
    episode: int
    duration: int        # seconds
    title: str

class CatalogItem(BaseModel):
    program_id: str
    title: str
    description: str     # 2-4 sentences, in program's language
    lang: str            # fr / en / de
    media: list[MediaItem]
```

### RecoRequest (API input)
```python
class RecoRequest(BaseModel):
    query: str           # max 300 chars
    lang: str            # fr / en / de
    size: int = 3        # 1-10
    duration: int | None = None  # preferred duration in seconds
```

### RecoResponse (API output)
```python
class RecoResponse(BaseModel):
    programs: list[str]  # program IDs, ranked
    medias: list[str]    # media IDs, ranked
```

## Configuration

Via pydantic-settings (`config.py`), configurable through environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_provider` | `"sentence-transformers"` | `"sentence-transformers"` or `"openai"` |
| `llm_provider` | `"mock"` | `"mock"` or `"openai"` |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Model name for the embedding provider |
| `openai_api_key` | `None` | Required only for OpenAI providers |
| `default_duration` | `600` | Fallback duration (seconds) when not specified |
| `duration_penalty` | `-1.0` | Penalty for media longer than preferred |
| `rrf_program_weights` | `[3, 2]` | Embedding, TF-IDF weights for program RRF |
| `rrf_program_k` | `5` | Program-level RRF overlap parameter |
| `rrf_media_weights` | `[3, 2, 3]` | Embedding, TF-IDF, duration weights for media RRF |
| `rrf_media_k` | `8` | Media-level RRF overlap parameter |
| `ann_metric` | `"angular"` | Annoy distance metric |
| `ann_n_trees` | `10` | Annoy index tree count |
| `ann_query_k` | `20` | Over-fetch size for ANN queries |
| `spacy_models` | `{fr: fr_core_news_sm, en: en_core_web_sm, de: de_core_news_sm}` | Per-language spaCy models |

## Provider Abstractions

### EmbeddingProvider (ABC)
```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Implementations:
- `SentenceTransformerProvider` — uses `sentence-transformers` library, default model `all-MiniLM-L6-v2`. Runs locally, no API key needed.
- `OpenAIEmbeddingProvider` — uses `openai` SDK. Works with both OpenAI and Azure OpenAI via `base_url` configuration.

### LLMProvider (ABC)
```python
class LLMProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], size: int) -> list[str]: ...
```

Implementations:
- `MockLLMProvider` — returns candidate IDs in input order (no-op reranking). Default for demo.
- `OpenAILLMProvider` — sends reranking prompt to OpenAI ChatCompletion, parses response.

### NLP (no abstraction)
spaCy is used directly. No provider ABC — there's no realistic alternative to swap in.

## Synthetic Catalog

`scripts/generate_catalog.py` generates `data/catalog.json`:

- ~200 programs: deterministic (seeded), realistic descriptions in each program's language
- Language distribution: ~100 FR, ~70 EN, ~30 DE
- ~1000 media items (episodes): 3-8 per program, durations 300-7200s
- ~15 topic areas (Technology, History, Science, Sports, Culture, Business, Comedy, Politics, Health, Education, Music, Travel, Crime, Environment, Philosophy) used to generate coherent descriptions — stored as a generation aid, not as a model field
- Descriptions must be meaningful enough for TF-IDF and embedding similarity (not lorem ipsum)

## API & CLI

### FastAPI App
- `POST /recommend` — accepts `RecoRequest`, returns `RecoResponse`
- `GET /health` — returns `{"status": "ok"}`

### CLI (via click or typer)
- `hybrid-recsys index` — build per-language indexes from `data/catalog.json`
- `hybrid-recsys serve` — start FastAPI server (uvicorn)
- `hybrid-recsys demo "query text"` — index + single query + print results (zero-to-output in one command)

## Testing

### Unit Tests
- `test_fusion.py` — RRF with known ranked lists and expected scores. Edge cases: single list, empty list, equal weights, item in all lists vs. one.
- `test_scorer.py` — Duration scoring with known deltas. Verify asymmetry, delta=0 boundary, configurable penalty.
- `test_tfidf.py` — Preprocessing: lemmatization, stopword removal, punctuation stripping per language.
- `test_pipeline.py` — Full pipeline with mock embedding provider (deterministic vectors) and mock LLM. Verify orchestration: correct provider calls, RRF params, LLM fallback.

### Integration Tests
- `test_api.py` — FastAPI TestClient with mock providers. Verify `/recommend` response shape and end-to-end flow. Verify `/health`.

## Out of Scope

- User history / personalization (no user profiles or listening stats)
- Category filtering (no algorithmic value for portfolio)
- Async providers (sync-only; FastAPI handles concurrency at endpoint level)
- Database layer (all data from JSON files)
- Azure-specific provider (OpenAI SDK's `base_url` covers Azure)

## Tech Stack

- Python 3.11+
- Poetry for dependency management
- pydantic + pydantic-settings for models and config
- FastAPI + uvicorn for API
- annoy for ANN search
- scikit-learn for TF-IDF
- sentence-transformers for local embeddings
- spaCy (fr/en/de small models) + NLTK stopwords for NLP
- openai SDK for OpenAI/Azure providers (optional)
- ruff for linting + formatting
- pytest for testing

## Quality Bar

- Type hints everywhere (strict)
- `ruff check` and `ruff format` pass
- Every public function has a docstring
- Demo flow works end-to-end with zero cloud dependencies: `hybrid-recsys index && hybrid-recsys demo "machine learning podcasts"`
- OpenAI/Azure providers are optional extras
- Tests cover mathematical correctness of RRF and duration scoring
