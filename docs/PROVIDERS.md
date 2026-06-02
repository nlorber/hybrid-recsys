# Provider Configuration Guide

All providers are selected via environment variables prefixed with `RECSYS_`.
Settings can be placed in a `.env` file at the project root or exported in the
shell before starting the server / running the CLI.

---

## Embedding Providers

Embedding providers convert text into dense vectors for ANN search. You must
re-run `hybrid-recsys index` whenever you switch providers, because the HNSW
indexes are built from the embeddings produced by the active provider.

### sentence-transformers (default)

Runs fully locally; no API key required.

```bash
RECSYS_EMBEDDING_PROVIDER=sentence-transformers
RECSYS_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2  # any model on HuggingFace Hub
```

The model is downloaded from HuggingFace on first use and cached locally.

### OpenAI

Requires the optional `openai` extra: `uv sync --extra openai`.

```bash
RECSYS_EMBEDDING_PROVIDER=openai
RECSYS_EMBEDDING_MODEL=text-embedding-3-small
RECSYS_EMBEDDING_API_KEY=sk-...
```

### Azure OpenAI

Same as OpenAI, but point the client at your Azure resource endpoint:

```bash
RECSYS_EMBEDDING_PROVIDER=openai
RECSYS_EMBEDDING_MODEL=text-embedding-3-small
RECSYS_EMBEDDING_API_KEY=sk-...
RECSYS_EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/
```

> **Note:** After switching embedding providers, always rebuild the indexes:
> ```bash
> uv run hybrid-recsys index
> ```
> Mixing vectors from different providers produces meaningless ANN results.

---

## LLM Providers

LLM providers are used for the optional re-ranking step. The pipeline falls back
to the RRF ranking whenever the LLM call fails or returns an unusable response.

### Mock (default)

Re-ranks candidates by keyword overlap between the query and program
descriptions. No configuration needed; useful for development, testing, and
environments where an LLM API is unavailable.

```bash
RECSYS_LLM_PROVIDER=mock
```

### OpenAI

Requires the optional `openai` extra: `uv sync --extra openai`.

```bash
RECSYS_LLM_PROVIDER=openai
RECSYS_LLM_MODEL=gpt-4o-mini       # optional; provider default if unset
RECSYS_LLM_API_KEY=sk-...
RECSYS_LLM_BASE_URL=               # optional; e.g. an OpenAI-compatible endpoint
```

### Anthropic (Claude)

Requires the optional `anthropic` extra: `uv sync --extra anthropic`.

```bash
RECSYS_LLM_PROVIDER=anthropic
RECSYS_LLM_MODEL=claude-haiku-4-5-20251001   # optional; this is the default
RECSYS_LLM_API_KEY=sk-ant-...                # falls back to ANTHROPIC_API_KEY
```

Embedding and LLM credentials are configured independently (`RECSYS_EMBEDDING_*`
vs `RECSYS_LLM_*`), so the two roles can use different vendors — for example local
`sentence-transformers` embeddings paired with a Claude re-ranker. Provider
selection is purely config-driven; the pipeline and API depend only on the
provider ABCs.

---

## Quick Reference

| Variable                   | Default                  | Description                                  |
|----------------------------|--------------------------|----------------------------------------------|
| `RECSYS_EMBEDDING_PROVIDER`| `sentence-transformers`  | Embedding backend (`sentence-transformers`, `openai`) |
| `RECSYS_EMBEDDING_MODEL`   | `paraphrase-multilingual-MiniLM-L12-v2` | Model name / deployment ID |
| `RECSYS_LLM_PROVIDER`      | `mock`                   | LLM backend (`mock`, `openai`, `anthropic`)  |
| `RECSYS_LLM_MODEL`         | *(provider default)*     | Re-rank model (e.g. `claude-haiku-4-5-20251001`) |
| `RECSYS_LLM_API_KEY`       | *(unset)*                | API key for the LLM provider                 |
| `RECSYS_LLM_BASE_URL`      | *(unset)*                | Override base URL for the LLM provider        |
| `RECSYS_EMBEDDING_API_KEY` | *(unset)*                | API key for the embedding provider           |
| `RECSYS_EMBEDDING_BASE_URL`| *(unset)*                | Override base URL for the embedding provider  |
| `RECSYS_DATA_DIR`          | `data`                   | Root directory for catalog and indexes       |
| `RECSYS_DEFAULT_DURATION`  | `600`                    | Fallback duration in seconds (10 min)        |
| `RECSYS_DURATION_PENALTY`  | `-1.0`                   | Score penalty for media longer than requested|
| `RECSYS_RRF_PROGRAM_K`     | `5`                      | RRF k parameter for program fusion           |
| `RECSYS_RRF_MEDIA_K`       | `8`                      | RRF k parameter for media fusion             |
| `RECSYS_ANN_METRIC`        | `cosine`                 | Distance metric (`cosine`, `euclidean`, `dot`) |
| `RECSYS_ANN_M`             | `16`                     | HNSW graph degree (higher = better recall, more memory) |
| `RECSYS_ANN_EF_CONSTRUCTION`| `200`                   | HNSW build-time candidate list size           |
| `RECSYS_ANN_QUERY_K`       | `20`                     | Candidates retrieved per ANN query           |
