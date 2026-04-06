# Provider Configuration Guide

All providers are selected via environment variables prefixed with `RECSYS_`.
Settings can be placed in a `.env` file at the project root or exported in the
shell before starting the server / running the CLI.

---

## Embedding Providers

Embedding providers convert text into dense vectors for ANN search. You must
re-run `hybrid-recsys index` whenever you switch providers, because the Annoy
indexes are built from the embeddings produced by the active provider.

### sentence-transformers (default)

Runs fully locally; no API key required.

```bash
RECSYS_EMBEDDING_PROVIDER=sentence-transformers
RECSYS_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2  # any model on HuggingFace Hub
```

The model is downloaded from HuggingFace on first use and cached locally.

### OpenAI

Requires the optional `openai` extra: `poetry install --with openai`.

```bash
RECSYS_EMBEDDING_PROVIDER=openai
RECSYS_EMBEDDING_MODEL=text-embedding-3-small
RECSYS_OPENAI_API_KEY=sk-...
```

### Azure OpenAI

Same as OpenAI, but point the client at your Azure resource endpoint:

```bash
RECSYS_EMBEDDING_PROVIDER=openai
RECSYS_EMBEDDING_MODEL=text-embedding-3-small
RECSYS_OPENAI_API_KEY=sk-...
RECSYS_OPENAI_BASE_URL=https://your-resource.openai.azure.com/
```

> **Note:** After switching embedding providers, always rebuild the indexes:
> ```bash
> poetry run hybrid-recsys index
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

Requires the optional `openai` extra: `poetry install --with openai`.

```bash
RECSYS_LLM_PROVIDER=openai
RECSYS_OPENAI_API_KEY=sk-...
```

The same `RECSYS_OPENAI_BASE_URL` override used for embeddings also applies here,
so a single base URL covers both providers when using Azure OpenAI.

---

## Quick Reference

| Variable                   | Default                  | Description                                  |
|----------------------------|--------------------------|----------------------------------------------|
| `RECSYS_EMBEDDING_PROVIDER`| `sentence-transformers`  | Embedding backend (`sentence-transformers`, `openai`) |
| `RECSYS_EMBEDDING_MODEL`   | `paraphrase-multilingual-MiniLM-L12-v2` | Model name / deployment ID |
| `RECSYS_LLM_PROVIDER`      | `mock`                   | LLM backend (`mock`, `openai`)               |
| `RECSYS_OPENAI_API_KEY`    | *(unset)*                | OpenAI or Azure OpenAI API key               |
| `RECSYS_OPENAI_BASE_URL`   | *(unset)*                | Override base URL (Azure OpenAI)             |
| `RECSYS_DATA_DIR`          | `data`                   | Root directory for catalog and indexes       |
| `RECSYS_DEFAULT_DURATION`  | `600`                    | Fallback duration in seconds (10 min)        |
| `RECSYS_DURATION_PENALTY`  | `-1.0`                   | Score penalty for media longer than requested|
| `RECSYS_RRF_PROGRAM_K`     | `5`                      | RRF k parameter for program fusion           |
| `RECSYS_RRF_MEDIA_K`       | `8`                      | RRF k parameter for media fusion             |
| `RECSYS_ANN_N_TREES`       | `10`                     | Number of trees in Annoy indexes             |
| `RECSYS_ANN_QUERY_K`       | `20`                     | Candidates retrieved per ANN query           |
