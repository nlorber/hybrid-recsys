# Design Document

## Overview

hybrid-recsys is a multilingual content recommendation engine that combines three
retrieval strategies into a single pipeline:

1. **Dual retrieval** — dense (embedding-based) and sparse (TF-IDF) search run
   in parallel over per-language Voyager HNSW indexes.
2. **Reciprocal Rank Fusion (RRF)** — merges the two ranked lists into one without
   requiring labeled training data.
3. **LLM re-ranking with fallback** — an optional LLM pass refines the fused list;
   failures fall back gracefully to the RRF order.

---

## Why Dual Retrieval?

Embeddings and TF-IDF are complementary signals:

| Signal     | Strengths                                   | Weaknesses                          |
|------------|---------------------------------------------|-------------------------------------|
| Embeddings | Capture semantic similarity, handle synonyms | Miss exact keyword matches          |
| TF-IDF     | Exact lexical matches, fast, deterministic  | Blind to synonyms and paraphrases   |

Running both and fusing their outputs reliably outperforms either alone on diverse
queries — a result well documented in the information retrieval literature.

---

## Why RRF Over Learned Fusion?

Reciprocal Rank Fusion is chosen because:

- **No labeled data required.** Learning a fusion model needs relevance judgements;
  RRF needs none.
- **Robust to score scale differences.** RRF only uses ranks, so it is immune to
  the scale mismatch between cosine similarities and TF-IDF scores.
- **Well-studied.** RRF was introduced in Cormack et al. (2009) and has been
  validated across many retrieval benchmarks.

### RRF Formula

```
score(item) = Σ  weight_L / (rank_in_L + k)
              L
```

- `rank_in_L` — 1-based position of the item in list L (absent items contribute 0)
- `k` — smoothing constant (default 5 for programs, 8 for media); higher values
  favour items appearing in multiple lists over items ranked very high in one list
- `weight_L` — per-list weight (default: embedding 3.0, TF-IDF 2.0)

---

## Why LLM Re-ranking With Fallback?

An LLM can reason about query intent beyond surface-level similarity, but:

- Network latency and API errors are real in production.
- LLMs can hallucinate or return malformed rankings.

The re-ranker therefore validates the LLM output and falls back to the RRF ranking
when the response is unusable. This keeps the pipeline reliable without abandoning
the quality uplift LLM re-ranking can provide.

---

## Why Voyager (HNSW) Over FAISS / ScaNN?

| Criterion          | Voyager (HNSW)            | FAISS / ScaNN              |
|--------------------|---------------------------|----------------------------|
| Dependencies       | Pre-built wheel           | C++ build toolchain        |
| Cosine similarity  | Native (`Space.Cosine`)   | Requires normalisation step|
| Scale              | Up to ~10 M items         | Billions of vectors        |
| Operational cost   | Single static file        | Server process or GPU      |
| Incremental update | Yes (add/mark-deleted)    | Rebuild required           |

For content catalogs in the range of thousands to tens of millions of items,
Voyager is fast enough and dramatically simpler to operate. FAISS or ScaNN become
relevant only when the catalog grows beyond that range or when sub-millisecond
latency at very large scale is required.

Voyager is Spotify's production HNSW library, released as the successor to Annoy.
It wraps the hnswlib algorithm with improved packaging, Python type stubs, and
batch-friendly APIs.

---

## Duration Scoring

Media items are scored by proximity to the caller's requested duration. The delta
is defined as `requested_duration - media_duration` (in seconds).

```
base  = 1 / (delta² + 1)

score = base                  if delta >= 0   (media is shorter than requested)
      = base + penalty        if delta < 0    (media is longer than requested)
```

The asymmetric penalty (default `–1.0`) discourages returning episodes that run
longer than what the listener asked for, while items that are slightly too short
receive only the natural `1/(delta²+1)` decay.

---

## Pipeline Architecture

```
                         ┌─────────────────────────────────────┐
                         │             query                   │
                         └──────────────────┬──────────────────┘
                                            │
                          ┌─────────────────┼─────────────────┐
                          ▼                                   ▼
                   embed (dense)                      TF-IDF vectorise
                          │                                   │
                          ▼                                   ▼
                 ANN search (Voyager)             ANN search (Voyager)
                  embedding index                    TF-IDF index
                          │                                   │
                          └──────────────┬────────────────────┘
                                         │
                                    RRF fusion
                                   (programs)
                                         │
                                    LLM rerank
                                  (with fallback)
                                         │
                                      programs
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                   emb media list  tfidf media list  duration scoring
                          │              │              │
                          └──────────────┼──────────────┘
                                         │
                                    RRF fusion
                                     (media)
                                         │
                                       media
```

---

## Indexing Strategy

Indexes are built per language at index time and stored as static files under
`data/index/<lang>/`. Each language bundle contains:

- `ann_embedding.voy` — Voyager HNSW index of program embedding vectors
- `ann_tfidf.voy` — Voyager HNSW index of TF-IDF sparse vectors projected to a dense space
- `tfidf_vectorizer.pkl` — fitted `TfidfVectorizer` (for query transformation)
- `metadata.json` — program IDs, descriptions, and media episode data

Building per-language indexes keeps query-time routing simple (one `dict` lookup)
and allows language-specific TF-IDF vocabularies without cross-language noise.

---

## Provider Abstraction

All external dependencies are hidden behind thin abstract base classes:

- `EmbeddingProvider` — `embed(text: str) -> list[float]`
- `LLMProvider` — `rerank(query, candidates, ...) -> list[str]`

Default implementations ship with the package (sentence-transformers and mock LLM).
Optional OpenAI / Azure OpenAI implementations are loaded lazily so that the
`openai` package is not required unless configured. See `docs/PROVIDERS.md` for
configuration details.

---

## Embedding Model Selection

The default embedding model is `paraphrase-multilingual-MiniLM-L12-v2`:

| Criterion            | `paraphrase-multilingual-MiniLM-L12-v2` | `all-MiniLM-L6-v2`     |
|----------------------|-----------------------------------------|------------------------|
| Languages            | 50+ (FR, DE, EN, …)                     | English-primary        |
| Dimensions           | 384                                     | 384                    |
| Training objective   | Paraphrase similarity, multilingual     | Semantic similarity, EN|
| HuggingFace Hub size | ~470 MB                                 | ~90 MB                 |

`all-MiniLM-L6-v2` was evaluated but produces significantly lower-quality
embeddings for French and German content, undermining the multilingual premise of
the system. The dimension is identical (384), so switching between the two models
only changes the vector values — the HNSW index must be rebuilt after any model
change: `hybrid-recsys index`.

Any model on HuggingFace Hub can be used by setting `RECSYS_EMBEDDING_MODEL`.
