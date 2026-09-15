"""Ablation study: dense-only vs TF-IDF-only vs hybrid retrieval.

Evaluates the same 20-query benchmark used in evaluate.py under three modes:
  - dense:   only the embedding ANN ranking is fed to RRF (TF-IDF list is empty)
  - sparse:  only the TF-IDF ANN ranking is fed to RRF (embedding list is empty)
  - hybrid:  both lists fused via RRF (default pipeline behaviour)

Uses the mock LLM provider so no API keys are required; LLM re-ranking is
bypassed (all three modes are equivalent at the re-ranking step: fewer candidates
than `size` → no LLM call).

Usage:
    uv run python scripts/eval_ablation.py

Requirements:
    - data/catalog.json must exist (run generate_catalog.py first)
    - data/index/ must have at least one language sub-directory (run hybrid-recsys index)

Output:
    Prints a markdown table of Precision / Recall / nDCG @3 and @5 per mode,
    averaged over all evaluated queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path so the script works from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Re-use query and topic definitions from evaluate.py
from evaluate import TOPIC_QUERIES, build_program_lang_map, infer_topic

from hybrid_recsys.config import Settings
from hybrid_recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k
from hybrid_recsys.models import RecoRequest, RecoResponse
from hybrid_recsys.providers.embeddings.sentence_tf import SentenceTransformerProvider
from hybrid_recsys.providers.llm.mock import MockLLMProvider
from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

# ---------------------------------------------------------------------------
# Ablation pipeline variants
# ---------------------------------------------------------------------------

Mode = str  # "dense" | "sparse" | "hybrid"


class AblationPipeline(RecommendationPipeline):
    """Pipeline variant that optionally zeroes out one of the two retrieval lists.

    - mode="dense":  TF-IDF list replaced with an empty list before RRF.
    - mode="sparse": Embedding list replaced with an empty list before RRF.
    - mode="hybrid": Standard behaviour (both lists passed to RRF).
    """

    def __init__(self, *args: Any, mode: Mode = "hybrid", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mode = mode

    def recommend(self, request: RecoRequest) -> RecoResponse:  # type: ignore[override]
        from hybrid_recsys.retrieval.ann_search import query_ann_index
        from hybrid_recsys.retrieval.fusion import reciprocal_rank_fusion
        from hybrid_recsys.retrieval.reranker import rerank_programs

        index = self._load_index(request.lang)

        query_embedding = self._embedder.embed(request.query)
        query_tfidf = self._tfidf.transform_query(
            request.query, index.tfidf_vectorizer, request.lang
        )

        k = min(self._settings.ann_query_k, len(index.program_ids))
        emb_indices = query_ann_index(index.ann_embedding, query_embedding, k)
        tfidf_indices = query_ann_index(index.ann_tfidf, query_tfidf, k)

        emb_programs = [index.program_ids[i] for i in emb_indices]
        tfidf_programs = [index.program_ids[i] for i in tfidf_indices]

        # Ablation: blank out one list
        if self._mode == "dense":
            tfidf_programs = []
        elif self._mode == "sparse":
            emb_programs = []

        program_rrf = reciprocal_rank_fusion(
            ranked_lists=[emb_programs, tfidf_programs],
            weights=self._settings.rrf_program_weights,
            k=self._settings.rrf_program_k,
        )

        programs = rerank_programs(
            llm=self._llm,
            query=request.query,
            rrf_ranking=program_rrf,
            descriptions=index.program_descriptions,
            size=request.size,
            lang=request.lang,
            timeout=self._settings.llm_rerank_timeout,
        )

        requested_duration = request.duration or self._settings.default_duration
        medias = self._rank_media(
            index=index,
            programs=programs,
            emb_programs=emb_programs,
            tfidf_programs=tfidf_programs,
            requested_duration=requested_duration,
            size=request.size,
        )

        return RecoResponse(programs=programs, medias=medias)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_ablation(
    catalog_path: Path,
    index_dir: Path,
    metrics_path: Path = Path("reports/ablation_metrics.json"),
) -> None:
    """Run the ablation evaluation, print a markdown table, and persist results."""

    class AblationSettings(Settings):
        @property
        def index_dir(self) -> Path:  # type: ignore[override]
            return index_dir

    settings = AblationSettings()
    embedder = SentenceTransformerProvider(settings.embedding_model)
    llm = MockLLMProvider()

    modes: list[Mode] = ["dense", "sparse", "hybrid"]
    pipelines = {mode: AblationPipeline(embedder, llm, settings, mode=mode) for mode in modes}

    with open(catalog_path) as f:
        catalog = json.load(f)

    program_topics: dict[str, str] = {}
    program_langs = build_program_lang_map(catalog)
    for prog in catalog["programs"]:
        topic = infer_topic(prog["description"])
        if topic:
            program_topics[prog["program_id"]] = topic

    k_values = [3, 5]

    # Collect per-mode per-k metrics
    mode_k_results: dict[Mode, dict[int, list[dict[str, float]]]] = {
        mode: {k: [] for k in k_values} for mode in modes
    }

    for lang, queries in TOPIC_QUERIES.items():
        if not (index_dir / lang).exists():
            print(f"  Skipping {lang}: no index found", flush=True)
            continue

        for qinfo in queries:
            query_text = str(qinfo["query"])
            expected_topics = set(str(t) for t in qinfo["topics"])  # type: ignore[arg-type]

            relevant = {
                pid
                for pid, topic in program_topics.items()
                if topic in expected_topics and program_langs.get(pid) == lang
            }
            if not relevant:
                continue

            for mode in modes:
                request = RecoRequest(query=query_text, lang=lang, size=max(k_values))
                response = pipelines[mode].recommend(request)

                for k in k_values:
                    p = precision_at_k(response.programs, relevant, k)
                    r = recall_at_k(response.programs, relevant, k)
                    n = ndcg_at_k(response.programs, relevant, k)
                    mode_k_results[mode][k].append({"precision": p, "recall": r, "ndcg": n})

    # Build summary table
    rows: list[dict[str, object]] = []
    for mode in modes:
        for k in k_values:
            entries = mode_k_results[mode][k]
            if not entries:
                continue
            avg_p = sum(e["precision"] for e in entries) / len(entries)
            avg_r = sum(e["recall"] for e in entries) / len(entries)
            avg_n = sum(e["ndcg"] for e in entries) / len(entries)
            rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "precision": avg_p,
                    "recall": avg_r,
                    "ndcg": avg_n,
                    "n_queries": len(entries),
                }
            )

    # Print markdown table
    print("\n## Ablation Results\n")
    print(
        "Averaged over all evaluated queries (mock LLM; no re-ranking). "
        "Relevance: program topic matches query topic.\n"
    )
    print(f"{'Mode':<8} {'@k':<4} {'Precision':>10} {'Recall':>8} {'nDCG':>8}  {'Queries':>8}")
    print("-" * 58)
    for row in rows:
        print(
            f"{row['mode']:<8} @{row['k']:<3} {row['precision']:>10.3f} "
            f"{row['recall']:>8.3f} {row['ndcg']:>8.3f}  {row['n_queries']:>8}"
        )

    # Produce a copy-pasteable markdown table for README
    print("\n### Copy-paste markdown table\n")
    print("| Mode    | @k | Precision | Recall | nDCG  |")
    print("|---------|----|-----------:|-------:|------:|")
    for row in rows:
        print(
            f"| {row['mode']:<7} | @{row['k']} "
            f"| {row['precision']:.3f}     | {row['recall']:.3f}  | {row['ndcg']:.3f} |"
        )

    # Persist a committed source of truth for the README "Retrieval Ablation" table.
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"\nWrote metrics to {metrics_path}")


if __name__ == "__main__":
    catalog_path = Path("data/catalog.json")
    index_dir = Path("data/index")

    if not catalog_path.exists():
        print(f"Catalog not found at {catalog_path}. Run generate_catalog.py first.")
        sys.exit(1)
    if not index_dir.exists():
        print(f"Indexes not found at {index_dir}. Run 'hybrid-recsys index' first.")
        sys.exit(1)

    print("Running ablation: dense vs sparse vs hybrid...")
    print("(Uses mock LLM — no API keys required)\n")
    run_ablation(catalog_path, index_dir)
