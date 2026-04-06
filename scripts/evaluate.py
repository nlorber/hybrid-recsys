"""Offline evaluation of recommendation quality using synthetic relevance judgments.

Computes precision@k, recall@k, and nDCG on topic-based relevance: a query about
a topic (e.g. "AI") should rank programs in that topic family higher.

Usage:
    poetry run python scripts/evaluate.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / len(top_k)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k results."""
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    score = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG at k. Returns 0 if no relevant items exist."""
    ideal = sorted(ranked[:k], key=lambda x: x in relevant, reverse=True)
    idcg = dcg_at_k(ideal, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


# ---------------------------------------------------------------------------
# Topic-based relevance judgments
# ---------------------------------------------------------------------------

TOPIC_QUERIES: dict[str, list[dict[str, object]]] = {
    "en": [
        {"query": "artificial intelligence machine learning", "topics": ["Technology"]},
        {"query": "ancient history civilization empires", "topics": ["History"]},
        {"query": "physics biology scientific discovery", "topics": ["Science"]},
        {"query": "football basketball sports athletes", "topics": ["Sports"]},
        {"query": "true crime investigation murder", "topics": ["Crime"]},
        {"query": "comedy humor funny jokes", "topics": ["Comedy"]},
        {"query": "climate change environment ecology", "topics": ["Environment"]},
        {"query": "philosophy ethics freedom justice", "topics": ["Philosophy"]},
        {"query": "music jazz classical albums", "topics": ["Music"]},
        {"query": "health nutrition exercise wellness", "topics": ["Health"]},
    ],
    "fr": [
        {"query": "intelligence artificielle apprentissage automatique", "topics": ["Technology"]},
        {"query": "histoire ancienne civilisation Rome", "topics": ["History"]},
        {"query": "physique biologie decouverte scientifique", "topics": ["Science"]},
        {"query": "football rugby sport athletes", "topics": ["Sports"]},
        {"query": "crime enquete meurtre judiciaire", "topics": ["Crime"]},
    ],
}

TOPIC_DESCRIPTION_KEYWORDS: dict[str, list[str]] = {
    "Technology": ["artificial intelligence", "machine learning", "tech", "software", "computing", "intelligence artificielle", "numérique", "cybersécurité", "künstliche intelligenz", "digital"],
    "History": ["history", "ancient", "civilization", "roman", "medieval", "histoire", "gaulois", "révolution", "civilisation", "geschichte", "imperien"],
    "Science": ["scientific", "physics", "biology", "astronomy", "science", "scientifique", "physique", "biologie", "astronomie", "wissenschaft"],
    "Sports": ["sports", "football", "rugby", "tennis", "athletic", "sport", "athlètes", "fußball"],
    "Culture": ["arts", "culture", "cinema", "literature", "museum", "cinéma", "littérature", "kunst", "kultur"],
    "Business": ["business", "entrepreneur", "economics", "finance", "entreprise", "management", "unternehmen", "wirtschaft"],
    "Comedy": ["comedy", "humor", "funny", "humour", "humoristique", "hilarante"],
    "Politics": ["politic", "policy", "government", "politique", "débat", "politisch"],
    "Health": ["health", "wellness", "nutrition", "exercise", "santé", "médecin", "gesundheit"],
    "Education": ["education", "learning", "teaching", "éducatif", "pédagogique", "bildung"],
    "Music": ["music", "jazz", "classical", "album", "musique", "musik"],
    "Travel": ["travel", "destination", "adventure", "voyage", "reisen"],
    "Crime": ["crime", "criminal", "investigation", "murder", "criminelle", "enquête", "kriminal"],
    "Environment": ["environment", "ecology", "climate", "biodiversity", "écologie", "environnement", "ökologie", "umwelt"],
    "Philosophy": ["philosophy", "ethics", "freedom", "justice", "philosophie", "liberté"],
}


def infer_topic(description: str) -> str | None:
    """Infer a program's topic from its description using keyword matching."""
    desc_lower = description.lower()
    best_topic = None
    best_score = 0
    for topic, keywords in TOPIC_DESCRIPTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic if best_score > 0 else None


def catalog_lang(catalog: dict, program_id: str) -> str | None:
    """Look up a program's language in the catalog."""
    for prog in catalog["programs"]:
        if prog["program_id"] == program_id:
            return prog["lang"]
    return None


def run_evaluation(catalog_path: Path, index_dir: Path) -> None:
    """Run offline evaluation and print results."""
    # Lazy imports to keep metric functions importable without heavy deps
    from hybrid_recsys.config import Settings
    from hybrid_recsys.models import RecoRequest
    from hybrid_recsys.providers.embeddings.sentence_tf import SentenceTransformerProvider
    from hybrid_recsys.providers.llm.mock import MockLLMProvider
    from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

    settings = Settings()
    embedder = SentenceTransformerProvider(settings.embedding_model)
    llm = MockLLMProvider()
    pipeline = RecommendationPipeline(embedder, llm, settings)

    with open(catalog_path) as f:
        catalog = json.load(f)

    # Build topic labels for each program
    program_topics: dict[str, str] = {}
    for prog in catalog["programs"]:
        topic = infer_topic(prog["description"])
        if topic:
            program_topics[prog["program_id"]] = topic

    k_values = [3, 5]
    results: list[dict] = []

    for lang, queries in TOPIC_QUERIES.items():
        if not (index_dir / lang).exists():
            print(f"Skipping {lang}: no index found at {index_dir / lang}")
            continue

        for qinfo in queries:
            query_text = str(qinfo["query"])
            expected_topics = set(qinfo["topics"])

            relevant = {
                pid for pid, topic in program_topics.items()
                if topic in expected_topics
                and catalog_lang(catalog, pid) == lang
            }

            if not relevant:
                continue

            request = RecoRequest(query=query_text, lang=lang, size=max(k_values))
            response = pipeline.recommend(request)

            for k in k_values:
                p = precision_at_k(response.programs, relevant, k)
                r = recall_at_k(response.programs, relevant, k)
                n = ndcg_at_k(response.programs, relevant, k)
                results.append({
                    "lang": lang,
                    "query": query_text[:50],
                    "k": k,
                    "precision": p,
                    "recall": r,
                    "ndcg": n,
                    "relevant_count": len(relevant),
                })

    # Print summary table
    print(f"\n{'Lang':<5} {'k':<3} {'P@k':>6} {'R@k':>6} {'nDCG':>6}  Query")
    print("-" * 75)
    for r in results:
        print(
            f"{r['lang']:<5} {r['k']:<3} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['ndcg']:>6.3f}  {r['query']}"
        )

    if results:
        for k in k_values:
            k_results = [r for r in results if r["k"] == k]
            avg_p = sum(r["precision"] for r in k_results) / len(k_results)
            avg_r = sum(r["recall"] for r in k_results) / len(k_results)
            avg_n = sum(r["ndcg"] for r in k_results) / len(k_results)
            print(f"\nAverage @{k}: precision={avg_p:.3f}  recall={avg_r:.3f}  nDCG={avg_n:.3f}")


if __name__ == "__main__":
    catalog_path = Path("data/catalog.json")
    index_dir = Path("data/index")

    if not catalog_path.exists():
        print(f"Catalog not found at {catalog_path}. Run generate_catalog.py first.")
        sys.exit(1)
    if not index_dir.exists():
        print(f"Indexes not found at {index_dir}. Run 'hybrid-recsys index' first.")
        sys.exit(1)

    run_evaluation(catalog_path, index_dir)
