"""Latency benchmark: times recommendation queries and reports p50/p95.

Usage:
    uv run python scripts/benchmark_latency.py [--n 100] [--lang en] [--size 3] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def run_benchmark(n: int, lang: str, size: int, json_out: Path | None = None) -> None:
    """Run n queries and report latency statistics."""
    from hybrid_recsys.config import Settings
    from hybrid_recsys.models import RecoRequest
    from hybrid_recsys.providers.embeddings.sentence_tf import SentenceTransformerProvider
    from hybrid_recsys.providers.llm.mock import MockLLMProvider
    from hybrid_recsys.retrieval.pipeline import RecommendationPipeline

    settings = Settings()
    if not settings.index_dir.exists():
        print(f"Indexes not found at {settings.index_dir}. Build them first.")
        sys.exit(1)

    embedder = SentenceTransformerProvider(settings.embedding_model)
    llm = MockLLMProvider()
    pipeline = RecommendationPipeline(embedder, llm, settings)

    queries = [
        "artificial intelligence",
        "true crime podcast",
        "history of Rome",
        "science for kids",
        "football highlights",
        "jazz music",
        "climate change",
        "philosophy of mind",
        "travel adventure",
        "health and nutrition",
        "comedy show",
        "business strategy",
    ]

    # Warm up (load index + model)
    warmup_req = RecoRequest(query="warmup", lang=lang, size=size)
    pipeline.recommend(warmup_req)

    latencies: list[float] = []
    for i in range(n):
        query = queries[i % len(queries)]
        request = RecoRequest(query=query, lang=lang, size=size)
        start = time.perf_counter()
        pipeline.recommend(request)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = int(len(latencies) * 0.95) - 1
    p95 = latencies[max(0, p95_idx)]
    mean = statistics.mean(latencies)

    print(f"\nLatency benchmark: {n} queries (lang={lang}, size={size})")
    print(f"  Mean:  {mean:>7.1f} ms")
    print(f"  p50:   {p50:>7.1f} ms")
    print(f"  p95:   {p95:>7.1f} ms")
    print(f"  Min:   {min(latencies):>7.1f} ms")
    print(f"  Max:   {max(latencies):>7.1f} ms")

    if json_out is not None:
        report = {
            "config": {
                "n_queries": n,
                "lang": lang,
                "size": size,
                "embedding_model": settings.embedding_model,
                "llm": "mock (keyword-overlap re-ranker)",
                "platform": platform.platform(),
            },
            "latency_ms": {
                "mean": round(mean, 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "min": round(min(latencies), 1),
                "max": round(max(latencies), 1),
            },
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark recommendation latency")
    parser.add_argument("--n", type=int, default=100, help="Number of queries")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--size", type=int, default=3, help="Results per query")
    parser.add_argument("--json-out", type=Path, default=None, help="Write results as JSON")
    args = parser.parse_args()
    run_benchmark(args.n, args.lang, args.size, args.json_out)
