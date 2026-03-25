"""
Query Latency Benchmark

Measures P50, P95, P99 latency for BM25, kNN, and Hybrid search,
broken down by language (en, zh-tw, zh-cn).

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --runs 20
    python scripts/benchmark_latency.py --language en
    python scripts/benchmark_latency.py --output data/benchmark/latency_baseline.json
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

from src.services.search_service import SearchService, SearchMode


TEST_QUERIES: Dict[str, List[str]] = {
    "en": [
        "technology podcast",
        "business news",
        "self improvement",
        "machine learning",
        "artificial intelligence",
        "startup founders",
        "book recommendations",
        "how to start a business",
    ],
    "zh-tw": [
        "科技新聞",
        "投資理財",
        "學英文",
        "AI 人工智慧",
        "如何開始投資",
    ],
    "zh-cn": [
        "科技资讯",
        "投资理财",
        "人工智能",
        "如何开始投资理财",
        "身心健康",
    ],
}


@dataclass
class LatencyStats:
    mode: str
    language: str
    runs: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float


@dataclass
class BenchmarkResult:
    timestamp: str
    environment: str
    total_runs_per_mode: int
    es_host: str
    by_language: Dict[str, Any]
    breakdown: Dict[str, Any]


def percentile(data: List[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def measure_latency(
    service: SearchService,
    queries: List[str],
    language: str,
    mode: SearchMode,
    runs: int = 10,
) -> tuple[List[float], Dict[str, List[float]]]:
    total_latencies = []
    per_query_latencies = {q: [] for q in queries}

    for _ in range(runs):
        for query in queries:
            start = time.perf_counter()

            if mode == SearchMode.BM25:
                service.search_bm25(query, size=10, language=language)
            elif mode == SearchMode.KNN:
                service.search_knn(query, size=10, language=language)
            else:
                service.search_hybrid(query, size=10, language=language)

            elapsed_ms = (time.perf_counter() - start) * 1000
            total_latencies.append(elapsed_ms)
            per_query_latencies[query].append(elapsed_ms)

    return total_latencies, per_query_latencies


def compute_stats(latencies: List[float], mode: str, language: str) -> LatencyStats:
    return LatencyStats(
        mode=mode,
        language=language,
        runs=len(latencies),
        min_ms=round(min(latencies), 2),
        max_ms=round(max(latencies), 2),
        mean_ms=round(statistics.mean(latencies), 2),
        p50_ms=round(percentile(latencies, 50), 2),
        p95_ms=round(percentile(latencies, 95), 2),
        p99_ms=round(percentile(latencies, 99), 2),
        std_ms=round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
    )


def run_benchmark(
    runs: int = 10,
    environment: str = "local",
    languages: List[str] = None,
) -> BenchmarkResult:
    if languages is None:
        languages = list(TEST_QUERIES.keys())

    service = SearchService()
    es_host = os.getenv("ES_HOST", "http://localhost:9200")

    print(f"ES Host: {es_host}")
    print(f"Environment: {environment}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Runs: {runs} per query")
    print()

    print("Warming up...")
    service.search_hybrid("test", size=10, language="en")
    print()

    by_language: Dict[str, Any] = {}
    breakdown: Dict[str, Any] = {}

    for lang in languages:
        queries = TEST_QUERIES[lang]
        print(f"--- {lang} ({len(queries)} queries) ---")
        lang_results: Dict[str, LatencyStats] = {}
        lang_breakdown: Dict[str, Any] = {}

        for mode in [SearchMode.BM25, SearchMode.KNN, SearchMode.HYBRID]:
            latencies, per_query = measure_latency(service, queries, lang, mode, runs)
            stats = compute_stats(latencies, mode.value, lang)
            lang_results[mode.value] = stats
            lang_breakdown[mode.value] = {
                q: {
                    "mean_ms": round(statistics.mean(times), 2),
                    "p50_ms": round(percentile(times, 50), 2),
                }
                for q, times in per_query.items()
            }
            print(f"  {mode.value:<8} P50={stats.p50_ms}ms  P95={stats.p95_ms}ms  P99={stats.p99_ms}ms")

        by_language[lang] = {
            "queries": len(queries),
            "bm25": asdict(lang_results["bm25"]),
            "knn": asdict(lang_results["knn"]),
            "hybrid": asdict(lang_results["hybrid"]),
        }
        breakdown[lang] = lang_breakdown
        print()

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        total_runs_per_mode=runs,
        es_host=es_host,
        by_language=by_language,
        breakdown=breakdown,
    )


def print_summary(result: BenchmarkResult) -> None:
    print("=" * 65)
    print(f"Query Latency Benchmark — {result.environment}")
    print("=" * 65)
    print(f"ES Host: {result.es_host}  |  Runs: {result.total_runs_per_mode} per query")
    print()

    header = f"  {'Language':<8} {'Mode':<8} {'P50':>9} {'P95':>9} {'P99':>9} {'Mean':>9}"
    print(header)
    print("  " + "-" * 55)

    for lang, data in result.by_language.items():
        for mode_key in ["bm25", "knn", "hybrid"]:
            s = data[mode_key]
            print(f"  {lang:<8} {mode_key:<8} {s['p50_ms']:>8.1f}ms {s['p95_ms']:>8.1f}ms {s['p99_ms']:>8.1f}ms {s['mean_ms']:>8.1f}ms")
        print()

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Query Latency Benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per query")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 5 runs per query (for use in evaluation suite)",
    )
    parser.add_argument("--env", type=str, default="local", help="Environment name")
    parser.add_argument(
        "--language",
        type=str,
        default="all",
        help="Language to benchmark: en, zh-tw, zh-cn, or all (default: all)",
    )
    parser.add_argument("--output", type=Path, help="Output JSON file path")

    args = parser.parse_args()

    runs = 5 if args.quick else args.runs

    if args.language == "all":
        languages = list(TEST_QUERIES.keys())
    elif args.language in TEST_QUERIES:
        languages = [args.language]
    else:
        parser.error(f"Unknown language '{args.language}'. Choose from: {', '.join(TEST_QUERIES)} or 'all'")

    result = run_benchmark(runs=runs, environment=args.env, languages=languages)
    print_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
