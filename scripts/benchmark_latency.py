"""
Query Latency Benchmark

Measures P50, P95, P99 latency for BM25, kNN, and Hybrid search.

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --runs 50
    python scripts/benchmark_latency.py --output data/benchmark/latency_local.json
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from src.services.search_service import SearchService, SearchMode


# Test queries (mix of Chinese and English)
TEST_QUERIES = [
    "科技新聞",
    "投資理財",
    "學英文",
    "AI 人工智慧",
    "technology podcast",
    "business news",
    "self improvement",
    "machine learning",
    "如何開始投資",
    "startup founders",
]


@dataclass
class LatencyStats:
    """Latency statistics for a search mode."""
    mode: str
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
    """Complete benchmark result."""
    timestamp: str
    environment: str
    total_runs_per_mode: int
    queries_per_run: int
    es_host: str
    bm25: LatencyStats
    knn: LatencyStats
    hybrid: LatencyStats
    breakdown: Dict[str, Any]


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def measure_latency(
    service: SearchService,
    queries: List[str],
    mode: SearchMode,
    runs: int = 10,
) -> tuple[List[float], Dict[str, List[float]]]:
    """
    Measure search latency.

    Returns:
        Tuple of (total_latencies, per_query_latencies)
    """
    total_latencies = []
    per_query_latencies = {q: [] for q in queries}

    for run in range(runs):
        for query in queries:
            start = time.perf_counter()

            if mode == SearchMode.BM25:
                service.search_bm25(query, size=10)
            elif mode == SearchMode.KNN:
                service.search_knn(query, size=10)
            else:
                service.search_hybrid(query, size=10)

            elapsed_ms = (time.perf_counter() - start) * 1000
            total_latencies.append(elapsed_ms)
            per_query_latencies[query].append(elapsed_ms)

    return total_latencies, per_query_latencies


def compute_stats(latencies: List[float], mode: str) -> LatencyStats:
    """Compute latency statistics."""
    return LatencyStats(
        mode=mode,
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
) -> BenchmarkResult:
    """Run the full benchmark."""
    service = SearchService()

    # Get ES host from environment
    import os
    es_host = os.getenv("ES_HOST", "http://localhost:9200")

    print(f"Running benchmark: {runs} runs x {len(TEST_QUERIES)} queries")
    print(f"ES Host: {es_host}")
    print(f"Environment: {environment}")
    print()

    # Warm up (first query loads the model)
    print("Warming up...")
    service.search_hybrid("test", size=10)
    print()

    results = {}
    breakdown = {}

    for mode in [SearchMode.BM25, SearchMode.KNN, SearchMode.HYBRID]:
        print(f"Testing {mode.value}...")
        latencies, per_query = measure_latency(service, TEST_QUERIES, mode, runs)
        stats = compute_stats(latencies, mode.value)
        results[mode.value] = stats
        breakdown[mode.value] = {
            q: {
                "mean_ms": round(statistics.mean(times), 2),
                "p50_ms": round(percentile(times, 50), 2),
            }
            for q, times in per_query.items()
        }
        print(f"  P50: {stats.p50_ms}ms, P99: {stats.p99_ms}ms")

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        total_runs_per_mode=runs,
        queries_per_run=len(TEST_QUERIES),
        es_host=es_host,
        bm25=results["bm25"],
        knn=results["knn"],
        hybrid=results["hybrid"],
        breakdown=breakdown,
    )


def print_summary(result: BenchmarkResult) -> None:
    """Print benchmark summary."""
    print()
    print("=" * 60)
    print(f"Query Latency Benchmark - {result.environment}")
    print("=" * 60)
    print(f"ES Host: {result.es_host}")
    print(f"Runs: {result.total_runs_per_mode} x {result.queries_per_run} queries")
    print()
    print(f"{'Mode':<10} {'P50':>10} {'P95':>10} {'P99':>10} {'Mean':>10}")
    print("-" * 50)

    for stats in [result.bm25, result.knn, result.hybrid]:
        print(f"{stats.mode:<10} {stats.p50_ms:>9.1f}ms {stats.p95_ms:>9.1f}ms {stats.p99_ms:>9.1f}ms {stats.mean_ms:>9.1f}ms")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query Latency Benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per query")
    parser.add_argument("--env", type=str, default="local", help="Environment name (local/cloud)")
    parser.add_argument("--output", type=Path, help="Output JSON file path")

    args = parser.parse_args()

    result = run_benchmark(runs=args.runs, environment=args.env)
    print_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
