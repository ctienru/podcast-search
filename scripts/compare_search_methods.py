"""
Compare BM25 vs kNN vs Hybrid Search Methods

Analyzes:
1. Result overlap between methods
2. Unique results from each method
3. Rank correlation
4. Cases where Hybrid helps vs doesn't help

Usage:
    python scripts/compare_search_methods.py
    python scripts/compare_search_methods.py --output data/evaluation/method_comparison.json
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Tuple

from src.services.search_service import SearchService, SearchMode, SearchResult


# Test queries
TEST_QUERIES = [
    # Chinese topic queries
    "科技新聞",
    "投資理財",
    "學英文",
    "創業故事",
    "心理健康",
    # Chinese entity queries
    "台灣政治",
    "AI 人工智慧",
    "股票分析",
    # English topic queries
    "technology podcast",
    "business news",
    "self improvement",
    "true crime",
    # English entity queries
    "artificial intelligence",
    "startup founders",
    "machine learning",
    # Mixed / long-tail
    "podcast 推薦",
    "如何開始投資",
    "how to start a business",
    # Edge cases
    "interview",
    "news",
]


@dataclass
class QueryComparison:
    """Comparison result for a single query."""
    query: str
    bm25_ids: List[str]
    knn_ids: List[str]
    hybrid_ids: List[str]
    bm25_only: List[str]  # In BM25 but not kNN
    knn_only: List[str]   # In kNN but not BM25
    overlap: List[str]    # In both BM25 and kNN
    hybrid_from_bm25: int  # Hybrid results that came from BM25
    hybrid_from_knn: int   # Hybrid results that came from kNN
    hybrid_from_both: int  # Hybrid results in both
    bm25_knn_jaccard: float
    bm25_hybrid_jaccard: float
    knn_hybrid_jaccard: float
    # RRF benefit analysis
    rrf_promoted: List[str]  # IDs that moved up in Hybrid vs both individual methods
    rrf_demoted: List[str]   # IDs that moved down


@dataclass
class MethodStats:
    """Aggregated statistics for method comparison."""
    total_queries: int
    avg_bm25_knn_jaccard: float
    avg_bm25_hybrid_jaccard: float
    avg_knn_hybrid_jaccard: float
    avg_bm25_only_count: float
    avg_knn_only_count: float
    avg_overlap_count: float
    queries_where_hybrid_helps: int
    queries_where_bm25_better: int
    queries_where_knn_better: int


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def analyze_query(
    service: SearchService,
    query: str,
    k: int = 10,
) -> QueryComparison:
    """Analyze search results for a single query."""
    # Get results from all three methods
    bm25_response = service.search_bm25(query, size=k)
    knn_response = service.search_knn(query, size=k)
    hybrid_response = service.search_hybrid(query, size=k)

    bm25_ids = [r.episode_id for r in bm25_response.results]
    knn_ids = [r.episode_id for r in knn_response.results]
    hybrid_ids = [r.episode_id for r in hybrid_response.results]

    bm25_set = set(bm25_ids)
    knn_set = set(knn_ids)
    hybrid_set = set(hybrid_ids)

    # Calculate overlaps
    bm25_only = list(bm25_set - knn_set)
    knn_only = list(knn_set - bm25_set)
    overlap = list(bm25_set & knn_set)

    # Analyze hybrid composition
    hybrid_from_bm25 = len([h for h in hybrid_ids if h in bm25_set and h not in knn_set])
    hybrid_from_knn = len([h for h in hybrid_ids if h in knn_set and h not in bm25_set])
    hybrid_from_both = len([h for h in hybrid_ids if h in bm25_set and h in knn_set])

    # Calculate Jaccard similarities
    bm25_knn_jaccard = jaccard_similarity(bm25_set, knn_set)
    bm25_hybrid_jaccard = jaccard_similarity(bm25_set, hybrid_set)
    knn_hybrid_jaccard = jaccard_similarity(knn_set, hybrid_set)

    # RRF benefit analysis
    # Find items that are in hybrid top-k but weren't in top-k of either method alone
    rrf_promoted = [h for h in hybrid_ids if h not in bm25_ids[:k] and h not in knn_ids[:k]]
    rrf_demoted = [b for b in bm25_ids[:5] if b not in hybrid_ids[:5]] + \
                  [n for n in knn_ids[:5] if n not in hybrid_ids[:5]]

    return QueryComparison(
        query=query,
        bm25_ids=bm25_ids,
        knn_ids=knn_ids,
        hybrid_ids=hybrid_ids,
        bm25_only=bm25_only,
        knn_only=knn_only,
        overlap=overlap,
        hybrid_from_bm25=hybrid_from_bm25,
        hybrid_from_knn=hybrid_from_knn,
        hybrid_from_both=hybrid_from_both,
        bm25_knn_jaccard=round(bm25_knn_jaccard, 4),
        bm25_hybrid_jaccard=round(bm25_hybrid_jaccard, 4),
        knn_hybrid_jaccard=round(knn_hybrid_jaccard, 4),
        rrf_promoted=rrf_promoted,
        rrf_demoted=list(set(rrf_demoted)),
    )


def compute_stats(comparisons: List[QueryComparison]) -> MethodStats:
    """Compute aggregated statistics."""
    n = len(comparisons)

    avg_bm25_knn = sum(c.bm25_knn_jaccard for c in comparisons) / n
    avg_bm25_hybrid = sum(c.bm25_hybrid_jaccard for c in comparisons) / n
    avg_knn_hybrid = sum(c.knn_hybrid_jaccard for c in comparisons) / n
    avg_bm25_only = sum(len(c.bm25_only) for c in comparisons) / n
    avg_knn_only = sum(len(c.knn_only) for c in comparisons) / n
    avg_overlap = sum(len(c.overlap) for c in comparisons) / n

    # Count queries where hybrid brings unique value
    hybrid_helps = sum(1 for c in comparisons if c.hybrid_from_bm25 > 0 and c.hybrid_from_knn > 0)

    # BM25 dominant = hybrid mostly from BM25
    bm25_better = sum(1 for c in comparisons if c.hybrid_from_bm25 > c.hybrid_from_knn)
    knn_better = sum(1 for c in comparisons if c.hybrid_from_knn > c.hybrid_from_bm25)

    return MethodStats(
        total_queries=n,
        avg_bm25_knn_jaccard=round(avg_bm25_knn, 4),
        avg_bm25_hybrid_jaccard=round(avg_bm25_hybrid, 4),
        avg_knn_hybrid_jaccard=round(avg_knn_hybrid, 4),
        avg_bm25_only_count=round(avg_bm25_only, 2),
        avg_knn_only_count=round(avg_knn_only, 2),
        avg_overlap_count=round(avg_overlap, 2),
        queries_where_hybrid_helps=hybrid_helps,
        queries_where_bm25_better=bm25_better,
        queries_where_knn_better=knn_better,
    )


def find_interesting_cases(comparisons: List[QueryComparison]) -> Dict:
    """Find interesting cases for blog/interview."""
    cases = {
        "high_disagreement": [],  # BM25 and kNN return very different results
        "high_agreement": [],      # BM25 and kNN agree
        "hybrid_adds_value": [],   # Hybrid brings results from both
        "bm25_dominant": [],       # Hybrid mostly follows BM25
        "knn_dominant": [],        # Hybrid mostly follows kNN
    }

    for c in comparisons:
        if c.bm25_knn_jaccard < 0.3:
            cases["high_disagreement"].append({
                "query": c.query,
                "jaccard": c.bm25_knn_jaccard,
                "bm25_only": len(c.bm25_only),
                "knn_only": len(c.knn_only),
            })
        elif c.bm25_knn_jaccard > 0.7:
            cases["high_agreement"].append({
                "query": c.query,
                "jaccard": c.bm25_knn_jaccard,
            })

        if c.hybrid_from_bm25 >= 2 and c.hybrid_from_knn >= 2:
            cases["hybrid_adds_value"].append({
                "query": c.query,
                "from_bm25": c.hybrid_from_bm25,
                "from_knn": c.hybrid_from_knn,
                "from_both": c.hybrid_from_both,
            })

        if c.hybrid_from_bm25 > c.hybrid_from_knn + 3:
            cases["bm25_dominant"].append({"query": c.query})
        elif c.hybrid_from_knn > c.hybrid_from_bm25 + 3:
            cases["knn_dominant"].append({"query": c.query})

    return cases


def run_comparison(k: int = 10) -> Tuple[List[QueryComparison], MethodStats, Dict]:
    """Run the full comparison."""
    service = SearchService()

    print(f"Comparing search methods for {len(TEST_QUERIES)} queries (k={k})")
    print()

    # Warm up
    print("Warming up...")
    service.search_hybrid("test", size=10)
    print()

    comparisons = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"[{i+1}/{len(TEST_QUERIES)}] {query}")
        comparison = analyze_query(service, query, k=k)
        comparisons.append(comparison)
        print(f"  BM25-kNN Jaccard: {comparison.bm25_knn_jaccard:.2%}")
        print(f"  Hybrid composition: BM25={comparison.hybrid_from_bm25}, kNN={comparison.hybrid_from_knn}, Both={comparison.hybrid_from_both}")

    stats = compute_stats(comparisons)
    cases = find_interesting_cases(comparisons)

    return comparisons, stats, cases


def print_summary(stats: MethodStats, cases: Dict) -> None:
    """Print summary."""
    print()
    print("=" * 60)
    print("Search Method Comparison Summary")
    print("=" * 60)
    print()
    print("Similarity (Jaccard):")
    print(f"  BM25 vs kNN:    {stats.avg_bm25_knn_jaccard:.2%}")
    print(f"  BM25 vs Hybrid: {stats.avg_bm25_hybrid_jaccard:.2%}")
    print(f"  kNN vs Hybrid:  {stats.avg_knn_hybrid_jaccard:.2%}")
    print()
    print("Result Composition (avg per query):")
    print(f"  BM25 only:  {stats.avg_bm25_only_count:.1f} results")
    print(f"  kNN only:   {stats.avg_knn_only_count:.1f} results")
    print(f"  Overlap:    {stats.avg_overlap_count:.1f} results")
    print()
    print("Hybrid Value:")
    print(f"  Queries where Hybrid combines both: {stats.queries_where_hybrid_helps}/{stats.total_queries}")
    print(f"  Queries where BM25 dominates:       {stats.queries_where_bm25_better}/{stats.total_queries}")
    print(f"  Queries where kNN dominates:        {stats.queries_where_knn_better}/{stats.total_queries}")
    print()
    print("Interesting Cases:")
    print(f"  High disagreement (Jaccard < 30%): {len(cases['high_disagreement'])} queries")
    for c in cases['high_disagreement'][:3]:
        print(f"    - \"{c['query']}\" (Jaccard={c['jaccard']:.2%})")
    print(f"  High agreement (Jaccard > 70%):    {len(cases['high_agreement'])} queries")
    print(f"  Hybrid adds value:                 {len(cases['hybrid_adds_value'])} queries")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare Search Methods")
    parser.add_argument("--k", type=int, default=10, help="Number of results to compare")
    parser.add_argument("--output", type=Path, help="Output JSON file path")

    args = parser.parse_args()

    comparisons, stats, cases = run_comparison(k=args.k)
    print_summary(stats, cases)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "k": args.k,
            "stats": asdict(stats),
            "interesting_cases": cases,
            "per_query": [asdict(c) for c in comparisons],
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
