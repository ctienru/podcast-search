#!/usr/bin/env python3
"""
Evaluate NDCG@k and MRR for Search Methods

Reads test queries and relevance judgments, executes searches,
and computes NDCG@10 and MRR for BM25 and Embedding methods.

Usage:
    python scripts/evaluate_ndcg_mrr.py \
        --queries data/evaluation/test_queries.json \
        --judgments data/evaluation/relevance_judgments.json \
        --output data/evaluation/ndcg_mrr_report.json

    # Include hybrid 
    python scripts/evaluate_ndcg_mrr.py \
        --queries data/evaluation/test_queries.json \
        --judgments data/evaluation/relevance_judgments_with_hybrid.json \
        --output data/evaluation/ndcg_mrr_report_phase3.json \
        --include-hybrid
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.evaluation.ranking_metrics import mrr, ndcg_at_k
from src.services.search_service import SearchService
from src.types import Language
from src.utils.logging import setup_logging


def _map_language(lang: str) -> Language:
    """Map query language labels to Language type used by search_knn().

    test_queries.json uses 'zh' (undifferentiated Chinese), which maps to the
    v1 zh-tw index. Update this mapping once language-split indices are in use.
    """
    return {"zh": "zh-tw", "mixed": "zh-tw"}.get(lang, lang)  # type: ignore[return-value]

logger = logging.getLogger(__name__)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judgments(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load relevance judgments from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("judgments", data)


def get_relevance_vector(
    result_ids: List[str],
    judgments: Dict[str, Any],
    k: int = 10,
) -> List[int]:
    """
    Build relevance vector for search results.

    Args:
        result_ids: List of episode IDs from search results
        judgments: Dict mapping episode_id -> judgment info
        k: Number of positions to consider

    Returns:
        List of relevance labels (0-3) for top-k results
    """
    relevances = []
    for ep_id in result_ids[:k]:
        judgment = judgments.get(ep_id, {})
        rel = judgment.get("relevance", 0)
        relevances.append(rel)

    # Pad with zeros if fewer than k results
    while len(relevances) < k:
        relevances.append(0)

    return relevances


def evaluate_single_query(
    query_text: str,
    search_service: SearchService,
    judgments: Dict[str, Any],
    k: int = 10,
    include_hybrid: bool = False,
    mrr_threshold: int = 2,
    language: str = "zh-tw",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single query against all methods.

    Returns:
        Dict mapping method -> {ndcg, mrr}
    """
    results = {}

    # BM25
    try:
        bm25_response = search_service.search_bm25(
            query_text, size=k, evaluation_mode=True
        )
        bm25_ids = [r.episode_id for r in bm25_response.results]
        bm25_rel = get_relevance_vector(bm25_ids, judgments, k)
        results["bm25"] = {
            "ndcg": ndcg_at_k(bm25_rel, k),
            "mrr": mrr(bm25_rel, threshold=mrr_threshold),
            "relevances": bm25_rel,
        }
    except Exception as e:
        logger.warning("bm25_eval_failed", extra={"query": query_text, "error": str(e)})
        results["bm25"] = {"ndcg": 0.0, "mrr": 0.0, "relevances": []}

    # Embedding (kNN)
    try:
        knn_response = search_service.search_knn(query_text, size=k, language=_map_language(language))
        knn_ids = [r.episode_id for r in knn_response.results]
        knn_rel = get_relevance_vector(knn_ids, judgments, k)
        results["embedding"] = {
            "ndcg": ndcg_at_k(knn_rel, k),
            "mrr": mrr(knn_rel, threshold=mrr_threshold),
            "relevances": knn_rel,
        }
    except Exception as e:
        logger.warning("embedding_eval_failed", extra={"query": query_text, "error": str(e)})
        results["embedding"] = {"ndcg": 0.0, "mrr": 0.0, "relevances": []}

    # Hybrid (optional)
    if include_hybrid:
        try:
            hybrid_response = search_service.search_hybrid(query_text, size=k)
            hybrid_ids = [r.episode_id for r in hybrid_response.results]
            hybrid_rel = get_relevance_vector(hybrid_ids, judgments, k)
            results["hybrid"] = {
                "ndcg": ndcg_at_k(hybrid_rel, k),
                "mrr": mrr(hybrid_rel, threshold=mrr_threshold),
                "relevances": hybrid_rel,
            }
        except Exception as e:
            logger.warning("hybrid_eval_failed", extra={"query": query_text, "error": str(e)})
            results["hybrid"] = {"ndcg": 0.0, "mrr": 0.0, "relevances": []}

    return results


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)

    arr = np.array(values)
    samples = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(samples, alpha * 100)),
        float(np.percentile(samples, (1 - alpha) * 100)),
    )


def aggregate_results(
    per_query_results: Dict[str, Dict[str, Dict[str, float]]],
    queries: List[Dict[str, Any]],
    methods: List[str],
    compute_ci: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate per-query results into summary statistics.

    Returns:
        Dict with overall, by_language, and by_category breakdowns
    """
    # Group queries by language and category
    by_language: Dict[str, List[str]] = defaultdict(list)
    by_category: Dict[str, List[str]] = defaultdict(list)

    query_to_meta = {q["query"]: q for q in queries}

    for query_text in per_query_results.keys():
        meta = query_to_meta.get(query_text, {})
        lang = meta.get("language", "unknown")
        cat = meta.get("category", "unknown")
        by_language[lang].append(query_text)
        by_category[cat].append(query_text)

    def compute_group_stats(query_list: List[str]) -> Dict[str, Any]:
        """Compute stats for a group of queries."""
        stats = {"queries": len(query_list)}

        for method in methods:
            ndcg_values = [
                per_query_results[q][method]["ndcg"]
                for q in query_list
                if method in per_query_results[q]
            ]
            mrr_values = [
                per_query_results[q][method]["mrr"]
                for q in query_list
                if method in per_query_results[q]
            ]

            method_stats = {
                "ndcg@10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
                "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
            }

            if compute_ci and len(ndcg_values) >= 5:
                ci_low, ci_high = bootstrap_ci(ndcg_values)
                method_stats["ci"] = [round(ci_low, 3), round(ci_high, 3)]

            stats[method] = method_stats

        return stats

    # Compute overall stats
    all_queries = list(per_query_results.keys())
    overall = compute_group_stats(all_queries)

    # Compute by_language stats
    by_lang_stats = {}
    for lang, query_list in sorted(by_language.items()):
        by_lang_stats[lang] = compute_group_stats(query_list)

    # Compute by_category stats
    by_cat_stats = {}
    for cat, query_list in sorted(by_category.items()):
        by_cat_stats[cat] = compute_group_stats(query_list)

    return {
        "overall": overall,
        "by_language": by_lang_stats,
        "by_category": by_cat_stats,
    }


def evaluate_all(
    queries: List[Dict[str, Any]],
    judgments: Dict[str, Dict[str, Any]],
    search_service: SearchService,
    k: int = 10,
    include_hybrid: bool = False,
    mrr_threshold: int = 2,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate all queries.

    Returns:
        Dict mapping query -> method -> {ndcg, mrr}
    """
    per_query_results = {}

    for i, q in enumerate(queries):
        query_text = q["query"]

        # Get judgments for this query
        query_judgments = judgments.get(query_text, {})

        if not query_judgments:
            logger.warning(
                "no_judgments_for_query",
                extra={"query": query_text},
            )
            continue

        logger.info(
            "evaluating_query",
            extra={
                "index": i + 1,
                "total": len(queries),
                "query": query_text,
                "judgment_count": len(query_judgments),
            },
        )

        results = evaluate_single_query(
            query_text=query_text,
            search_service=search_service,
            judgments=query_judgments,
            k=k,
            include_hybrid=include_hybrid,
            mrr_threshold=mrr_threshold,
            language=q.get("language", "zh-tw"),
        )

        per_query_results[query_text] = results

    return per_query_results


def save_report(
    per_query_results: Dict[str, Dict[str, Dict[str, float]]],
    aggregated: Dict[str, Any],
    output_path: Path,
    k: int,
    mrr_threshold: int,
    include_hybrid: bool,
) -> None:
    """Save evaluation report."""
    # Build per_query section (without relevances for cleaner output)
    per_query_clean = {}
    for query, methods in per_query_results.items():
        per_query_clean[query] = {
            method: {"ndcg@10": round(m["ndcg"], 4), "mrr": round(m["mrr"], 4)}
            for method, m in methods.items()
        }

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": 3 if include_hybrid else 2,
            "total_queries": len(per_query_results),
            "k": k,
            "mrr_threshold": mrr_threshold,
            "methods": ["bm25", "embedding"] + (["hybrid"] if include_hybrid else []),
        },
        **aggregated,
        "per_query": per_query_clean,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("report_saved", extra={"path": str(output_path)})


def print_summary(aggregated: Dict[str, Any], methods: List[str]) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("NDCG/MRR Evaluation Report")
    print("=" * 60)

    # Overall
    print("\n📊 Overall Results:")
    overall = aggregated["overall"]
    print(f"   Total queries: {overall['queries']}")
    print()

    header = "   {:12} {:>10} {:>10}".format("Method", "NDCG@10", "MRR")
    print(header)
    print("   " + "-" * 32)
    for method in methods:
        if method in overall:
            m = overall[method]
            ci_str = ""
            if "ci" in m:
                ci_str = f" [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
            print(f"   {method:12} {m['ndcg@10']:>10.4f} {m['mrr']:>10.4f}{ci_str}")

    # By language
    print("\n📈 By Language:")
    for lang, stats in aggregated["by_language"].items():
        print(f"\n   {lang.upper()} (n={stats['queries']}):")
        for method in methods:
            if method in stats:
                m = stats[method]
                print(f"      {method}: NDCG={m['ndcg@10']:.4f}, MRR={m['mrr']:.4f}")

    # By category
    print("\n📋 By Category:")
    for cat, stats in aggregated["by_category"].items():
        print(f"\n   {cat} (n={stats['queries']}):")
        for method in methods:
            if method in stats:
                m = stats[method]
                print(f"      {method}: NDCG={m['ndcg@10']:.4f}, MRR={m['mrr']:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NDCG and MRR for search methods")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/test_queries.json"),
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("data/evaluation/relevance_judgments.json"),
        help="Path to relevance judgments JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/ndcg_mrr_report.json"),
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to evaluate (default: 10)",
    )
    parser.add_argument(
        "--mrr-threshold",
        type=int,
        default=2,
        help="MRR relevance threshold (default: 2)",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include hybrid search evaluation",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Skip bootstrap confidence interval computation",
    )
    args = parser.parse_args()

    setup_logging()

    # Suppress noisy loggers
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Load data
    queries = load_queries(args.queries)
    judgments = load_judgments(args.judgments)

    logger.info(
        "data_loaded",
        extra={
            "queries": len(queries),
            "judgment_queries": len(judgments),
        },
    )

    # Initialize search service
    search_service = SearchService()

    # Evaluate
    methods = ["bm25", "embedding"] + (["hybrid"] if args.include_hybrid else [])

    per_query_results = evaluate_all(
        queries=queries,
        judgments=judgments,
        search_service=search_service,
        k=args.k,
        include_hybrid=args.include_hybrid,
        mrr_threshold=args.mrr_threshold,
    )

    # Aggregate
    aggregated = aggregate_results(
        per_query_results=per_query_results,
        queries=queries,
        methods=methods,
        compute_ci=not args.no_ci,
    )

    # Save report
    save_report(
        per_query_results=per_query_results,
        aggregated=aggregated,
        output_path=args.output,
        k=args.k,
        mrr_threshold=args.mrr_threshold,
        include_hybrid=args.include_hybrid,
    )

    # Print summary
    print_summary(aggregated, methods)

    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
