#!/usr/bin/env python3
"""
Build Annotation Pool for Relevance Judgments

For each query, execute BM25 and Embedding (kNN) search, then merge results
into an annotation pool. Each episode appears only once per query with:
- sources: which search methods retrieved it
- ranks: the rank from each method

Usage:
    python scripts/build_annotation_pool.py \
        --queries data/evaluation/test_queries.json \
        --output data/evaluation/annotation_pool.json

    # Include hybrid
    python scripts/build_annotation_pool.py \
        --queries data/evaluation/test_queries.json \
        --output data/evaluation/annotation_pool_with_hybrid.json \
        --include-hybrid
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.services.search_service import SearchService
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_annotation_pool(
    queries: List[Dict[str, Any]],
    search_service: SearchService,
    k: int = 10,
    include_hybrid: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Build annotation pool from search results.

    Args:
        queries: List of query dicts with 'query' key
        search_service: SearchService instance
        k: Number of results per method
        include_hybrid: Whether to include hybrid search

    Returns:
        Dict mapping query -> {episode_id -> episode_info}
    """
    pool = {}

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info(
            "processing_query",
            extra={"index": i + 1, "total": len(queries), "query": query_text},
        )

        # Collect results from each method
        methods_results = []

        # BM25 search (evaluation mode: no time_decay, no language filter)
        try:
            bm25_response = search_service.search_bm25(
                query_text, size=k, evaluation_mode=True
            )
            methods_results.append(("bm25", bm25_response.results))
            logger.debug(
                "bm25_results",
                extra={"query": query_text, "count": len(bm25_response.results)},
            )
        except Exception as e:
            logger.warning(
                "bm25_search_failed",
                extra={"query": query_text, "error": str(e)},
            )
            methods_results.append(("bm25", []))

        # kNN (embedding) search
        try:
            knn_response = search_service.search_knn(query_text, size=k)
            methods_results.append(("embedding", knn_response.results))
            logger.debug(
                "embedding_results",
                extra={"query": query_text, "count": len(knn_response.results)},
            )
        except Exception as e:
            logger.warning(
                "embedding_search_failed",
                extra={"query": query_text, "error": str(e)},
            )
            methods_results.append(("embedding", []))

        # Hybrid search (optional, Phase 3)
        if include_hybrid:
            try:
                hybrid_response = search_service.search_hybrid(query_text, size=k)
                methods_results.append(("hybrid", hybrid_response.results))
                logger.debug(
                    "hybrid_results",
                    extra={"query": query_text, "count": len(hybrid_response.results)},
                )
            except Exception as e:
                logger.warning(
                    "hybrid_search_failed",
                    extra={"query": query_text, "error": str(e)},
                )
                methods_results.append(("hybrid", []))

        # Merge results (de-duplicate by episode_id)
        unique: Dict[str, Dict[str, Any]] = {}

        for method_name, results in methods_results:
            for rank, result in enumerate(results):
                ep_id = result.episode_id

                if ep_id not in unique:
                    unique[ep_id] = {
                        "episode_id": ep_id,
                        "title": result.title,
                        "description": result.description,
                        "sources": [],
                        "ranks": {},
                    }

                if method_name not in unique[ep_id]["sources"]:
                    unique[ep_id]["sources"].append(method_name)

                # Record rank (1-indexed)
                unique[ep_id]["ranks"][method_name] = rank + 1

        pool[query_text] = unique

        logger.info(
            "query_pool_built",
            extra={
                "query": query_text,
                "unique_episodes": len(unique),
            },
        )

    return pool


def save_annotation_pool(
    pool: Dict[str, Dict[str, Any]],
    output_path: Path,
    queries: List[Dict[str, Any]],
    include_hybrid: bool,
) -> None:
    """Save annotation pool to JSON file with metadata."""
    # Calculate stats
    total_episodes = sum(len(episodes) for episodes in pool.values())

    output = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(queries),
            "total_episodes": total_episodes,
            "include_hybrid": include_hybrid,
            "methods": ["bm25", "embedding"] + (["hybrid"] if include_hybrid else []),
        },
        "pool": pool,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "annotation_pool_saved",
        extra={
            "path": str(output_path),
            "queries": len(queries),
            "episodes": total_episodes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Build annotation pool for evaluation")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/test_queries.json"),
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/annotation_pool.json"),
        help="Output path for annotation pool",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results per search method (default: 10)",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include hybrid search results",
    )
    args = parser.parse_args()

    setup_logging()

    # Suppress noisy loggers
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Load queries
    queries = load_queries(args.queries)
    logger.info("queries_loaded", extra={"count": len(queries), "path": str(args.queries)})

    # Initialize search service
    search_service = SearchService()

    # Build annotation pool
    pool = build_annotation_pool(
        queries=queries,
        search_service=search_service,
        k=args.k,
        include_hybrid=args.include_hybrid,
    )

    # Save results
    save_annotation_pool(
        pool=pool,
        output_path=args.output,
        queries=queries,
        include_hybrid=args.include_hybrid,
    )

    print(f"\nAnnotation pool saved to: {args.output}")
    print(f"Total queries: {len(queries)}")
    print(f"Total unique episodes: {sum(len(e) for e in pool.values())}")


if __name__ == "__main__":
    main()
