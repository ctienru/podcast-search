#!/usr/bin/env python3
"""
Annotate Annotation Pool with Cross-Encoder

Reads annotation_pool.json and uses a cross-encoder model to score
each (query, episode) pair, then converts scores to relevance labels.

Usage:
    python scripts/annotate_with_cross_encoder.py \
        --pool data/evaluation/annotation_pool.json \
        --output data/evaluation/relevance_judgments.json

    # Incremental annotation
    python scripts/annotate_with_cross_encoder.py \
        --pool data/evaluation/annotation_pool_with_hybrid.json \
        --existing data/evaluation/relevance_judgments.json \
        --output data/evaluation/relevance_judgments_with_hybrid.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sentence_transformers
import torch

from src.evaluation.cross_encoder_judge import CrossEncoderJudge
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_annotation_pool(path: Path) -> Dict[str, Any]:
    """Load annotation pool from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_judgments(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load existing judgments for incremental annotation."""
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("judgments", {})


def build_annotation_text(episode: Dict[str, Any]) -> str:
    """Build text for annotation from episode data."""
    title = episode.get("title", "") or ""
    description = episode.get("description", "") or ""

    # Combine title and description (title-weighted)
    # Cross-encoder sees both, so no need for explicit weighting
    text = f"{title}\n\n{description}"

    # Truncate to reasonable length (cross-encoder has token limit)
    max_chars = 1500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def annotate_pool(
    pool: Dict[str, Dict[str, Any]],
    judge: CrossEncoderJudge,
    existing: Dict[str, Dict[str, Any]],
    batch_size: int = 32,
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Annotate all episodes in the pool.

    Args:
        pool: Annotation pool {query -> {episode_id -> info}}
        judge: CrossEncoderJudge instance
        existing: Existing judgments to preserve
        batch_size: Batch size for inference

    Returns:
        Tuple of (judgments dict, warnings list)
    """
    judgments: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    total_queries = len(pool)
    total_new = 0
    total_reused = 0

    for i, (query, episodes) in enumerate(pool.items()):
        logger.info(
            "annotating_query",
            extra={
                "index": i + 1,
                "total": total_queries,
                "query": query,
                "episodes": len(episodes),
            },
        )

        # Check existing judgments
        existing_for_query = existing.get(query, {})

        # Separate new and existing episodes
        new_episodes = {}
        reused_episodes = {}

        for ep_id, ep_info in episodes.items():
            if ep_id in existing_for_query:
                reused_episodes[ep_id] = existing_for_query[ep_id]
                total_reused += 1
            else:
                new_episodes[ep_id] = ep_info

        # Annotate new episodes
        if new_episodes:
            ep_ids = list(new_episodes.keys())
            texts = [build_annotation_text(new_episodes[ep_id]) for ep_id in ep_ids]

            # Batch score
            scores = judge.score_batch(query, texts, batch_size=batch_size)
            labels = judge.scores_to_labels(scores)

            for ep_id, score, label in zip(ep_ids, scores, labels):
                reused_episodes[ep_id] = {
                    "score": float(score),
                    "relevance": int(label),
                }
                total_new += 1

        judgments[query] = reused_episodes

        # Check for anomalies
        all_labels = [j["relevance"] for j in reused_episodes.values()]
        if len(set(all_labels)) == 1 and len(all_labels) > 5:
            warnings.append(f"Query '{query}' has no label variance (all labels = {all_labels[0]})")

    logger.info(
        "annotation_complete",
        extra={
            "new_annotations": total_new,
            "reused_annotations": total_reused,
            "total_queries": total_queries,
        },
    )

    return judgments, warnings


def compute_stats(judgments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics about the judgments."""
    all_scores = []
    all_labels = []
    by_query_variance = []

    for query, eps in judgments.items():
        query_labels = [j["relevance"] for j in eps.values()]
        query_scores = [j["score"] for j in eps.values()]

        all_labels.extend(query_labels)
        all_scores.extend(query_scores)

        if query_labels:
            by_query_variance.append(np.var(query_labels))

    if not all_scores:
        return {}

    arr_scores = np.array(all_scores)
    arr_labels = np.array(all_labels)

    return {
        "total_judgments": len(all_scores),
        "score_stats": {
            "min": float(arr_scores.min()),
            "max": float(arr_scores.max()),
            "mean": float(arr_scores.mean()),
            "std": float(arr_scores.std()),
        },
        "label_distribution": {
            str(i): int((arr_labels == i).sum()) for i in range(4)
        },
        "avg_query_label_variance": float(np.mean(by_query_variance)) if by_query_variance else 0,
    }


def save_judgments(
    judgments: Dict[str, Dict[str, Any]],
    warnings: List[str],
    output_path: Path,
    model_name: str,
    seed: int,
) -> None:
    """Save judgments with metadata."""
    stats = compute_stats(judgments)

    output = {
        "meta": {
            "model": model_name,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
        },
        "judgments": judgments,
        "warnings": warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "judgments_saved",
        extra={
            "path": str(output_path),
            "total_judgments": stats.get("total_judgments", 0),
            "warnings_count": len(warnings),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Annotate pool with cross-encoder")
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("data/evaluation/annotation_pool.json"),
        help="Path to annotation pool JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/relevance_judgments.json"),
        help="Output path for relevance judgments",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=None,
        help="Path to existing judgments for incremental annotation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    setup_logging()

    # Suppress noisy loggers
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Load annotation pool
    data = load_annotation_pool(args.pool)
    pool = data.get("pool", data)  # Support both wrapped and raw format

    logger.info(
        "pool_loaded",
        extra={"queries": len(pool), "path": str(args.pool)},
    )

    # Load existing judgments
    existing = load_existing_judgments(args.existing)
    if existing:
        logger.info(
            "existing_judgments_loaded",
            extra={"queries": len(existing)},
        )

    # Initialize judge
    judge = CrossEncoderJudge(model_name=args.model, seed=args.seed)

    # Annotate
    judgments, warnings = annotate_pool(
        pool=pool,
        judge=judge,
        existing=existing,
        batch_size=args.batch_size,
    )

    # Save
    save_judgments(
        judgments=judgments,
        warnings=warnings,
        output_path=args.output,
        model_name=args.model,
        seed=args.seed,
    )

    # Print summary
    stats = compute_stats(judgments)
    print(f"\nJudgments saved to: {args.output}")
    print(f"Total judgments: {stats.get('total_judgments', 0)}")
    print(f"Label distribution: {stats.get('label_distribution', {})}")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
