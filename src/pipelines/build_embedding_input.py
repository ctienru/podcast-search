"""
Build Embedding Input Pipeline

CLEANED JSON -> EMBEDDING_INPUT JSON

Read Layer 2 from podcast-search/data/cleaned/,
transform and output to podcast-search/data/embedding_input/
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.embedding.input_builder import EmbeddingInputBuilder, EmbeddingInputConfig
from src.storage import storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class BuildEmbeddingInputPipeline:
    """
    Pipeline: CLEANED JSON -> EMBEDDING_INPUT JSON

    1. Read Layer 2 CLEANED JSON
    2. Build embedding input (title weighting + truncation)
    3. Output Layer 3 JSON

    Usage:
        pipeline = BuildEmbeddingInputPipeline(
            cleaned_dir=Path("data/cleaned/episodes"),
            output_dir=Path("data/embedding_input/episodes"),
        )
        pipeline.run()
    """

    def __init__(
        self,
        cleaned_dir: Path,
        output_dir: Path,
        config: Optional[EmbeddingInputConfig] = None,
        shows_lookup: Optional[dict] = None,
    ):
        """
        Args:
            cleaned_dir: Directory containing Layer 2 CLEANED JSON files
            output_dir: Output directory for Layer 3 EMBEDDING_INPUT JSON files
            config: Optional embedding input configuration
            shows_lookup: Optional dict mapping show_id -> show_title
        """
        self.cleaned_dir = cleaned_dir
        self.output_dir = output_dir
        self.builder = EmbeddingInputBuilder(config)
        self.shows_lookup = shows_lookup or {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run the embedding input build pipeline.

        Returns:
            Stats dict with processing results
        """
        stats = {
            "episodes_processed": 0,
            "episodes_failed": 0,
            "truncated_count": 0,
            "total_estimated_tokens": 0,
            "shows_found": 0,
            "shows_missing": 0,
        }

        # Get list of cleaned JSON files
        cleaned_files = sorted(self.cleaned_dir.glob("*.json"))

        logger.info(
            "build_embedding_input_start",
            extra={"total_files": len(cleaned_files)},
        )

        for cleaned_path in cleaned_files:
            try:
                result = self._process_file(cleaned_path)
                stats["episodes_processed"] += 1
                stats["total_estimated_tokens"] += result["estimated_tokens"]
                if result["was_truncated"]:
                    stats["truncated_count"] += 1
                if result["has_show_title"]:
                    stats["shows_found"] += 1
                else:
                    stats["shows_missing"] += 1

                # Log progress every 1000 files
                if stats["episodes_processed"] % 1000 == 0:
                    logger.info(
                        "build_progress",
                        extra={"processed": stats["episodes_processed"]},
                    )

            except Exception as e:
                stats["episodes_failed"] += 1
                logger.warning(
                    "build_embedding_input_failed",
                    extra={"file": str(cleaned_path), "error": str(e)},
                )

        # Calculate averages
        if stats["episodes_processed"] > 0:
            stats["avg_estimated_tokens"] = (
                stats["total_estimated_tokens"] / stats["episodes_processed"]
            )
            stats["truncation_rate"] = (
                stats["truncated_count"] / stats["episodes_processed"]
            )
        else:
            stats["avg_estimated_tokens"] = 0
            stats["truncation_rate"] = 0

        logger.info(
            "build_embedding_input_complete",
            extra=stats,
        )

        return stats

    def _process_file(self, cleaned_path: Path) -> dict:
        """Process a single cleaned JSON file."""
        # Load cleaned JSON
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_episode = json.load(f)

        # Look up show_title
        show_id = cleaned_episode.get("show_id")
        show_title = self.shows_lookup.get(show_id) if show_id else None

        # Build embedding input with show_title
        embedding_input = self.builder.build(cleaned_episode, show_title=show_title)

        # Convert to dict
        result = self.builder.to_layer3_dict(embedding_input)

        # Add metadata
        result["build_meta"] = {
            "pipeline_version": "embed-input-v1.0",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_file": cleaned_path.name,
            "source_cleaning_version": cleaned_episode.get("cleaning_meta", {}).get(
                "pipeline_version", "unknown"
            ),
        }

        # Save to output
        self._save_output(result, cleaned_path.stem)

        return {
            "estimated_tokens": embedding_input.estimated_tokens,
            "was_truncated": embedding_input.was_truncated,
            "has_show_title": show_title is not None,
        }

    def _save_output(self, result: dict, filename_stem: str) -> None:
        """Save embedding input to JSON file."""
        output_path = self.output_dir / f"{filename_stem}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def load_shows_lookup() -> dict:
    """
    Load all shows from storage and return a dict mapping show_id -> title.
    """
    shows_lookup = {}
    show_ids = storage.list_show_ids()
    logger.info("loading_shows", extra={"count": len(show_ids)})

    for show_id in show_ids:
        try:
            show = storage.load_show(show_id)
            if show.get("title"):
                shows_lookup[show_id] = show["title"]
        except Exception as e:
            logger.warning("load_show_failed", extra={"show_id": show_id, "error": str(e)})

    logger.info("shows_loaded", extra={"count": len(shows_lookup)})
    return shows_lookup


def run(
    cleaned_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    title_weight: int = 3,
    max_tokens: int = 256,
) -> dict:
    """
    Run the build embedding input pipeline.

    Args:
        cleaned_dir: Path to Layer 2 CLEANED directory
        output_dir: Path to output directory for Layer 3
        title_weight: Number of times to repeat title (default: 3)
        max_tokens: Maximum estimated tokens (default: 256)

    Returns:
        Stats dict
    """
    setup_logging()

    # Default paths
    if cleaned_dir is None:
        cleaned_dir = str(
            Path(__file__).parent.parent.parent / "data" / "cleaned" / "episodes"
        )

    if output_dir is None:
        output_dir = str(
            Path(__file__).parent.parent.parent / "data" / "embedding_input" / "episodes"
        )

    # Load shows lookup for show_title
    shows_lookup = load_shows_lookup()

    config = EmbeddingInputConfig(
        title_weight=title_weight,
        max_tokens=max_tokens,
    )

    pipeline = BuildEmbeddingInputPipeline(
        cleaned_dir=Path(cleaned_dir),
        output_dir=Path(output_dir),
        config=config,
        shows_lookup=shows_lookup,
    )

    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding input from cleaned episodes")
    parser.add_argument(
        "--cleaned-dir",
        type=str,
        help="Path to Layer 2 CLEANED directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory for Layer 3",
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=3,
        help="Number of times to repeat title (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum estimated tokens (default: 256)",
    )

    args = parser.parse_args()

    stats = run(
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        title_weight=args.title_weight,
        max_tokens=args.max_tokens,
    )

    print(f"\nPipeline complete!")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Episodes failed: {stats['episodes_failed']}")
    print(f"Shows found: {stats['shows_found']}")
    print(f"Shows missing: {stats['shows_missing']}")
    print(f"Truncated: {stats['truncated_count']} ({stats['truncation_rate']:.1%})")
    print(f"Avg estimated tokens: {stats['avg_estimated_tokens']:.1f}")
