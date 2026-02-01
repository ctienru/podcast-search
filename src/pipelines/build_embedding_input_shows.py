"""
Build Embedding Input Pipeline for Shows

CANONICAL SHOW -> EMBEDDING_INPUT JSON

Read shows from storage and build embedding input for vector search.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.storage import storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ShowEmbeddingInputConfig:
    """Configuration for show embedding input"""

    # Title weighting
    title_weight: int = 3  # Number of times to repeat title

    # Length limits
    max_tokens: int = 256  # Maximum token count (estimated)
    chars_per_token: float = 2.5  # Estimated for mixed CJK/English text

    # Model configuration
    model_family: str = "sentence-transformers"
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    normalize_embeddings: bool = True


@dataclass
class ShowEmbeddingInput:
    """Embedding input structure for shows"""

    show_id: str
    text: str  # Final text to feed to model

    # Component details (for debugging)
    title: str
    title_weight: int
    description_used: str
    max_tokens: int
    estimated_tokens: int
    was_truncated: bool

    # Model configuration
    model_family: str
    model_name: str
    normalize_embeddings: bool


class ShowEmbeddingInputBuilder:
    """
    Build embedding input for shows.

    Text format: "{title} {title} {title} {description}"
    """

    def __init__(self, config: Optional[ShowEmbeddingInputConfig] = None):
        self.config = config or ShowEmbeddingInputConfig()

    def build(self, show: dict) -> ShowEmbeddingInput:
        """
        Build embedding input from canonical show data.

        Args:
            show: Canonical show dict from storage

        Returns:
            ShowEmbeddingInput with final text for embedding model
        """
        show_id = show["show_id"]
        title = show.get("title", "")
        description = show.get("description", "") or ""

        # Step 1: Title weighting (repeat N times)
        weighted_title = " ".join([title] * self.config.title_weight)

        # Step 2: Calculate available description length
        title_chars = len(weighted_title)
        max_chars = int(self.config.max_tokens * self.config.chars_per_token)
        available_chars = max(0, max_chars - title_chars - 1)  # -1 for space

        # Step 3: Truncate description
        was_truncated = len(description) > available_chars
        description_used = description[:available_chars] if was_truncated else description

        # Step 4: Combine final text
        if description_used:
            text = f"{weighted_title} {description_used}"
        else:
            text = weighted_title

        # Estimate token count
        estimated_tokens = int(len(text) / self.config.chars_per_token)

        return ShowEmbeddingInput(
            show_id=show_id,
            text=text,
            title=title,
            title_weight=self.config.title_weight,
            description_used=description_used,
            max_tokens=self.config.max_tokens,
            estimated_tokens=estimated_tokens,
            was_truncated=was_truncated,
            model_family=self.config.model_family,
            model_name=self.config.model_name,
            normalize_embeddings=self.config.normalize_embeddings,
        )

    def to_dict(self, embedding_input: ShowEmbeddingInput) -> dict:
        """Convert ShowEmbeddingInput to JSON structure"""
        return {
            "show_id": embedding_input.show_id,
            "embedding_input": {
                "text": embedding_input.text,
                "components": {
                    "title": embedding_input.title,
                    "title_weight": embedding_input.title_weight,
                    "description_used": embedding_input.description_used,
                    "max_tokens": embedding_input.max_tokens,
                    "estimated_tokens": embedding_input.estimated_tokens,
                    "was_truncated": embedding_input.was_truncated,
                },
                "model_config": {
                    "model_family": embedding_input.model_family,
                    "model_name": embedding_input.model_name,
                    "normalize_embeddings": embedding_input.normalize_embeddings,
                },
            },
        }


class BuildShowEmbeddingInputPipeline:
    """
    Pipeline: CANONICAL SHOW -> EMBEDDING_INPUT JSON

    1. Read canonical shows from storage
    2. Build embedding input (title weighting + description truncation)
    3. Output JSON files
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[ShowEmbeddingInputConfig] = None,
    ):
        self.output_dir = output_dir
        self.builder = ShowEmbeddingInputBuilder(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run the embedding input build pipeline.

        Returns:
            Stats dict with processing results
        """
        stats = {
            "shows_processed": 0,
            "shows_failed": 0,
            "shows_without_description": 0,
            "truncated_count": 0,
            "total_estimated_tokens": 0,
        }

        show_ids = storage.list_show_ids()
        logger.info(
            "build_show_embedding_input_start",
            extra={"total_shows": len(show_ids)},
        )

        for show_id in show_ids:
            try:
                result = self._process_show(show_id)
                if result is None:
                    stats["shows_failed"] += 1
                    continue

                stats["shows_processed"] += 1
                stats["total_estimated_tokens"] += result["estimated_tokens"]

                if result["was_truncated"]:
                    stats["truncated_count"] += 1
                if not result["has_description"]:
                    stats["shows_without_description"] += 1

                # Log progress every 500 shows
                if stats["shows_processed"] % 500 == 0:
                    logger.info(
                        "build_progress",
                        extra={"processed": stats["shows_processed"]},
                    )

            except Exception as e:
                stats["shows_failed"] += 1
                logger.warning(
                    "build_show_embedding_input_failed",
                    extra={"show_id": show_id, "error": str(e)},
                )

        # Calculate averages
        if stats["shows_processed"] > 0:
            stats["avg_estimated_tokens"] = (
                stats["total_estimated_tokens"] / stats["shows_processed"]
            )
            stats["truncation_rate"] = (
                stats["truncated_count"] / stats["shows_processed"]
            )
            stats["no_description_rate"] = (
                stats["shows_without_description"] / stats["shows_processed"]
            )
        else:
            stats["avg_estimated_tokens"] = 0
            stats["truncation_rate"] = 0
            stats["no_description_rate"] = 0

        logger.info(
            "build_show_embedding_input_complete",
            extra=stats,
        )

        return stats

    def _process_show(self, show_id: str) -> Optional[dict]:
        """Process a single show."""
        show = storage.load_show(show_id)
        if not show:
            logger.warning(
                "show_not_found",
                extra={"show_id": show_id},
            )
            return None

        # Build embedding input
        embedding_input = self.builder.build(show)
        result = self.builder.to_dict(embedding_input)

        # Add metadata
        result["build_meta"] = {
            "pipeline_version": "show-embed-input-v1.0",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save to output
        self._save_output(result, show_id)

        return {
            "estimated_tokens": embedding_input.estimated_tokens,
            "was_truncated": embedding_input.was_truncated,
            "has_description": bool(embedding_input.description_used),
        }

    def _save_output(self, result: dict, show_id: str) -> None:
        """Save embedding input to JSON file."""
        # Convert show_id to safe filename
        safe_filename = show_id.replace(":", "_").replace("/", "_")
        output_path = self.output_dir / f"{safe_filename}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def run(
    output_dir: Optional[str] = None,
    title_weight: int = 3,
    max_tokens: int = 256,
) -> dict:
    """
    Run the build show embedding input pipeline.

    Args:
        output_dir: Path to output directory
        title_weight: Number of times to repeat title (default: 3)
        max_tokens: Maximum estimated tokens (default: 256)

    Returns:
        Stats dict
    """
    setup_logging()

    # Default path
    if output_dir is None:
        output_dir = str(
            Path(__file__).parent.parent.parent / "data" / "embedding_input" / "shows"
        )

    config = ShowEmbeddingInputConfig(
        title_weight=title_weight,
        max_tokens=max_tokens,
    )

    pipeline = BuildShowEmbeddingInputPipeline(
        output_dir=Path(output_dir),
        config=config,
    )

    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding input for shows")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory",
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
        output_dir=args.output_dir,
        title_weight=args.title_weight,
        max_tokens=args.max_tokens,
    )

    print(f"\nPipeline complete!")
    print(f"Shows processed: {stats['shows_processed']}")
    print(f"Shows failed: {stats['shows_failed']}")
    print(f"Shows without description: {stats['shows_without_description']} ({stats['no_description_rate']:.1%})")
    print(f"Truncated: {stats['truncated_count']} ({stats['truncation_rate']:.1%})")
    print(f"Avg estimated tokens: {stats['avg_estimated_tokens']:.1f}")
