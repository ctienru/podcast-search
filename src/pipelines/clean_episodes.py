"""
Clean Episodes Pipeline

RAW XML -> CLEANED JSON

Read RSS XML from podcast-crawler/data/raw/rss/,
clean and output to podcast-search/data/cleaned/episodes/
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.cleaning.rss_parser import RSSParser, RawEpisode
from src.cleaning.text_cleaner import PodcastTextCleaner
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class CleanEpisodesPipeline:
    """
    Pipeline: RAW XML -> CLEANED JSON

    1. Parse RSS XML (Layer 1)
    2. Build show-level paragraph frequency table
    3. Clean each episode
    4. Output Layer 2 JSON

    Usage:
        pipeline = CleanEpisodesPipeline(
            raw_rss_dir=Path("../podcast-crawler/data/raw/rss"),
            output_dir=Path("data/cleaned/episodes"),
        )
        pipeline.run()
    """

    def __init__(
        self,
        raw_rss_dir: Path,
        output_dir: Path,
        show_ids: Optional[list[str]] = None,
    ):
        """
        Args:
            raw_rss_dir: Directory containing RSS XML files
            output_dir: Output directory for cleaned JSON files
            show_ids: Optional list of show IDs to process (for testing)
        """
        self.raw_rss_dir = raw_rss_dir
        self.output_dir = output_dir
        self.show_ids = show_ids

        self.parser = RSSParser()
        self.cleaner = PodcastTextCleaner()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run the cleaning pipeline.

        Returns:
            Stats dict with processing results
        """
        stats = {
            "shows_processed": 0,
            "episodes_processed": 0,
            "episodes_failed": 0,
            "total_paragraphs": 0,
            "removed_paragraphs": 0,
            "removal_breakdown": {},
        }

        # Get list of XML files to process
        if self.show_ids:
            xml_files = [
                self.raw_rss_dir / f"{show_id}.xml"
                for show_id in self.show_ids
                if (self.raw_rss_dir / f"{show_id}.xml").exists()
            ]
        else:
            xml_files = sorted(self.raw_rss_dir.glob("*.xml"))

        logger.info(
            "clean_pipeline_start",
            extra={"total_shows": len(xml_files)},
        )

        for xml_path in xml_files:
            try:
                show_stats = self._process_show(xml_path)
                stats["shows_processed"] += 1
                stats["episodes_processed"] += show_stats["episodes_processed"]
                stats["episodes_failed"] += show_stats["episodes_failed"]
                stats["total_paragraphs"] += show_stats["total_paragraphs"]
                stats["removed_paragraphs"] += show_stats["removed_paragraphs"]

                # Merge removal breakdown
                for reason, count in show_stats["removal_breakdown"].items():
                    stats["removal_breakdown"][reason] = (
                        stats["removal_breakdown"].get(reason, 0) + count
                    )

            except Exception as e:
                logger.exception(
                    "show_processing_failed",
                    extra={"xml_path": str(xml_path), "error": str(e)},
                )

        logger.info(
            "clean_pipeline_complete",
            extra=stats,
        )

        return stats

    def _process_show(self, xml_path: Path) -> dict:
        """Process a single show's RSS XML file."""
        show_stats = {
            "episodes_processed": 0,
            "episodes_failed": 0,
            "total_paragraphs": 0,
            "removed_paragraphs": 0,
            "removal_breakdown": {},
        }

        # Parse RSS
        show, episodes = self.parser.parse_file(xml_path)
        show_id = show.show_id
        show_language = show.language  # From RSS <language> tag

        logger.info(
            "processing_show",
            extra={"show_id": show_id, "episode_count": len(episodes)},
        )

        # Build frequency table for this show (for detecting repeated boilerplate)
        episode_dicts = [
            {
                "description": ep.description,
                "content_encoded": ep.content_encoded,
            }
            for ep in episodes
        ]
        self.cleaner.build_frequency_table(show_id, episode_dicts)

        # Process each episode
        for episode in episodes:
            try:
                cleaned = self._clean_episode(episode, language=show_language)
                self._save_cleaned(cleaned)

                show_stats["episodes_processed"] += 1
                show_stats["total_paragraphs"] += cleaned["cleaned"]["stats"][
                    "total_paragraphs"
                ]
                show_stats["removed_paragraphs"] += cleaned["cleaned"]["stats"][
                    "removed_paragraphs"
                ]

                # Merge removal breakdown
                for reason, count in cleaned["cleaned"]["stats"][
                    "removal_breakdown"
                ].items():
                    show_stats["removal_breakdown"][reason] = (
                        show_stats["removal_breakdown"].get(reason, 0) + count
                    )

            except Exception as e:
                show_stats["episodes_failed"] += 1
                logger.warning(
                    "episode_cleaning_failed",
                    extra={
                        "episode_id": episode.episode_id,
                        "error": str(e),
                    },
                )

        return show_stats

    def _clean_episode(
        self, episode: RawEpisode, language: Optional[str] = None
    ) -> dict:
        """Clean a single episode and return Layer 2 dict."""
        cleaned = self.cleaner.clean_episode(
            episode_id=episode.episode_id,
            show_id=episode.show_id,
            title=episode.title,
            description=episode.description or "",
            content_encoded=episode.content_encoded,
        )

        # Convert to dict and add timestamp
        result = self.cleaner.to_layer2_dict(cleaned)
        result["cleaning_meta"]["cleaned_at"] = datetime.now(timezone.utc).isoformat()

        # Add original metadata for reference
        result["original_meta"] = {
            "guid": episode.guid,
            "pub_date": episode.pub_date,
            "duration": episode.duration,
            "audio_url": episode.audio_url,
            "language": language,  # From RSS <language> tag
            "itunes_summary": episode.itunes_summary,
            "creator": episode.creator,
            "episode_type": episode.episode_type,
        }

        return result

    def _save_cleaned(self, cleaned: dict) -> None:
        """Save cleaned episode to JSON file."""
        episode_id = cleaned["episode_id"]

        # Sanitize filename (replace : with _)
        safe_filename = episode_id.replace(":", "_") + ".json"
        output_path = self.output_dir / safe_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)


def run(
    raw_rss_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    show_ids: Optional[list[str]] = None,
) -> dict:
    """
    Run the clean episodes pipeline.

    Args:
        raw_rss_dir: Path to RSS XML directory (default: ../podcast-crawler/data/raw/rss)
        output_dir: Path to output directory (default: data/cleaned/episodes)
        show_ids: Optional list of show IDs to process

    Returns:
        Stats dict
    """
    setup_logging()

    # Default paths
    if raw_rss_dir is None:
        # Assume we're running from podcast-search/
        raw_rss_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "podcast-crawler"
            / "data"
            / "raw"
            / "rss"
        )

    if output_dir is None:
        output_dir = str(
            Path(__file__).parent.parent.parent / "data" / "cleaned" / "episodes"
        )

    pipeline = CleanEpisodesPipeline(
        raw_rss_dir=Path(raw_rss_dir),
        output_dir=Path(output_dir),
        show_ids=show_ids,
    )

    return pipeline.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean podcast episodes")
    parser.add_argument(
        "--raw-rss-dir",
        type=str,
        help="Path to RSS XML directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--show-ids",
        type=str,
        nargs="+",
        help="Specific show IDs to process",
    )

    args = parser.parse_args()

    stats = run(
        raw_rss_dir=args.raw_rss_dir,
        output_dir=args.output_dir,
        show_ids=args.show_ids,
    )

    print(f"\nPipeline complete!")
    print(f"Shows processed: {stats['shows_processed']}")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Episodes failed: {stats['episodes_failed']}")
    print(f"Total paragraphs: {stats['total_paragraphs']}")
    print(f"Removed paragraphs: {stats['removed_paragraphs']}")
    print(f"Removal breakdown: {stats['removal_breakdown']}")
