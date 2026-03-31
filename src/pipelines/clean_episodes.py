"""
Clean Episodes Pipeline

RAW XML -> CLEANED JSON

Read RSS XML from podcast-crawler/data/raw/rss/,
clean and output to podcast-search/data/cleaned/episodes/
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from src.cleaning.rss_parser import RSSParser, RawEpisode
from src.cleaning.text_cleaner import PodcastTextCleaner
from src.config.settings import ENABLE_LANGUAGE_SPLIT
from src.storage.base import StorageBase
from src.storage.factory import create_storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _process_show_worker(args: tuple) -> dict:
    """Module-level worker for ProcessPoolExecutor — must be top-level to be picklable.

    Each subprocess creates its own RSSParser + PodcastTextCleaner so there is no
    shared mutable state between workers.
    """
    xml_path, target_index_by_show, output_dir = args
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = RSSParser()
    cleaner = PodcastTextCleaner()

    show_stats = {
        "episodes_processed": 0,
        "episodes_failed": 0,
        "total_paragraphs": 0,
        "removed_paragraphs": 0,
        "removal_breakdown": {},
    }

    try:
        show, episodes = parser.parse_file(xml_path)
        show_id = show.show_id
        show_language = show.language
        target_index = target_index_by_show.get(show_id)

        episode_dicts = [
            {"description": ep.description, "content_encoded": ep.content_encoded}
            for ep in episodes
        ]
        cleaner.build_frequency_table(show_id, episode_dicts)

        for episode in episodes:
            try:
                cleaned = cleaner.clean_episode(
                    episode_id=episode.episode_id,
                    show_id=episode.show_id,
                    title=episode.title,
                    description=episode.description or "",
                    content_encoded=episode.content_encoded,
                )
                result = cleaner.to_layer2_dict(cleaned)
                result["cleaning_meta"]["cleaned_at"] = datetime.now(timezone.utc).isoformat()
                if target_index is not None:
                    result["target_index"] = target_index
                result["original_meta"] = {
                    "guid": episode.guid,
                    "pub_date": episode.pub_date,
                    "duration": episode.duration,
                    "audio_url": episode.audio_url,
                    "language": show_language,
                    "image_url": episode.image_url,
                    "itunes_summary": episode.itunes_summary,
                    "creator": episode.creator,
                    "episode_type": episode.episode_type,
                    "chapters": episode.chapters,
                }

                safe_filename = result["episode_id"].replace(":", "_") + ".json"
                with open(output_dir / safe_filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                show_stats["episodes_processed"] += 1
                ep_stats = result["cleaned"]["stats"]
                show_stats["total_paragraphs"] += ep_stats["total_paragraphs"]
                show_stats["removed_paragraphs"] += ep_stats["removed_paragraphs"]
                for reason, count in ep_stats["removal_breakdown"].items():
                    show_stats["removal_breakdown"][reason] = (
                        show_stats["removal_breakdown"].get(reason, 0) + count
                    )

            except Exception as e:
                show_stats["episodes_failed"] += 1
                logging.getLogger(__name__).warning(
                    "episode_cleaning_failed",
                    extra={"episode_id": episode.episode_id, "error": str(e)},
                )

    except Exception as e:
        logging.getLogger(__name__).exception(
            "show_processing_failed",
            extra={"xml_path": str(xml_path), "error": str(e)},
        )

    return show_stats


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
        storage: Optional[StorageBase] = None,
        enable_language_split: Optional[bool] = None,
    ):
        """
        Args:
            raw_rss_dir: Directory containing RSS XML files
            output_dir: Output directory for cleaned JSON files
            show_ids: Optional list of show IDs to process (for testing)
            storage: Storage backend for show metadata (default: create_storage())
            enable_language_split: Override ENABLE_LANGUAGE_SPLIT env var (for testing)
        """
        self.raw_rss_dir = raw_rss_dir
        self.output_dir = output_dir
        self.show_ids = show_ids
        self.storage = storage or create_storage()
        self.enable_language_split = (
            enable_language_split if enable_language_split is not None
            else ENABLE_LANGUAGE_SPLIT
        )

        self.parser = RSSParser()
        self.cleaner = PodcastTextCleaner()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run the cleaning pipeline.

        Returns:
            Stats dict with processing results (includes 'elapsed_sec' key).
        """
        t0 = time.perf_counter()
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

        # v2: pre-load show→target_index mapping from SQLite
        target_index_by_show: Dict[str, str] = {}
        if self.enable_language_split:
            for show in self.storage.get_shows():
                target_index_by_show[show.show_id] = show.target_index
            logger.info(
                "target_index_map_loaded",
                extra={"show_count": len(target_index_by_show)},
            )

        max_workers = min(8, os.cpu_count() or 1)
        logger.info(
            "clean_pipeline_start",
            extra={"total_shows": len(xml_files), "max_workers": max_workers},
        )

        worker_args = [
            (str(xml_path), target_index_by_show, str(self.output_dir))
            for xml_path in xml_files
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_show_worker, args): args[0] for args in worker_args}
            with tqdm(total=len(futures), desc="Cleaning shows", unit="show") as pbar:
                for future in as_completed(futures):
                    try:
                        show_stats = future.result()
                        stats["shows_processed"] += 1
                        stats["episodes_processed"] += show_stats["episodes_processed"]
                        stats["episodes_failed"] += show_stats["episodes_failed"]
                        stats["total_paragraphs"] += show_stats["total_paragraphs"]
                        stats["removed_paragraphs"] += show_stats["removed_paragraphs"]
                        for reason, count in show_stats["removal_breakdown"].items():
                            stats["removal_breakdown"][reason] = (
                                stats["removal_breakdown"].get(reason, 0) + count
                            )
                        pbar.set_postfix(episodes=stats["episodes_processed"], failed=stats["episodes_failed"])
                    except Exception as e:
                        logger.exception(
                            "show_processing_failed",
                            extra={"xml_path": futures[future], "error": str(e)},
                        )
                    finally:
                        pbar.update(1)

        stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        logger.info(
            "clean_pipeline_complete",
            extra=stats,
        )

        return stats

    def _process_show(
        self,
        xml_path: Path,
        target_index_by_show: Dict[str, str],
    ) -> dict:
        """Process a single show's RSS XML file.

        Args:
            xml_path: Path to the RSS XML file.
            target_index_by_show: Mapping of show_id → target_index from SQLite.
                Empty dict when ENABLE_LANGUAGE_SPLIT=False.
        """
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
        target_index = target_index_by_show.get(show_id)

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
                cleaned = self._clean_episode(
                    episode, language=show_language, target_index=target_index
                )
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
        self,
        episode: RawEpisode,
        language: Optional[str] = None,
        target_index: Optional[str] = None,
    ) -> dict:
        """Clean a single episode and return Layer 2 dict.

        Args:
            episode: Raw episode parsed from RSS XML.
            language: Language code from RSS <language> tag.
            target_index: v2 routing key from SQLite (e.g. "podcast-episodes-zh-tw").
                None when ENABLE_LANGUAGE_SPLIT=False.
        """
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

        # v2: propagate target_index so embed_and_ingest can route to the right index
        if target_index is not None:
            result["target_index"] = target_index

        # Add original metadata for reference
        result["original_meta"] = {
            "guid": episode.guid,
            "pub_date": episode.pub_date,
            "duration": episode.duration,
            "audio_url": episode.audio_url,
            "language": language,  # From RSS <language> tag
            "image_url": episode.image_url,  # Episode-specific image
            "itunes_summary": episode.itunes_summary,
            "creator": episode.creator,
            "episode_type": episode.episode_type,
            "chapters": episode.chapters,  # PSC chapters
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
    print(f"Shows processed:    {stats['shows_processed']}")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Episodes failed:    {stats['episodes_failed']}")
    print(f"Total paragraphs:   {stats['total_paragraphs']}")
    print(f"Removed paragraphs: {stats['removed_paragraphs']}")
    print(f"Removal breakdown:  {stats['removal_breakdown']}")
    print(f"Elapsed:            {stats['elapsed_sec']}s")
