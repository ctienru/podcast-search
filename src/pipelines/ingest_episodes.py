"""
Ingest Episodes Pipeline

Read cleaned episodes and ingest them to Elasticsearch.

Source: data/cleaned/episodes/*.json
Target: Elasticsearch episodes index
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from elasticsearch.helpers import streaming_bulk

from src.config.settings import PROJECT_ROOT
from src.services.es_service import ElasticsearchService
from src.storage import storage
from src.utils.logging import setup_logging
from src.utils.parsers import normalize_language, parse_duration, parse_pub_date

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class IngestEpisodesPipeline:
    """
    Ingest cleaned episode documents into Elasticsearch `episodes` index.

    - Reads cleaned episode JSONs from data/cleaned/episodes/
    - Projects them into search-optimized documents
    - Bulk indexes into ES using alias

    IMPORTANT: Run ingest_shows.py BEFORE ingest_episodes.py to ensure
    show data is available for embedding into episode documents.
    """

    INDEX_ALIAS = "episodes"

    def __init__(
        self,
        cleaned_dir: Optional[Path] = None,
        es_service: Optional[ElasticsearchService] = None,
    ) -> None:
        # Cleaned episodes are in podcast-search/data, not DATA_DIR (which points to crawler)
        self.cleaned_dir = cleaned_dir or PROJECT_ROOT / "data" / "cleaned" / "episodes"
        self.es = es_service or ElasticsearchService()

        # Pre-load all shows into memory for efficient lookup during episode ingestion
        self._show_cache: Dict[str, Dict] = {}
        self._load_show_cache()

    def _load_show_cache(self) -> None:
        """Pre-load all show data into memory cache."""
        show_ids = storage.list_show_ids()
        loaded = 0
        for show_id in show_ids:
            try:
                show_data = storage.load_show(show_id)
                if show_data:
                    self._show_cache[show_id] = show_data
                    loaded += 1
            except Exception as e:
                logger.warning(
                    "show_cache_load_failed",
                    extra={"show_id": show_id, "error": str(e)},
                )

        logger.info(
            "show_cache_loaded",
            extra={"count": loaded, "total_ids": len(show_ids)},
        )

    def _get_show_data(self, show_id: str) -> Optional[Dict]:
        """Get show data from cache."""
        return self._show_cache.get(show_id)

    # ---------- load ----------

    def list_cleaned_episode_files(self) -> list[Path]:
        """List all cleaned episode JSON files."""
        if not self.cleaned_dir.exists():
            return []
        return sorted(self.cleaned_dir.glob("*.json"))

    def load_cleaned_episodes(self) -> Iterable[Dict]:
        """Load cleaned episode JSONs from filesystem."""
        for json_path in self.list_cleaned_episode_files():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    yield data
            except Exception as e:
                logger.warning(
                    "cleaned_episode_load_failed",
                    extra={"path": str(json_path), "error": str(e)},
                )

    # ---------- transform ----------

    def to_es_doc(self, cleaned_episode: Dict) -> Dict:
        """
        Project cleaned episode into Elasticsearch document.

        Input format (from clean_episodes.py):
        {
            "episode_id": "...",
            "show_id": "...",
            "cleaned": {
                "normalized": {"title": "...", "description": "..."},
                "paragraphs": [...],
                "stats": {...}
            },
            "original_meta": {
                "guid": "...",
                "pub_date": "...",
                "duration": "...",
                "audio_url": "..."
            }
        }
        """
        episode_id = cleaned_episode["episode_id"]
        show_id = cleaned_episode["show_id"]
        cleaned = cleaned_episode.get("cleaned", {})
        normalized = cleaned.get("normalized", {})
        original_meta = cleaned_episode.get("original_meta", {})

        # Load show data from cache
        show_data = self._get_show_data(show_id) if show_id else None
        if show_id and not show_data:
            logger.warning(
                "show_not_found_for_episode",
                extra={"episode_id": episode_id, "show_id": show_id},
            )

        # Build show object
        show_obj = {"show_id": show_id}
        if show_data:
            show_obj["title"] = show_data.get("title")
            show_obj["publisher"] = show_data.get("author")

            image = show_data.get("image") or {}
            show_obj["image_url"] = image.get("url")

            external_urls = show_data.get("external_urls") or {}
            if external_urls:
                show_obj["external_urls"] = external_urls

        # Parse duration to seconds
        duration_sec = parse_duration(original_meta.get("duration"))

        return {
            "_index": self.INDEX_ALIAS,
            "_id": episode_id,
            "_source": {
                "episode_id": episode_id,

                # Content (from cleaned)
                "title": normalized.get("title"),
                "description": normalized.get("description"),

                # Metadata (from original_meta)
                "published_at": parse_pub_date(original_meta.get("pub_date")),
                "duration_sec": duration_sec,
                "language": normalize_language(original_meta.get("language")),

                # Audio
                "audio": {
                    "url": original_meta.get("audio_url"),
                },

                # Show (embedded from show cache)
                "show": show_obj,
            },
        }

    # ---------- bulk ----------

    def build_actions(self, episodes: Iterable[Dict]):
        """Build ES bulk actions from cleaned episodes."""
        for episode in episodes:
            try:
                yield self.to_es_doc(episode)
            except Exception:
                episode_id = episode.get("episode_id") if episode else None
                logger.exception(
                    "build_episode_doc_failed",
                    extra={"episode_id": episode_id},
                )

    # ---------- orchestration ----------

    def run(self) -> None:
        """Run the ingest pipeline."""
        episode_files = self.list_cleaned_episode_files()
        total_count = len(episode_files)

        logger.info(
            "episodes_to_process",
            extra={"count": total_count, "source_dir": str(self.cleaned_dir)},
        )

        if not episode_files:
            logger.warning("no_cleaned_episodes_to_ingest")
            return

        # Use streaming_bulk to process episodes
        success = 0
        errors = []

        for ok, item in streaming_bulk(
            self.es.client,
            self.build_actions(self.load_cleaned_episodes()),
            chunk_size=500,
            raise_on_error=False,
        ):
            if ok:
                success += 1
                if success % 5000 == 0:
                    logger.info(
                        "ingest_progress",
                        extra={"processed": success, "total": total_count},
                    )
            else:
                errors.append(item)

        logger.info(
            "episodes_ingested",
            extra={
                "success": success,
                "errors": len(errors),
                "total": total_count,
            },
        )

        if errors:
            logger.warning(
                "episode_ingest_errors_sample",
                extra={"sample": errors[:5]},
            )


def run() -> None:
    """Standalone entry point."""
    setup_logging()

    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    IngestEpisodesPipeline().run()


if __name__ == "__main__":
    run()
