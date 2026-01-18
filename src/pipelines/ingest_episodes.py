import logging
import re
from html import unescape
from typing import Dict, Iterable, Optional

from elasticsearch import helpers
from elasticsearch.helpers import streaming_bulk

from src.services.es_service import ElasticsearchService
from src.storage import storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class IngestEpisodesPipeline:
    """
    Ingest canonical episode documents into Elasticsearch `episodes` index (alias).

    - Reads canonical episode JSONs from storage
    - Projects them into search-optimized documents
    - Bulk indexes into ES using alias

    IMPORTANT: Run ingest_shows.py BEFORE ingest_episodes.py to ensure
    show data is available for embedding into episode documents.
    """

    INDEX_ALIAS = "episodes"

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        # Pre-load all shows into memory for efficient lookup during episode ingestion
        self._show_cache: Dict[str, Dict] = {}
        self._load_show_cache()

    def _load_show_cache(self) -> None:
        """
        Pre-load all show data into memory cache.
        This ensures show info is available when embedding into episodes.
        """
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

    # ---------- helpers ----------

    @staticmethod
    def _clean_html(text: Optional[str]) -> Optional[str]:
        """
        Remove HTML tags and decode HTML entities from text.
        Preserves paragraph breaks as double newlines.
        """
        if not text:
            return text

        # Replace common block-level tags with newlines
        text = re.sub(r'</(p|div|br)>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities (e.g., &amp; → &, &lt; → <)
        text = unescape(text)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines → double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces → single space
        text = text.strip()

        return text if text else None

    # ---------- load ----------

    def load_episodes(self) -> Iterable[Dict]:
        for episode_id in storage.list_episode_ids():
            data = storage.load_episode(episode_id)
            if not data:
                logger.warning(
                    "episode_not_found_in_storage",
                    extra={"episode_id": episode_id},
                )
                continue
            yield data

    # ---------- transform ----------

    def to_es_doc(self, episode: Dict) -> Dict:
        """
        Project canonical episode into Elasticsearch document
        according to episodes mapping.
        """

        audio = episode.get("audio") or {}
        external_ids = episode.get("external_ids") or {}

        show_id = episode.get("show_id")

        # Load show data from cache to embed into episode
        show_data = None
        if show_id:
            show_data = self._get_show_data(show_id)
            if not show_data:
                logger.warning(
                    "show_not_found_for_episode",
                    extra={"episode_id": episode.get("episode_id"), "show_id": show_id},
                )

        # Build show object with complete info
        show_obj = {"show_id": show_id}
        if show_data:
            show_obj["title"] = show_data.get("title")
            show_obj["publisher"] = show_data.get("author")

            # Get image URL from show's image object
            image = show_data.get("image") or {}
            show_obj["image_url"] = image.get("url")

            # Get external URLs
            external_urls = show_data.get("external_urls") or {}
            if external_urls:
                show_obj["external_urls"] = external_urls

        return {
            "_index": self.INDEX_ALIAS,
            "_id": episode["episode_id"],

            "_source": {
                "episode_id": episode["episode_id"],

                # ---- external ids ----
                # passthrough canonical external_ids
                "external_ids": external_ids,

                # ---- content ----
                "title": episode.get("title"),
                "description": self._clean_html(episode.get("description")),
                "language": episode.get("language"),

                "published_at": episode.get("published_at"),
                "duration_sec": episode.get("duration_sec"),

                # ---- audio ----
                "audio": {
                    "url": audio.get("url"),
                    "type": audio.get("type"),
                    "length_bytes": audio.get("length_bytes"),
                },

                # ---- show (STRICT OBJECT) ----
                "show": show_obj,

                # ---- media ----
                "image_url": episode.get("image", {}).get("url"),

                # ---- timestamps ----
                "created_at": episode.get("created_at"),
                "updated_at": episode.get("updated_at"),
            },
        }

    # ---------- bulk ----------

    def build_actions(self, episodes: Iterable[Dict]):
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
        # Count episode IDs for progress tracking without loading all data into memory
        episode_ids = list(storage.list_episode_ids())
        total_count = len(episode_ids)

        logger.info(
            "episodes_to_process",
            extra={"count": total_count},
        )

        if not episode_ids:
            logger.warning("no_episodes_to_ingest")
            return

        # Use streaming_bulk to process episodes without loading all into memory
        success = 0
        errors = []

        for ok, item in streaming_bulk(
            self.es.client,
            self.build_actions(self.load_episodes()),
            chunk_size=500,
            raise_on_error=False,
        ):
            if ok:
                success += 1
                # Report progress every 5000 episodes
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
    setup_logging()

    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    IngestEpisodesPipeline().run()


if __name__ == "__main__":
    run()