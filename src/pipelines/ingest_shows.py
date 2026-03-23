import logging
from typing import Dict, Iterable, Optional

from elasticsearch import helpers

from src.services.es_service import ElasticsearchService
from src.storage.base import StorageBase
from src.storage.factory import create_storage
from src.types import Show
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class IngestShowsPipeline:
    """
    Ingest canonical show documents into Elasticsearch `shows` index (alias).

    Reads shows from SQLiteStorage (v2), projects them into search-optimized
    documents, and bulk indexes into ES using the shows alias.
    """

    INDEX_ALIAS = "shows"

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
        storage: Optional[StorageBase] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        self.storage = storage or create_storage()

    # ---------- load ----------

    @staticmethod
    def _show_to_dict(show: Show) -> Dict:
        """Convert v2 Show dataclass to a dict compatible with to_es_doc.

        Fields not available in SQLite (external_urls, image, etc.) default
        to None and will be omitted from the ES document.
        """
        return {
            "show_id": show.show_id,
            "title": show.title,
            "author": show.author,
            "language": show.language_detected,
            "updated_at": show.updated_at,
        }

    def load_shows(self) -> Iterable[Dict]:
        """Load all canonical shows from SQLiteStorage.

        Reads Show dataclasses via get_shows() and converts them to dicts
        for downstream transformation.
        """
        for show in self.storage.get_shows():
            yield self._show_to_dict(show)

    # ---------- transform ----------

    def to_es_doc(self, show: Dict) -> Dict:
        """
        Project canonical show into Elasticsearch document
        according to shows mapping.
        """
        external_urls = show.get("external_urls") or {}
        image = show.get("image") or {}
        episode_stats = show.get("episode_stats") or {}

        # Use provider from show data, default to "apple_podcasts" for backward compatibility
        provider = show.get("provider", "apple_podcasts")

        # Map provider to external_urls key (crawler uses different naming)
        # crawler: {"provider": "apple", "external_urls": {"apple_podcasts": "..."}}
        external_url_key_map = {
            "apple": "apple_podcasts",
        }
        external_url_key = external_url_key_map.get(provider, provider)
        external_url = external_urls.get(external_url_key)

        return {
            "_index": self.INDEX_ALIAS,
            "_id": show["show_id"],
            "_source": {
                "show_id": show["show_id"],

                # ---- external ids ----
                "external_ids": {
                    provider: show.get("external_id"),
                },

                # ---- external urls ----
                "external_urls": {
                    provider: external_url,
                },

                # ---- content ----
                "title": show.get("title"),
                "publisher": show.get("author"),
                "description": show.get("description"),
                "subtitle": show.get("subtitle"),

                # ---- metadata ----
                "categories": show.get("categories"),
                "explicit": show.get("explicit"),
                "show_type": show.get("show_type"),
                "website_url": show.get("website_url"),

                "language": show.get("language"),

                # ---- episode stats ----
                "episode_count": episode_stats.get("episode_count"),
                "last_episode_at": episode_stats.get("last_episode_at"),

                # ---- ranking ----
                "popularity_score": None,

                # ---- media ----
                "image_url": image.get("url"),

                # ---- timestamps ----
                "created_at": show.get("created_at"),
                "updated_at": show.get("updated_at"),
            },
        }

    # ---------- bulk ----------

    def build_actions(self, shows: Iterable[Dict]):
        for show in shows:
            try:
                yield self.to_es_doc(show)
            except Exception:
                logger.exception(
                    "build_show_doc_failed",
                    extra={"show_id": show.get("show_id")},
                )

    # ---------- orchestration ----------

    def run(self) -> None:
        shows = list(self.load_shows())
        logger.info(
            "shows_loaded",
            extra={"count": len(shows)},
        )

        if not shows:
            logger.warning("no_shows_to_ingest")
            return

        success, errors = helpers.bulk(
            self.es.client,
            self.build_actions(shows),
            raise_on_error=False,
        )

        logger.info(
            "shows_ingested",
            extra={
                "success": success,
                "errors": len(errors),
            },
        )

        if errors:
            logger.warning(
                "show_ingest_errors_sample",
                extra={"sample": errors[:5]},
            )


def run() -> None:
    setup_logging()

    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    IngestShowsPipeline().run()


if __name__ == "__main__":
    run()