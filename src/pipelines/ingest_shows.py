import logging
from typing import Dict, Iterable, Optional

from elasticsearch import helpers

from src.services.es_service import ElasticsearchService
from src.storage import storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class IngestShowsPipeline:
    """
    Ingest canonical show documents into Elasticsearch `shows` index (alias).

    - Reads canonical show JSONs from storage
    - Projects them into search-optimized documents
    - Bulk indexes into ES using alias
    """

    INDEX_ALIAS = "shows"

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()

    # ---------- load ----------

    def load_shows(self) -> Iterable[Dict]:
        """
        Load all canonical shows from storage.
        """
        for show_id in storage.list_show_ids():
            data = storage.load_show(show_id)
            if not data:
                logger.warning(
                    "show_not_found_in_storage",
                    extra={"show_id": show_id},
                )
                continue
            yield data

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
                    provider: external_urls.get(provider),
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