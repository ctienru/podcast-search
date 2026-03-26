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
        sync_repo=None,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        self.storage = storage or create_storage()
        self.sync_repo = sync_repo

    # ---------- load ----------

    @staticmethod
    def _show_to_dict(show: Show) -> Dict:
        """Convert Show dataclass to a dict for to_es_doc().

        Adapts the flat Show fields to the nested structure to_es_doc() expects:
        - image_url  → {"image": {"url": ...}} so to_es_doc can call image.get("url")
        - episode_count / last_episode_at → {"episode_stats": {...}}
        - external_urls / provider / external_id passed through as-is
        """
        return {
            "show_id":      show.show_id,
            "provider":     show.provider or "apple_podcasts",
            "external_id":  show.external_id,
            "title":        show.title,
            "author":       show.author,
            "description":  show.description,
            "language":     show.language_detected,
            "updated_at":   show.updated_at,
            "image":        {"url": show.image_url} if show.image_url else {},
            "external_urls": dict(show.external_urls),
            "episode_stats": {
                "episode_count":   show.episode_count,
                "last_episode_at": show.last_episode_at,
            },
            "categories": list(show.categories),
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
                    external_url_key: external_url,
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

    def run(self, shows: Optional[list] = None) -> None:
        if shows is None:
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

        if self.sync_repo is not None:
            error_ids = {
                e.get("index", {}).get("_id")
                for e in errors
            }
            for show in shows:
                show_id = show.get("show_id") if isinstance(show, dict) else show["show_id"]
                if show_id not in error_ids:
                    self.sync_repo.mark_done(
                        entity_type="show",
                        entity_id=show_id,
                        index_alias=self.INDEX_ALIAS,
                        content_hash=show.get("content_hash") if isinstance(show, dict) else None,
                        source_updated_at=show.get("updated_at") if isinstance(show, dict) else None,
                    )


def run() -> None:
    setup_logging()

    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    IngestShowsPipeline().run()


if __name__ == "__main__":
    run()