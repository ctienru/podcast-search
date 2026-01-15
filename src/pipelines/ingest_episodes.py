import logging
from typing import Dict, Iterable, Optional

from elasticsearch import helpers

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
    """

    INDEX_ALIAS = "episodes"

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()

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
                "description": episode.get("description"),
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
                "show": {
                    "show_id": show_id,
                },

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
                logger.exception(
                    "build_episode_doc_failed",
                    extra={"episode_id": episode.get("episode_id")},
                )

    # ---------- orchestration ----------

    def run(self) -> None:
        episodes = list(self.load_episodes())

        logger.info(
            "episodes_loaded",
            extra={"count": len(episodes)},
        )

        if not episodes:
            logger.warning("no_episodes_to_ingest")
            return

        success, errors = helpers.bulk(
            self.es.client,
            self.build_actions(episodes),
            raise_on_error=False,
        )

        logger.info(
            "episodes_ingested",
            extra={
                "success": success,
                "errors": len(errors),
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