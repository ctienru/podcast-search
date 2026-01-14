import logging
from typing import Iterable, Dict

from elasticsearch import helpers

from src.services.es_service import ElasticsearchService
from src.storage import storage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

class IngestEpisodesPipeline:
    """
    Ingest canonical episode documents into Elasticsearch.

    Strategy:
    - Use bulk + op_type=create
    - Existing documents are skipped automatically (409 conflict)
    - No per-document exists check (fast & scalable)
    """

    INDEX_NAME = "episodes"

    def __init__(
        self,
        es_service: ElasticsearchService | None = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()

    # ---------- data loading ----------

    def iter_episode_docs(self) -> Iterable[Dict]:
        episode_ids = list(storage.list_episode_ids())

        if not episode_ids:
            logger.warning("no_episodes_to_ingest")
            return

        logger.info(
            "episodes_loaded",
            extra={"count": len(episode_ids)},
        )

        for episode_id in episode_ids:
            doc = storage.load_episode(episode_id)
            if not doc:
                logger.warning(
                    "episode_load_failed",
                    extra={"episode_id": episode_id},
                )
                continue

            yield doc

    # ---------- bulk actions ----------

    def build_actions(self) -> Iterable[Dict]:
        for episode in self.iter_episode_docs():
            yield {
                "_index": self.INDEX_NAME,
                "_id": episode["episode_id"],
                "_op_type": "create",   # ⭐ 핵심
                "_source": episode,
            }

    # ---------- orchestration ----------

    def run(self) -> None:
        logger.info(
            "episode_ingest_start",
            extra={"index": self.INDEX_NAME},
        )

        success, errors = helpers.bulk(
            self.es.client,
            self.build_actions(),
            chunk_size=500,
            raise_on_error=False,
            stats_only=False,
        )

        created = success
        skipped = 0
        failed = 0

        for err in errors:
            action, info = next(iter(err.items()))

            status = info.get("status")
            if status == 409:
                skipped += 1
            else:
                failed += 1

        logger.info(
            "episode_ingest_finished",
            extra={
                "ingest_created": created,
                "ingest_skipped": skipped,
                "ingest_failed": failed,
            },
        )

        if failed:
            logger.warning(
                "episode_ingest_failed_samples",
                extra={"sample": errors[:3]},
            )


def run() -> None:
    setup_logging()

    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)

    IngestEpisodesPipeline().run()


if __name__ == "__main__":
    run()