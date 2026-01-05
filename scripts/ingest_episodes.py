import json
import logging
from pathlib import Path
from typing import Iterable, Dict

from elasticsearch import Elasticsearch, helpers

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class EpisodeIngestor:
    INDEX_NAME = "episodes"

    def __init__(self, es: Elasticsearch):
        self.es = es

    def iter_episode_files(self) -> Iterable[Path]:
        episodes_dir = config.paths.data_dir / "episodes"
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

        return episodes_dir.glob("*.json")

    def build_actions(self):
        for path in self.iter_episode_files():
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            show_id = payload["show_id"]
            episodes = payload.get("episodes", [])

            for idx, ep in enumerate(episodes):
                doc_id = f"{show_id}_{idx}"

                yield {
                    "_index": self.INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "show_id": str(show_id),
                        "title": ep.get("title"),
                        "description": ep.get("description"),
                        "pub_date": ep.get("pub_date"),
                    },
                }

    def run(self) -> None:
        success, errors = helpers.bulk(
            self.es,
            self.build_actions(),
            chunk_size=500,
            raise_on_error=False,
        )

        logger.info(
            "Episode ingestion finished: success=%d errors=%d",
            success,
            len(errors),
        )

        if errors:
            logger.warning("Some episode documents failed to ingest")
            for err in errors[:5]:
                logger.warning(err)


def main() -> None:
    es = Elasticsearch(config.es.url)

    try:
        info = es.info()
        logger.info(
            "Connected to ES cluster=%s version=%s",
            info["cluster_name"],
            info["version"]["number"],
        )
    except Exception as e:
        raise RuntimeError("Failed to connect to Elasticsearch") from e

    ingestor = EpisodeIngestor(es)
    ingestor.run()


if __name__ == "__main__":
    main()