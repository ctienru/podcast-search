import json
import logging
from typing import Iterable, Dict

from elasticsearch import Elasticsearch, helpers

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class ShowIngestor:
    INDEX_NAME = "shows"

    def __init__(self, es: Elasticsearch):
        self.es = es

    def load_shows(self) -> list[Dict]:
        path = config.paths.data_dir / "shows.json"
        if not path.exists():
            raise FileNotFoundError(f"Shows data not found: {path}")

        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def build_actions(self, shows: Iterable[Dict]):
        for show in shows:
            yield {
                "_index": self.INDEX_NAME,
                "_id": str(show["id"]),
                "_source": {
                    "id": str(show["id"]),
                    "title": show.get("title"),
                    "author": show.get("author"),
                    "region": show.get("region"),
                },
            }

    def run(self) -> None:
        shows = self.load_shows()
        logger.info("Loaded %d shows", len(shows))

        success, errors = helpers.bulk(
            self.es,
            self.build_actions(shows),
            raise_on_error=False,
        )

        logger.info(
            "Show ingestion finished: success=%d errors=%d",
            success,
            len(errors),
        )

        if errors:
            logger.warning("Some show documents failed to ingest")
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

    ingestor = ShowIngestor(es)
    ingestor.run()


if __name__ == "__main__":
    main()