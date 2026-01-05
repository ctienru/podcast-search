import json
import logging

from elasticsearch import Elasticsearch

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def load_mapping(path):
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def recreate_index(es: Elasticsearch, index_name: str, mapping_path):
    logger.info("Recreating index: %s", index_name)

    if es.indices.exists(index=index_name):
        logger.info("Deleting existing index: %s", index_name)
        es.indices.delete(index=index_name)

    mapping = load_mapping(mapping_path)
    es.indices.create(index=index_name, body=mapping)

    logger.info("Index created: %s", index_name)


def main() -> None:
    es = Elasticsearch(config.es.url)

    try:
        info = es.info()
        logger.info(
            "Connected to Elasticsearch cluster=%s version=%s",
            info["cluster_name"],
            info["version"]["number"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Elasticsearch: {config.es.url}"
        ) from e

    recreate_index(es, "shows", config.paths.shows_mapping)
    recreate_index(es, "episodes", config.paths.episodes_mapping)

    logger.info("All indices created successfully")


if __name__ == "__main__":
    main()