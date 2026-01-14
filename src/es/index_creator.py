import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class IndexCreator:
    """
    Responsible for creating Elasticsearch indices
    based on mapping definitions.
    """

    def __init__(self, es_client, mapping_loader):
        self.es = es_client
        self.mapping_loader = mapping_loader

    def create_index(self, index_name: str) -> None:
        """
        Create index if it does not exist.

        - Idempotent
        - Mapping-driven
        """
        if self.es.indices.exists(index=index_name):
            logger.info(
                "index_exists",
                extra={"index": index_name},
            )
            return

        mapping: Dict[str, Any] = self.mapping_loader.load(index_name)

        logger.info(
            "creating_index",
            extra={
                "index": index_name,
                "has_settings": "settings" in mapping,
            },
        )

        self.es.indices.create(
            index=index_name,
            body=mapping,
        )

        logger.info(
            "index_created",
            extra={"index": index_name},
        )