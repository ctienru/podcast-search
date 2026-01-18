import logging

from elastic_transport import HeadApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, ConnectionError

from src.config.settings import ES_HOST

logger = logging.getLogger(__name__)


class ElasticsearchService:
    def __init__(self, host: str = ES_HOST):
        self.client = Elasticsearch(hosts=[host])

    def index_exists(self, name: str) -> bool:
        return self.client.indices.exists(index=name)

    def alias_exists(self, name: str) -> bool:
        return self.client.indices.exists_alias(name=name)

    def create_index(self, index: str, body: dict) -> bool:
        """
        Create an index with the given mapping.

        Returns True if created successfully, False if already exists.
        Raises other exceptions.
        """
        try:
            self.client.indices.create(index=index, body=body)
            logger.info("index_created", extra={"index": index})
            return True
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.warning("index_already_exists", extra={"index": index})
                return False
            logger.error(
                "index_create_failed",
                extra={"index": index, "error": str(e)},
            )
            raise
        except ConnectionError as e:
            logger.error(
                "elasticsearch_connection_failed",
                extra={"error": str(e)},
            )
            raise

    def reindex(self, source: str, dest: str) -> bool:
        """
        Reindex from source to destination.

        Returns True if successful, False otherwise.
        """
        try:
            result = self.client.reindex(
                body={
                    "source": {"index": source},
                    "dest": {"index": dest},
                },
                wait_for_completion=True,
            )
            logger.info(
                "reindex_completed",
                extra={"source": source, "dest": dest, "total": result.get("total", 0)},
            )
            return True
        except RequestError as e:
            logger.error(
                "reindex_failed",
                extra={"source": source, "dest": dest, "error": str(e)},
            )
            raise
        except ConnectionError as e:
            logger.error(
                "elasticsearch_connection_failed",
                extra={"error": str(e)},
            )
            raise

    def update_aliases(self, actions: list[dict]) -> bool:
        """
        Update index aliases.

        Returns True if successful, False otherwise.
        """
        try:
            self.client.indices.update_aliases(
                body={"actions": actions}
            )
            logger.info("aliases_updated", extra={"actions_count": len(actions)})
            return True
        except RequestError as e:
            logger.error(
                "alias_update_failed",
                extra={"error": str(e)},
            )
            raise
        except ConnectionError as e:
            logger.error(
                "elasticsearch_connection_failed",
                extra={"error": str(e)},
            )
            raise

    def delete_index(self, index: str) -> bool:
        """
        Delete an index.

        Returns True if deleted successfully, False if not found.
        """
        try:
            self.client.indices.delete(index=index)
            logger.info("index_deleted", extra={"index": index})
            return True
        except RequestError as e:
            if "index_not_found_exception" in str(e):
                logger.warning("index_not_found_for_deletion", extra={"index": index})
                return False
            logger.error(
                "index_delete_failed",
                extra={"index": index, "error": str(e)},
            )
            raise
        except ConnectionError as e:
            logger.error(
                "elasticsearch_connection_failed",
                extra={"error": str(e)},
            )
            raise

    def document_exists(self, index: str, doc_id: str) -> HeadApiResponse:
        """
        Check whether a document exists in the given index.

        Uses lightweight HEAD request.
        """
        return self.client.exists(
            index=index,
            id=doc_id,
        )