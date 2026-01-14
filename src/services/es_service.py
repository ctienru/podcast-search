import logging
from elasticsearch import Elasticsearch

from src.config.settings import ES_HOST

logger = logging.getLogger(__name__)


class ElasticsearchService:
    def __init__(self, host: str = ES_HOST):
        self.client = Elasticsearch(hosts=[host])

    def index_exists(self, name: str) -> bool:
        return self.client.indices.exists(index=name)

    def alias_exists(self, name: str) -> bool:
        return self.client.indices.exists_alias(name=name)

    def create_index(self, index: str, body: dict) -> None:
        self.client.indices.create(index=index, body=body)

    def reindex(self, source: str, dest: str) -> None:
        self.client.reindex(
            body={
                "source": {"index": source},
                "dest": {"index": dest},
            },
            wait_for_completion=True,
        )

    def update_aliases(self, actions: list[dict]) -> None:
        self.client.indices.update_aliases(
            body={"actions": actions}
        )

    def delete_index(self, index: str) -> None:
        self.client.indices.delete(index=index)