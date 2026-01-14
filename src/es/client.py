import os
from elasticsearch import Elasticsearch


def get_es_client() -> Elasticsearch:
    """
    Create Elasticsearch client from environment variables.
    """
    host = os.getenv("ES_HOST", "http://localhost:9200")

    return Elasticsearch(hosts=[host])