import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

def get_es_client() -> Elasticsearch:
    """
    Create Elasticsearch client from environment variables.

    Supports:
    - Local development: ES_HOST only (no authentication)
    - Remote ES: ES_HOST + ES_API_KEY (API Key authentication)
    - Remote ES: ES_HOST + ES_USER + ES_PASSWORD (Basic authentication)
    """
    host = os.getenv("ES_HOST", "http://localhost:9200")
    api_key = os.getenv("ES_API_KEY")
    es_user = os.getenv("ES_USER")
    es_password = os.getenv("ES_PASSWORD")

    if api_key:
        # Remote ES with API Key authentication
        return Elasticsearch(
            hosts=[host],
            api_key=api_key,
        )
    elif es_user and es_password:
        # Remote ES with Basic authentication
        return Elasticsearch(
            hosts=[host],
            basic_auth=(es_user, es_password),
        )
    else:
        # Local development without authentication
        return Elasticsearch(hosts=[host])
