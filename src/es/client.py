import os
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
logger = logging.getLogger(__name__)


def get_es_client() -> Elasticsearch:
    """
    Create Elasticsearch client from environment variables.

    Environment Variables:
    - ES_HOST: Elasticsearch URL (default: http://localhost:9200)

    Authentication (choose one):

    1. API Key with ID + Secret (recommended, no special characters):
       - ES_API_KEY_ID: The "id" from API key response
       - ES_API_KEY_SECRET: The "api_key" from API key response

    2. API Key encoded (base64, may have special char issues):
       - ES_API_KEY: The "encoded" value (base64 string)

    3. Basic Auth:
       - ES_USER: Username (e.g., "elastic")
       - ES_PASSWORD: Password

    Example .env for API Key (recommended):
        ES_HOST=https://es.example.com
        ES_API_KEY_ID=VuaCfGcBCdbkQm-e5aOx
        ES_API_KEY_SECRET=ui2lp2axTNmsyakw9tvNnw

    Example .env for Basic Auth:
        ES_HOST=https://es.example.com
        ES_USER=elastic
        ES_PASSWORD=your-password
    """
    host = os.getenv("ES_HOST", "http://localhost:9200")

    # API Key with separate ID and secret (recommended)
    api_key_id = os.getenv("ES_API_KEY_ID")
    api_key_secret = os.getenv("ES_API_KEY_SECRET")

    # API Key encoded (base64)
    api_key_encoded = os.getenv("ES_API_KEY") or os.getenv("ELASTICSEARCH_API_KEY")

    # Basic auth
    es_user = os.getenv("ES_USER") or os.getenv("ELASTICSEARCH_USERNAME")
    es_password = os.getenv("ES_PASSWORD") or os.getenv("ELASTICSEARCH_PASSWORD")

    _timeout_kwargs = dict(
        request_timeout=60,
        retry_on_timeout=True,
        max_retries=3,
    )

    if api_key_id and api_key_secret:
        # Use tuple format (no base64 encoding needed)
        logger.info(f"Connecting to ES at {host} with API Key (id: {api_key_id[:8]}...)")
        return Elasticsearch(
            hosts=[host],
            api_key=(api_key_id, api_key_secret),
            **_timeout_kwargs,
        )
    elif api_key_encoded:
        logger.info(f"Connecting to ES at {host} with API Key (encoded)")
        return Elasticsearch(
            hosts=[host],
            api_key=api_key_encoded,
            **_timeout_kwargs,
        )
    elif es_user and es_password:
        logger.info(f"Connecting to ES at {host} with Basic Auth (user: {es_user})")
        return Elasticsearch(
            hosts=[host],
            basic_auth=(es_user, es_password),
            **_timeout_kwargs,
        )
    else:
        logger.info(f"Connecting to ES at {host} without authentication (local dev)")
        return Elasticsearch(hosts=[host], **_timeout_kwargs)
