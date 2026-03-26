from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.getenv("DATA_DIR", PROJECT_ROOT / "data")
)

if not DATA_DIR.is_absolute():
    DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()
else:
    DATA_DIR = DATA_DIR.resolve()

# Static assets (versioned in repo)
MAPPINGS_DIR = PROJECT_ROOT / "mappings"

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Index management settings
INDEX_VERSION = int(os.getenv("INDEX_VERSION", "1"))
REINDEX = os.getenv("REINDEX", "false").lower() == "true"
ALLOW_DELETE_BASE_INDEX = os.getenv("ALLOW_DELETE_BASE_INDEX", "false").lower() == "true"

# Elasticsearch authentication (for remote ES)
ES_API_KEY = os.getenv("ES_API_KEY")

# Sync settings
SYNC_MODE = os.getenv("SYNC_MODE", "incremental")  # full, incremental, backfill, single
BACKFILL_FROM = os.getenv("BACKFILL_FROM")  # Timestamp for backfill mode

# v2: Language split feature flag
ENABLE_LANGUAGE_SPLIT: bool = os.getenv("ENABLE_LANGUAGE_SPLIT", "true").lower() == "true"

# v2: SQLite data source (produced by podcast-crawler v2)
SQLITE_PATH: Path = Path(os.getenv("SQLITE_PATH", str(DATA_DIR / "crawler.db")))

# v2: ES index alias names (backend always uses aliases, never raw index names)
INDEX_ZH_TW: str = os.getenv("INDEX_ZH_TW", "episodes-zh-tw")
INDEX_ZH_CN: str = os.getenv("INDEX_ZH_CN", "episodes-zh-cn")
INDEX_EN: str = os.getenv("INDEX_EN", "episodes-en")

# v2: Behavioral log paths
QUERY_LOG_PATH: Path = Path(os.getenv("QUERY_LOG_PATH", "logs/query_log.jsonl"))
CLICK_LOG_PATH: Path = Path(os.getenv("CLICK_LOG_PATH", "logs/click_log.jsonl"))
INGEST_CURSOR_PATH: Path = Path(os.getenv("INGEST_CURSOR_PATH", "data/ingest_cursor.json"))

# v2: Local embedding vector cache (written by embed_episodes, read by embed_and_ingest --from-cache)
EMBEDDING_CACHE_DIR: Path = Path(os.getenv("EMBEDDING_CACHE_DIR", "data/embeddings"))

# Embedding version: encodes both model and text assembly rules.
# Bump text-vN when the text passed to the encoder changes (e.g. new fields, new template).
# Used in cache JSON as "{model_name}/{EMBEDDING_TEXT_VERSION}".
EMBEDDING_TEXT_VERSION: str = os.getenv("EMBEDDING_TEXT_VERSION", "text-v1")
