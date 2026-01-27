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
