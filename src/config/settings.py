from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Static assets (versioned in repo)
MAPPINGS_DIR = PROJECT_ROOT / "mappings"

# Ingest data (can be local folder or cloud later)
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
DATA_BACKEND = os.getenv("DATA_BACKEND", "local")

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")