from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.getenv("DATA_DIR", PROJECT_ROOT / "data")
)

print("DATA_DIR", DATA_DIR)

if not DATA_DIR.is_absolute():
    DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()
else:
    DATA_DIR = DATA_DIR.resolve()

# Static assets (versioned in repo)
MAPPINGS_DIR = PROJECT_ROOT / "mappings"

# Ingest data (can be local folder or cloud later)
DATA_BACKEND = os.getenv("DATA_BACKEND", "local")

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
