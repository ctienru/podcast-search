import os

from src.config.settings import DATA_DIR
from src.storage.local import LocalSearchDataStorage

_backend = os.getenv("DATA_BACKEND", "local")

if _backend == "local":
    storage = LocalSearchDataStorage(DATA_DIR)
else:
    raise ValueError(f"Unsupported DATA_BACKEND={_backend}")