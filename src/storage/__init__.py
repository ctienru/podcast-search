import os
from pathlib import Path

from src.storage.local import LocalSearchDataStorage

_backend = os.getenv("DATA_BACKEND", "local")

if _backend == "local":
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    storage = LocalSearchDataStorage(data_dir)
else:
    raise ValueError(f"Unsupported DATA_BACKEND={_backend}")