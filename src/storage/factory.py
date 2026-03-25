from src.config import settings
from src.storage.base import StorageBase
from src.storage.local import LocalStorage
from src.storage.sqlite import SQLiteStorage


def create_storage() -> StorageBase:
    """Return the appropriate storage backend based on the feature flag.

    Pattern: Factory — pipeline code imports this function, not concrete classes.
    Switching between v1 and v2 storage requires only changing the env var.

    Returns:
        SQLiteStorage when ENABLE_LANGUAGE_SPLIT=true,
        LocalStorage otherwise.
    """
    if settings.ENABLE_LANGUAGE_SPLIT:
        return SQLiteStorage(db_path=settings.SQLITE_PATH)
    return LocalStorage(data_dir=settings.DATA_DIR)
