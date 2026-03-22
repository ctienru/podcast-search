from src.config.settings import DATA_DIR
from src.storage.base import StorageBase
from src.storage.local import LocalStorage

storage: LocalStorage = LocalStorage(DATA_DIR)
