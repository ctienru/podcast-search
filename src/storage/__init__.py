from src.config.settings import DATA_DIR
from src.storage.base import SearchDataStorage
from src.storage.local import LocalSearchDataStorage

storage: SearchDataStorage = LocalSearchDataStorage(DATA_DIR)