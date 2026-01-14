from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any


class SearchDataStorage(ABC):
    """
    Abstract storage interface for reading canonical crawler output.
    Read-only by design.
    """

    # ---------- shows ----------

    @abstractmethod
    def list_show_ids(self) -> Iterable[str]:
        """
        Return iterable of show IDs.
        """
        pass

    @abstractmethod
    def load_show(self, show_id: str) -> Dict[str, Any]:
        """
        Load a canonical show record.
        """
        pass

    # ---------- episodes ----------

    @abstractmethod
    def list_episode_ids(self) -> Iterable[str]:
        """
        Return iterable of episode IDs.
        """
        pass

    @abstractmethod
    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        Load a canonical episode record.
        """
        pass