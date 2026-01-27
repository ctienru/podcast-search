from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, Optional, List


class SearchDataStorage(ABC):
    """
    Abstract storage interface for reading crawler output.
    Read-only by design.

    Note: Episodes are no longer stored in normalized format.
    podcast-search reads raw RSS XML and cleans/parses episodes itself.
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

    # ---------- manifests ----------

    @abstractmethod
    def list_manifests(self) -> List[str]:
        """
        List all manifest timestamps (sorted).
        """
        pass

    @abstractmethod
    def load_manifest(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Load a manifest by timestamp.
        """
        pass

    # ---------- sync cursor ----------

    @abstractmethod
    def load_sync_cursor(self) -> Optional[Dict[str, Any]]:
        """
        Load the sync cursor (last synced manifest timestamp).
        """
        pass

    @abstractmethod
    def save_sync_cursor(self, data: Dict[str, Any]) -> None:
        """
        Save the sync cursor.
        """
        pass