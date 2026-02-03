import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from src.storage.base import SearchDataStorage

logger = logging.getLogger(__name__)


class LocalSearchDataStorage(SearchDataStorage):
    """
    Read crawler output from local filesystem.

    Expected layout (from crawler):

    data/
      normalized/
        shows/
          {show_id}.json
        episodes/
          {episode_id}.json
      manifests/
        {timestamp}.json
        sync-cursor.json
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.shows_dir = data_dir / "normalized" / "shows"
        self.episodes_dir = data_dir / "normalized" / "episodes"
        self.manifests_dir = data_dir / "manifests"

    # ---------- shows ----------

    def list_show_ids(self) -> list[str]:
        """
        List all show IDs from normalized crawler output.

        Returns empty list if directory does not exist.
        """
        if not self.shows_dir.exists():
            return []

        return sorted(
            path.stem
            for path in self.shows_dir.iterdir()
            if path.is_file() and path.suffix == ".json"
        )

    def load_show(self, show_id: str) -> Dict[str, Any]:
        path = self.shows_dir / f"{show_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"show_not_found: {path}")

        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- episodes ----------

    def list_episode_ids(self) -> Generator[str, None, None]:
        """
        List all episode IDs from normalized crawler output.

        Returns generator of episode IDs. Returns empty generator if directory does not exist.
        """
        if not self.episodes_dir.exists():
            return

        for path in self.episodes_dir.iterdir():
            if path.is_file() and path.suffix == ".json":
                yield path.stem

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        Load an episode by ID.

        Raises FileNotFoundError if episode doesn't exist.
        """
        path = self.episodes_dir / f"{episode_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"episode_not_found: {path}")

        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- manifests ----------

    def list_manifests(self) -> List[str]:
        """
        List all manifest timestamps (sorted).
        Returns list of timestamps like ["2026-01-18T06-00-00Z", "2026-01-18T12-00-00Z"]
        """
        if not self.manifests_dir.exists():
            return []

        timestamps = []
        for path in self.manifests_dir.iterdir():
            if path.is_file() and path.suffix == ".json":
                filename = path.stem
                # Skip sync-cursor.json
                if filename != "sync-cursor":
                    timestamps.append(filename)

        return sorted(timestamps)

    def load_manifest(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """Load a manifest by timestamp."""
        path = self.manifests_dir / f"{timestamp}.json"
        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- sync cursor ----------

    def load_sync_cursor(self) -> Optional[Dict[str, Any]]:
        """
        Load the sync cursor (last synced manifest timestamp).
        Returns None if no cursor exists (first run).
        """
        path = self.manifests_dir / "sync-cursor.json"
        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def save_sync_cursor(self, data: Dict[str, Any]) -> None:
        """
        Save the sync cursor.

        Expected format:
        {
            "last_synced_manifest": "2026-01-18T11-00-00Z",
            "last_synced_at": "2026-01-18T11:05:00Z"
        }
        """
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifests_dir / "sync-cursor.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "sync_cursor_saved",
            extra={"last_synced_manifest": data.get("last_synced_manifest")},
        )