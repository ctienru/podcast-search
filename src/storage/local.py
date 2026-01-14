import json
from pathlib import Path
from typing import Iterable, Dict, Any

from podcast_search.storage.base import SearchDataStorage


class LocalSearchDataStorage(SearchDataStorage):
    """
    Read canonical crawler output from local filesystem.

    Expected layout (from crawler):

    data/
      normalized/
        shows/
          {show_id}.json
        episodes/
          {episode_id}.json
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.shows_dir = data_dir / "normalized" / "shows"
        self.episodes_dir = data_dir / "normalized" / "episodes"

    # ---------- shows ----------

    def list_show_ids(self) -> Iterable[str]:
        if not self.shows_dir.exists():
            return []

        return (
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

    def list_episode_ids(self) -> Iterable[str]:
        if not self.episodes_dir.exists():
            return []

        return (
            path.stem
            for path in self.episodes_dir.iterdir()
            if path.is_file() and path.suffix == ".json"
        )

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        path = self.episodes_dir / f"{episode_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"episode_not_found: {path}")

        return json.loads(path.read_text(encoding="utf-8"))