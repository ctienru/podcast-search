from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.storage.base import StorageBase
from src.types import Language, Show

logger = logging.getLogger(__name__)


class LocalStorage(StorageBase):
    """Read crawler output from the local filesystem.

    Implements StorageBase using the v1 directory layout:

        data/
          normalized/
            shows/
              {show_id}.json
            episodes/
              {episode_id}.json
          manifests/
            {timestamp}.json
            sync-cursor.json

    get_shows() and get_shows_updated_since() read from normalized/shows/.
    v1 show JSON files do not contain language_detected or target_index, so
    those fields default to uncertain values. Use SQLiteStorage (v2) for
    accurate language routing.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.shows_dir = data_dir / "normalized" / "shows"
        self.episodes_dir = data_dir / "normalized" / "episodes"
        self.manifests_dir = data_dir / "manifests"

    # ── StorageBase interface ────────────────────────────────────────────────

    def get_shows(self, language: Language | None = None) -> Iterator[Show]:
        """Yield Show objects from normalized/shows/.

        v1 JSON files lack language_detected and target_index; those fields
        default to uncertain placeholders. Use SQLiteStorage for v2 routing.
        """
        return self.get_shows_updated_since(since="", language=language)

    def get_shows_updated_since(
        self,
        since: str,
        language: Language | None = None,
    ) -> Iterator[Show]:
        """Yield Show objects updated after the given timestamp.

        v1 JSON files may not have updated_at; when missing the show is always
        included. Passing since="" returns all shows (same as get_shows).
        """
        if not self.shows_dir.exists():
            return

        for path in sorted(self.shows_dir.iterdir()):
            if not path.is_file() or path.suffix != ".json":
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(
                    "local_show_load_failed",
                    extra={"path": str(path), "error": str(exc)},
                )
                continue

            updated_at = data.get("updated_at", "")
            if since and updated_at and updated_at <= since:
                continue

            detected: Language = data.get("language_detected", "zh-tw")  # type: ignore[assignment]
            if language is not None and detected != language:
                continue

            yield Show(
                show_id=data.get("show_id", path.stem),
                title=data.get("title", ""),
                author=data.get("author", ""),
                language_detected=detected,
                language_confidence=data.get("language_confidence", 0.0),
                language_uncertain=bool(data.get("language_uncertain", True)),
                target_index=data.get("target_index", ""),
                rss_feed_url=data.get("rss_feed_url", ""),
                updated_at=updated_at,
            )

    # ── Legacy helpers (used by v1 pipelines until Commits 10–12) ───────────

    def list_show_ids(self) -> list[str]:
        """List all show IDs from normalized/shows/."""
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

    def list_episode_ids(self) -> Generator[str, None, None]:
        if not self.episodes_dir.exists():
            return
        for path in self.episodes_dir.iterdir():
            if path.is_file() and path.suffix == ".json":
                yield path.stem

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        path = self.episodes_dir / f"{episode_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"episode_not_found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_manifests(self) -> List[str]:
        if not self.manifests_dir.exists():
            return []
        return sorted(
            path.stem
            for path in self.manifests_dir.iterdir()
            if path.is_file() and path.suffix == ".json" and path.stem != "sync-cursor"
        )

    def load_manifest(self, timestamp: str) -> Optional[Dict[str, Any]]:
        path = self.manifests_dir / f"{timestamp}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_sync_cursor(self) -> Optional[Dict[str, Any]]:
        path = self.manifests_dir / "sync-cursor.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_sync_cursor(self, data: Dict[str, Any]) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifests_dir / "sync-cursor.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "sync_cursor_saved",
            extra={"last_synced_manifest": data.get("last_synced_manifest")},
        )
