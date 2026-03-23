from __future__ import annotations

import json as _json
from collections.abc import Iterator
from pathlib import Path

from sqlite_utils import Database

from src.storage.base import StorageBase
from src.types import Language, Show


def _parse_json(value: str | None, fallback):
    """Safely parse a SQLite JSON column that may be NULL or malformed.

    Args:
        value:    Raw string value from SQLite (may be None).
        fallback: Returned when value is None, empty, or not valid JSON.
    """
    if not value:
        return fallback
    try:
        return _json.loads(value)
    except (ValueError, TypeError):
        return fallback


class SQLiteStorage(StorageBase):
    """Read shows from a SQLite database produced by podcast-crawler v2.

    Pattern: Repository — hides SQLite query details behind StorageBase.
    Callers depend on the interface, not the schema.

    Uses sqlite-utils (consistent with podcast-crawler) so all queries are
    parameterized automatically — no raw SQL string building.

    Args:
        db_path: Path to the crawler.db SQLite file.
    """

    _WHERE_BASE = "target_index IS NOT NULL AND language_detected IS NOT NULL"

    def __init__(self, db_path: Path) -> None:
        self._db = Database(db_path)

    def get_shows(self, language: Language | None = None) -> Iterator[Show]:
        """Yield shows from the 'shows' table, optionally filtered by language.

        Only rows where both target_index and language_detected are non-null
        are returned.
        """
        return self.get_shows_updated_since(since="", language=language)

    def get_shows_updated_since(
        self,
        since: str,
        language: Language | None = None,
    ) -> Iterator[Show]:
        """Yield shows updated after the given timestamp.

        Passing since="" returns all eligible shows (same as get_shows).

        Args:
            since:    ISO 8601 UTC timestamp. Yields shows where updated_at > since.
                      Empty string means no filter.
            language: Optional language filter.

        Yields:
            Show dataclass instances.
        """
        where = self._WHERE_BASE
        params: dict = {}

        if since:
            where += " AND updated_at > :since"
            params["since"] = since

        if language is not None:
            where += " AND language_detected = :language"
            params["language"] = language

        for row in self._db["shows"].rows_where(where, params):
            image = _parse_json(row.get("image"), {})
            external_urls = _parse_json(row.get("external_urls"), {})
            raw_categories = _parse_json(row.get("categories"), [])

            yield Show(
                show_id=row["show_id"],
                title=row["title"],
                author=row["author"],
                language_detected=row["language_detected"],
                language_confidence=row["language_confidence"],
                language_uncertain=bool(row["language_uncertain"]),
                target_index=row["target_index"],
                rss_feed_url=row["rss_feed_url"],
                updated_at=row["updated_at"],
                provider=row.get("provider") or "",
                external_id=row.get("external_id") or "",
                description=row.get("description"),
                image_url=image.get("url") if isinstance(image, dict) else None,
                external_urls=external_urls if isinstance(external_urls, dict) else {},
                episode_count=row.get("episode_count"),
                last_episode_at=row.get("last_episode_at"),
                categories=tuple(raw_categories) if isinstance(raw_categories, list) else (),
            )
