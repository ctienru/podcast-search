"""Sync state writer for podcast-search.

Writes to the search_sync_state table in crawler.db after successful ES ingest.
The table is created by podcast-crawler's _ensure_schema(); this class assumes
it already exists (crawler always runs before search).

For safety, CREATE TABLE IF NOT EXISTS is also run on first use so the search
side can work even if crawler hasn't been run yet.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlite_utils import Database


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SyncStateRepository:
    """Minimal sync state writer for use in podcast-search pipelines."""

    _TABLE = "search_sync_state"

    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
              entity_type       TEXT NOT NULL,
              entity_id         TEXT NOT NULL,
              index_alias       TEXT,
              backing_index     TEXT,
              index_version     TEXT,
              content_hash      TEXT,
              source_updated_at TEXT,
              embedding_model   TEXT,
              embedding_version TEXT,
              sync_status       TEXT DEFAULT 'pending',
              last_synced_at    TEXT,
              last_error        TEXT,
              PRIMARY KEY (entity_type, entity_id)
            )
        """)
        self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_sync_status
            ON {self._TABLE} (sync_status)
        """)
        self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_entity_type
            ON {self._TABLE} (entity_type)
        """)

    def mark_done(
        self,
        entity_type: str,
        entity_id: str,
        index_alias: str,
        content_hash: Optional[str] = None,
        source_updated_at: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Record a successful ES sync."""
        now = _utc_now_iso()
        self._db.execute(
            f"""
            INSERT INTO {self._TABLE}
              (entity_type, entity_id, index_alias, content_hash, source_updated_at,
               embedding_model, sync_status, last_synced_at, last_error)
            VALUES (?, ?, ?, ?, ?, ?, 'synced', ?, NULL)
            ON CONFLICT(entity_type, entity_id) DO UPDATE SET
              index_alias       = excluded.index_alias,
              content_hash      = excluded.content_hash,
              source_updated_at = excluded.source_updated_at,
              embedding_model   = excluded.embedding_model,
              sync_status       = 'synced',
              last_synced_at    = excluded.last_synced_at,
              last_error        = NULL
            """,
            [entity_type, entity_id, index_alias, content_hash, source_updated_at, embedding_model, now],
        )
