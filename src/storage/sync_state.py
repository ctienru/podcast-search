"""Sync state writer for podcast-search.

Writes to the search_sync_state table in crawler.db after successful ES ingest.
The table is normally created by podcast-crawler's _ensure_schema(), but this
class also ensures it exists by running CREATE TABLE IF NOT EXISTS on first use.
This create-if-missing behavior allows the search side to work even if crawler
has not been run yet.
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
        # Migration: if table exists without environment column, recreate with new PK
        table_names = [r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if self._TABLE in table_names:
            cols = [row[1] for row in self._db.execute(f"PRAGMA table_info({self._TABLE})").fetchall()]
            if "environment" not in cols:
                conn = self._db.conn
                conn.execute("BEGIN")
                try:
                    conn.execute(f"ALTER TABLE {self._TABLE} RENAME TO {self._TABLE}_old")
                    conn.execute(f"""
                        CREATE TABLE {self._TABLE} (
                          entity_type       TEXT NOT NULL,
                          entity_id         TEXT NOT NULL,
                          environment       TEXT NOT NULL DEFAULT 'default',
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
                          PRIMARY KEY (entity_type, entity_id, environment)
                        )
                    """)
                    conn.execute(f"""
                        INSERT INTO {self._TABLE}
                          (entity_type, entity_id, environment, index_alias, backing_index,
                           index_version, content_hash, source_updated_at, embedding_model,
                           embedding_version, sync_status, last_synced_at, last_error)
                        SELECT entity_type, entity_id, 'default', index_alias, backing_index,
                           index_version, content_hash, source_updated_at, embedding_model,
                           embedding_version, sync_status, last_synced_at, last_error
                        FROM {self._TABLE}_old
                    """)
                    conn.execute(f"DROP TABLE {self._TABLE}_old")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._TABLE} (
              entity_type       TEXT NOT NULL,
              entity_id         TEXT NOT NULL,
              environment       TEXT NOT NULL DEFAULT 'default',
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
              PRIMARY KEY (entity_type, entity_id, environment)
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
        self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_environment
            ON {self._TABLE} (environment)
        """)

    def commit(self) -> None:
        """Commit pending writes to the database."""
        self._db.conn.commit()

    def mark_done(
        self,
        entity_type: str,
        entity_id: str,
        index_alias: str,
        content_hash: Optional[str] = None,
        source_updated_at: Optional[str] = None,
        embedding_model: Optional[str] = None,
        environment: str = "default",
    ) -> None:
        """Record a successful ES sync."""
        now = _utc_now_iso()
        self._db.execute(
            f"""
            INSERT INTO {self._TABLE}
              (entity_type, entity_id, environment, index_alias, content_hash,
               source_updated_at, embedding_model, sync_status, last_synced_at, last_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'synced', ?, NULL)
            ON CONFLICT(entity_type, entity_id, environment) DO UPDATE SET
              index_alias       = excluded.index_alias,
              content_hash      = excluded.content_hash,
              source_updated_at = excluded.source_updated_at,
              embedding_model   = excluded.embedding_model,
              sync_status       = 'synced',
              last_synced_at    = excluded.last_synced_at,
              last_error        = NULL
            """,
            [entity_type, entity_id, environment, index_alias, content_hash, source_updated_at, embedding_model, now],
        )
