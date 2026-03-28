"""Episode embedding status writer for podcast-search.

Updates episodes.embedding_status in crawler.db after embedding vectors are
computed by embed_episodes. This is separate from search_sync_state, which
tracks ES ingest status.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlite_utils import Database


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EpisodeStatusRepository:
    """Thin writer for updating embedding status on episode rows."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def mark_embedded_batch(
        self,
        episode_ids: list[str],
        model: str,
        version: str,
        embedded_at: str,
    ) -> int:
        """Mark a batch of episodes as embedding_status='done'.

        Args:
            episode_ids:  Episode IDs to update.
            model:        Embedding model name (e.g. 'BAAI/bge-base-zh-v1.5').
            version:      Embedding version string (e.g. 'BAAI/bge-base-zh-v1.5/text-v1').
            embedded_at:  UTC ISO timestamp of when the embedding was computed.

        Returns:
            Number of rows updated.
        """
        if not episode_ids:
            return 0
        placeholders = ",".join("?" * len(episode_ids))
        now = _utc_now_iso()
        result = self._db.execute(
            f"""UPDATE episodes SET
                  embedding_status  = 'done',
                  embedding_model   = ?,
                  embedding_version = ?,
                  last_embedded_at  = ?,
                  updated_at        = ?
                WHERE episode_id IN ({placeholders})""",
            [model, version, embedded_at, now] + list(episode_ids),
        )
        self._db.conn.commit()
        return result.rowcount
