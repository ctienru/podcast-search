"""Episode embedding status writer for podcast-search.

Updates episodes.embedding_status in crawler.db after embedding vectors are
computed by embed_episodes. This is separate from search_sync_state, which
tracks ES ingest status.
"""

from datetime import datetime, timezone

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
            model:        Embedding model name (e.g. 'paraphrase-multilingual-MiniLM-L12-v2').
            version:      Embedding version string (e.g. 'paraphrase-multilingual-MiniLM-L12-v2/text-v1').
            embedded_at:  UTC ISO timestamp of when the embedding was computed.

        Returns:
            Number of rows updated.
        """
        if not episode_ids:
            return 0
        # SQLite has a hard limit on bound parameters (~999). Chunk to stay safe.
        MAX_CHUNK = 900
        total = 0
        now = _utc_now_iso()
        for i in range(0, len(episode_ids), MAX_CHUNK):
            chunk = episode_ids[i : i + MAX_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            result = self._db.execute(
                f"""UPDATE episodes SET
                      embedding_status  = 'done',
                      embedding_model   = ?,
                      embedding_version = ?,
                      last_embedded_at  = ?,
                      updated_at        = ?
                    WHERE episode_id IN ({placeholders})""",
                [model, version, embedded_at, now] + list(chunk),
            )
            total += result.rowcount
        self._db.conn.commit()
        return total

    def mark_embedding_metadata_only(
        self,
        episode_ids: list[str],
        model: str,
        version: str,
        embedded_at: str,
    ) -> int:
        """Update embedding metadata without touching `embedding_status`.

        This is the Phase 2a per-show commit path: after a fallback rebuild
        produced a fresh cache and the corresponding ES bulk actions for
        that show all succeeded, we record which model and version just
        wrote those vectors — but we deliberately do NOT set
        `embedding_status='done'` because that field's semantics are
        decoupled from ES state until Phase 2b performs the cleanup.
        Writing 'done' in Phase 2a would spread the stale semantics
        instead of waiting for the truth-source decision.

        Use `mark_embedded_batch` for the legacy `embed_episodes` path,
        which still sets `embedding_status='done'` under the pre-Phase-2a
        contract.

        Args:
            episode_ids: Episode IDs to update.
            model:       Embedding model name.
            version:     Embedding version string. Phase 2a callers pass
                         the identity's `embedding_version` value.
            embedded_at: UTC ISO timestamp of when the embedding was
                         computed (typically the rebuild primitive's
                         `new_last_embedded_at`).

        Returns:
            Number of rows updated across all chunks.
        """
        if not episode_ids:
            return 0
        MAX_CHUNK = 900
        total = 0
        now = _utc_now_iso()
        for i in range(0, len(episode_ids), MAX_CHUNK):
            chunk = episode_ids[i : i + MAX_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            result = self._db.execute(
                f"""UPDATE episodes SET
                      embedding_model   = ?,
                      embedding_version = ?,
                      last_embedded_at  = ?,
                      updated_at        = ?
                    WHERE episode_id IN ({placeholders})""",
                [model, version, embedded_at, now] + list(chunk),
            )
            total += result.rowcount
        self._db.conn.commit()
        return total
