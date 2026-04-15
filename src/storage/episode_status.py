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

    def mark_embedded_daily(
        self,
        episode_ids: list[str],
        model: str,
        version: str,
        embedded_at: str,
    ) -> int:
        """Artifact-ready commit: write full embedding metadata AND
        `embedding_status='done'`.

        Phase 2b-A V1e-A sanctioned callers:
          - `embed_and_ingest.py` CB1 per-show commit — only for shows
            with rebuild_ok AND show_bulk_ok. Cache-hit-only shows
            never pass through this writer (no fresh embedded_at; would
            destroy lineage); their status reconciliation is the
            backfill script's job.
          - `scripts/force_embed.py` — operator override writes the
            artifact-ready half of the two-commit-boundary design,
            then advertises the canonical handoff command so the
            local-synced half (owned by embed_and_ingest) can follow.

        Do NOT use this for:
          - Fallback / force_embed metadata-only paths where status
            must stay untouched — use `mark_embedding_metadata_only`.
          - Legacy standalone `embed_episodes.py` — keep using
            `mark_embedded_batch`.

        Args:
            episode_ids: Episode IDs from the rebuild_ok + bulk_ok
                         shows.
            model:       Embedding model name from
                         `rebuild_result.identity_used.model_name`.
            version:     Embedding version from
                         `rebuild_result.identity_used.embedding_version`
                         (identity's canonical form, not legacy
                         `<model>/text-v1`).
            embedded_at: UTC ISO timestamp from
                         `rebuild_result.new_last_embedded_at` — the
                         actual moment the vector was produced.

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

        Under Phase 2b-A V1e-A this primitive stays on the force_embed
        and fallback-rebuild paths where `embedding_status` must remain
        untouched. The daily pipeline CB1 commit uses
        `mark_embedded_daily` instead, which additionally sets
        `embedding_status='done'`. Do not migrate force_embed's writer
        in Step 6 — that's Step 7's decision.

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
