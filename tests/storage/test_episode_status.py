"""Tests for `src/storage/episode_status.py` writers.

Step 6 introduces `mark_embedded_daily` — the Phase 2b-A V1e-A daily
pipeline artifact-ready writer. These tests cover:

- `mark_embedded_daily` writes `embedding_status='done'` plus all
  embedding metadata on the target rows.
- Field parity with `mark_embedding_metadata_only` minus the status
  column (same model / version / last_embedded_at / updated_at
  behaviour) — prevents drift between the two writers.
- `mark_embedding_metadata_only` regression: still leaves
  `embedding_status` untouched (Phase 2a contract).
- `mark_embedded_batch` regression: body not modified by Step 6
  (legacy `embed_episodes` path unchanged).
- Empty list / chunking: `mark_embedded_daily` returns 0 for [] and
  handles > MAX_CHUNK correctly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from sqlite_utils import Database

from src.storage.episode_status import EpisodeStatusRepository


def _make_db(tmp_path: Path, rows: list[dict]) -> Database:
    db_path = tmp_path / "podcast.sqlite"
    db = Database(str(db_path))
    db["episodes"].create(
        {
            "episode_id": str,
            "embedding_model": str,
            "embedding_version": str,
            "last_embedded_at": str,
            "embedding_status": str,
            "updated_at": str,
        },
        pk="episode_id",
    )
    db["episodes"].insert_all(rows)
    return db


class TestMarkEmbeddedDaily:
    def test_writes_status_done_and_all_metadata(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": None, "updated_at": None},
            {"episode_id": "ep:2", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": None, "updated_at": None},
        ])
        repo = EpisodeStatusRepository(db)
        count = repo.mark_embedded_daily(
            ["ep:1", "ep:2"],
            model="mm-minilm",
            version="text-v1",
            embedded_at="2026-04-15T00:00:00Z",
        )
        assert count == 2
        for row in db.execute(
            "SELECT episode_id, embedding_model, embedding_version, "
            "last_embedded_at, embedding_status, updated_at "
            "FROM episodes ORDER BY episode_id"
        ):
            assert row[1] == "mm-minilm"
            assert row[2] == "text-v1"
            assert row[3] == "2026-04-15T00:00:00Z"
            assert row[4] == "done"
            assert row[5]  # updated_at populated

    def test_empty_list_returns_zero_and_no_write(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": None, "updated_at": None},
        ])
        repo = EpisodeStatusRepository(db)
        count = repo.mark_embedded_daily(
            [], model="m", version="v", embedded_at="t",
        )
        assert count == 0
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id='ep:1'"
        ))
        assert row[0] is None

    def test_chunks_large_batches_above_max_chunk(self, tmp_path: Path) -> None:
        """MAX_CHUNK is 900; a batch over that must round-trip successfully
        without tripping SQLite's parameter limit."""
        episode_ids = [f"ep:{i}" for i in range(1500)]
        db = _make_db(tmp_path, [
            {"episode_id": eid, "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": None, "updated_at": None}
            for eid in episode_ids
        ])
        repo = EpisodeStatusRepository(db)
        count = repo.mark_embedded_daily(
            episode_ids, model="m", version="v", embedded_at="t",
        )
        assert count == 1500
        (done_count,) = next(db.execute(
            "SELECT COUNT(*) FROM episodes WHERE embedding_status='done'"
        ))
        assert done_count == 1500


class TestMetadataOnlyRegression:
    """`mark_embedding_metadata_only` must remain the Phase 2a contract:
    writes metadata but NEVER touches `embedding_status`. Force_embed
    and fallback-rebuild paths still depend on this."""

    def test_does_not_touch_embedding_status(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": "pending", "updated_at": None},
        ])
        repo = EpisodeStatusRepository(db)
        repo.mark_embedding_metadata_only(
            ["ep:1"], model="m", version="v", embedded_at="t",
        )
        row = next(db.execute(
            "SELECT embedding_status, embedding_model FROM episodes "
            "WHERE episode_id='ep:1'"
        ))
        assert row[0] == "pending"  # status preserved
        assert row[1] == "m"  # metadata written

    def test_also_preserves_done_status(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": "done", "updated_at": None},
        ])
        repo = EpisodeStatusRepository(db)
        repo.mark_embedding_metadata_only(
            ["ep:1"], model="m", version="v", embedded_at="t",
        )
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id='ep:1'"
        ))
        assert row[0] == "done"


class TestBatchRegression:
    """`mark_embedded_batch` stays untouched for the legacy
    `embed_episodes.py` standalone path. Step 6 must not refactor its
    body."""

    def test_signature_unchanged(self) -> None:
        sig = inspect.signature(EpisodeStatusRepository.mark_embedded_batch)
        param_names = list(sig.parameters)
        # Includes `self` plus the four public kwargs.
        assert param_names == ["self", "episode_ids", "model", "version", "embedded_at"]

    def test_sets_status_done(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_model": None,
             "embedding_version": None, "last_embedded_at": None,
             "embedding_status": None, "updated_at": None},
        ])
        repo = EpisodeStatusRepository(db)
        repo.mark_embedded_batch(
            ["ep:1"], model="m", version="v", embedded_at="t",
        )
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id='ep:1'"
        ))
        assert row[0] == "done"


class TestFieldParity:
    """`mark_embedded_daily` writes exactly the same metadata columns
    as `mark_embedding_metadata_only`, differing only by also setting
    `embedding_status='done'`. Any future drift between the two must
    either be intentional (update both) or flagged here."""

    def test_daily_and_metadata_only_touch_same_metadata_columns(
        self, tmp_path: Path
    ) -> None:
        db = _make_db(tmp_path, [
            {"episode_id": "ep:a", "embedding_model": "old",
             "embedding_version": "old-v", "last_embedded_at": "old",
             "embedding_status": None, "updated_at": "old"},
            {"episode_id": "ep:b", "embedding_model": "old",
             "embedding_version": "old-v", "last_embedded_at": "old",
             "embedding_status": None, "updated_at": "old"},
        ])
        repo = EpisodeStatusRepository(db)
        repo.mark_embedding_metadata_only(
            ["ep:a"], model="new-m", version="new-v", embedded_at="new-t",
        )
        repo.mark_embedded_daily(
            ["ep:b"], model="new-m", version="new-v", embedded_at="new-t",
        )
        ra = next(db.execute(
            "SELECT embedding_model, embedding_version, last_embedded_at, "
            "embedding_status FROM episodes WHERE episode_id='ep:a'"
        ))
        rb = next(db.execute(
            "SELECT embedding_model, embedding_version, last_embedded_at, "
            "embedding_status FROM episodes WHERE episode_id='ep:b'"
        ))
        # Metadata columns identical…
        assert ra[:3] == rb[:3] == ("new-m", "new-v", "new-t")
        # …status column is the intentional divergence.
        assert ra[3] is None
        assert rb[3] == "done"


class TestWiringAstAudit:
    """Phase 2b-A wiring contract after Step 7:
    `mark_embedded_daily` is used by `embed_and_ingest.py` (Step 6)
    AND `scripts/force_embed.py` (Step 7). `embed_episodes.py` keeps
    `mark_embedded_batch` for the legacy standalone path."""

    def test_embed_and_ingest_calls_mark_embedded_daily(self) -> None:
        src = Path("src/pipelines/embed_and_ingest.py").read_text()
        assert "mark_embedded_daily" in src
        assert "mark_embedding_metadata_only" not in src

    def test_force_embed_calls_mark_embedded_daily(self) -> None:
        src = Path("scripts/force_embed.py").read_text()
        assert "mark_embedded_daily" in src
        assert "mark_embedding_metadata_only" not in src

    def test_embed_episodes_still_uses_embedded_batch(self) -> None:
        src = Path("src/pipelines/embed_episodes.py").read_text()
        assert "mark_embedded_batch" in src
        assert "mark_embedded_daily" not in src
