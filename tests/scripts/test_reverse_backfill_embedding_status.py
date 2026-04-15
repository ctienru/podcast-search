"""Tests for `scripts/reverse_backfill_embedding_status.py` (Step 5).

Covers the seven exit codes, dry-run safety, pre/post fingerprint
gating, drift override, path-mismatch warning, and reverse-after-reverse
idempotency.

Each test constructs a small snapshot via `phase2b_snapshot.write_snapshot`
with an explicit `db_fingerprint` metadata block matching whatever DB
state the scenario requires. That keeps the reverse script under test
decoupled from the backfill apply path and lets us exercise each
fingerprint-match branch in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlite_utils import Database

from scripts import reverse_backfill_embedding_status as rev
from src.storage import phase2b_snapshot


_SNAPSHOT_TYPE = "backfill_embedding_status"


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a minimal episodes table and return the DB path."""
    db_path = tmp_path / "podcast.sqlite"
    db = Database(str(db_path))
    db["episodes"].create(
        {
            "episode_id": str,
            "embedding_status": str,
            "updated_at": str,
        },
        pk="episode_id",
    )
    if rows:
        db["episodes"].insert_all(rows)
    db.conn.close()
    return db_path


def _current_fp(db_path: Path) -> dict[str, Any]:
    return rev._compute_fingerprint(db_path)


def _write_snapshot(
    path: Path,
    *,
    rows: list[dict],
    pre_apply_fp: dict,
    post_apply_fp: dict,
    rel_path: str = "data/podcast.sqlite",
) -> None:
    phase2b_snapshot.write_snapshot(
        path,
        snapshot_type=_SNAPSHOT_TYPE,
        rows=rows,
        metadata={
            "script_git_sha": "0" * 40,
            "db_fingerprint": {
                "path": rel_path,
                "pre_apply": pre_apply_fp,
                "post_apply": post_apply_fp,
            },
            "total_rows_scanned": len(rows),
        },
    )


class TestSnapshotLoading:
    def test_missing_snapshot_exit_2(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [])
        rc = rev.main([
            "--snapshot", str(tmp_path / "does_not_exist.json"),
            "--db-path", str(db_path),
        ])
        assert rc == 2

    def test_invalid_snapshot_type_exit_3(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [])
        snap = tmp_path / "wrong-type.json"
        phase2b_snapshot.write_snapshot(
            snap,
            snapshot_type="some_other_migration",
            rows=[],
        )
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path)])
        assert rc == 3

    def test_missing_db_fingerprint_exit_3(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [])
        snap = tmp_path / "no-fingerprint.json"
        # Snapshot is valid at the universal layer but missing the
        # backfill-specific db_fingerprint block.
        phase2b_snapshot.write_snapshot(
            snap,
            snapshot_type=_SNAPSHOT_TYPE,
            rows=[],
        )
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path)])
        assert rc == 3


class TestFingerprintGating:
    def test_db_equal_post_apply_reverses_successfully(
        self, tmp_path: Path
    ) -> None:
        """Freshly applied state: DB sha256 matches `post_apply`; reverse
        runs, each row's status returns to its snapshot `pre` value."""
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
            {"episode_id": "ep:2", "embedding_status": "pending", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        # Construct an artificial `pre_apply` sha (different from current)
        # so the no-op branch isn't accidentally hit.
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[
                {"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None},
                {"episode_id": "ep:2", "show_id": "s", "pre_embedding_status": "pending"},
            ],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )

        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc == 0

        db = Database(str(db_path))
        got = {r[0]: r[1] for r in db.execute(
            "SELECT episode_id, embedding_status FROM episodes ORDER BY episode_id"
        )}
        assert got == {"ep:1": None, "ep:2": "pending"}

    def test_db_equal_pre_apply_is_noop(self, tmp_path: Path) -> None:
        """DB already in pre-apply state (e.g. restored from backup,
        or previously reversed). Exit 0 with no writes."""
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "pending", "updated_at": "mtime-keep"},
        ])
        current_fp = _current_fp(db_path)
        # The recorded `pre_apply` matches current; post_apply is a
        # different value (anything that isn't current).
        fake_post = {"file_sha256": "f" * 64, "file_size_bytes": 99999, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": "pending"}],
            pre_apply_fp=current_fp,
            post_apply_fp=fake_post,
        )

        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path)])
        assert rc == 0
        # updated_at must not have been touched — no-op means no write.
        db = Database(str(db_path))
        row = next(db.execute(
            "SELECT updated_at FROM episodes WHERE episode_id=?", ["ep:1"]
        ))
        assert row[0] == "mtime-keep"

    def test_fingerprint_drift_without_flag_exit_5(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        # Neither pre nor post matches current.
        pre_fp = {"file_sha256": "a" * 64, "file_size_bytes": 1, "mtime": ""}
        post_fp = {"file_sha256": "b" * 64, "file_size_bytes": 2, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path)])
        assert rc == 5

    def test_fingerprint_drift_with_flag_and_reason_allows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        pre_fp = {"file_sha256": "a" * 64, "file_size_bytes": 1, "mtime": ""}
        post_fp = {"file_sha256": "b" * 64, "file_size_bytes": 2, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        monkeypatch.setenv(rev._DRIFT_REASON_ENV, "operator acknowledged out-of-band edit")

        rc = rev.main([
            "--snapshot", str(snap), "--db-path", str(db_path),
            "--allow-fingerprint-drift", "--apply",
        ])
        assert rc == 0
        db = Database(str(db_path))
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?", ["ep:1"]
        ))
        assert row[0] is None

    def test_drift_flag_without_reason_exit_5(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        pre_fp = {"file_sha256": "a" * 64, "file_size_bytes": 1, "mtime": ""}
        post_fp = {"file_sha256": "b" * 64, "file_size_bytes": 2, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        monkeypatch.delenv(rev._DRIFT_REASON_ENV, raising=False)
        # Non-TTY stdin in pytest means prompt returns None too.
        rc = rev.main([
            "--snapshot", str(snap), "--db-path", str(db_path),
            "--allow-fingerprint-drift", "--apply",
        ])
        assert rc == 5

    def test_path_mismatch_but_hash_matches_still_succeeds(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CI / cross-checkout friendly: if sha256 + size match
        post_apply but the snapshot's recorded relative path differs
        from current, log a warning and proceed.

        `setup_logging()` clears root handlers which would wipe out
        caplog's handler; neutralise it for this log-assertion test."""
        monkeypatch.setattr(rev, "setup_logging", lambda: None)
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
            rel_path="some/other/location/podcast.sqlite",
        )

        with caplog.at_level("WARNING", logger="scripts.reverse_backfill_embedding_status"):
            rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc == 0
        assert "reverse_db_path_differs_from_snapshot" in caplog.text


class TestDryRun:
    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": "keep"},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        rc = rev.main([
            "--snapshot", str(snap), "--db-path", str(db_path), "--dry-run",
        ])
        assert rc == 0
        db = Database(str(db_path))
        row = next(db.execute(
            "SELECT embedding_status, updated_at FROM episodes WHERE episode_id=?",
            ["ep:1"],
        ))
        assert row[0] == "done"  # unchanged
        assert row[1] == "keep"  # updated_at untouched


class TestPostVerify:
    def test_concurrent_modification_mismatches_exit_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate: reverse writes, but between our UPDATE and verify
        another process flips the row. The verify step must see the
        mismatch and exit 4."""
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )

        original_verify = rev._verify_rows

        def tamper_then_verify(db, rows):
            # Before the verify runs, a phantom writer sets status to
            # something other than the expected pre value.
            db.conn.execute(
                "UPDATE episodes SET embedding_status=? WHERE episode_id=?",
                ("done", "ep:1"),
            )
            db.conn.commit()
            return original_verify(db, rows)

        monkeypatch.setattr(rev, "_verify_rows", tamper_then_verify)
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc == 4


class TestIdempotency:
    def test_reverse_after_reverse_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sequential reverses: the first writes, then an operator
        (simulated) restores the DB to pre_apply fingerprint via backup.
        The second run detects DB == pre_apply and exits no-op.

        SQLite on-disk layout doesn't round-trip to byte-identical files
        after a write cycle (page allocations, WAL, etc.), so we
        monkey-patch `_compute_fingerprint` for the second invocation
        to return the declared pre_apply value — the production
        equivalent is `cp backup.sqlite podcast.sqlite` between runs."""
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "p" * 64, "file_size_bytes": 1, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )

        rc1 = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc1 == 0

        # Between runs, an operator restored DB to pre_apply state.
        # Stamp a sentinel on the row so we can detect any stray write.
        db = Database(str(db_path))
        db.conn.execute(
            "UPDATE episodes SET updated_at=? WHERE episode_id=?",
            ("noop-sentinel", "ep:1"),
        )
        db.conn.commit()
        db.conn.close()
        monkeypatch.setattr(rev, "_compute_fingerprint", lambda _: pre_fp)

        rc2 = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc2 == 0
        db = Database(str(db_path))
        row = next(db.execute(
            "SELECT updated_at FROM episodes WHERE episode_id=?",
            ["ep:1"],
        ))
        assert row[0] == "noop-sentinel"  # no-op confirmed


class TestSnapshotRowValidation:
    """Step 8 copilot-review fix: a snapshot row missing `episode_id`
    must cause exit 3 (schema invalid) rather than crashing with
    KeyError inside `_reverse_rows`."""

    def test_row_missing_episode_id_exits_3(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        # Malformed: row dict is missing `episode_id`.
        _write_snapshot(
            snap,
            rows=[{"show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc == 3
        # DB untouched.
        db = Database(str(db_path))
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id='ep:1'"
        ))
        assert row[0] == "done"

    def test_row_not_a_dict_exits_3(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=["not_a_dict"],  # type: ignore[list-item]
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        rc = rev.main(["--snapshot", str(snap), "--db-path", str(db_path), "--apply"])
        assert rc == 3


class TestVerifyRowsRowAbsent:
    """Step 8 copilot-review fix: `_verify_rows` distinguishes a row
    absent from the DB from a row whose DB value is legitimately NULL.
    Previously both wrote `actual: null` and were ambiguous."""

    def test_mismatch_record_has_row_absent_boolean(
        self, tmp_path: Path
    ) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:here", "embedding_status": "done", "updated_at": ""},
        ])
        db = Database(str(db_path))
        # Snapshot references a row that isn't in DB → should surface
        # as row_absent=True in the mismatch list.
        mismatches = rev._verify_rows(db, [
            {"episode_id": "ep:gone", "show_id": "s", "pre_embedding_status": None},
        ])
        assert len(mismatches) == 1
        entry = mismatches[0]
        assert entry["episode_id"] == "ep:gone"
        assert entry["row_absent"] is True
        assert entry["actual"] is None

    def test_null_status_with_expected_value_is_not_row_absent(
        self, tmp_path: Path
    ) -> None:
        """The row exists with a NULL status but the snapshot expected
        'pending'. This is a real mismatch, not a missing row — the
        distinction matters for operator triage."""
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": None, "updated_at": ""},
        ])
        db = Database(str(db_path))
        mismatches = rev._verify_rows(db, [
            {"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": "pending"},
        ])
        assert len(mismatches) == 1
        entry = mismatches[0]
        assert entry["row_absent"] is False
        assert entry["actual"] is None
        assert entry["expected"] == "pending"


class TestJsonReport:
    def test_apply_report_contains_row_count_and_mode(
        self, tmp_path: Path
    ) -> None:
        db_path = _make_db(tmp_path, [
            {"episode_id": "ep:1", "embedding_status": "done", "updated_at": ""},
        ])
        post_fp = _current_fp(db_path)
        pre_fp = {"file_sha256": "0" * 64, "file_size_bytes": 0, "mtime": ""}
        snap = tmp_path / "snap.json"
        _write_snapshot(
            snap,
            rows=[{"episode_id": "ep:1", "show_id": "s", "pre_embedding_status": None}],
            pre_apply_fp=pre_fp,
            post_apply_fp=post_fp,
        )
        report_path = tmp_path / "report.json"
        rc = rev.main([
            "--snapshot", str(snap), "--db-path", str(db_path),
            "--json-report", str(report_path), "--apply",
        ])
        assert rc == 0
        report = json.loads(report_path.read_text())
        assert report["mode"] == "apply"
        assert report["rows_restored"] == 1
        assert report["mismatches"] == []
