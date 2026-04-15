"""Tests for `scripts/backfill_embedding_status.py` (Step 3 + Step 4).

Step 3 covers CT1–CT3 classification (one fixture per §3.6 category).
Step 4 adds CT4 (episode coverage), `--apply` mode with snapshot +
db_fingerprint, per-class apply policy (hard-fail block, anomaly
threshold), and the two-field concurrent-write guard (§3.2a).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlite_utils import Database

from scripts import backfill_embedding_status as script
from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE
from src.pipelines.embedding_paths import cache_path_for
from src.storage import phase2b_snapshot


_MODEL = next(iter(_DIM_TABLE))
_DIMS = _DIM_TABLE[_MODEL]
_VERSION = "text-v1"
_IDENTITY = EmbeddingIdentity(
    model_name=_MODEL,
    embedding_version=_VERSION,
    embedding_dimensions=_DIMS,
)


def _write_cache_payload(
    cache_dir: Path,
    show_id: str,
    *,
    identity: EmbeddingIdentity = _IDENTITY,
    episodes: dict | None = None,
    raw_text: str | None = None,
    override_identity_fields: dict | None = None,
) -> Path:
    path = cache_path_for(cache_dir, identity, show_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_text is not None:
        path.write_text(raw_text, encoding="utf-8")
        return path
    entry = {
        "show_id": show_id,
        "model_name": identity.model_name,
        "embedding_version": identity.embedding_version,
        "embedding_dimensions": identity.embedding_dimensions,
        "episodes": episodes if episodes is not None else {},
    }
    if override_identity_fields:
        entry.update(override_identity_fields)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _make_db(tmp_path: Path, rows: list[dict]) -> Database:
    db_path = tmp_path / "podcast.sqlite"
    db = Database(str(db_path))
    db["episodes"].create(
        {
            "episode_id": str,
            "show_id": str,
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


class TestCategoryFixtures:
    def test_pass_fixture(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:pass:1",
                "show_id": "show:pass",
                "embedding_model": _MODEL,
                "embedding_version": _VERSION,
                "last_embedded_at": "2026-04-01T00:00:00Z",
                "embedding_status": None,
                "updated_at": "",
            }],
        )
        _write_cache_payload(cache_dir, "show:pass", episodes={"ep:pass:1": [0.1] * _DIMS})

        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.PASS.value] == 1

    def test_neutral_metadata_absent_fixture(self, tmp_path: Path) -> None:
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:neutral:1",
                "show_id": "show:neutral",
                "embedding_model": None,
                "embedding_version": None,
                "last_embedded_at": None,
                "embedding_status": None,
                "updated_at": "",
            }],
        )
        result = script.classify_all(db, cache_dir=tmp_path / "cache")
        assert result.counts[script.Category.NEUTRAL_METADATA_ABSENT.value] == 1
        # Neutral rows must not contribute to the anomaly/hard-fail counts.
        assert result.counts.get(script.Category.ANOMALY_CACHE_MISSING.value, 0) == 0

    def test_anomaly_partial_metadata_fixture(self, tmp_path: Path) -> None:
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:partial:1",
                "show_id": "show:partial",
                "embedding_model": _MODEL,
                "embedding_version": None,  # One missing field causes partial anomaly
                "last_embedded_at": None,
                "embedding_status": None,
                "updated_at": "",
            }],
        )
        result = script.classify_all(db, cache_dir=tmp_path / "cache")
        assert result.counts[script.Category.ANOMALY_PARTIAL_METADATA.value] == 1
        assert len(result.anomalies) == 1
        assert result.anomalies[0]["episode_id"] == "ep:partial:1"

    def test_anomaly_cache_missing_fixture(self, tmp_path: Path) -> None:
        # DB metadata complete but no cache file under versioned path.
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:anomaly:1",
                "show_id": "show:anomaly",
                "embedding_model": _MODEL,
                "embedding_version": _VERSION,
                "last_embedded_at": "2026-04-01T00:00:00Z",
                "embedding_status": "done",
                "updated_at": "",
            }],
        )
        result = script.classify_all(db, cache_dir=tmp_path / "cache")
        assert result.counts[script.Category.ANOMALY_CACHE_MISSING.value] == 1
        # Anomalies must be captured in the structured list, not merged silently.
        assert len(result.anomalies) == 1
        assert result.anomalies[0]["episode_id"] == "ep:anomaly:1"

    def test_fail_payload_unreadable_fixture(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:unreadable:1",
                "show_id": "show:unreadable",
                "embedding_model": _MODEL,
                "embedding_version": _VERSION,
                "last_embedded_at": "2026-04-01T00:00:00Z",
                "embedding_status": None,
                "updated_at": "",
            }],
        )
        _write_cache_payload(cache_dir, "show:unreadable", raw_text="{ not valid json")

        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.FAIL_PAYLOAD_UNREADABLE.value] == 1

    def test_fail_payload_identity_mismatch_fixture(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [{
                "episode_id": "ep:mismatch:1",
                "show_id": "show:mismatch",
                "embedding_model": _MODEL,
                "embedding_version": _VERSION,
                "last_embedded_at": "2026-04-01T00:00:00Z",
                "embedding_status": None,
                "updated_at": "",
            }],
        )
        # Write a payload at the path the row identity would select, but with
        # divergent identity fields inside — CT2 payload check must fail.
        _write_cache_payload(
            cache_dir, "show:mismatch",
            override_identity_fields={"embedding_version": "text-v2"},
        )

        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.FAIL_PAYLOAD_IDENTITY_MISMATCH.value] == 1


class TestReportShape:
    def test_report_has_required_machine_readable_fields(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path, [])
        result = script.classify_all(db, cache_dir=tmp_path / "cache")
        report = script.build_report(result, mode="dry-run")
        for key in (
            "mode", "ran_at", "total_rows_scanned", "counts", "samples",
            "anomalies", "metadata_complete_count",
            "anomaly_cache_missing_count", "anomaly_cache_missing_pct",
            "hard_fail_count",
        ):
            assert key in report, f"missing {key}"
        # Every category must appear with at least 0.
        for cat in script.Category:
            assert cat.value in report["counts"]

    def test_denominator_excludes_neutral(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [
                # neutral — must not contribute to denominator
                {"episode_id": "ep:n:1", "show_id": "show:n",
                 "embedding_model": None, "embedding_version": None,
                 "last_embedded_at": None, "embedding_status": None, "updated_at": ""},
                # anomaly (denominator includes this, numerator == 1)
                {"episode_id": "ep:a:1", "show_id": "show:a",
                 "embedding_model": _MODEL, "embedding_version": _VERSION,
                 "last_embedded_at": "2026-04-01T00:00:00Z",
                 "embedding_status": "done", "updated_at": ""},
            ],
        )
        result = script.classify_all(db, cache_dir=cache_dir)
        report = script.build_report(result, mode="dry-run")
        assert report["metadata_complete_count"] == 1
        assert report["anomaly_cache_missing_count"] == 1
        assert report["anomaly_cache_missing_pct"] == 1.0

    def test_denominator_excludes_partial_metadata(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [
                # partial — must not contribute to denominator
                {"episode_id": "ep:p:1", "show_id": "show:p",
                 "embedding_model": _MODEL, "embedding_version": None,
                 "last_embedded_at": None, "embedding_status": None, "updated_at": ""},
                # anomaly (denominator includes this, numerator == 1)
                {"episode_id": "ep:a:1", "show_id": "show:a",
                 "embedding_model": _MODEL, "embedding_version": _VERSION,
                 "last_embedded_at": "2026-04-01T00:00:00Z",
                 "embedding_status": "done", "updated_at": ""},
            ],
        )
        result = script.classify_all(db, cache_dir=cache_dir)
        report = script.build_report(result, mode="dry-run")
        assert report["metadata_complete_count"] == 1
        assert report["anomaly_cache_missing_count"] == 1
        assert report["anomaly_cache_missing_pct"] == 1.0


class TestCliSurface:
    def test_dry_run_and_apply_are_mutually_exclusive(self, tmp_path: Path) -> None:
        db_path = tmp_path / "podcast.sqlite"
        Database(str(db_path))["episodes"].create(
            {"episode_id": str}, pk="episode_id"
        )
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--dry-run", "--apply", "--db-path", str(db_path)])
        # argparse exits with 2 on mutex violation.
        assert excinfo.value.code == 2

    def test_main_exit_4_when_episodes_table_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.sqlite"
        Database(str(db_path))  # empty DB
        rc = script.main(["--dry-run", "--db-path", str(db_path),
                          "--cache-dir", str(tmp_path / "cache")])
        assert rc == 4


class TestCacheReuse:
    def test_multiple_rows_same_show_reuse_cache_load(self, tmp_path: Path) -> None:
        """Two episodes from the same show must share one cache load.
        Implementation invariant: cache_by_key is keyed by (show_id,
        identity); this test ensures the correct category is assigned
        to both rows and the grouping path does not classify them
        differently."""
        cache_dir = tmp_path / "cache"
        db = _make_db(
            tmp_path,
            [
                {"episode_id": "ep:x:1", "show_id": "show:x",
                 "embedding_model": _MODEL, "embedding_version": _VERSION,
                 "last_embedded_at": "2026-04-01T00:00:00Z",
                 "embedding_status": None, "updated_at": ""},
                {"episode_id": "ep:x:2", "show_id": "show:x",
                 "embedding_model": _MODEL, "embedding_version": _VERSION,
                 "last_embedded_at": "2026-04-01T00:00:00Z",
                 "embedding_status": None, "updated_at": ""},
            ],
        )
        # No cache file — both episodes should land in anomaly.
        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.ANOMALY_CACHE_MISSING.value] == 2


class TestLimit:
    def test_limit_caps_scan(self, tmp_path: Path) -> None:
        db = _make_db(
            tmp_path,
            [{"episode_id": f"ep:{i}", "show_id": "show:z",
              "embedding_model": None, "embedding_version": None,
              "last_embedded_at": None, "embedding_status": None,
              "updated_at": ""} for i in range(5)],
        )
        result = script.classify_all(db, cache_dir=tmp_path / "cache", limit=2)
        assert result.total_rows_scanned == 2


# ---------------------------------------------------------------------------
# Step 4 — CT4 + apply + snapshot + concurrency guard
# ---------------------------------------------------------------------------


def _pass_row(episode_id: str, show_id: str, *, status: str | None = None) -> dict:
    return {
        "episode_id": episode_id,
        "show_id": show_id,
        "embedding_model": _MODEL,
        "embedding_version": _VERSION,
        "last_embedded_at": "2026-04-01T00:00:00Z",
        "embedding_status": status,
        "updated_at": "",
    }


def _run_cli_apply(
    tmp_path: Path,
    db: Database,
    *,
    cache_dir: Path,
    threshold: float | None = None,
    snapshot_dir: Path | None = None,
) -> tuple[int, Path]:
    snap_dir = snapshot_dir if snapshot_dir is not None else tmp_path / "snapshots"
    argv = ["--apply",
            "--db-path", str(db.conn.execute("PRAGMA database_list").fetchone()[2]),
            "--cache-dir", str(cache_dir),
            "--snapshot-dir", str(snap_dir)]
    if threshold is not None:
        argv.extend(["--anomaly-threshold-pct", str(threshold)])
    # Close the test DB handle so main() can open it cleanly.
    db.conn.close()
    rc = script.main(argv)
    return rc, snap_dir


class TestCT4:
    def test_fail_episode_entry_missing_fixture(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:ct4:1", "show:ct4")])
        # Payload is well-formed but the row's episode_id is not in the
        # episodes dict — partial rebuild residue.
        _write_cache_payload(cache_dir, "show:ct4",
                             episodes={"ep:other:1": [0.0] * _DIMS})

        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.FAIL_EPISODE_ENTRY_MISSING.value] == 1
        assert (
            script._CATEGORY_CLASS[script.Category.FAIL_EPISODE_ENTRY_MISSING]
            is script.Klass.HARD_FAIL
        )

    def test_empty_episodes_dict_still_fails_ct4(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:ct4:2", "show:ct4b")])
        _write_cache_payload(cache_dir, "show:ct4b", episodes={})

        result = script.classify_all(db, cache_dir=cache_dir)
        assert result.counts[script.Category.FAIL_EPISODE_ENTRY_MISSING.value] == 1


class TestApplySnapshot:
    def test_apply_writes_snapshot_with_db_fingerprint_four_subfields(
        self, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        rc, snap_dir = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 0

        snapshots = list(snap_dir.glob("backfill-*.json"))
        assert len(snapshots) == 1, snapshots
        data = phase2b_snapshot.read_snapshot(snapshots[0])
        phase2b_snapshot.validate_schema(data, expected_type="backfill_embedding_status")

        fp = data["db_fingerprint"]
        assert "path" in fp
        assert "pre_apply" in fp and "post_apply" in fp
        for side in ("pre_apply", "post_apply"):
            for key in ("file_sha256", "file_size_bytes", "mtime"):
                assert key in fp[side], f"missing {side}.{key}"
        # pre and post differ because the UPDATE mutates the DB.
        assert fp["pre_apply"]["file_sha256"] != fp["post_apply"]["file_sha256"]
        # rows_to_change captures the one row that actually changed.
        assert len(data["rows"]) == 1
        assert data["rows"][0]["episode_id"] == "ep:ok:1"
        assert data["rows"][0]["pre_embedding_status"] is None
        # Script-level metadata surfaces in the final snapshot.
        assert data["changed_count"] == 1
        assert data["skipped_concurrent_write_count"] == 0

    def test_apply_status_is_set_to_done_for_pass_row(
        self, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        rc, _ = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 0
        # Re-open to observe post-apply state.
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:ok:1"],
        ))
        assert row[0] == "done"

    def test_apply_snapshot_write_failure_blocks_db_update(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        def _boom(*a, **kw):
            raise OSError("disk full (simulated)")

        monkeypatch.setattr(script, "write_snapshot", _boom)
        rc, _ = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 3
        # DB must be untouched.
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:ok:1"],
        ))
        assert row[0] is None

    def test_apply_final_snapshot_rewrite_failure_exits_6(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 8 copilot-review fix: if the post-UPDATE snapshot
        rewrite fails, the DB has already been mutated. We must exit 6
        (not 0 with a warning) and leave the probe snapshot — with
        pre_apply fingerprint — in place so operators can restore from
        backup manually."""
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        # Let the first (probe) write succeed; fail the final rewrite.
        real_write = script.write_snapshot
        call_count = {"n": 0}

        def _write_then_fail(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_write(*args, **kwargs)
            raise OSError("disk went read-only mid-apply (simulated)")

        monkeypatch.setattr(script, "write_snapshot", _write_then_fail)
        rc, snap_dir = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 6

        # DB IS mutated — we don't roll back past the apply.
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:ok:1"],
        ))
        assert row[0] == "done"

        # Probe snapshot (with post_apply=None) remains; tmp file cleaned up.
        snapshots = list(snap_dir.glob("backfill-*.json"))
        assert len(snapshots) == 1
        data = json.loads(snapshots[0].read_text())
        assert data["db_fingerprint"]["post_apply"] is None
        assert data["db_fingerprint"]["pre_apply"] is not None
        # No lingering .tmp file.
        assert not list(snap_dir.glob("*.tmp"))


class TestApplyGates:
    def test_hard_fail_blocks_apply(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:ct4:1", "show:ct4")])
        _write_cache_payload(cache_dir, "show:ct4",
                             episodes={"ep:other:1": [0.0] * _DIMS})

        rc, snap_dir = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 2
        # No snapshot written.
        assert not any(snap_dir.glob("backfill-*.json")) if snap_dir.exists() else True
        # DB row untouched.
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:ct4:1"],
        ))
        assert row[0] is None

    def test_anomaly_threshold_exceeded_blocks_apply(
        self, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        # Only metadata-complete row is an anomaly (100% > default 10%).
        db = _make_db(tmp_path, [
            _pass_row("ep:anom:1", "show:anom", status="done"),
        ])
        # No cache written → ANOMALY_CACHE_MISSING.
        rc, snap_dir = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 2
        assert not any(snap_dir.glob("backfill-*.json")) if snap_dir.exists() else True
        # DB row untouched (still 'done').
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:anom:1"],
        ))
        assert row[0] == "done"

    def test_anomaly_threshold_override_allows_apply(
        self, tmp_path: Path
    ) -> None:
        """With a permissive threshold the same anomaly fixture must land
        the anomaly row at 'pending' (downgrade-from-'done' is a feature
        per §3.2a, not a bug)."""
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [
            _pass_row("ep:anom:1", "show:anom", status="done"),
        ])
        rc, _ = _run_cli_apply(tmp_path, db, cache_dir=cache_dir, threshold=100.0)
        assert rc == 0
        after = Database(str(db_path))
        row = next(after.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:anom:1"],
        ))
        assert row[0] == "pending"

    def test_threshold_flag_cannot_override_hard_fail(
        self, tmp_path: Path
    ) -> None:
        """A permissive --anomaly-threshold-pct must still fail apply
        when a hard_fail row exists. The flag controls the anomaly class
        only; hard-fail is non-overridable."""
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:ct4:1", "show:ct4")])
        _write_cache_payload(cache_dir, "show:ct4",
                             episodes={"ep:other:1": [0.0] * _DIMS})
        rc, _ = _run_cli_apply(tmp_path, db, cache_dir=cache_dir, threshold=100.0)
        assert rc == 2


class TestConcurrentWriteGuard:
    """Exercise the two-field `WHERE ... IS ...` guard directly by
    running classify → then mutating the DB row behind the result's
    back → then calling `_apply_updates` with the stale read markers."""

    def test_skips_row_when_last_embedded_at_changed(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:race:1", "show:race")])
        _write_cache_payload(cache_dir, "show:race",
                             episodes={"ep:race:1": [0.1] * _DIMS})

        result = script.classify_all(db, cache_dir=cache_dir)
        decisions = script._collect_apply_decisions(result)
        assert len(decisions) == 1

        # Simulate a concurrent `force_embed`: it re-embeds and bumps
        # last_embedded_at. Our guard should prevent overwriting.
        db.conn.execute(
            "UPDATE episodes SET last_embedded_at=? WHERE episode_id=?",
            ("2099-01-01T00:00:00Z", "ep:race:1"),
        )
        db.conn.commit()

        written, skipped = script._apply_updates(db, decisions)
        assert written == []
        assert skipped == 1
        # Row status remains whatever it was before (None).
        row = next(db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id=?",
            ["ep:race:1"],
        ))
        assert row[0] is None

    def test_skips_row_when_embedding_status_changed(self, tmp_path: Path) -> None:
        """Guards the case where a concurrent writer flips the status
        field without touching last_embedded_at — a single-field guard
        on last_embedded_at would miss this, hence AND-of-both."""
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:race:2", "show:race2")])
        _write_cache_payload(cache_dir, "show:race2",
                             episodes={"ep:race:2": [0.1] * _DIMS})

        result = script.classify_all(db, cache_dir=cache_dir)
        decisions = script._collect_apply_decisions(result)
        assert len(decisions) == 1

        db.conn.execute(
            "UPDATE episodes SET embedding_status=? WHERE episode_id=?",
            ("pending", "ep:race:2"),
        )
        db.conn.commit()

        written, skipped = script._apply_updates(db, decisions)
        assert written == []
        assert skipped == 1


class TestIdempotencyAndTargetSkip:
    def test_apply_twice_is_idempotent(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        db_path = tmp_path / "podcast.sqlite"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        rc1, _ = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc1 == 0
        # Second apply: row is already 'done', so nothing to change.
        second_db = Database(str(db_path))
        rc2, snap_dir = _run_cli_apply(tmp_path, second_db,
                                       cache_dir=cache_dir,
                                       snapshot_dir=tmp_path / "snapshots2")
        assert rc2 == 0
        snapshots = list((tmp_path / "snapshots2").glob("backfill-*.json"))
        assert len(snapshots) == 1
        data = phase2b_snapshot.read_snapshot(snapshots[0])
        assert data["rows"] == []
        assert data["changed_count"] == 0

    def test_row_already_at_target_is_not_in_snapshot(
        self, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        # Mix: one row already 'done' and classified PASS (no-op) +
        # one row 'pending' and classified PASS (must become 'done').
        db = _make_db(
            tmp_path,
            [
                _pass_row("ep:ok:already", "show:ok", status="done"),
                _pass_row("ep:ok:new", "show:ok", status=None),
            ],
        )
        _write_cache_payload(
            cache_dir, "show:ok",
            episodes={
                "ep:ok:already": [0.1] * _DIMS,
                "ep:ok:new": [0.2] * _DIMS,
            },
        )

        rc, snap_dir = _run_cli_apply(tmp_path, db, cache_dir=cache_dir)
        assert rc == 0
        snapshots = list(snap_dir.glob("backfill-*.json"))
        data = phase2b_snapshot.read_snapshot(snapshots[0])
        ids = {r["episode_id"] for r in data["rows"]}
        assert ids == {"ep:ok:new"}
        assert data["changed_count"] == 1


class TestJsonReportSchemaParity:
    def test_dry_run_report_keys_subset_of_apply_report_keys(
        self, tmp_path: Path
    ) -> None:
        """Dry-run and apply JSON reports share the audit-core schema
        (apply extends with changed/skipped/snapshot fields). Consumers
        can read every classification metric from either mode."""
        cache_dir = tmp_path / "cache"
        db = _make_db(tmp_path, [_pass_row("ep:ok:1", "show:ok")])
        _write_cache_payload(cache_dir, "show:ok",
                             episodes={"ep:ok:1": [0.1] * _DIMS})

        dry_report = script.build_report(
            script.classify_all(db, cache_dir=cache_dir),
            mode="dry-run",
        )
        audit_core = {
            "mode", "ran_at", "total_rows_scanned", "counts", "samples",
            "anomalies", "metadata_complete_count",
            "anomaly_cache_missing_count", "anomaly_cache_missing_pct",
            "hard_fail_count",
        }
        assert audit_core.issubset(dry_report.keys())

        # Build an apply-mode report the same way _run_apply does, then
        # ensure the core keys are present and extend-only fields can be
        # attached without collision.
        apply_report = script.build_report(
            script.classify_all(db, cache_dir=cache_dir),
            mode="apply",
        )
        apply_report.update({
            "changed_count": 0,
            "unchanged_count": 1,
            "skipped_concurrent_write_count": 0,
            "snapshot_path": "x",
        })
        assert audit_core.issubset(apply_report.keys())
