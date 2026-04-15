"""Tests for `scripts/normalize_embedding_version.py`.

Covers the six cases from 2b-A impl doc Step 2: dry-run leaves the DB
untouched, apply rewrites legacy rows, a second apply is a no-op,
audit/report fields are present, `--limit` caps the scan, and a
missing `episodes` table / opening failure returns the documented exit
codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlite_utils import Database

from scripts import normalize_embedding_version as script


_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_LEGACY = f"{_MODEL}/text-v1"
_CLEAN = "text-v1"


@pytest.fixture()
def episodes_db(tmp_path: Path) -> Database:
    """An episodes table pre-populated with one legacy and one clean row."""
    db_path = tmp_path / "podcast.sqlite"
    db = Database(str(db_path))
    db["episodes"].create(
        {
            "episode_id": str,
            "embedding_model": str,
            "embedding_version": str,
            "updated_at": str,
        },
        pk="episode_id",
    )
    db["episodes"].insert_all(
        [
            {"episode_id": "ep:a", "embedding_model": _MODEL, "embedding_version": _LEGACY, "updated_at": ""},
            {"episode_id": "ep:b", "embedding_model": _MODEL, "embedding_version": _LEGACY, "updated_at": ""},
            {"episode_id": "ep:c", "embedding_model": _MODEL, "embedding_version": _CLEAN, "updated_at": ""},
            {"episode_id": "ep:d", "embedding_model": _MODEL, "embedding_version": None, "updated_at": ""},
        ]
    )
    return db


class TestDryRun:
    def test_dry_run_does_not_touch_db(self, episodes_db: Database) -> None:
        before = [r["embedding_version"] for r in episodes_db["episodes"].rows]

        report = script.normalize(episodes_db, dry_run=True)

        after = [r["embedding_version"] for r in episodes_db["episodes"].rows]
        assert before == after
        assert report["mode"] == "dry-run"
        assert report["updated_count"] == 2
        assert report["skipped_count"] == 0

    def test_dry_run_after_distribution_is_projected(self, episodes_db: Database) -> None:
        report = script.normalize(episodes_db, dry_run=True)
        # The projection collapses legacy into clean without actually writing.
        assert report["after_distribution"].get(_CLEAN, 0) == 3
        assert _LEGACY not in report["after_distribution"]


class TestApply:
    def test_apply_updates_legacy_rows(self, episodes_db: Database) -> None:
        report = script.normalize(episodes_db, dry_run=False)

        versions = {
            r["episode_id"]: r["embedding_version"] for r in episodes_db["episodes"].rows
        }
        assert versions["ep:a"] == _CLEAN
        assert versions["ep:b"] == _CLEAN
        assert versions["ep:c"] == _CLEAN
        assert versions["ep:d"] is None

        assert report["updated_count"] == 2
        assert report["skipped_count"] == 0
        assert report["after_distribution"].get(_CLEAN) == 3

    def test_apply_is_idempotent(self, episodes_db: Database) -> None:
        script.normalize(episodes_db, dry_run=False)

        second = script.normalize(episodes_db, dry_run=False)
        assert second["updated_count"] == 0
        assert second["candidate_count"] == 0

    def test_apply_writes_updated_at_for_changed_rows_only(self, episodes_db: Database) -> None:
        script.normalize(episodes_db, dry_run=False)

        rows = {r["episode_id"]: r for r in episodes_db["episodes"].rows}
        assert rows["ep:a"]["updated_at"] != ""
        assert rows["ep:b"]["updated_at"] != ""
        # Clean / NULL rows must not have `updated_at` touched.
        assert rows["ep:c"]["updated_at"] == ""
        assert rows["ep:d"]["updated_at"] == ""


class TestReportShape:
    def test_report_has_documented_fields(self, episodes_db: Database) -> None:
        report = script.normalize(episodes_db, dry_run=True)
        for key in (
            "mode", "ran_at", "limit", "before_distribution", "after_distribution",
            "candidate_count", "updated_count", "skipped_count",
        ):
            assert key in report, f"missing {key}"


class TestLimit:
    def test_limit_caps_candidates(self, episodes_db: Database) -> None:
        report = script.normalize(episodes_db, dry_run=True, limit=1)
        assert report["candidate_count"] == 1
        assert report["updated_count"] == 1


class TestMainExitCodes:
    def test_exit_2_when_episodes_table_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.sqlite"
        Database(str(db_path))  # creates an empty DB file
        rc = script.main(["--dry-run", "--db-path", str(db_path)])
        assert rc == 2

    def test_exit_0_on_successful_dry_run(
        self, episodes_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db_path = Path(episodes_db.conn.execute("PRAGMA database_list").fetchall()[0][2])
        report_path = tmp_path / "report.json"
        rc = script.main(["--dry-run", "--db-path", str(db_path), "--json-report", str(report_path)])
        assert rc == 0

        report = json.loads(report_path.read_text())
        assert report["mode"] == "dry-run"
        assert report["updated_count"] == 2
