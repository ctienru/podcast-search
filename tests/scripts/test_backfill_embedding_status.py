"""Tests for `scripts/backfill_embedding_status.py` (Step 3 — dry-run
classification against CT1–CT3).

One fixture per category from §3.6 covers the happy-path taxonomy;
two additional tests cover the CLI surface (`--apply` rejected,
`--help` mentions the apply-mode limitation) and one exercises the
grouping/cache-reuse assumption by putting multiple rows in the same
show.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlite_utils import Database

from scripts import backfill_embedding_status as script
from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE
from src.pipelines.embedding_paths import cache_path_for


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
    def test_apply_flag_is_rejected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "podcast.sqlite"
        Database(str(db_path))["episodes"].create(
            {"episode_id": str}, pk="episode_id"
        )
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--apply", "--db-path", str(db_path)])
        # argparse exits with 2 on unrecognized arguments.
        assert excinfo.value.code == 2

    def test_help_mentions_apply_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            script.main(["--help"])
        captured = capsys.readouterr().out
        # argparse may wrap the phrase across lines in the help block;
        # collapse whitespace before checking to stay tolerant.
        collapsed = " ".join(captured.split())
        assert "apply mode unavailable" in collapsed

    def test_main_exit_2_when_episodes_table_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.sqlite"
        Database(str(db_path))  # empty DB
        rc = script.main(["--dry-run", "--db-path", str(db_path),
                          "--cache-dir", str(tmp_path / "cache")])
        assert rc == 2


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
