"""Phase 2a drift detection CLI tests.

Covers:
- `check_episode_drift` classifies DB rows against expected identity
  and tolerates the legacy `<model>/text-v1` embedding_version format.
- Rows with `embedding_model IS NULL` are skipped (not drift).
- Rows whose show has an unknown target_index are counted separately
  (`unresolvable_language_count`), never conflated with drift.
- `summarize_show_impact` produces a deterministic (ascending, deduped)
  `affected_show_ids` and a top-N list with stable ordering.
- `summarize_sync_state_distribution` groups correctly and normalizes
  legacy version formats in the output.
- `main` exits 0 whether drift exists or not (Phase 2a is report-only).
- `--json` output is valid JSON and includes the expected top-level keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlite_utils import Database

from src.pipelines.embedding_catalog import MODEL_MAP
from src.pipelines.embedding_identity import _DIM_TABLE
from src.tools.check_drift import (
    _normalize_version,
    build_report,
    check_episode_drift,
    format_report_text,
    main,
    report_to_dict,
    summarize_show_impact,
    summarize_sync_state_distribution,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

MODEL = MODEL_MAP["zh"]  # paraphrase-multilingual-MiniLM-L12-v2
DIMS = _DIM_TABLE[MODEL]  # 384


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "crawler.db")
    db.executescript("""
        CREATE TABLE shows (
            show_id TEXT PRIMARY KEY,
            target_index TEXT
        );
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            show_id TEXT,
            embedding_status TEXT,
            embedding_model TEXT,
            embedding_version TEXT,
            last_embedded_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE search_sync_state (
            entity_type TEXT,
            entity_id TEXT,
            environment TEXT,
            index_alias TEXT,
            content_hash TEXT,
            source_updated_at TEXT,
            embedding_model TEXT,
            embedding_version TEXT,
            sync_status TEXT,
            last_synced_at TEXT,
            last_error TEXT
        );
    """)
    return db


def _add_show(db: Database, show_id: str, target_index: str = "podcast-episodes-zh-tw") -> None:
    db["shows"].insert({"show_id": show_id, "target_index": target_index}, pk="show_id")


def _add_episode(
    db: Database,
    *,
    episode_id: str,
    show_id: str,
    embedding_model: str | None = MODEL,
    embedding_version: str | None = "text-v1",
) -> None:
    db["episodes"].insert({
        "episode_id": episode_id,
        "show_id": show_id,
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
    }, pk="episode_id")


def _add_sync_state(
    db: Database,
    *,
    entity_id: str,
    environment: str = "local",
    model: str = MODEL,
    version: str = "text-v1",
    status: str = "synced",
) -> None:
    db["search_sync_state"].insert({
        "entity_type": "episode",
        "entity_id": entity_id,
        "environment": environment,
        "index_alias": "episodes-zh-tw",
        "embedding_model": model,
        "embedding_version": version,
        "sync_status": status,
    })


# ── _normalize_version ──────────────────────────────────────────────────────

class TestNormalizeVersion:
    def test_strips_legacy_model_prefix(self) -> None:
        assert _normalize_version(f"{MODEL}/text-v1") == "text-v1"

    def test_bare_version_unchanged(self) -> None:
        assert _normalize_version("text-v1") == "text-v1"

    def test_none_passes_through(self) -> None:
        assert _normalize_version(None) is None


# ── check_episode_drift ─────────────────────────────────────────────────────

class TestCheckEpisodeDrift:
    def test_ok_row_is_not_drift(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, episode_id="ep:1", show_id="show:1")

        report, drifting = check_episode_drift(db)

        assert report.episodes_with_metadata == 1
        assert report.ok_count == 1
        assert report.drift_counts == {}
        assert drifting == []

    def test_legacy_embedding_version_format_tolerated(self, tmp_path: Path) -> None:
        """DB rows written before Phase 2a carry `<model>/text-v1` — not drift."""
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(
            db,
            episode_id="ep:1",
            show_id="show:1",
            embedding_version=f"{MODEL}/text-v1",
        )

        report, drifting = check_episode_drift(db)

        assert report.ok_count == 1
        assert report.drift_counts == {}
        assert drifting == []

    def test_model_mismatch_is_counted(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, episode_id="ep:1", show_id="show:1", embedding_model="old-model")

        report, drifting = check_episode_drift(db)

        assert report.ok_count == 0
        assert report.drift_counts == {"model_mismatch": 1}
        assert drifting == ["show:1"]

    def test_version_mismatch_is_counted(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, episode_id="ep:1", show_id="show:1", embedding_version="text-v0")

        report, drifting = check_episode_drift(db)

        assert report.drift_counts == {"version_mismatch": 1}
        assert drifting == ["show:1"]

    def test_both_model_and_version_mismatch_is_multiple(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(
            db,
            episode_id="ep:1", show_id="show:1",
            embedding_model="old-model", embedding_version="text-v0",
        )

        report, drifting = check_episode_drift(db)

        assert report.drift_counts == {"multiple": 1}

    def test_null_embedding_metadata_skipped(self, tmp_path: Path) -> None:
        """Unembedded rows are NOT scanned — they're expected to be null."""
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, episode_id="ep:1", show_id="show:1", embedding_model=None)

        report, drifting = check_episode_drift(db)

        assert report.episodes_with_metadata == 0
        assert report.ok_count == 0
        assert report.drift_counts == {}

    def test_unresolvable_language_is_counted_separately(self, tmp_path: Path) -> None:
        """A show with an unknown target_index must not leak into drift counts."""
        db = _make_db(tmp_path)
        _add_show(db, "show:1", target_index="podcast-episodes-unknown")
        _add_episode(db, episode_id="ep:1", show_id="show:1")

        report, drifting = check_episode_drift(db)

        assert report.unresolvable_language_count == 1
        assert report.drift_counts == {}
        assert drifting == []


# ── summarize_show_impact ───────────────────────────────────────────────────

class TestShowImpactSummary:
    def test_affected_show_ids_are_ascending_and_deduped(self) -> None:
        out = summarize_show_impact(["show:b", "show:a", "show:b", "show:c"])

        assert out.affected_show_ids == ["show:a", "show:b", "show:c"]
        assert out.episode_truth_line_count == 4
        assert out.show_artifact_line_count == 3

    def test_top_shows_by_drift_sorted_by_count_desc_then_id_asc(self) -> None:
        out = summarize_show_impact(
            ["show:b", "show:a", "show:a", "show:c", "show:c"]
        )

        # a=2, c=2, b=1 → by count desc then id asc: (a,2), (c,2), (b,1)
        assert out.top_shows_by_drift == [("show:a", 2), ("show:c", 2), ("show:b", 1)]

    def test_empty_input_yields_empty_summary(self) -> None:
        out = summarize_show_impact([])

        assert out.episode_truth_line_count == 0
        assert out.show_artifact_line_count == 0
        assert out.affected_show_ids == []
        assert out.top_shows_by_drift == []


# ── summarize_sync_state_distribution ──────────────────────────────────────

class TestSyncStateDistribution:
    def test_groups_by_environment_model_version_and_status(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        # Two 'synced' entries under same env/model/version
        _add_sync_state(db, entity_id="ep:1")
        _add_sync_state(db, entity_id="ep:2")
        # Different environment
        _add_sync_state(db, entity_id="ep:3", environment="production")

        dist = summarize_sync_state_distribution(db)

        assert any(
            row["environment"] == "local"
            and row["embedding_model"] == MODEL
            and row["embedding_version"] == "text-v1"
            and row["sync_status"] == "synced"
            and row["count"] == 2
            for row in dist.rows
        )
        assert any(
            row["environment"] == "production" and row["count"] == 1
            for row in dist.rows
        )

    def test_normalizes_legacy_version_in_output(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_sync_state(db, entity_id="ep:1", version=f"{MODEL}/text-v1")

        dist = summarize_sync_state_distribution(db)

        assert dist.rows[0]["embedding_version"] == "text-v1"

    def test_missing_table_returns_empty(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "empty.db")  # no tables

        dist = summarize_sync_state_distribution(db)

        assert dist.rows == []

    def test_output_is_stable_across_two_calls(self, tmp_path: Path) -> None:
        """Deterministic ORDER BY makes diffing between two runs trivial."""
        db = _make_db(tmp_path)
        _add_sync_state(db, entity_id="ep:2", environment="production")
        _add_sync_state(db, entity_id="ep:1", environment="local")

        first = summarize_sync_state_distribution(db)
        second = summarize_sync_state_distribution(db)

        assert first.rows == second.rows


# ── build_report + CLI ─────────────────────────────────────────────────────

class TestBuildReport:
    def test_report_composition_end_to_end(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:ok")
        _add_show(db, "show:drift")
        _add_episode(db, episode_id="ep:ok", show_id="show:ok")
        _add_episode(
            db, episode_id="ep:drift", show_id="show:drift",
            embedding_model="old-model",
        )
        _add_sync_state(db, entity_id="ep:ok")

        report = build_report(db)

        assert report.expected_identity_by_language["zh-tw"]["model_name"] == MODEL
        assert report.episode_drift.episodes_with_metadata == 2
        assert report.episode_drift.ok_count == 1
        assert report.episode_drift.drift_counts == {"model_mismatch": 1}
        assert report.show_impact.affected_show_ids == ["show:drift"]
        assert len(report.sync_state_distribution.rows) == 1


class TestFormatReportText:
    def test_contains_each_section_header(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        report = build_report(db)
        text = format_report_text(report)
        for header in (
            "Expected identity",
            "Episode drift",
            "Show impact",
            "sync_state distribution",
        ):
            assert header in text


class TestMainCLI:
    def test_main_returns_zero_even_with_drift(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Phase 2a is report-only — exit code is always 0."""
        db_path = tmp_path / "crawler.db"
        db = _make_db(db_path.parent)
        db["shows"].insert({"show_id": "show:1", "target_index": "podcast-episodes-zh-tw"}, pk="show_id")
        db["episodes"].insert({
            "episode_id": "ep:1", "show_id": "show:1",
            "embedding_model": "wrong-model", "embedding_version": "text-v1",
        }, pk="episode_id")

        monkeypatch.setattr("sys.argv", ["check_drift.py", "--db", str(db_path)])
        rc = main()
        assert rc == 0

        output = capsys.readouterr().out
        assert "model_mismatch" in output

    def test_main_json_output_is_valid_json(self, tmp_path: Path, monkeypatch, capsys) -> None:
        db_path = tmp_path / "crawler.db"
        _make_db(db_path.parent)  # creates tables at tmp_path/crawler.db

        monkeypatch.setattr("sys.argv", ["check_drift.py", "--db", str(db_path), "--json"])
        rc = main()
        assert rc == 0

        payload = json.loads(capsys.readouterr().out)
        assert set(payload.keys()) >= {
            "expected_identity_by_language",
            "episode_drift",
            "show_impact",
            "sync_state_distribution",
        }
