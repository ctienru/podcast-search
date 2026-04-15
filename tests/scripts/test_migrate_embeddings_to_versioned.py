"""Phase 2a migration script tests.

Covers each of the five classification buckets, idempotency, dry-run
behavior, embedding_version normalization, and the CLI exit code that
enforces the migration gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.migrate_embeddings_to_versioned import (
    MigrationCategory,
    _canonical_hash,
    _normalize_embedding_version,
    classify_and_migrate,
    main,
    run_migration,
)
from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE
from src.pipelines.embedding_paths import cache_path_for


# ── Helpers ─────────────────────────────────────────────────────────────────

MODEL_KNOWN = "paraphrase-multilingual-MiniLM-L12-v2"
DIMS_KNOWN = _DIM_TABLE[MODEL_KNOWN]


def _flat(
    tmp_path: Path,
    show_id: str,
    *,
    model_name: str | None = MODEL_KNOWN,
    embedding_version: str | None = f"{MODEL_KNOWN}/text-v1",
    dims: int = DIMS_KNOWN,
    embedded_at: str = "2026-04-01T00:00:00Z",
    episodes: dict | None = None,
) -> Path:
    """Write a flat cache file at <tmp_path>/<show_id>.json."""
    entry: dict = {
        "show_id": show_id,
        "embedded_at": embedded_at,
        "episodes": episodes or {"ep:1": [0.1] * dims},
    }
    if model_name is not None:
        entry["model_name"] = model_name
    if embedding_version is not None:
        entry["embedding_version"] = embedding_version
    path = tmp_path / f"{show_id}.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _clean_identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name=MODEL_KNOWN,
        embedding_version="text-v1",
        embedding_dimensions=DIMS_KNOWN,
    )


# ── Version normalization ──────────────────────────────────────────────────

class TestNormalizeEmbeddingVersion:
    def test_strips_legacy_model_prefix(self) -> None:
        assert _normalize_embedding_version(f"{MODEL_KNOWN}/text-v1") == "text-v1"

    def test_keeps_bare_version_unchanged(self) -> None:
        assert _normalize_embedding_version("text-v1") == "text-v1"

    def test_handles_multiple_segments(self) -> None:
        """Only the final segment after `/` is kept."""
        assert _normalize_embedding_version("org/model/text-v2") == "text-v2"


# ── Category: MIGRATED_OK ──────────────────────────────────────────────────

class TestMigratedOK:
    def test_creates_versioned_file_with_three_identity_fields(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1")

        category = classify_and_migrate(flat, tmp_path)

        assert category == MigrationCategory.MIGRATED_OK
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        assert versioned_path.exists()
        entry = json.loads(versioned_path.read_text())
        assert entry["model_name"] == MODEL_KNOWN
        assert entry["embedding_version"] == "text-v1"  # normalized
        assert entry["embedding_dimensions"] == DIMS_KNOWN

    def test_preserves_episodes_and_embedded_at(self, tmp_path: Path) -> None:
        flat = _flat(
            tmp_path, "show:1",
            embedded_at="2026-03-01T00:00:00Z",
            episodes={"ep:a": [0.5] * DIMS_KNOWN, "ep:b": [0.9] * DIMS_KNOWN},
        )

        classify_and_migrate(flat, tmp_path)

        entry = json.loads(cache_path_for(tmp_path, _clean_identity(), "show:1").read_text())
        assert entry["embedded_at"] == "2026-03-01T00:00:00Z"
        assert set(entry["episodes"].keys()) == {"ep:a", "ep:b"}

    def test_leaves_flat_file_in_place(self, tmp_path: Path) -> None:
        """Phase 2a strategy A: legacy flat files may remain (runtime ignores)."""
        flat = _flat(tmp_path, "show:1")

        classify_and_migrate(flat, tmp_path)

        assert flat.exists(), "migration must not delete flat files (strategy A)"


# ── Category: UNMIGRATED_MISSING_METADATA ──────────────────────────────────

class TestUnmigratedMissingMetadata:
    def test_flat_without_model_name(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1", model_name=None)
        assert classify_and_migrate(flat, tmp_path) == MigrationCategory.UNMIGRATED_MISSING_METADATA

    def test_flat_without_embedding_version(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1", embedding_version=None)
        assert classify_and_migrate(flat, tmp_path) == MigrationCategory.UNMIGRATED_MISSING_METADATA

    def test_unreadable_json_classified_as_missing(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        assert classify_and_migrate(bad, tmp_path) == MigrationCategory.UNMIGRATED_MISSING_METADATA

    def test_does_not_write_versioned_file(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1", model_name=None)
        classify_and_migrate(flat, tmp_path)
        # No subdirectory created under cache_dir
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs == []


# ── Category: UNMIGRATED_INFER_FAILED ──────────────────────────────────────

class TestUnmigratedInferFailed:
    def test_unknown_model_keeps_flat_untouched(self, tmp_path: Path) -> None:
        flat = _flat(
            tmp_path, "show:1",
            model_name="unknown-model-not-in-dim-table",
            embedding_version="unknown-model-not-in-dim-table/text-v1",
        )

        category = classify_and_migrate(flat, tmp_path)

        assert category == MigrationCategory.UNMIGRATED_INFER_FAILED
        # No versioned file created for any slug
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs == []


# ── Category: CONFLICT_EXISTING_SAME (idempotency) ─────────────────────────

class TestConflictExistingSame:
    def test_idempotent_second_run_detects_same_content(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1")

        first = classify_and_migrate(flat, tmp_path)
        second = classify_and_migrate(flat, tmp_path)

        assert first == MigrationCategory.MIGRATED_OK
        assert second == MigrationCategory.CONFLICT_EXISTING_SAME

    def test_differing_embedded_at_does_not_flag_as_conflict(self, tmp_path: Path) -> None:
        """Canonical hash excludes embedded_at — same semantic content passes."""
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        versioned_path.parent.mkdir(parents=True)
        versioned_path.write_text(json.dumps({
            "show_id": "show:1",
            "model_name": MODEL_KNOWN,
            "embedding_version": "text-v1",
            "embedding_dimensions": DIMS_KNOWN,
            "embedded_at": "2026-05-01T00:00:00Z",  # newer
            "episodes": {"ep:1": [0.1] * DIMS_KNOWN},
        }), encoding="utf-8")

        flat = _flat(
            tmp_path, "show:1",
            embedded_at="2026-04-01T00:00:00Z",  # older
            episodes={"ep:1": [0.1] * DIMS_KNOWN},
        )

        assert classify_and_migrate(flat, tmp_path) == MigrationCategory.CONFLICT_EXISTING_SAME


# ── Category: CONFLICT_EXISTING_DIFFERENT (MG1 halt) ───────────────────────

class TestConflictExistingDifferent:
    def test_different_episodes_flagged(self, tmp_path: Path) -> None:
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        versioned_path.parent.mkdir(parents=True)
        versioned_path.write_text(json.dumps({
            "show_id": "show:1",
            "model_name": MODEL_KNOWN,
            "embedding_version": "text-v1",
            "embedding_dimensions": DIMS_KNOWN,
            "embedded_at": "2026-05-01Z",
            "episodes": {"ep:other": [0.0] * DIMS_KNOWN},  # DIFFERENT episode
        }), encoding="utf-8")

        flat = _flat(tmp_path, "show:1", episodes={"ep:1": [0.1] * DIMS_KNOWN})

        assert classify_and_migrate(flat, tmp_path) == MigrationCategory.CONFLICT_EXISTING_DIFFERENT

    def test_corrupt_existing_versioned_flagged_as_different(self, tmp_path: Path) -> None:
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        versioned_path.parent.mkdir(parents=True)
        versioned_path.write_text("{{ not json")  # corrupt

        flat = _flat(tmp_path, "show:1")

        assert classify_and_migrate(flat, tmp_path) == MigrationCategory.CONFLICT_EXISTING_DIFFERENT


# ── Dry-run ────────────────────────────────────────────────────────────────

class TestDryRun:
    def test_classifies_without_writing_versioned_file(self, tmp_path: Path) -> None:
        flat = _flat(tmp_path, "show:1")

        category = classify_and_migrate(flat, tmp_path, dry_run=True)

        assert category == MigrationCategory.MIGRATED_OK
        assert not cache_path_for(tmp_path, _clean_identity(), "show:1").exists()


# ── run_migration + report ─────────────────────────────────────────────────

class TestRunMigration:
    def test_mixed_input_populates_counts_and_samples(self, tmp_path: Path) -> None:
        _flat(tmp_path, "show:ok:1")
        _flat(tmp_path, "show:ok:2")
        _flat(tmp_path, "show:missing", model_name=None)
        _flat(
            tmp_path, "show:infer",
            model_name="unknown-x", embedding_version="unknown-x/text-v1",
        )

        result = run_migration(tmp_path)
        report = result.to_report()

        assert report["total_scanned"] == 4
        assert report["counts"]["migrated_ok"] == 2
        assert report["counts"]["unmigrated_missing_metadata"] == 1
        assert report["counts"]["unmigrated_infer_failed"] == 1
        assert set(report["samples"]["migrated_ok"]) == {"show:ok:1", "show:ok:2"}
        assert report["migration_complete"] is True

    def test_migration_complete_flag_false_on_conflict(self, tmp_path: Path) -> None:
        # Pre-seed a conflicting versioned file
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        versioned_path.parent.mkdir(parents=True)
        versioned_path.write_text(json.dumps({
            "show_id": "show:1", "model_name": MODEL_KNOWN,
            "embedding_version": "text-v1", "embedding_dimensions": DIMS_KNOWN,
            "embedded_at": "x",
            "episodes": {"ep:conflict": [0.0] * DIMS_KNOWN},
        }), encoding="utf-8")
        _flat(tmp_path, "show:1", episodes={"ep:1": [0.0] * DIMS_KNOWN})

        result = run_migration(tmp_path)

        assert result.counts["conflict_existing_different"] == 1
        assert result.migration_complete is False

    def test_ignores_versioned_subdirectory_files(self, tmp_path: Path) -> None:
        """Already-migrated files under <slug>/ must not be re-processed."""
        _flat(tmp_path, "show:1")
        # First migration lands a versioned file
        run_migration(tmp_path)

        # Second run: only the flat file is scanned; the versioned file
        # under the subdir is not re-processed as an input.
        result = run_migration(tmp_path)
        assert result.total_scanned == 1  # just the flat

    def test_empty_cache_dir_returns_zero_scanned(self, tmp_path: Path) -> None:
        result = run_migration(tmp_path)
        assert result.total_scanned == 0
        assert result.migration_complete is True  # vacuously

    def test_nonexistent_cache_dir_does_not_raise(self, tmp_path: Path) -> None:
        result = run_migration(tmp_path / "does-not-exist")
        assert result.total_scanned == 0


# ── main() exit codes ──────────────────────────────────────────────────────

class TestMainExitCode:
    def test_exit_zero_on_clean_migration(self, tmp_path: Path, monkeypatch) -> None:
        _flat(tmp_path, "show:1")
        report_path = tmp_path / "report.json"

        rc = self._run_main(
            ["--cache-dir", str(tmp_path), "--report", str(report_path)],
            monkeypatch,
        )

        assert rc == 0
        report = json.loads(report_path.read_text())
        assert report["counts"]["migrated_ok"] == 1

    def test_exit_two_on_conflict_existing_different(self, tmp_path: Path, monkeypatch) -> None:
        versioned_path = cache_path_for(tmp_path, _clean_identity(), "show:1")
        versioned_path.parent.mkdir(parents=True)
        versioned_path.write_text(json.dumps({
            "show_id": "show:1", "model_name": MODEL_KNOWN,
            "embedding_version": "text-v1", "embedding_dimensions": DIMS_KNOWN,
            "embedded_at": "x",
            "episodes": {"ep:A": [0.0] * DIMS_KNOWN},
        }), encoding="utf-8")
        _flat(tmp_path, "show:1", episodes={"ep:B": [0.0] * DIMS_KNOWN})
        report_path = tmp_path / "report.json"

        rc = self._run_main(
            ["--cache-dir", str(tmp_path), "--report", str(report_path)],
            monkeypatch,
        )

        assert rc == 2

    def test_dry_run_writes_report_and_leaves_nothing_on_disk(self, tmp_path: Path, monkeypatch) -> None:
        _flat(tmp_path, "show:1")
        report_path = tmp_path / "report.json"

        rc = self._run_main(
            ["--cache-dir", str(tmp_path), "--report", str(report_path), "--dry-run"],
            monkeypatch,
        )

        assert rc == 0
        assert report_path.exists()
        assert not cache_path_for(tmp_path, _clean_identity(), "show:1").exists()

    @staticmethod
    def _run_main(argv: list[str], monkeypatch) -> int:
        """Invoke main() with a synthetic argv vector."""
        monkeypatch.setattr("sys.argv", ["migrate_embeddings_to_versioned.py", *argv])
        return main()


# ── Canonical hash ─────────────────────────────────────────────────────────

class TestCanonicalHash:
    def test_same_episodes_different_key_order_yields_same_hash(self) -> None:
        a = {"show_id": "s", "model_name": MODEL_KNOWN,
             "embedding_version": "text-v1", "embedding_dimensions": DIMS_KNOWN,
             "embedded_at": "x", "episodes": {"ep:1": [1.0], "ep:2": [2.0]}}
        b = {"show_id": "s", "model_name": MODEL_KNOWN,
             "embedding_version": "text-v1", "embedding_dimensions": DIMS_KNOWN,
             "embedded_at": "y", "episodes": {"ep:2": [2.0], "ep:1": [1.0]}}
        assert _canonical_hash(a) == _canonical_hash(b)

    def test_different_episodes_yields_different_hash(self) -> None:
        a = {"show_id": "s", "model_name": "m", "embedding_version": "v",
             "embedding_dimensions": 1, "embedded_at": "x",
             "episodes": {"ep:1": [1.0]}}
        b = {"show_id": "s", "model_name": "m", "embedding_version": "v",
             "embedding_dimensions": 1, "embedded_at": "x",
             "episodes": {"ep:1": [2.0]}}
        assert _canonical_hash(a) != _canonical_hash(b)
