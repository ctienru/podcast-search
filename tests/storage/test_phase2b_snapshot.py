"""Tests for the shared Phase 2b snapshot utility.

Covers the five cases listed in the 2b-A impl doc Step 1a spec:
round-trip write/read, snapshot_type mismatch, malformed JSON, missing
required keys, and an empty `rows` list being legal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage.phase2b_snapshot import (
    SnapshotReadError,
    SnapshotSchemaError,
    read_snapshot,
    validate_schema,
    write_snapshot,
)


class TestRoundTrip:
    def test_write_then_read_preserves_rows_and_type(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        rows = [{"episode_id": "ep:a", "pre": "pending"}, {"episode_id": "ep:b", "pre": None}]
        write_snapshot(path, snapshot_type="backfill_embedding_status", rows=rows)

        data = read_snapshot(path)

        assert data["snapshot_type"] == "backfill_embedding_status"
        assert data["rows"] == rows
        assert "created_at" in data
        validate_schema(data, expected_type="backfill_embedding_status")

    def test_metadata_is_flat_merged(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        write_snapshot(
            path,
            snapshot_type="backfill_embedding_status",
            rows=[],
            metadata={"script_git_sha": "abc123", "total_rows_scanned": 7},
        )
        data = read_snapshot(path)
        assert data["script_git_sha"] == "abc123"
        assert data["total_rows_scanned"] == 7


class TestValidateSchema:
    def test_snapshot_type_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "snap.json"
        write_snapshot(path, snapshot_type="type_a", rows=[])
        data = read_snapshot(path)

        with pytest.raises(SnapshotSchemaError, match="snapshot_type mismatch"):
            validate_schema(data, expected_type="type_b")

    def test_missing_required_key_raises(self) -> None:
        data = {"snapshot_type": "x", "rows": []}  # missing `created_at`
        with pytest.raises(SnapshotSchemaError, match="missing required keys"):
            validate_schema(data, expected_type="x")

    def test_rows_not_list_raises(self) -> None:
        data = {"snapshot_type": "x", "created_at": "2026-04-15T00:00:00Z", "rows": {}}
        with pytest.raises(SnapshotSchemaError, match="rows must be a list"):
            validate_schema(data, expected_type="x")


class TestReadErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotReadError, match="not found"):
            read_snapshot(tmp_path / "does-not-exist.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(SnapshotReadError, match="not valid JSON"):
            read_snapshot(path)

    def test_non_object_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "arr.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(SnapshotReadError, match="must be a JSON object"):
            read_snapshot(path)


class TestEmptyRowsLegal:
    def test_empty_rows_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        write_snapshot(path, snapshot_type="backfill_embedding_status", rows=[])
        data = read_snapshot(path)
        validate_schema(data, expected_type="backfill_embedding_status")
        assert data["rows"] == []


class TestMetadataCollisionRejected:
    def test_metadata_cannot_overwrite_reserved_keys(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotSchemaError, match="reserved top-level keys"):
            write_snapshot(
                tmp_path / "x.json",
                snapshot_type="t",
                rows=[],
                metadata={"snapshot_type": "evil-override"},
            )
