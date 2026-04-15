"""Shared snapshot utility for Phase 2b reversible migrations.

Phase 2b backfill / normalization scripts that intend to be reversible
write a row-level pre-run snapshot before mutating the DB. This module is
the minimal IO + schema-validation layer shared by those scripts; the
specific snapshot content (fields beyond the universal ones) is the
caller's responsibility and lives in `metadata`.

Universal snapshot shape (enforced here)::

    {
      "snapshot_type": "<caller-chosen string, e.g. backfill_embedding_status>",
      "created_at":    "<ISO 8601 UTC timestamp>",
      "rows":          [ {..caller-defined row..}, ... ],
      ...caller metadata flat-merged into top level...
    }

`validate_schema` checks `snapshot_type` equality and required-key
presence; callers layer their own schema checks on top (for example the
backfill script's `db_fingerprint` sub-object).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REQUIRED_KEYS: tuple[str, ...] = ("snapshot_type", "created_at", "rows")


class SnapshotError(Exception):
    """Base for all snapshot-related errors."""


class SnapshotSchemaError(SnapshotError):
    """Raised when a snapshot's top-level shape fails validation."""


class SnapshotReadError(SnapshotError):
    """Raised when a snapshot file is missing or not valid JSON."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_snapshot(
    path: Path,
    *,
    snapshot_type: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a snapshot to `path`, creating parent dirs as needed.

    `metadata` is flat-merged into the top-level dict alongside the
    universal keys; a caller collision on any of the universal keys
    raises `SnapshotSchemaError` rather than silently overwriting.
    """
    payload: dict[str, Any] = {
        "snapshot_type": snapshot_type,
        "created_at": _utc_now_iso(),
        "rows": rows,
    }
    if metadata:
        overlapping = set(metadata) & set(_REQUIRED_KEYS)
        if overlapping:
            raise SnapshotSchemaError(
                f"metadata collides with reserved top-level keys: {sorted(overlapping)}"
            )
        payload.update(metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_snapshot(path: Path) -> dict[str, Any]:
    """Read a snapshot JSON file and return the raw dict.

    Does not validate `snapshot_type`; callers wanting a type guarantee
    must call `validate_schema` afterwards.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SnapshotReadError(f"snapshot file not found: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SnapshotReadError(f"snapshot is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise SnapshotReadError(
            f"snapshot root must be a JSON object, got {type(data).__name__}: {path}"
        )
    return data


def validate_schema(data: dict[str, Any], *, expected_type: str) -> None:
    """Raise `SnapshotSchemaError` if the snapshot fails universal checks.

    Checks:
    - all `_REQUIRED_KEYS` present
    - `snapshot_type` matches `expected_type`
    - `rows` is a list
    """
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise SnapshotSchemaError(f"snapshot missing required keys: {missing}")
    if data["snapshot_type"] != expected_type:
        raise SnapshotSchemaError(
            f"snapshot_type mismatch: expected {expected_type!r}, "
            f"got {data['snapshot_type']!r}"
        )
    if not isinstance(data["rows"], list):
        raise SnapshotSchemaError(
            f"snapshot rows must be a list, got {type(data['rows']).__name__}"
        )
