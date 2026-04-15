"""Reverse a prior `backfill_embedding_status --apply` using its snapshot.

Step 5 of the 2b-A implementation. Reads the snapshot file produced by
the apply run and restores each row's `embedding_status` to the
`pre_embedding_status` recorded at apply time. Uses the snapshot's
`db_fingerprint.{pre_apply, post_apply}` to decide legality:

- current DB == `post_apply` → normal reverse path, execute
- current DB == `pre_apply`  → no-op (DB already at pre-apply state),
                               exit 0 with log
- neither                    → drift; exit 5 unless
                               `--allow-fingerprint-drift` is passed
                               along with a reason (env var
                               `PHASE2B_REVERSE_DRIFT_REASON` or TTY
                               prompt)

Path mismatch (sha256 + size still match `post_apply` but the recorded
relative path differs) is a warning, not a blocker — CI runners and
moved checkouts shouldn't fail reverse.

CLI::

    python -m scripts.reverse_backfill_embedding_status
        --snapshot PATH
        [--dry-run | --apply]
        [--allow-fingerprint-drift]
        [--json-report PATH]
        [--db-path PATH]

Exit codes::

    0   OK (reverse applied, or DB already at pre_apply — no-op)
    1   DB-level error
    2   Snapshot missing / unreadable
    3   Snapshot schema invalid (wrong snapshot_type or missing
        required keys — including `db_fingerprint` shape)
    4   Post-reverse verification failed (some row's DB value after
        reverse doesn't match its snapshot `pre_embedding_status`)
    5   Fingerprint drift (DB != post_apply AND != pre_apply) without
        `--allow-fingerprint-drift`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlite_utils import Database

from src.config import settings
from src.storage.phase2b_snapshot import (
    SnapshotError,
    SnapshotReadError,
    SnapshotSchemaError,
    read_snapshot,
    validate_schema,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_SNAPSHOT_TYPE = "backfill_embedding_status"
_DRIFT_REASON_ENV = "PHASE2B_REVERSE_DRIFT_REASON"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_fingerprint(db_path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with db_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    stat = db_path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "file_sha256": h.hexdigest(),
        "file_size_bytes": stat.st_size,
        "mtime": mtime_iso,
    }


def _fingerprint_matches(current: dict[str, Any], target: dict[str, Any]) -> bool:
    """Lineage match uses sha256 + size only; mtime is diagnostic."""
    return (
        current.get("file_sha256") == target.get("file_sha256")
        and current.get("file_size_bytes") == target.get("file_size_bytes")
    )


def _validate_fingerprint_shape(fp: dict[str, Any]) -> None:
    """Raise SnapshotSchemaError if db_fingerprint doesn't carry the
    required shape. The universal snapshot validator in
    phase2b_snapshot.py only guards the top-level; this is the
    backfill-specific layer."""
    if not isinstance(fp, dict):
        raise SnapshotSchemaError("db_fingerprint missing or not an object")
    for side in ("pre_apply", "post_apply"):
        sub = fp.get(side)
        if not isinstance(sub, dict):
            raise SnapshotSchemaError(f"db_fingerprint.{side} missing")
        for key in ("file_sha256", "file_size_bytes"):
            if key not in sub:
                raise SnapshotSchemaError(f"db_fingerprint.{side}.{key} missing")


def _validate_rows_shape(rows: Any) -> None:
    """Per-row schema guard so a malformed snapshot exits 3 instead of
    crashing with KeyError inside `_reverse_rows`. Only `episode_id` is
    mandatory — `pre_embedding_status` may legitimately be absent (it
    then round-trips as NULL)."""
    if not isinstance(rows, list):
        raise SnapshotSchemaError(f"rows must be a list, got {type(rows).__name__}")
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            raise SnapshotSchemaError(
                f"rows[{idx}] must be a dict, got {type(r).__name__}"
            )
        if "episode_id" not in r:
            raise SnapshotSchemaError(f"rows[{idx}] missing episode_id")


def _relative_db_path(db_path: Path) -> str:
    try:
        return str(db_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(db_path)


def _resolve_drift_reason() -> str | None:
    """Operator must supply a reason when using --allow-fingerprint-drift.

    Source preference: env var (scriptable + test-friendly) → TTY prompt.
    If neither available, returns None and the caller should fail.
    """
    env_reason = os.environ.get(_DRIFT_REASON_ENV, "").strip()
    if env_reason:
        return env_reason
    if sys.stdin.isatty():
        try:
            reason = input("Reason for fingerprint drift override: ").strip()
        except EOFError:
            return None
        return reason or None
    return None


def _reverse_rows(
    db: Database,
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> int:
    """Restore each row's embedding_status to its snapshot pre value.

    Returns the number of rows the UPDATE actually affected. Unlike
    apply, reverse does NOT guard on read markers — the snapshot is the
    source of truth for what should exist after reverse, and concurrent
    modification is surfaced by the post-verify step (exit 4).
    """
    if dry_run:
        return 0
    now_iso = _utc_now_iso()
    affected = 0
    with db.conn:
        for r in rows:
            cur = db.conn.execute(
                "UPDATE episodes "
                "SET embedding_status = :pre, updated_at = :now "
                "WHERE episode_id = :episode_id",
                {
                    "pre": r.get("pre_embedding_status"),
                    "now": now_iso,
                    "episode_id": r["episode_id"],
                },
            )
            affected += cur.rowcount or 0
    return affected


def _verify_rows(
    db: Database,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a list of mismatches: for each snapshot row, assert the
    DB value equals `pre_embedding_status`. Rows absent from the DB are
    flagged with a dedicated `row_absent=True` field so operators can
    tell "row vanished from DB" apart from "DB value is NULL" — both
    would otherwise land as `actual: null` in the JSON report."""
    mismatches: list[dict[str, Any]] = []
    for r in rows:
        cursor = db.conn.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id = ?",
            (r["episode_id"],),
        )
        found = cursor.fetchone()
        row_absent = found is None
        actual = None if row_absent else found[0]
        expected = r.get("pre_embedding_status")
        if row_absent or actual != expected:
            mismatches.append({
                "episode_id": r["episode_id"],
                "expected": expected,
                "actual": actual,
                "row_absent": row_absent,
            })
    return mismatches


def _write_json_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse a backfill_embedding_status apply by restoring each "
            "row's embedding_status to its snapshot pre value."
        ),
    )
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Path to the snapshot produced by apply.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run",
                      help="Verify snapshot + fingerprint; no DB writes.")
    mode.add_argument("--apply", dest="mode", action="store_const", const="apply",
                      help="Execute the reverse (default).")
    parser.set_defaults(mode="apply")

    parser.add_argument("--allow-fingerprint-drift", action="store_true",
                        help="Permit execution when DB fingerprint matches "
                             "neither pre_apply nor post_apply. Reason "
                             "must be supplied via env var "
                             f"{_DRIFT_REASON_ENV} or TTY prompt.")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="Write a JSON summary of the reverse run.")
    parser.add_argument("--db-path", type=Path,
                        default=Path(settings.DATA_DIR) / "podcast.sqlite",
                        help="SQLite DB path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()

    # --- Load and validate snapshot ---
    try:
        data = read_snapshot(args.snapshot)
    except SnapshotReadError as exc:
        logger.error("snapshot_unreadable",
                     extra={"path": str(args.snapshot), "error": repr(exc)})
        return 2
    try:
        validate_schema(data, expected_type=_SNAPSHOT_TYPE)
    except SnapshotSchemaError as exc:
        logger.error("snapshot_schema_invalid",
                     extra={"path": str(args.snapshot), "error": repr(exc)})
        return 3

    fp = data.get("db_fingerprint")
    try:
        _validate_fingerprint_shape(fp)
    except SnapshotSchemaError as exc:
        logger.error("snapshot_fingerprint_invalid",
                     extra={"path": str(args.snapshot), "error": repr(exc)})
        return 3

    try:
        _validate_rows_shape(data.get("rows"))
    except SnapshotSchemaError as exc:
        logger.error("snapshot_rows_invalid",
                     extra={"path": str(args.snapshot), "error": repr(exc)})
        return 3

    # --- Open DB ---
    try:
        db = Database(str(args.db_path))
    except sqlite3.Error as exc:
        logger.error("db_open_failed", extra={"error": repr(exc)})
        return 1

    # --- Fingerprint gate ---
    current_fp = _compute_fingerprint(args.db_path)
    post_apply_fp = fp["post_apply"]
    pre_apply_fp = fp["pre_apply"]

    # No-op: DB already in pre-apply state (prior reverse, restore from
    # backup, etc.). Reverse has nothing to do; exit 0.
    if _fingerprint_matches(current_fp, pre_apply_fp):
        logger.info("reverse_noop_db_already_at_pre_apply",
                    extra={"db_path": str(args.db_path)})
        return 0

    # Path mismatch is warning-only; lineage rests on sha256 + size.
    current_rel = _relative_db_path(args.db_path)
    if current_rel != fp.get("path"):
        logger.warning(
            "reverse_db_path_differs_from_snapshot",
            extra={"current_path": current_rel, "snapshot_path": fp.get("path")},
        )

    drift_reason: str | None = None
    if not _fingerprint_matches(current_fp, post_apply_fp):
        if not args.allow_fingerprint_drift:
            logger.error(
                "reverse_fingerprint_drift",
                extra={
                    "current_sha256": current_fp["file_sha256"],
                    "post_apply_sha256": post_apply_fp.get("file_sha256"),
                    "pre_apply_sha256": pre_apply_fp.get("file_sha256"),
                },
            )
            return 5
        drift_reason = _resolve_drift_reason()
        if not drift_reason:
            logger.error("reverse_drift_override_requires_reason",
                         extra={"env_var": _DRIFT_REASON_ENV})
            return 5
        logger.warning("reverse_fingerprint_drift_overridden",
                       extra={"reason": drift_reason})

    rows = data["rows"]

    # --- Dry-run short-circuit ---
    if args.mode == "dry-run":
        logger.info("reverse_dry_run_would_restore",
                    extra={"row_count": len(rows)})
        _write_json_report(args.json_report, {
            "mode": "dry-run",
            "ran_at": _utc_now_iso(),
            "snapshot": str(args.snapshot),
            "rows_to_restore": len(rows),
            "drift_reason": drift_reason,
        })
        return 0

    # --- Execute reverse ---
    try:
        _reverse_rows(db, rows, dry_run=False)
    except sqlite3.Error as exc:
        logger.error("reverse_db_error", extra={"error": repr(exc)})
        return 1

    # --- Post-verify ---
    mismatches = _verify_rows(db, rows)
    report: dict[str, Any] = {
        "mode": "apply",
        "ran_at": _utc_now_iso(),
        "snapshot": str(args.snapshot),
        "rows_restored": len(rows) - len(mismatches),
        "mismatches": mismatches,
        "drift_reason": drift_reason,
    }
    if mismatches:
        logger.error("reverse_verify_mismatch",
                     extra={"mismatch_count": len(mismatches)})
        _write_json_report(args.json_report, report)
        return 4

    logger.info("reverse_complete",
                extra={"row_count": len(rows)})
    _write_json_report(args.json_report, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
