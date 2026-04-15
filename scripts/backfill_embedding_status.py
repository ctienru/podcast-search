"""Backfill `episodes.embedding_status` from cache + DB truth (CT1–CT4).

Step 4 of the 2b-A implementation: classify every `episodes` row against
the four correctness contracts and optionally apply the classification
result back to the DB with snapshot-backed reversibility.

Correctness contracts:

- CT1: row metadata complete enough to form an `EmbeddingIdentity`
- CT2: row identity == path identity == payload identity
- CT3: cache payload JSON-readable
- CT4: payload's `episodes` dict contains the row's `episode_id`

CLI::

    python -m scripts.backfill_embedding_status [--dry-run | --apply]
                                                 [--limit N]
                                                 [--snapshot-dir PATH]
                                                 [--anomaly-threshold-pct P]
                                                 [--json-report PATH]

Exit codes::

    0   Completed (dry-run reported, or apply succeeded).
    1   DB-level error.
    2   Apply blocked: a hard-fail row exists OR
        anomaly_cache_missing_pct exceeds threshold.
    3   Snapshot write failed (pre-UPDATE probe); DB untouched.
    4   Pre-condition failed (`episodes` table missing — Phase 2a
        migration has not run).
    6   Snapshot finalisation failed after DB was mutated. DB is in
        the new state but the on-disk snapshot still carries the
        probe content (post_apply=null). The pre_apply fingerprint
        is still trustworthy for a manual restore-from-backup, but
        the automated `reverse_backfill_embedding_status.py` path
        cannot accept this snapshot without operator intervention.

Audit category taxonomy (per 2b-A impl doc §3.6):

- ``normal``: ``pass``, ``neutral_metadata_absent``
- ``anomaly``: ``anomaly_partial_metadata``, ``anomaly_cache_missing``
- ``hard_fail``: ``fail_payload_unreadable``,
  ``fail_payload_identity_mismatch``, ``fail_episode_entry_missing``

`anomaly_cache_missing_pct` denominator = `metadata_complete_count`
(rows passing CT1).

Snapshot semantics (per §3.3):

The ``--apply`` path writes a snapshot before touching the DB
(``db_fingerprint.pre_apply``), then rewrites it post-UPDATE with
``db_fingerprint.post_apply`` and the rows that actually wrote (rows
skipped by the two-field concurrent-write guard are excluded). The
reverse script uses pre/post fingerprints to decide legality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from sqlite_utils import Database

from src.config import settings
from src.pipelines.embedding_identity import EmbeddingIdentity
from src.pipelines.embedding_identity_adapter import (
    IdentityAdapterError,
    identity_from_payload,
    identity_from_row,
)
from src.pipelines.embedding_paths import cache_path_for
from src.storage.phase2b_snapshot import (
    SnapshotError,
    write_snapshot,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_SAMPLE_LIMIT = 10
_SNAPSHOT_TYPE = "backfill_embedding_status"
_DEFAULT_ANOMALY_THRESHOLD_PCT = 10.0


class Category(str, Enum):
    PASS = "pass"
    NEUTRAL_METADATA_ABSENT = "neutral_metadata_absent"
    ANOMALY_PARTIAL_METADATA = "anomaly_partial_metadata"
    ANOMALY_CACHE_MISSING = "anomaly_cache_missing"
    FAIL_PAYLOAD_UNREADABLE = "fail_payload_unreadable"
    FAIL_PAYLOAD_IDENTITY_MISMATCH = "fail_payload_identity_mismatch"
    FAIL_EPISODE_ENTRY_MISSING = "fail_episode_entry_missing"


class Klass(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    HARD_FAIL = "hard_fail"


_CATEGORY_CLASS: dict[Category, Klass] = {
    Category.PASS: Klass.NORMAL,
    Category.NEUTRAL_METADATA_ABSENT: Klass.NORMAL,
    Category.ANOMALY_PARTIAL_METADATA: Klass.ANOMALY,
    Category.ANOMALY_CACHE_MISSING: Klass.ANOMALY,
    Category.FAIL_PAYLOAD_UNREADABLE: Klass.HARD_FAIL,
    Category.FAIL_PAYLOAD_IDENTITY_MISMATCH: Klass.HARD_FAIL,
    Category.FAIL_EPISODE_ENTRY_MISSING: Klass.HARD_FAIL,
}

# Target `embedding_status` value derived purely from category. Hard-fail
# categories are absent here because apply is blocked before reaching
# any per-row decision when a hard-fail row exists.
_CATEGORY_TARGET: dict[Category, str] = {
    Category.PASS: "done",
    Category.NEUTRAL_METADATA_ABSENT: "pending",
    Category.ANOMALY_PARTIAL_METADATA: "pending",
    Category.ANOMALY_CACHE_MISSING: "pending",
}


@dataclass
class ClassifiedRow:
    row: dict[str, Any]
    category: Category


@dataclass
class ClassificationResult:
    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    anomalies: list[dict[str, str]] = field(default_factory=list)
    total_rows_scanned: int = 0
    # Per-row classification retained so `--apply` can iterate without
    # re-running classify. The memory cost is one dict reference per row
    # (row dicts are already held during classify), which is acceptable
    # at the scale this script runs on.
    rows: list[ClassifiedRow] = field(default_factory=list)

    def record(self, *, row: dict[str, Any], category: Category) -> None:
        self.counts[category.value] += 1
        bucket = self.samples[category.value]
        if len(bucket) < _SAMPLE_LIMIT:
            bucket.append(row["episode_id"])
        if _CATEGORY_CLASS[category] is Klass.ANOMALY:
            self.anomalies.append(
                {
                    "episode_id": row["episode_id"],
                    "show_id": row.get("show_id", ""),
                    "reason": category.value,
                }
            )
        self.rows.append(ClassifiedRow(row=row, category=category))


def _row_is_fully_blank(row: dict[str, Any]) -> bool:
    return (
        (row.get("embedding_model") in (None, ""))
        and (row.get("embedding_version") in (None, ""))
        and (row.get("last_embedded_at") in (None, ""))
    )


def _classify_row(
    row: dict[str, Any],
    cache_dir: Path,
    cache_by_key: dict[tuple[str, EmbeddingIdentity], tuple[str, dict | None]],
) -> Category:
    """Classify a single row against CT1–CT4.

    `cache_by_key` caches the (status, payload) outcome per
    (show_id, identity) so the cache file and its parse are resolved
    at most once regardless of how many episodes a show has.
    """
    # CT1: row metadata complete enough to form an identity.
    try:
        row_identity = identity_from_row(row)
    except IdentityAdapterError:
        if _row_is_fully_blank(row):
            return Category.NEUTRAL_METADATA_ABSENT
        return Category.ANOMALY_PARTIAL_METADATA

    key = (row["show_id"], row_identity)
    if key not in cache_by_key:
        cache_by_key[key] = _resolve_cache(cache_dir, row_identity, row["show_id"])

    status, payload = cache_by_key[key]
    if status == "not_found":
        return Category.ANOMALY_CACHE_MISSING
    if status == "unreadable":
        return Category.FAIL_PAYLOAD_UNREADABLE
    if status in ("payload_bad", "mismatch"):
        return Category.FAIL_PAYLOAD_IDENTITY_MISMATCH

    # CT3 passed; payload is usable. CT4: this row's episode_id must
    # have an entry in the payload's `episodes` dict. A show cache that
    # exists but is missing this row is a partial-rebuild residue.
    episodes = (payload or {}).get("episodes")
    if not isinstance(episodes, dict) or row["episode_id"] not in episodes:
        return Category.FAIL_EPISODE_ENTRY_MISSING

    return Category.PASS


def _resolve_cache(
    cache_dir: Path, identity: EmbeddingIdentity, show_id: str
) -> tuple[str, dict | None]:
    path = cache_path_for(cache_dir, identity, show_id)
    if not path.exists():
        return ("not_found", None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — any parse failure is CT3 unreadable
        return ("unreadable", None)
    try:
        payload_identity = identity_from_payload(payload)
    except IdentityAdapterError:
        return ("payload_bad", payload)
    if payload_identity != identity:
        return ("mismatch", payload)
    return ("ok", payload)


def _iter_rows(db: Database, limit: int | None) -> Iterable[dict[str, Any]]:
    select_cols = (
        "episode_id, show_id, embedding_model, embedding_version, "
        "last_embedded_at, embedding_status"
    )
    sql = f"SELECT {select_cols} FROM episodes"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in db.execute(sql):
        keys = ("episode_id", "show_id", "embedding_model", "embedding_version",
                "last_embedded_at", "embedding_status")
        yield dict(zip(keys, row))


def classify_all(
    db: Database,
    *,
    cache_dir: Path,
    limit: int | None = None,
) -> ClassificationResult:
    result = ClassificationResult()
    cache_by_key: dict[tuple[str, EmbeddingIdentity], tuple[str, dict | None]] = {}
    for row in _iter_rows(db, limit):
        result.total_rows_scanned += 1
        category = _classify_row(row, cache_dir, cache_by_key)
        result.record(row=row, category=category)
    return result


def _metadata_complete_count(counts: Counter) -> int:
    """CT1 denominator: rows whose metadata was complete enough to form
    an identity (i.e. excluding neutral and partial-metadata rows)."""
    return sum(
        v for k, v in counts.items()
        if k not in (
            Category.NEUTRAL_METADATA_ABSENT.value,
            Category.ANOMALY_PARTIAL_METADATA.value,
        )
    )


def _hard_fail_count(counts: Counter) -> int:
    return sum(
        v for k, v in counts.items()
        if _CATEGORY_CLASS[Category(k)] is Klass.HARD_FAIL
    )


def build_report(result: ClassificationResult, *, mode: str) -> dict[str, Any]:
    now = _utc_now_iso()
    counts = dict(result.counts)
    for cat in Category:
        counts.setdefault(cat.value, 0)  # always emit zero rather than omit

    metadata_complete = _metadata_complete_count(result.counts)
    anomaly_missing = counts[Category.ANOMALY_CACHE_MISSING.value]
    pct = (anomaly_missing / metadata_complete) if metadata_complete > 0 else 0.0

    return {
        "mode": mode,
        "ran_at": now,
        "total_rows_scanned": result.total_rows_scanned,
        "counts": counts,
        "samples": {k: list(v) for k, v in result.samples.items()},
        "anomalies": list(result.anomalies),
        "metadata_complete_count": metadata_complete,
        "anomaly_cache_missing_count": anomaly_missing,
        "anomaly_cache_missing_pct": pct,
        "hard_fail_count": _hard_fail_count(result.counts),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"[backfill_embedding_status] mode={report['mode']} at {report['ran_at']}")
    print(f"  total rows scanned: {report['total_rows_scanned']}")
    print(f"  metadata_complete_count: {report['metadata_complete_count']}")
    print(
        f"  anomaly_cache_missing_count: {report['anomaly_cache_missing_count']} "
        f"({report['anomaly_cache_missing_pct']:.2%})"
    )
    print(f"  hard_fail_count: {report['hard_fail_count']}")
    print("  per-category counts:")
    for k in (
        Category.PASS.value,
        Category.NEUTRAL_METADATA_ABSENT.value,
        Category.ANOMALY_PARTIAL_METADATA.value,
        Category.ANOMALY_CACHE_MISSING.value,
        Category.FAIL_PAYLOAD_UNREADABLE.value,
        Category.FAIL_PAYLOAD_IDENTITY_MISMATCH.value,
        Category.FAIL_EPISODE_ENTRY_MISSING.value,
    ):
        print(f"    {k}: {report['counts'].get(k, 0)}")
    if "changed_count" in report:
        print(f"  changed_count: {report['changed_count']}")
        print(f"  unchanged_count: {report['unchanged_count']}")
        print(f"  skipped_concurrent_write_count: {report['skipped_concurrent_write_count']}")
    if report["anomalies"]:
        shown = min(len(report["anomalies"]), _SAMPLE_LIMIT)
        print(f"  anomalies (first {shown}):")
        for entry in report["anomalies"][:shown]:
            print(f"    {entry}")


# ---------------------------------------------------------------------------
# Apply-mode helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_fingerprint(db_path: Path) -> dict[str, Any]:
    """Return sha256 + size + mtime for the DB file. Used to anchor the
    snapshot's pre_apply / post_apply lineage."""
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


def _relative_db_path(db_path: Path) -> str:
    """Best-effort relative path anchored at CWD; falls back to absolute
    string. The reverse script treats `path` difference as a warning, so
    exact canonicalisation isn't load-bearing."""
    try:
        return str(db_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(db_path)


def _git_sha_or_unknown() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if len(out) == 40:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return "unknown"


def _default_snapshot_dir() -> Path:
    """`podcast-search/data/phase2b-snapshots/`, computed relative to this
    script rather than settings.DATA_DIR (which points to the crawler's
    data dir, shared via symlinking but not semantically the right home
    for search-local snapshots)."""
    return Path(__file__).resolve().parents[1] / "data" / "phase2b-snapshots"


@dataclass
class ApplyDecision:
    """One row's apply-time decision. Only rows where `target != current`
    survive to this stage; hard-fail rows never reach here because apply
    is gated out before we iterate."""
    episode_id: str
    show_id: str
    pre_embedding_status: str | None
    target_status: str
    # Read-time markers for the two-field concurrent-write guard (§3.2a).
    read_last_embedded_at: str | None
    read_embedding_status: str | None


def _collect_apply_decisions(
    result: ClassificationResult,
) -> list[ApplyDecision]:
    decisions: list[ApplyDecision] = []
    for cr in result.rows:
        target = _CATEGORY_TARGET.get(cr.category)
        if target is None:
            continue  # hard-fail rows — apply would already be blocked
        current = cr.row.get("embedding_status")
        if current == target:
            continue  # unchanged; reconciliation has nothing to do
        decisions.append(
            ApplyDecision(
                episode_id=cr.row["episode_id"],
                show_id=cr.row.get("show_id", ""),
                pre_embedding_status=current,
                target_status=target,
                read_last_embedded_at=cr.row.get("last_embedded_at"),
                read_embedding_status=current,
            )
        )
    return decisions


def _snapshot_rows(decisions: list[ApplyDecision]) -> list[dict[str, Any]]:
    """Public snapshot shape (§3.3 rows_to_change)."""
    return [
        {
            "episode_id": d.episode_id,
            "show_id": d.show_id,
            "pre_embedding_status": d.pre_embedding_status,
        }
        for d in decisions
    ]


def _apply_updates(
    db: Database,
    decisions: list[ApplyDecision],
) -> tuple[list[ApplyDecision], int]:
    """Execute per-row UPDATE with two-field concurrent-write guard.

    Returns (actually_written_decisions, skipped_concurrent_count).
    Uses `WHERE last_embedded_at IS :x AND embedding_status IS :y` so
    that a concurrent writer who touches either field between our SELECT
    and UPDATE lands us on `rowcount == 0` (skipped, not an error). `IS`
    rather than `=` to let NULL markers compare equal.
    """
    now_iso = _utc_now_iso()
    written: list[ApplyDecision] = []
    skipped = 0
    # Single transaction: any sqlite3 error inside rolls back the batch.
    with db.conn:
        for d in decisions:
            cur = db.conn.execute(
                "UPDATE episodes "
                "SET embedding_status = :target, updated_at = :now "
                "WHERE episode_id = :episode_id "
                "  AND last_embedded_at IS :read_last "
                "  AND embedding_status IS :read_status",
                {
                    "target": d.target_status,
                    "now": now_iso,
                    "episode_id": d.episode_id,
                    "read_last": d.read_last_embedded_at,
                    "read_status": d.read_embedding_status,
                },
            )
            if cur.rowcount == 1:
                written.append(d)
            else:
                skipped += 1
    return written, skipped


def _write_json_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _run_apply(
    db: Database,
    *,
    args: argparse.Namespace,
) -> int:
    result = classify_all(db, cache_dir=args.cache_dir, limit=args.limit)
    report = build_report(result, mode="apply")

    # Gate 1: any hard-fail row blocks apply (§3.7). Never overridable.
    if report["hard_fail_count"] > 0:
        logger.error(
            "apply_blocked_hard_fail",
            extra={"hard_fail_count": report["hard_fail_count"]},
        )
        report["apply_blocked_reason"] = "hard_fail"
        _print_report(report)
        _write_json_report(args.json_report, report)
        return 2

    # Gate 2: anomaly_cache_missing_pct > threshold (§3.7).
    # `anomaly_cache_missing_pct` in the report is a fraction (0–1);
    # `--anomaly-threshold-pct` is an argparse percentage (0–100).
    anomaly_cache_missing_pct_points = report["anomaly_cache_missing_pct"] * 100.0
    if anomaly_cache_missing_pct_points > args.anomaly_threshold_pct:
        logger.error(
            "apply_blocked_anomaly_threshold",
            extra={
                "anomaly_cache_missing_pct": report["anomaly_cache_missing_pct"],
                "threshold_pct": args.anomaly_threshold_pct,
            },
        )
        report["apply_blocked_reason"] = "anomaly_threshold_exceeded"
        _print_report(report)
        _write_json_report(args.json_report, report)
        return 2

    # Decide which rows to touch.
    decisions = _collect_apply_decisions(result)

    # Pre-UPDATE snapshot probe: if we can't write a snapshot, we don't
    # touch the DB. This is the failure mode that exit code 3 covers.
    snapshot_path = args.snapshot_dir / f"backfill-{_utc_now_iso()}.json"
    pre_fp = _compute_fingerprint(args.db_path)
    probe_metadata = {
        "script_git_sha": _git_sha_or_unknown(),
        "db_fingerprint": {
            "path": _relative_db_path(args.db_path),
            "pre_apply": pre_fp,
            "post_apply": None,
        },
        "total_rows_scanned": result.total_rows_scanned,
    }
    try:
        write_snapshot(
            snapshot_path,
            snapshot_type=_SNAPSHOT_TYPE,
            rows=_snapshot_rows(decisions),
            metadata=probe_metadata,
        )
    except (SnapshotError, OSError) as exc:
        logger.error("snapshot_probe_write_failed",
                     extra={"error": repr(exc), "path": str(snapshot_path)})
        report["apply_blocked_reason"] = "snapshot_write_failed"
        _print_report(report)
        _write_json_report(args.json_report, report)
        return 3

    # Execute UPDATEs with two-field guard.
    try:
        written, skipped_concurrent = _apply_updates(db, decisions)
    except sqlite3.Error as exc:
        logger.error("apply_db_error", extra={"error": repr(exc)})
        return 1

    # Post-UPDATE snapshot: rewrite with post_apply fingerprint and the
    # filtered rows that actually wrote. Written atomically through a
    # temp file + os.replace so a mid-write failure cannot leave the
    # snapshot partially overwritten (the probe version stays on disk
    # until the rename completes). If the rename itself fails, the DB
    # is already mutated — exit 6 so operators know the automated
    # reverse path requires manual repair (the pre_apply fingerprint
    # remains trustworthy for restore-from-backup).
    post_fp = _compute_fingerprint(args.db_path)
    final_metadata = {
        "script_git_sha": probe_metadata["script_git_sha"],
        "db_fingerprint": {
            "path": probe_metadata["db_fingerprint"]["path"],
            "pre_apply": pre_fp,
            "post_apply": post_fp,
        },
        "total_rows_scanned": result.total_rows_scanned,
        "changed_count": len(written),
        "skipped_concurrent_write_count": skipped_concurrent,
    }
    tmp_snapshot_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    try:
        write_snapshot(
            tmp_snapshot_path,
            snapshot_type=_SNAPSHOT_TYPE,
            rows=_snapshot_rows(written),
            metadata=final_metadata,
        )
        os.replace(tmp_snapshot_path, snapshot_path)
    except (SnapshotError, OSError) as exc:
        # Clean up the tmp file if it was partially written; best-
        # effort — we don't care if the unlink itself fails, the real
        # signal is the exit 6 plus the stable probe snapshot.
        try:
            tmp_snapshot_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error(
            "snapshot_final_write_failed",
            extra={
                "error": repr(exc),
                "path": str(snapshot_path),
                "db_state": "mutated",
                "recoverable_via": "manual restore from pre_apply snapshot backup",
            },
        )
        report["apply_blocked_reason"] = "snapshot_final_write_failed"
        report["snapshot_path"] = str(snapshot_path)
        _print_report(report)
        _write_json_report(args.json_report, report)
        return 6

    unchanged = result.total_rows_scanned - len(decisions)
    report["changed_count"] = len(written)
    report["unchanged_count"] = unchanged
    report["skipped_concurrent_write_count"] = skipped_concurrent
    report["snapshot_path"] = str(snapshot_path)

    _print_report(report)
    _write_json_report(args.json_report, report)

    logger.info(
        "backfill_apply_complete",
        extra={
            "changed_count": len(written),
            "unchanged_count": unchanged,
            "skipped_concurrent_write_count": skipped_concurrent,
            "snapshot_path": str(snapshot_path),
        },
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify episodes.embedding_status against CT1–CT4 and either "
            "report (--dry-run, default) or reconcile the DB (--apply). "
            "Apply writes a snapshot before touching any row; use the "
            "reverse script bound to that snapshot to roll back."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run",
                      help="Classify and report; no DB writes (default).")
    mode.add_argument("--apply", dest="mode", action="store_const", const="apply",
                      help="Classify, write pre-apply snapshot, then reconcile DB.")
    parser.set_defaults(mode="dry-run")

    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of rows scanned.")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="Write the JSON report to this path.")
    parser.add_argument("--snapshot-dir", type=Path, default=_default_snapshot_dir(),
                        help="Directory for the pre-apply snapshot (apply mode).")
    parser.add_argument(
        "--anomaly-threshold-pct", type=float,
        default=_DEFAULT_ANOMALY_THRESHOLD_PCT,
        help=(
            "Block --apply when anomaly_cache_missing exceeds this percent "
            "of metadata_complete_count (default 10.0). Only applies to "
            "anomaly_cache_missing; hard-fail categories cannot be overridden."
        ),
    )
    parser.add_argument("--db-path", type=Path,
                        default=Path(settings.DATA_DIR) / "podcast.sqlite",
                        help="SQLite DB path.")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(settings.DATA_DIR) / "embeddings",
                        help="Embedding cache root directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()

    try:
        db = Database(str(args.db_path))
    except sqlite3.Error as exc:
        logger.error("db_open_failed", extra={"error": repr(exc)})
        return 1

    if "episodes" not in db.table_names():
        logger.error("precondition_failed_episodes_table_missing",
                     extra={"path": str(args.db_path)})
        return 4

    if args.mode == "apply":
        return _run_apply(db, args=args)

    # Dry-run path.
    try:
        result = classify_all(db, cache_dir=args.cache_dir, limit=args.limit)
    except sqlite3.Error as exc:
        logger.error("classify_db_error", extra={"error": repr(exc)})
        return 1

    report = build_report(result, mode="dry-run")
    _print_report(report)
    _write_json_report(args.json_report, report)

    logger.info(
        "backfill_classify_complete",
        extra={
            "counts": report["counts"],
            "metadata_complete_count": report["metadata_complete_count"],
            "anomaly_cache_missing_count": report["anomaly_cache_missing_count"],
            "hard_fail_count": report["hard_fail_count"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
