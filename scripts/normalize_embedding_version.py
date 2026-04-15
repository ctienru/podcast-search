"""Normalize legacy `embedding_version` values in `episodes`.

Phase 2a cache files previously stored `embedding_version` as
``<model>/text-v1`` while the in-repo identity resolver produces the
bare ``text-v1``. The one-shot cache migration already rewrote the on-
disk payloads; this script closes the loop on the DB side by rewriting
any ``episodes.embedding_version`` row that still carries the legacy
``<model>/`` prefix. Running it is idempotent — rows already in the
clean form are untouched and counted as ``skipped``.

CLI::

    python -m scripts.normalize_embedding_version [--dry-run | --apply]
                                                   [--limit N]
                                                   [--json-report PATH]

Exit codes (2b-A impl doc §3.1)::

    0   Normalization succeeded (or dry-run completed).
    1   DB-level error (connection failed, UPDATE blew up).
    2   Pre-condition failed: `episodes` table missing — Phase 2a
        migration has not run, so there is nothing to normalize
        against.

The JSON report and the human-readable stdout carry the same counts
(`before_distribution`, `updated_count`, `skipped_count`, and an
`after_distribution` computed post-run so dry-run reports what the
state *would* look like).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlite_utils import Database

from src.config import settings
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_LEGACY_SEPARATOR = "/"


def _clean(raw: str) -> str:
    return raw.rsplit(_LEGACY_SEPARATOR, 1)[-1] if _LEGACY_SEPARATOR in raw else raw


def _version_distribution(db: Database) -> dict[str, int]:
    counts = Counter()
    for row in db.execute(
        "SELECT embedding_version, COUNT(*) FROM episodes "
        "WHERE embedding_version IS NOT NULL "
        "GROUP BY embedding_version"
    ):
        counts[row[0]] = row[1]
    return dict(counts)


def normalize(
    db: Database,
    *,
    dry_run: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the normalization and return a structured result dict.

    The caller decides whether to persist or print. The return dict is
    also the canonical JSON report shape.
    """
    before = _version_distribution(db)

    select_sql = (
        "SELECT episode_id, embedding_version FROM episodes "
        "WHERE embedding_version IS NOT NULL AND embedding_version LIKE '%/%'"
    )
    if limit is not None:
        select_sql += f" LIMIT {int(limit)}"

    candidates: list[tuple[str, str]] = [
        (row[0], row[1]) for row in db.execute(select_sql)
    ]

    updated_count = 0
    skipped_count = 0
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if not dry_run and candidates:
        conn = db.conn
        with conn:
            for episode_id, raw in candidates:
                clean_value = _clean(raw)
                if clean_value == raw:
                    skipped_count += 1
                    continue
                cur = conn.execute(
                    "UPDATE episodes SET embedding_version = ?, updated_at = ? "
                    "WHERE episode_id = ? AND embedding_version = ?",
                    (clean_value, now, episode_id, raw),
                )
                if cur.rowcount == 1:
                    updated_count += 1
                else:
                    # Row value changed between SELECT and UPDATE; don't
                    # touch it — operator can re-run for a fresh scan.
                    skipped_count += 1
    else:
        for _, raw in candidates:
            if _clean(raw) == raw:
                skipped_count += 1
            else:
                updated_count += 1

    after = _version_distribution(db) if not dry_run else _projected_after(before, candidates)

    return {
        "mode": "dry-run" if dry_run else "apply",
        "ran_at": now,
        "limit": limit,
        "before_distribution": before,
        "after_distribution": after,
        "candidate_count": len(candidates),
        "updated_count": updated_count,
        "skipped_count": skipped_count,
    }


def _projected_after(before: dict[str, int], candidates: list[tuple[str, str]]) -> dict[str, int]:
    """Project the post-apply distribution for dry-run reporting.

    Reflects the exact changes that would be made to the candidates,
    so that `--limit` causes a consistent partial projection.
    """
    projected = dict(before)
    for _, raw in candidates:
        clean_value = _clean(raw)
        if clean_value != raw:
            projected[raw] -= 1
            projected[clean_value] = projected.get(clean_value, 0) + 1
            if projected[raw] == 0:
                del projected[raw]
    return projected


def _print_report(report: dict[str, Any]) -> None:
    print(f"[normalize_embedding_version] mode={report['mode']} at {report['ran_at']}")
    print(f"  candidates matched (LIKE '%/%'): {report['candidate_count']}")
    print(f"  updated: {report['updated_count']}   skipped: {report['skipped_count']}")
    if report["limit"] is not None:
        print(f"  --limit applied: {report['limit']}")
    print("  before distribution:")
    for k, v in sorted(report["before_distribution"].items()):
        print(f"    {k}: {v}")
    print("  after distribution:")
    for k, v in sorted(report["after_distribution"].items()):
        print(f"    {k}: {v}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize legacy <model>/text-v1 in episodes.embedding_version.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Report only; no DB writes (default).")
    mode.add_argument("--apply", action="store_true", help="Apply normalization.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of candidate rows scanned in this run.",
    )
    parser.add_argument(
        "--json-report", type=Path, default=None,
        help="Write the structured report to this JSON path.",
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path(settings.DATA_DIR) / "podcast.sqlite",
        help="SQLite DB path (default: settings.DATA_DIR/podcast.sqlite).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()

    dry_run = not args.apply  # default when neither is set

    try:
        db = Database(str(args.db_path))
    except sqlite3.Error as exc:
        logger.error("db_open_failed", extra={"path": str(args.db_path), "error": repr(exc)})
        return 1

    if "episodes" not in db.table_names():
        logger.error(
            "precondition_failed_episodes_table_missing",
            extra={"path": str(args.db_path)},
        )
        return 2

    try:
        report = normalize(db, dry_run=dry_run, limit=args.limit)
    except sqlite3.Error as exc:
        logger.error("normalize_db_error", extra={"error": repr(exc)})
        return 1

    _print_report(report)

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "normalize_complete",
        extra={
            "mode": report["mode"],
            "updated_count": report["updated_count"],
            "skipped_count": report["skipped_count"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
