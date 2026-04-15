"""Backfill `episodes.embedding_status` from cache + DB truth (CT1–CT3).

This is Step 3 of the 2b-A implementation: classify every `episodes`
row against three of the four correctness contracts (CT1 row metadata
complete, CT2 path + payload identity match, CT3 payload readable) and
report the result. `--apply` is deliberately absent at this step; it is
added in Step 4 alongside CT4 (episode-entry coverage), snapshot, and
the per-class apply policy.

CLI::

    python -m scripts.backfill_embedding_status --dry-run
                                                 [--limit N]
                                                 [--json-report PATH]

Exit codes at this step::

    0   Dry-run completed, report written.
    1   DB-level error.
    2   Pre-condition failed (`episodes` table missing — Phase 2a
        migration has not run).
    4   Step 0 blocking verification failed (kept for parity with the
        Step 4 CLI surface; reserved, not used yet).

The audit category taxonomy (per 2b-A impl doc §3.6) is machine-
readable in the JSON report and grouped by `Class`:

- ``normal``: ``pass``, ``neutral_metadata_absent``
- ``anomaly``: ``anomaly_cache_missing`` (DB metadata present but
  versioned cache file absent — data inconsistency, not pre-embed)
- ``hard_fail``: ``fail_payload_unreadable``,
  ``fail_payload_identity_mismatch`` (``fail_episode_entry_missing``
  lands in Step 4 when CT4 is implemented)

`anomaly_cache_missing_pct` is reported with the denominator
``metadata_complete_count`` (rows passing CT1), per §3.7: using the
universe of rows whose metadata *claims* they were embedded is the only
denominator that reflects "of the rows that should have a cache, how
many are missing one".
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
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
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_SAMPLE_LIMIT = 10


class Category(str, Enum):
    PASS = "pass"
    NEUTRAL_METADATA_ABSENT = "neutral_metadata_absent"
    ANOMALY_CACHE_MISSING = "anomaly_cache_missing"
    FAIL_PAYLOAD_UNREADABLE = "fail_payload_unreadable"
    FAIL_PAYLOAD_IDENTITY_MISMATCH = "fail_payload_identity_mismatch"


class Klass(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    HARD_FAIL = "hard_fail"


_CATEGORY_CLASS: dict[Category, Klass] = {
    Category.PASS: Klass.NORMAL,
    Category.NEUTRAL_METADATA_ABSENT: Klass.NORMAL,
    Category.ANOMALY_CACHE_MISSING: Klass.ANOMALY,
    Category.FAIL_PAYLOAD_UNREADABLE: Klass.HARD_FAIL,
    Category.FAIL_PAYLOAD_IDENTITY_MISMATCH: Klass.HARD_FAIL,
}


@dataclass
class Classification:
    category: Category
    # The identity we succeeded in building from the row, when applicable.
    row_identity: EmbeddingIdentity | None = None


@dataclass
class ClassificationResult:
    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    anomalies: list[dict[str, str]] = field(default_factory=list)
    total_rows_scanned: int = 0

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
    """Classify a single row against CT1–CT3.

    `cache_by_key` caches the (status, payload) outcome per
    (show_id, identity) so the cache file and its parse are resolved
    at most once regardless of how many episodes a show has.
    """
    # CT1: row metadata complete enough to form an identity.
    try:
        row_identity = identity_from_row(row)
    except IdentityAdapterError:
        # Fully blank → plain pre-embed neutral; a mix of present-and-
        # absent lands here too. Partial metadata is rare in practice and
        # treated the same way: write 'pending' and leave the row for a
        # re-embed on the next run.
        if _row_is_fully_blank(row):
            return Category.NEUTRAL_METADATA_ABSENT
        return Category.NEUTRAL_METADATA_ABSENT

    key = (row["show_id"], row_identity)
    if key not in cache_by_key:
        cache_by_key[key] = _resolve_cache(cache_dir, row_identity, row["show_id"])

    status, _payload = cache_by_key[key]
    if status == "not_found":
        return Category.ANOMALY_CACHE_MISSING
    if status == "unreadable":
        return Category.FAIL_PAYLOAD_UNREADABLE
    if status in ("payload_bad", "mismatch"):
        return Category.FAIL_PAYLOAD_IDENTITY_MISMATCH
    # status == "ok" → CT1–CT3 all satisfied. CT4 is Step 4's job.
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
    """CT1 denominator: rows where metadata was complete enough to form
    an identity (i.e. everything except `neutral_metadata_absent`)."""
    return sum(
        v for k, v in counts.items() if k != Category.NEUTRAL_METADATA_ABSENT.value
    )


def _hard_fail_count(counts: Counter) -> int:
    return sum(
        v for k, v in counts.items()
        if _CATEGORY_CLASS[Category(k)] is Klass.HARD_FAIL
    )


def build_report(result: ClassificationResult, *, mode: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
        Category.ANOMALY_CACHE_MISSING.value,
        Category.FAIL_PAYLOAD_UNREADABLE.value,
        Category.FAIL_PAYLOAD_IDENTITY_MISMATCH.value,
    ):
        print(f"    {k}: {report['counts'].get(k, 0)}")
    if report["anomalies"]:
        shown = min(len(report["anomalies"]), _SAMPLE_LIMIT)
        print(f"  anomalies (first {shown}):")
        for entry in report["anomalies"][:shown]:
            print(f"    {entry}")


_APPLY_UNAVAILABLE_HELP = (
    "apply mode unavailable until episode coverage validation (CT4), "
    "snapshot, and per-class apply policy are implemented in Step 4"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify episodes.embedding_status against CT1–CT3 and report. "
            f"{_APPLY_UNAVAILABLE_HELP}."
        ),
    )
    # Only --dry-run at this step; --apply is intentionally not declared so
    # argparse rejects it with "unrecognized argument".
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Report only; no DB writes (currently the only supported mode).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of rows scanned.")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="Write the JSON report to this path.")
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
        return 2

    try:
        result = classify_all(db, cache_dir=args.cache_dir, limit=args.limit)
    except sqlite3.Error as exc:
        logger.error("classify_db_error", extra={"error": repr(exc)})
        return 1

    report = build_report(result, mode="dry-run")

    _print_report(report)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

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
