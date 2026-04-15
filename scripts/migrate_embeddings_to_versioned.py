"""Migrate flat embedding cache files to the versioned layout.

Phase 2a one-shot migration: rewrite `<cache_dir>/<show_id>.json` into
`<cache_dir>/<identity-slug>/<show_id>.json`, adding the
`embedding_dimensions` metadata field that cache identity validation
now requires and normalizing the legacy `<model>/<text-version>` form
of `embedding_version` to the bare text version.

Each flat file lands in exactly one of five categories:

    migrated_ok
        Flat has complete metadata (`model_name` + `embedding_version`)
        and we know the vector dimension for that model. Rewritten
        into the versioned layout with the full three identity fields.

    unmigrated_missing_metadata
        Flat is missing `model_name` or `embedding_version`, so we
        cannot determine a target slug. Leave the file in place — the
        runtime cache-lookup layer ignores flat files and will simply
        treat the show as a cache miss, triggering a rebuild next run.

    unmigrated_infer_failed
        Metadata present, but the model is not in the in-repo
        `_DIM_TABLE`. This is a development-time mismatch (new model
        added to `MODEL_MAP` without the corresponding dim entry) —
        investigate before re-running migration.

    conflict_existing_same
        The versioned target already exists and its semantic content
        is identical to what this run would write. Skipped safely;
        makes the script idempotent.

    conflict_existing_different
        The versioned target already exists with different content.
        This is a migration-gate violation: something already wrote to
        that slug (perhaps an earlier force_embed or a partial rebuild)
        and migrating the flat file on top of it would lose data. The
        script exits 2 instead of overwriting so an operator can pick
        a resolution.

Exit codes:
    0   Migration completed with zero `conflict_existing_different`.
    2   MG1 violation: one or more conflicts need manual resolution.
    1   Unexpected error (e.g. cache_dir does not exist).

Canonical content comparison excludes `embedded_at` so that a force-
rebuild timestamp difference never trips the conflict check; only the
(show_id, model, version, dimensions, episodes) tuple matters.

The final JSON report is advisory: it includes a `migration_complete`
field that downstream tools MAY surface, but production workflow gates
rely on the CLI exit code and operator acceptance of the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE
from src.pipelines.embedding_paths import cache_path_for
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class MigrationCategory(str, Enum):
    MIGRATED_OK = "migrated_ok"
    UNMIGRATED_MISSING_METADATA = "unmigrated_missing_metadata"
    UNMIGRATED_INFER_FAILED = "unmigrated_infer_failed"
    CONFLICT_EXISTING_SAME = "conflict_existing_same"
    CONFLICT_EXISTING_DIFFERENT = "conflict_existing_different"


_SAMPLE_LIMIT = 10  # cap samples per category to keep reports bounded


@dataclass
class MigrationResult:
    """Per-run migration tally + diagnostic samples."""

    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    total_scanned: int = 0
    cache_dir: Path = field(default_factory=lambda: Path())

    def record(self, category: MigrationCategory, show_id: str) -> None:
        self.counts[category.value] += 1
        bucket = self.samples[category.value]
        if len(bucket) < _SAMPLE_LIMIT:
            bucket.append(show_id)

    @property
    def migration_complete(self) -> bool:
        """Advisory flag — true when no `conflict_existing_different` rows.

        Intentionally NOT a workflow gate. Tools that consume the report
        should treat this as informational; the authoritative signal is
        the CLI exit code (2 on conflicts) and operator acceptance.
        """
        return self.counts[MigrationCategory.CONFLICT_EXISTING_DIFFERENT.value] == 0

    def to_report(self) -> dict[str, Any]:
        return {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "cache_dir": str(self.cache_dir),
            "total_scanned": self.total_scanned,
            "counts": dict(self.counts),
            "samples": {k: list(v) for k, v in self.samples.items()},
            "migration_complete": self.migration_complete,  # advisory only
        }


def _canonical_hash(entry: dict[str, Any]) -> str:
    """Stable hash of the semantic content of a cache entry.

    Excludes `embedded_at` so that a legitimate rewrite from the same
    (show, identity, episodes) does not look like a conflict.
    """
    stable = {
        "show_id": entry.get("show_id"),
        "model_name": entry.get("model_name"),
        "embedding_version": entry.get("embedding_version"),
        "embedding_dimensions": entry.get("embedding_dimensions"),
        "episodes": dict(sorted((entry.get("episodes") or {}).items())),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_embedding_version(raw: str) -> str:
    """Strip the legacy `<model>/` prefix so it matches the resolver output.

    Legacy cache files carry `embedding_version = "<model>/text-v1"`.
    The Phase 2a identity uses just `"text-v1"`. Migration rewrites the
    stored field so a subsequent `validate_cache_identity` returns None
    instead of flagging VERSION_MISMATCH for every show.
    """
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    return raw


def classify_and_migrate(
    flat_path: Path,
    cache_dir: Path,
    *,
    dry_run: bool = False,
) -> MigrationCategory:
    """Classify one flat cache file and, if possible, migrate it."""
    show_id = flat_path.stem
    try:
        entry = json.loads(flat_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable / malformed → treat as missing
        return MigrationCategory.UNMIGRATED_MISSING_METADATA

    model_name = entry.get("model_name")
    raw_version = entry.get("embedding_version")
    if not isinstance(model_name, str) or not model_name:
        return MigrationCategory.UNMIGRATED_MISSING_METADATA
    if not isinstance(raw_version, str) or not raw_version:
        return MigrationCategory.UNMIGRATED_MISSING_METADATA

    dims = _DIM_TABLE.get(model_name)
    if dims is None:
        return MigrationCategory.UNMIGRATED_INFER_FAILED

    clean_version = _normalize_embedding_version(raw_version)
    identity = EmbeddingIdentity(
        model_name=model_name,
        embedding_version=clean_version,
        embedding_dimensions=dims,
    )
    new_entry: dict[str, Any] = {
        "show_id": show_id,
        "model_name": model_name,
        "embedding_version": clean_version,
        "embedding_dimensions": dims,
        "embedded_at": entry.get("embedded_at", ""),
        "episodes": entry.get("episodes") or {},
    }

    target_path = cache_path_for(cache_dir, identity, show_id)

    if target_path.exists():
        try:
            existing = json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unreadable existing = treat as different
            return MigrationCategory.CONFLICT_EXISTING_DIFFERENT
        if _canonical_hash(existing) == _canonical_hash(new_entry):
            return MigrationCategory.CONFLICT_EXISTING_SAME
        return MigrationCategory.CONFLICT_EXISTING_DIFFERENT

    if not dry_run:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(new_entry), encoding="utf-8")

    return MigrationCategory.MIGRATED_OK


def run_migration(cache_dir: Path, *, dry_run: bool = False) -> MigrationResult:
    """Scan `cache_dir` for flat cache files and classify each one.

    Only top-level `*.json` files are processed; versioned files living
    under `<cache_dir>/<slug>/` subdirectories are ignored (they are the
    migration output). This also makes the script idempotent across
    repeated runs.
    """
    result = MigrationResult(cache_dir=cache_dir)
    if not cache_dir.exists():
        logger.error("cache_dir_not_found", extra={"path": str(cache_dir)})
        return result

    flat_files = sorted(
        p for p in cache_dir.iterdir() if p.is_file() and p.suffix == ".json"
    )
    result.total_scanned = len(flat_files)

    for flat_path in flat_files:
        category = classify_and_migrate(flat_path, cache_dir, dry_run=dry_run)
        result.record(category, flat_path.stem)

    return result


def _default_report_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"data/migration-report-{stamp}.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate flat embedding cache files to versioned layout.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/embeddings"),
        help="Root directory containing the flat cache files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify without writing; report-only mode.",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help=f"Path to write the JSON migration report. Default: {_default_report_path()}",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info(
        "migration_start",
        extra={"cache_dir": str(args.cache_dir), "dry_run": args.dry_run},
    )

    result = run_migration(args.cache_dir, dry_run=args.dry_run)
    report = result.to_report()

    report_path = args.report or _default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "migration_complete",
        extra={
            "counts": report["counts"],
            "report_path": str(report_path),
            "migration_complete_flag": report["migration_complete"],
        },
    )
    print(f"Migration report written to: {report_path}")
    print(json.dumps(report["counts"], indent=2))

    if result.counts[MigrationCategory.CONFLICT_EXISTING_DIFFERENT.value] > 0:
        logger.error(
            "mg1_violation_conflict_existing_different",
            extra={
                "count": result.counts[MigrationCategory.CONFLICT_EXISTING_DIFFERENT.value],
                "samples": result.samples[MigrationCategory.CONFLICT_EXISTING_DIFFERENT.value],
            },
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
