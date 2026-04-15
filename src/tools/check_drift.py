"""Read-only drift detection + sync_state observability tool.

This tool is deliberately `report-only` in Phase 2a — it never changes
exit code based on whether drift exists. An operator or CI runs it to
understand the current state; a later phase promotes it into a hard
gate when the truth-source decision (embedding_status backfill vs
retire) has been made. Phase 5 will also wrap this behind a
`triage.sh model-drift` subcommand.

Three pieces, chosen so each one can be consumed in isolation:

  check_episode_drift(db, *, identity_by_language)
      Scan every `episodes` row that carries embedding metadata, compare
      each row's stored model/version against the expected identity for
      that episode's show (via the show's target_index). Produce a
      tally per DriftKind — not a rebuild plan, just a triage view.

  summarize_sync_state_distribution(db)
      Group `search_sync_state` rows by (environment, embedding_model,
      embedding_version) and report counts. This surfaces what each
      environment currently believes is synced so the operator can
      spot cross-env divergence.

  ShowImpactSummary
      Translates episode-level drift into the dual-track view the
      Phase 2a Artifact Scope Contract uses: episode truth line
      (per-episode counts) vs show artifact line (per-show counts
      + a deterministic ascending / deduped list of affected show ids).

The three pieces are composed by the `main()` CLI into a single report
that can be printed as a human-readable summary or emitted as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from sqlite_utils import Database

from src.config import settings
from src.pipelines.embedding_catalog import MODEL_MAP
from src.pipelines.embedding_identity import EmbeddingIdentity, resolve_expected_identity
from src.pipelines.embedding_paths import DriftKind
from src.types import Language
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


_TARGET_INDEX_TO_LANGUAGE: dict[str, Language] = {
    "podcast-episodes-zh-tw": "zh-tw",
    "podcast-episodes-zh-cn": "zh-cn",
    "podcast-episodes-en": "en",
}


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeDriftReport:
    """Aggregate drift view at the episode-row level."""

    # Total episode rows inspected (only rows with embedding metadata set).
    episodes_with_metadata: int
    # Rows whose row-level identity matches the expected identity for
    # their language — broken out for a sanity-check sum.
    ok_count: int
    # Counts per drift kind (MODEL_MISMATCH / VERSION_MISMATCH / ...).
    drift_counts: dict[str, int]
    # Rows whose show could not be mapped to a language (e.g. unknown
    # target_index). Reported separately so a drift value is not
    # conflated with "unclassifiable".
    unresolvable_language_count: int


@dataclass(frozen=True)
class SyncStateDistribution:
    """Row counts grouped by (environment, embedding_model, embedding_version).

    Sorted lexicographically on the grouping tuple so two runs produce
    byte-identical JSON output.
    """

    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ShowImpactSummary:
    """Dual-track view over the same underlying drift.

    episode_truth_line_count: total distinct episodes drifting.
    show_artifact_line_count: total distinct shows with at least one
                              drifting episode.
    affected_show_ids: deterministic (ascending, dedup'd) list of the
                       show ids that contain at least one drifting
                       episode. Useful for feeding the rebuild tool.
    top_shows_by_drift: up to `_TOP_N` (show_id, drifting_episode_count)
                        pairs, sorted by count descending then id
                        ascending.
    """

    episode_truth_line_count: int
    show_artifact_line_count: int
    affected_show_ids: list[str]
    top_shows_by_drift: list[tuple[str, int]]


_TOP_N = 10


@dataclass(frozen=True)
class DriftReport:
    """Full CLI report — composition of the three pieces + current expected identity."""

    expected_identity_by_language: dict[str, dict[str, Any]]
    episode_drift: EpisodeDriftReport
    show_impact: ShowImpactSummary
    sync_state_distribution: SyncStateDistribution


# ── Core computation ────────────────────────────────────────────────────────


def _normalize_version(raw: Optional[str]) -> Optional[str]:
    """Accept both cleaned (`text-v1`) and legacy (`<model>/text-v1`) forms.

    A pre-Phase-2a DB row will carry the legacy form; Phase 2a writes
    the cleaned form. Comparisons in this tool strip the legacy prefix
    so a drift report does not light up just because of format drift.
    """
    if raw is None:
        return None
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    return raw


def _resolve_language_for_target_index(target_index: Optional[str]) -> Optional[Language]:
    if not target_index:
        return None
    return _TARGET_INDEX_TO_LANGUAGE.get(target_index)


def _identity_by_language() -> dict[Language, EmbeddingIdentity]:
    """Resolve expected identity for every language the catalog supports."""
    out: dict[Language, EmbeddingIdentity] = {}
    for lang in ("zh-tw", "zh-cn", "en"):
        out[lang] = resolve_expected_identity(language=lang)
    return out


def check_episode_drift(
    db: Database,
    *,
    identity_by_language: Optional[dict[Language, EmbeddingIdentity]] = None,
) -> tuple[EpisodeDriftReport, list[str]]:
    """Return (report, affected_episode_show_id_pairs).

    The second element is the raw per-row set used to build
    `ShowImpactSummary` — kept as a flat list of show_ids for each
    drifting episode so the summary step does the de-duplication.

    Behavior:
      - Rows with `embedding_model IS NULL` are skipped (not yet embedded).
      - Rows whose show has no language mapping are counted as
        `unresolvable_language_count` and do not contribute to
        `drift_counts` or `ok_count`.
      - `embedding_version` comparisons are tolerant of the legacy
        `<model>/<version>` form so a pre-Phase-2a row is not flagged.
    """
    identities = identity_by_language or _identity_by_language()

    rows = db.execute(
        """
        SELECT e.episode_id, e.show_id, e.embedding_model, e.embedding_version,
               s.target_index
        FROM episodes AS e
        LEFT JOIN shows AS s ON s.show_id = e.show_id
        WHERE e.embedding_model IS NOT NULL
        """
    ).fetchall()

    drift_counts: Counter = Counter()
    ok_count = 0
    unresolvable = 0
    drifting_show_ids: list[str] = []

    for episode_id, show_id, row_model, row_version, target_index in rows:
        language = _resolve_language_for_target_index(target_index)
        if language is None:
            unresolvable += 1
            continue
        expected = identities.get(language)
        if expected is None:
            unresolvable += 1
            continue

        row_version_clean = _normalize_version(row_version)

        drifts: list[DriftKind] = []
        if row_model != expected.model_name:
            drifts.append(DriftKind.MODEL_MISMATCH)
        if row_version_clean != expected.embedding_version:
            drifts.append(DriftKind.VERSION_MISMATCH)

        if not drifts:
            ok_count += 1
            continue

        kind = drifts[0] if len(drifts) == 1 else DriftKind.MULTIPLE
        drift_counts[kind.value] += 1
        if show_id:
            drifting_show_ids.append(show_id)

    report = EpisodeDriftReport(
        episodes_with_metadata=len(rows),
        ok_count=ok_count,
        drift_counts=dict(drift_counts),
        unresolvable_language_count=unresolvable,
    )
    return report, drifting_show_ids


def summarize_sync_state_distribution(db: Database) -> SyncStateDistribution:
    """Group sync_state rows by (environment, model, version) and count them."""
    if "search_sync_state" not in db.table_names():
        return SyncStateDistribution(rows=[])

    rows = db.execute(
        """
        SELECT environment, embedding_model, embedding_version, sync_status,
               COUNT(*) AS n
        FROM search_sync_state
        GROUP BY environment, embedding_model, embedding_version, sync_status
        ORDER BY environment, embedding_model, embedding_version, sync_status
        """
    ).fetchall()

    out: list[dict[str, Any]] = []
    for environment, model, version, status, count in rows:
        out.append({
            "environment": environment,
            "embedding_model": model,
            "embedding_version": _normalize_version(version),
            "sync_status": status,
            "count": count,
        })
    return SyncStateDistribution(rows=out)


def summarize_show_impact(drifting_show_ids: list[str]) -> ShowImpactSummary:
    """Translate the flat list of drifting show_ids into a dual-track summary.

    `affected_show_ids` is sorted ascending and de-duplicated so the
    output is stable across runs. `top_shows_by_drift` surfaces the
    shows that contribute the most drifting episodes — useful when a
    handful of large shows dominate the rebuild cost.
    """
    counts: Counter = Counter(drifting_show_ids)
    affected = sorted(set(drifting_show_ids))
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_TOP_N]
    return ShowImpactSummary(
        episode_truth_line_count=sum(counts.values()),
        show_artifact_line_count=len(affected),
        affected_show_ids=affected,
        top_shows_by_drift=top,
    )


def build_report(db: Database) -> DriftReport:
    identities = _identity_by_language()
    episode_drift, drifting_show_ids = check_episode_drift(db, identity_by_language=identities)
    show_impact = summarize_show_impact(drifting_show_ids)
    distribution = summarize_sync_state_distribution(db)
    expected = {
        lang: {
            "model_name": ident.model_name,
            "embedding_version": ident.embedding_version,
            "embedding_dimensions": ident.embedding_dimensions,
        }
        for lang, ident in identities.items()
    }
    return DriftReport(
        expected_identity_by_language=expected,
        episode_drift=episode_drift,
        show_impact=show_impact,
        sync_state_distribution=distribution,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def report_to_dict(report: DriftReport) -> dict[str, Any]:
    """Flatten a `DriftReport` into JSON-friendly dicts."""
    return {
        "expected_identity_by_language": report.expected_identity_by_language,
        "episode_drift": asdict(report.episode_drift),
        "show_impact": asdict(report.show_impact),
        "sync_state_distribution": report.sync_state_distribution.rows,
    }


def format_report_text(report: DriftReport) -> str:
    lines = ["=== check_drift report ==="]
    lines.append("")
    lines.append("Expected identity (by language):")
    for lang, ident in report.expected_identity_by_language.items():
        lines.append(
            f"  {lang}: model={ident['model_name']} "
            f"version={ident['embedding_version']} "
            f"dims={ident['embedding_dimensions']}"
        )
    ed = report.episode_drift
    lines.append("")
    lines.append("Episode drift:")
    lines.append(f"  episodes_with_metadata: {ed.episodes_with_metadata}")
    lines.append(f"  ok_count:               {ed.ok_count}")
    lines.append(f"  unresolvable_language:  {ed.unresolvable_language_count}")
    if ed.drift_counts:
        for kind, n in sorted(ed.drift_counts.items()):
            lines.append(f"  {kind}: {n}")
    else:
        lines.append("  drift_counts: (none)")

    si = report.show_impact
    lines.append("")
    lines.append("Show impact (dual-track):")
    lines.append(f"  episode_truth_line_count: {si.episode_truth_line_count}")
    lines.append(f"  show_artifact_line_count: {si.show_artifact_line_count}")
    sample = si.affected_show_ids[:_TOP_N]
    lines.append(f"  affected_show_ids (first {_TOP_N}): {sample}")
    if si.top_shows_by_drift:
        lines.append("  top shows by drifting episode count:")
        for show_id, n in si.top_shows_by_drift:
            lines.append(f"    {show_id}: {n}")

    lines.append("")
    lines.append("sync_state distribution:")
    if not report.sync_state_distribution.rows:
        lines.append("  (no rows)")
    else:
        for row in report.sync_state_distribution.rows:
            lines.append(
                f"  env={row['environment']} "
                f"model={row['embedding_model']} "
                f"version={row['embedding_version']} "
                f"status={row['sync_status']} "
                f"count={row['count']}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect embedding drift across episodes and sync_state (report-only).",
    )
    parser.add_argument(
        "--db", type=Path, default=settings.SQLITE_PATH,
        help="Path to the crawler SQLite database.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON report on stdout instead of the human summary.",
    )
    args = parser.parse_args()

    setup_logging()
    db = Database(str(args.db))
    report = build_report(db)

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))
    else:
        print(format_report_text(report))

    # Phase 2a: report-only. Exit 0 regardless of drift. Later phases
    # may introduce a `--strict` flag that returns non-zero on drift.
    return 0


if __name__ == "__main__":
    sys.exit(main())
