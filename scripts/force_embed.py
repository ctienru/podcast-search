"""Manual override tool for rebuilding a show's embedding cache.

An operator runs this when they know the production cache drifted from
the currently-deployed model (usually flagged by `check_drift`) and
explicitly want to overwrite the affected shows. The tool is the sole
consumer of the `--allow-model-drift` flag — the scheduled pipeline
never sees it.

What this tool does:
  - Resolve the operator's selection (`--show-ids` and/or
    `--episode-ids`) into a unique set of shows. Episode ids are
    selection-only: giving one episode of a show rebuilds the whole
    show, because show is the rewrite unit.
  - For each show, call `rebuild_show_cache()` — the same primitive
    the daily pipeline uses when it falls back on a cache identity
    mismatch, so both paths produce byte-identical cache content for
    the same (show, identity) pair.
  - Commit DB embedding metadata (`model` / `version` /
    `last_embedded_at`) per-show as soon as the primitive reports ok.
    There is no bulk ingest stage in this tool, so the primitive's ok
    is the whole commit signal.

What this tool never does:
  - Never writes `embedding_status='done'`. Phase 2a keeps that field
    decoupled from actual ES state until the semantic cleanup decision
    is made, and this tool must not spread the stale semantics.
  - Never writes `search_sync_state` and never calls ES bulk. Those
    are owned by the daily pipeline.
  - Never re-imports `embed_episodes` — the primitive chain already
    owns chunking and runtime, and reaching into the batch entry point
    would fork the text / runtime behavior.

Exit codes (picked so CI can distinguish operator errors from systemic
halts without parsing the summary):

  0  All requested shows rebuilt successfully (or dry-run accepted).
  1  Neither `--show-ids` nor `--episode-ids` supplied.
  2  Missing the required `--allow-model-drift` flag. argparse's
     default on a required option happens to be 2, which matches
     this slot intentionally.
  3  None of the supplied ids resolved to a rebuildable show.
  4  At least one show failed with a per-show recoverable error
     (ET2). Processing continued for the rest.
  5  A systemic halt (ET1 — vector dimension contract violation)
     aborted the run. Earlier shows that already finished are
     preserved; the in-flight show and anything after it were not
     committed.

Preserve semantics on ET1 (matches §4.4):
  - Previously-ok shows keep both their freshly-written cache file
    and their committed DB metadata.
  - The in-flight show has no DB metadata commit (the exception is
    raised inside the runtime primitive before the cache write, so
    no cache file is materialized for it either).
  - No subsequent show is attempted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlite_utils import Database

from src.config import settings
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
    resolve_expected_identity,
)
from src.pipelines.show_rebuild import ShowRebuildResult, rebuild_show_cache
from src.storage.episode_status import EpisodeStatusRepository
from src.types import Language
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# Keep in sync with `check_drift._TARGET_INDEX_TO_LANGUAGE`. The two
# scripts intentionally duplicate this table rather than sharing a
# helper so neither drags the other's import surface along. A regression
# test pins them to the same mapping.
_TARGET_INDEX_TO_LANGUAGE: dict[str, Language] = {
    "podcast-episodes-zh-tw": "zh-tw",
    "podcast-episodes-zh-cn": "zh-cn",
    "podcast-episodes-en": "en",
}


# Exit code slots — named so the summary stays readable.
EXIT_OK = 0
EXIT_EMPTY_INPUT = 1
EXIT_MISSING_ALLOW_FLAG = 2  # argparse enforces this slot
EXIT_ALL_UNRESOLVABLE = 3
EXIT_PER_SHOW_FAILURE = 4
EXIT_SYSTEMIC_HALT = 5


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedShow:
    """A show selected for rebuild plus its resolved language."""

    show_id: str
    language: Language


@dataclass
class ForceEmbedSummary:
    """Captured state at the end of the run. Printed verbatim and
    returned to tests so they can assert on specific fields without
    parsing stdout."""

    requested_show_ids: int = 0
    requested_episode_ids: int = 0
    unresolvable_episode_ids: list[str] = field(default_factory=list)
    resolved_unique_shows: int = 0
    rebuild_attempted: int = 0
    rebuild_succeeded: int = 0
    rebuild_failed: list[tuple[str, str]] = field(default_factory=list)
    rebuild_systemic_halted: bool = False
    db_metadata_updated: int = 0
    exit_code: int = EXIT_OK

    def format(self) -> str:
        lines = ["=== force_embed summary ==="]
        lines.append(f"requested_show_ids:       {self.requested_show_ids}")
        lines.append(f"requested_episode_ids:    {self.requested_episode_ids}")
        lines.append(
            "unresolvable_episode_ids: "
            f"{self.unresolvable_episode_ids} ({len(self.unresolvable_episode_ids)})"
        )
        lines.append(f"resolved_unique_shows:    {self.resolved_unique_shows}")
        lines.append(f"rebuild_attempted:        {self.rebuild_attempted}")
        lines.append(f"rebuild_succeeded:        {self.rebuild_succeeded}")
        failed_ids = [sid for sid, _ in self.rebuild_failed]
        lines.append(
            f"rebuild_failed:           {failed_ids} ({len(self.rebuild_failed)})"
        )
        lines.append(f"rebuild_systemic_halted:  {self.rebuild_systemic_halted}")
        lines.append(f"db_metadata_updated:      {self.db_metadata_updated}")
        lines.append(f"exit_code:                {self.exit_code}")
        return "\n".join(lines)


# ── Selection → resolved shows ──────────────────────────────────────────────


def _split_csv(raw: Optional[str]) -> list[str]:
    """Split a comma-separated CLI value into a deduped ordered list."""
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw.split(","):
        val = tok.strip()
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def _language_for_target_index(target_index: Optional[str]) -> Optional[Language]:
    if not target_index:
        return None
    return _TARGET_INDEX_TO_LANGUAGE.get(target_index)


def resolve_selection(
    db: Database,
    *,
    show_ids: list[str],
    episode_ids: list[str],
) -> tuple[list[ResolvedShow], list[str]]:
    """Translate the operator's selection into a concrete plan.

    Returns `(resolved_shows, unresolvable_episode_ids)`:
      - `resolved_shows` is sorted ascending by `show_id` so repeated
        runs attempt shows in the same order. That is what makes
        "preserve semantics" observable — on ET1 you know exactly
        which shows were processed.
      - `unresolvable_episode_ids` collects every input id whose row
        was missing from `episodes`, whose show had no `target_index`,
        or whose `target_index` points at an unknown language. Those
        inputs do not block other selections, but if the set of
        resolved shows turns out to be empty the caller reports
        exit 3 and names them.

    Shows passed via `--show-ids` are also dropped into
    `unresolvable_episode_ids`-like handling when their language
    cannot be resolved; they are reported in the summary so the
    operator can see why a requested show was skipped.
    """
    unresolvable: list[str] = []
    collected: dict[str, Language] = {}  # show_id → language

    # Resolve --show-ids directly.
    if show_ids:
        rows = db.execute(
            f"""
            SELECT show_id, target_index
            FROM shows
            WHERE show_id IN ({','.join('?' * len(show_ids))})
            """,
            show_ids,
        ).fetchall()
        found = {r[0]: r[1] for r in rows}
        for sid in show_ids:
            target_index = found.get(sid)
            lang = _language_for_target_index(target_index)
            if lang is None:
                unresolvable.append(sid)
                continue
            collected.setdefault(sid, lang)

    # Resolve --episode-ids → show_id by joining episodes → shows.
    if episode_ids:
        rows = db.execute(
            f"""
            SELECT e.episode_id, e.show_id, s.target_index
            FROM episodes AS e
            LEFT JOIN shows AS s ON s.show_id = e.show_id
            WHERE e.episode_id IN ({','.join('?' * len(episode_ids))})
            """,
            episode_ids,
        ).fetchall()
        by_episode = {r[0]: (r[1], r[2]) for r in rows}
        for eid in episode_ids:
            row = by_episode.get(eid)
            if row is None:
                unresolvable.append(eid)
                continue
            sid, target_index = row
            lang = _language_for_target_index(target_index)
            if lang is None or sid is None:
                unresolvable.append(eid)
                continue
            collected.setdefault(sid, lang)

    resolved = [
        ResolvedShow(show_id=sid, language=lang)
        for sid, lang in sorted(collected.items())
    ]
    return resolved, unresolvable


# ── Per-show execution ──────────────────────────────────────────────────────


def _episode_ids_for_show(db: Database, show_id: str) -> list[str]:
    """Every episode row of `show_id` — used as the commit target.

    The rebuild primitive owns which episodes are embedded (it reads
    the embedding_input directory and rebuilds them all). The DB
    commit mirrors the show's full episode set so a partial commit
    cannot drift metadata away from what the cache now reflects.
    """
    rows = db.execute(
        "SELECT episode_id FROM episodes WHERE show_id = ?",
        [show_id],
    ).fetchall()
    return [r[0] for r in rows]


def _commit_show_metadata(
    *,
    db: Database,
    episode_status_repo: EpisodeStatusRepository,
    show_id: str,
    result: ShowRebuildResult,
) -> int:
    """Record the rebuild on every episode row of the show.

    force_embed's commit trigger is deliberately lighter than
    `embed_and_ingest`'s: there is no bulk ingest stage, so the
    primitive's `status == "ok"` is enough. Returns the number of DB
    rows updated; the caller folds it into `db_metadata_updated`.
    """
    episode_ids = _episode_ids_for_show(db, show_id)
    if not episode_ids:
        return 0
    assert result.new_last_embedded_at is not None  # status == "ok" invariant
    return episode_status_repo.mark_embedding_metadata_only(
        episode_ids=episode_ids,
        model=result.identity_used.model_name,
        version=result.identity_used.embedding_version,
        embedded_at=result.new_last_embedded_at.isoformat(),
    )


def run_force_embed(
    *,
    db: Database,
    resolved_shows: list[ResolvedShow],
    cache_dir: Path,
    embedding_input_dir: Optional[Path],
    dry_run: bool,
) -> ForceEmbedSummary:
    """Orchestrate the per-show rebuild loop.

    Split from `main` so tests can drive it with an in-memory DB and a
    patched rebuild primitive without wrestling argparse.
    """
    summary = ForceEmbedSummary()
    summary.resolved_unique_shows = len(resolved_shows)

    if dry_run:
        for rs in resolved_shows:
            logger.info(
                "force_embed_dry_run_plan",
                extra={"show_id": rs.show_id, "language": rs.language},
            )
        return summary

    episode_status_repo = EpisodeStatusRepository(db)

    for resolved in resolved_shows:
        identity: EmbeddingIdentity = resolve_expected_identity(
            language=resolved.language,
        )
        summary.rebuild_attempted += 1

        try:
            result = rebuild_show_cache(
                show_id=resolved.show_id,
                identity=identity,
                language=resolved.language,
                cache_dir=cache_dir,
                embedding_input_dir=embedding_input_dir,
            )
        except EmbeddingDimensionContractViolation as exc:
            # ET1 systemic halt. Don't roll back already-committed
            # shows — their cache files and DB metadata reflect real,
            # validated embeddings. The in-flight show hit the
            # exception inside the runtime before any cache write,
            # so there is nothing to preserve for it. Subsequent
            # shows are not attempted.
            logger.error(
                "force_embed_systemic_halt",
                extra={
                    "show_id": resolved.show_id,
                    "expected_dim": exc.expected,
                    "actual_dim": exc.actual,
                    "model_name": exc.model_name,
                },
            )
            summary.rebuild_systemic_halted = True
            summary.exit_code = EXIT_SYSTEMIC_HALT
            return summary

        if result.status == "failed":
            summary.rebuild_failed.append(
                (resolved.show_id, result.error_code or "unknown"),
            )
            continue

        summary.rebuild_succeeded += 1
        updated = _commit_show_metadata(
            db=db,
            episode_status_repo=episode_status_repo,
            show_id=resolved.show_id,
            result=result,
        )
        summary.db_metadata_updated += updated

    if summary.rebuild_failed:
        summary.exit_code = EXIT_PER_SHOW_FAILURE
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually rebuild a show's embedding cache (Phase 2a override)."
        ),
    )
    parser.add_argument(
        "--allow-model-drift",
        action="store_true",
        required=True,
        help=(
            "Required acknowledgement that the operator knows this "
            "command overwrites cache entries that may currently "
            "disagree with the deployed model."
        ),
    )
    parser.add_argument(
        "--show-ids",
        default="",
        help="Comma-separated show ids to rebuild.",
    )
    parser.add_argument(
        "--episode-ids",
        default="",
        help=(
            "Comma-separated episode ids. Selection-only: the owning "
            "show is rebuilt in full."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the selection and print the plan without "
            "touching the cache or the database."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.SQLITE_PATH,
        help="Path to the crawler SQLite database.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.EMBEDDING_CACHE_DIR,
        help="Root directory containing versioned cache slugs.",
    )
    parser.add_argument(
        "--embedding-input-dir",
        type=Path,
        default=None,
        help=(
            "Override for the embedding-input directory the rebuild "
            "primitive reads. Defaults to the primitive's standard "
            "location."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()

    show_ids = _split_csv(args.show_ids)
    episode_ids = _split_csv(args.episode_ids)

    if not show_ids and not episode_ids:
        print(
            "error: must provide --show-ids and/or --episode-ids",
            file=sys.stderr,
        )
        return EXIT_EMPTY_INPUT

    db = Database(str(args.db))
    resolved_shows, unresolvable = resolve_selection(
        db,
        show_ids=show_ids,
        episode_ids=episode_ids,
    )

    if not resolved_shows:
        summary = ForceEmbedSummary(
            requested_show_ids=len(show_ids),
            requested_episode_ids=len(episode_ids),
            unresolvable_episode_ids=unresolvable,
            exit_code=EXIT_ALL_UNRESOLVABLE,
        )
        print(summary.format())
        return summary.exit_code

    summary = run_force_embed(
        db=db,
        resolved_shows=resolved_shows,
        cache_dir=args.cache_dir,
        embedding_input_dir=args.embedding_input_dir,
        dry_run=args.dry_run,
    )
    summary.requested_show_ids = len(show_ids)
    summary.requested_episode_ids = len(episode_ids)
    summary.unresolvable_episode_ids = unresolvable

    print(summary.format())
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
