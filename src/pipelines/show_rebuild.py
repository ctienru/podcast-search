"""Show-level embedding rebuild primitive — Phase 2a.

One function that takes a show_id + expected identity and produces a
versioned, internally-consistent cache file for that show. Shared by
two callers:

  * `embed_and_ingest.py` — when a cache identity mismatch is detected,
    the pipeline falls back to rebuilding that show's cache on the spot
    rather than skipping or failing hard.
  * `force_embed.py` — a manual override tool the operator runs after
    explicitly acknowledging model drift.

Splitting this out is deliberate: both callers need to produce
byte-identical cache content for the same (show, identity) pair,
and the only way to guarantee that is to share the write path.

What this primitive does:
  - Load the show's embedding inputs from the configured input
    directory (or accept pre-loaded inputs from the caller).
  - Call the shared chunking primitive.
  - Call the shared runtime primitive (which self-validates vector
    dimensions — any mismatch raises systemic halt and is NOT caught
    here so it bubbles past per-show error handling).
  - Atomically write the versioned cache file (tmp + rename) with the
    three identity fields populated so `validate_cache_identity` will
    find an internally consistent entry on the next run.
  - Return a lean result object describing what happened.

What this primitive does NOT do (the caller owns these):
  - Never writes `search_sync_state`.
  - Never writes ES bulk.
  - Never sets `episodes.embedding_status='done'`.
  - Never updates `episodes.embedding_model`, `embedding_version`, or
    `last_embedded_at`. Those are commit decisions that depend on
    caller-specific conditions (e.g. whether the subsequent bulk ingest
    succeeded), so this primitive returns them as a result payload and
    the caller commits per its own policy.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from src.embedding.backend import EmbeddingBackend
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
)
from src.pipelines.embedding_paths import cache_path_for
from src.pipelines.embedding_runtime import embed_texts
from src.pipelines.embedding_text import prepare_chunks_for_show
from src.types import Language

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")


class RebuildErrorCode(str, Enum):
    """Failure categories that the primitive reports back to the caller.

    Systemic violations (e.g. wrong vector dimension) are never encoded
    here — they raise `EmbeddingDimensionContractViolation` and bubble
    past the primitive so the caller cannot accidentally swallow them.
    """

    STORAGE_READ_FAILURE = "storage_read_failure"
    ZERO_EPISODE_IN_CANDIDATE = "zero_episode_in_candidate"
    CACHE_WRITE_FAILURE = "cache_write_failure"
    EMBEDDING_RUNTIME_ERROR = "embedding_runtime_error"


@dataclass(frozen=True)
class ShowRebuildResult:
    """Outcome of a single `rebuild_show_cache` call.

    Invariants (enforced by tests):
      - When `status == "ok"`:
          `cache_written is True`,
          `new_last_embedded_at is not None`,
          `error_code is None`,
          `error_message is None`.
      - When `status == "failed"`:
          `cache_written is False`,
          `new_last_embedded_at is None`,
          `error_code is not None`,
          `error_message is not None`.

    The three identity fields live inside `identity_used` — there is no
    separate `new_embedding_model` or `new_embedding_version` so callers
    cannot drift them from the identity that was actually used.
    """

    show_id: str
    status: Literal["ok", "failed"]
    cache_written: bool
    episode_count: int
    identity_used: EmbeddingIdentity
    new_last_embedded_at: Optional[datetime]
    error_code: Optional[str]
    error_message: Optional[str]


def rebuild_show_cache(
    *,
    show_id: str,
    identity: EmbeddingIdentity,
    language: Language,
    cache_dir: Path,
    episode_inputs: Optional[Sequence[Mapping[str, Any]]] = None,
    embedding_input_dir: Optional[Path] = None,
    backend: Optional[EmbeddingBackend] = None,
) -> ShowRebuildResult:
    """Rebuild the versioned cache file for `show_id` under `identity`.

    Args:
        show_id: The show to rebuild.
        identity: The expected embedding identity. Used to resolve the
            cache output path, stamped into the written metadata, and
            enforced by the runtime primitive on every returned vector.
        language: The language passed to the runtime primitive so it
            selects the right model. The caller typically derives this
            from the show's target_index.
        cache_dir: The root directory under which the versioned cache
            slug subdirectory lives.
        episode_inputs: Pre-loaded embedding-input records for this show.
            If None, the primitive reads from `embedding_input_dir`.
        embedding_input_dir: Directory containing per-episode
            embedding-input JSON files. Defaults to
            `data/embedding_input/episodes/`.
        backend: Optional encoder backend override, primarily for tests.

    Returns:
        `ShowRebuildResult` describing the outcome. Either all the "ok"
        invariants hold or all the "failed" invariants hold.

    Raises:
        EmbeddingDimensionContractViolation: The runtime primitive
            detected a vector dimension mismatch. Systemic halt —
            callers MUST NOT catch this at the per-show boundary.
    """
    # ── Step 1: load inputs ──────────────────────────────────────────
    if episode_inputs is None:
        input_dir = embedding_input_dir or DEFAULT_EMBEDDING_INPUT_DIR
        try:
            episode_inputs = _load_episode_inputs_for_show(show_id, input_dir)
        except Exception as exc:  # noqa: BLE001 — any load failure maps to the same code
            logger.warning(
                "show_rebuild_storage_read_failure",
                extra={"show_id": show_id, "error": repr(exc)},
            )
            return _failed(
                show_id=show_id,
                identity=identity,
                episode_count=0,
                code=RebuildErrorCode.STORAGE_READ_FAILURE,
                message=f"failed to read embedding inputs for {show_id}: {exc}",
            )

    if not episode_inputs:
        return _failed(
            show_id=show_id,
            identity=identity,
            episode_count=0,
            code=RebuildErrorCode.ZERO_EPISODE_IN_CANDIDATE,
            message=(
                f"no embedding inputs found for show_id={show_id}; "
                f"investigate candidate selection or show→episode linkage"
            ),
        )

    # ── Step 2: chunk ────────────────────────────────────────────────
    chunks = prepare_chunks_for_show(
        show_id=show_id,
        episode_inputs=episode_inputs,
        identity=identity,
    )

    if not chunks:
        # All records were filtered (missing episode_id or empty text).
        # Treat the same as zero-episode — investigate upstream.
        return _failed(
            show_id=show_id,
            identity=identity,
            episode_count=0,
            code=RebuildErrorCode.ZERO_EPISODE_IN_CANDIDATE,
            message=(
                f"prepare_chunks_for_show produced 0 chunks from "
                f"{len(episode_inputs)} inputs for show_id={show_id}"
            ),
        )

    # ── Step 3: encode (ET1 bubbles; other errors map to ET2) ───────
    try:
        vectors = embed_texts(
            texts=[chunk.text for chunk in chunks],
            language=language,
            identity=identity,
            backend=backend,
        )
    except EmbeddingDimensionContractViolation:
        # Systemic halt — never swallow at per-show boundary.
        raise
    except Exception as exc:  # noqa: BLE001 — all other runtime failures map to ET2
        logger.warning(
            "show_rebuild_embedding_runtime_error",
            extra={"show_id": show_id, "error": repr(exc)},
        )
        return _failed(
            show_id=show_id,
            identity=identity,
            episode_count=len(chunks),
            code=RebuildErrorCode.EMBEDDING_RUNTIME_ERROR,
            message=f"embed_texts failed for show_id={show_id}: {exc}",
        )

    # ── Step 4: atomic cache write ──────────────────────────────────
    embedded_at = datetime.now(timezone.utc)
    cache_entry: dict[str, Any] = {
        "show_id": show_id,
        "model_name": identity.model_name,
        "embedding_version": identity.embedding_version,
        "embedding_dimensions": identity.embedding_dimensions,
        "embedded_at": embedded_at.isoformat(),
        "episodes": {chunk.episode_id: vec for chunk, vec in zip(chunks, vectors)},
    }

    target_path = cache_path_for(cache_dir, identity, show_id)
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(target_path, cache_entry)
    except Exception as exc:  # noqa: BLE001 — any write failure maps to the same code
        logger.warning(
            "show_rebuild_cache_write_failure",
            extra={"show_id": show_id, "path": str(target_path), "error": repr(exc)},
        )
        return _failed(
            show_id=show_id,
            identity=identity,
            episode_count=len(chunks),
            code=RebuildErrorCode.CACHE_WRITE_FAILURE,
            message=f"cache write failed at {target_path}: {exc}",
        )

    logger.debug(
        "show_rebuild_ok",
        extra={
            "show_id": show_id,
            "episode_count": len(chunks),
            "path": str(target_path),
        },
    )

    return ShowRebuildResult(
        show_id=show_id,
        status="ok",
        cache_written=True,
        episode_count=len(chunks),
        identity_used=identity,
        new_last_embedded_at=embedded_at,
        error_code=None,
        error_message=None,
    )


# ── Internals ────────────────────────────────────────────────────────────────


def _failed(
    *,
    show_id: str,
    identity: EmbeddingIdentity,
    episode_count: int,
    code: RebuildErrorCode,
    message: str,
) -> ShowRebuildResult:
    """Build a `status="failed"` result with the invariants enforced by type."""
    return ShowRebuildResult(
        show_id=show_id,
        status="failed",
        cache_written=False,
        episode_count=episode_count,
        identity_used=identity,
        new_last_embedded_at=None,
        error_code=code.value,
        error_message=message,
    )


def _load_episode_inputs_for_show(
    show_id: str, input_dir: Path,
) -> list[dict[str, Any]]:
    """Scan `input_dir` for embedding-input JSONs belonging to `show_id`.

    Each input file is a JSON with at least `show_id` and `episode_id`
    fields; `show_rebuild` filters by `show_id`. Files that fail to parse
    are skipped (they would have been skipped by embed_episodes too).
    """
    if not input_dir.exists():
        return []
    matches: list[dict[str, Any]] = []
    for path in input_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                record = json.load(fh)
        except Exception:  # noqa: BLE001 — unreadable records are not fatal
            continue
        if isinstance(record, dict) and record.get("show_id") == show_id:
            matches.append(record)
    return matches


def _atomic_write_json(target: Path, data: dict[str, Any]) -> None:
    """Write `data` as JSON to `target` atomically via tmp file + rename.

    The tmp file lives in the same directory as the target so that
    `Path.replace` performs a rename within the same filesystem (atomic
    on POSIX; best-effort on Windows). Rename is only reached when the
    write succeeds, so a crash mid-write leaves the tmp file orphaned
    but the target either absent or at its previous good content.
    """
    target_dir = target.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target_dir,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp_path.replace(target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        raise
