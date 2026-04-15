"""Embedding cache path helpers + identity validation.

Phase 2a §3.5 IC1–IC2 (Content Identity Contract) + §3.8 (architecture).

This module owns:
- Cache path slug construction from `EmbeddingIdentity` (IC1).
- Cache entry parsing and identity validation, including payload vector length
  cross-check against metadata dims (IC2: "三欄位 + payload vector 長度 ==
  metadata dims").
- `DriftKind` / `FoundParseState` / `IdentityMismatch` — the structured drift
  event surface consumed by `validate_cache_identity`, `check_drift` CLI, and
  the `embed_and_ingest` fallback path (RA2 / SUM5 log).

No runtime side effects beyond filesystem path construction. Does NOT read
cache files (caller opens and passes a parsed dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.pipelines.embedding_identity import EmbeddingIdentity


class DriftKind(str, Enum):
    """Categorical reason a cache entry's identity disagrees with expected.

    Ordering of emission from `validate_cache_identity`: when multiple
    categories apply at once, `MULTIPLE` is returned and the individual
    kinds are recoverable from the `IdentityMismatch` fields (found_* vs
    expected_*).
    """

    MODEL_MISMATCH = "model_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    VECTOR_LENGTH_MISMATCH = "vector_length_mismatch"
    MULTIPLE = "multiple"


class FoundParseState(str, Enum):
    """Describes how much of the cache entry was machine-parsable.

    Distinct from `DriftKind`: parse state captures missing / corrupt
    metadata, while drift kind captures "metadata present but wrong".
    Populated regardless of whether a mismatch is detected.

    Phase 2a v4: renamed from `found_state` for clarity.
    """

    OK = "ok"
    MISSING_MODEL = "missing_model"
    MISSING_VERSION = "missing_version"
    MISSING_DIMS = "missing_dims"
    EMPTY_EPISODES = "empty_episodes"


@dataclass(frozen=True)
class IdentityMismatch:
    """Structured drift event for SUM5 log + ShowImpactSummary."""

    drift_kind: DriftKind
    found_parse_state: FoundParseState
    expected_model: str
    expected_version: str
    expected_dims: int
    found_model: str | None
    found_version: str | None
    found_dims: int | None
    vector_length_observed: int | None


def _safe(s: str) -> str:
    """Path-safe transform of an identity field (`/` → `--`)."""
    return s.replace("/", "--")


def cache_path_for(cache_dir: Path, identity: EmbeddingIdentity, show_id: str) -> Path:
    """Return versioned cache path for a show under the given identity (IC1).

    Slug: `f"{model_safe}__{version_safe}__dim{dims}"`. Show-level filename
    (not episode-level) — Phase 2a compatibility compromise documented in
    main spec §3.3.3 #2.
    """
    slug = (
        f"{_safe(identity.model_name)}__"
        f"{_safe(identity.embedding_version)}__"
        f"dim{identity.embedding_dimensions}"
    )
    return cache_dir / slug / f"{show_id}.json"


def validate_cache_identity(
    entry: dict[str, Any],
    expected: EmbeddingIdentity,
) -> IdentityMismatch | None:
    """Validate a loaded cache entry against expected identity (IC2).

    IC2 dual check:
    1. Metadata fields (`model_name`, `embedding_version`,
       `embedding_dimensions`) equal to `expected`.
    2. Payload vector length == `embedding_dimensions` metadata
       (internal-consistency check; catches poisoned cache files where
       metadata claims dim D but vectors are a different length).

    Returns:
        None if identity fully matches (and `embedding_dimensions` == vector
        length when vectors are present).
        `IdentityMismatch` describing the divergence otherwise. When more
        than one divergence applies, `drift_kind = MULTIPLE` and the caller
        can reconstruct the specifics from found_* fields.
    """
    found_model = entry.get("model_name")
    found_version = entry.get("embedding_version")
    found_dims = entry.get("embedding_dimensions")

    parse_state = FoundParseState.OK
    if found_model is None:
        parse_state = FoundParseState.MISSING_MODEL
    elif found_version is None:
        parse_state = FoundParseState.MISSING_VERSION
    elif found_dims is None:
        parse_state = FoundParseState.MISSING_DIMS

    # Observe the first episode's vector length to cross-check metadata dims.
    episodes = entry.get("episodes") or {}
    vec_len: int | None = None
    if isinstance(episodes, dict) and episodes:
        first = next(iter(episodes.values()))
        if isinstance(first, list):
            vec_len = len(first)

    if vec_len is None and parse_state == FoundParseState.OK:
        parse_state = FoundParseState.EMPTY_EPISODES

    drifts: list[DriftKind] = []
    if found_model != expected.model_name:
        drifts.append(DriftKind.MODEL_MISMATCH)
    if found_version != expected.embedding_version:
        drifts.append(DriftKind.VERSION_MISMATCH)
    if found_dims != expected.embedding_dimensions:
        drifts.append(DriftKind.DIMENSION_MISMATCH)
    # Intra-cache consistency: payload length vs metadata dims. Only meaningful
    # when both are observable; an empty cache or missing dims skips this.
    if (
        vec_len is not None
        and isinstance(found_dims, int)
        and vec_len != found_dims
    ):
        drifts.append(DriftKind.VECTOR_LENGTH_MISMATCH)

    if not drifts:
        return None

    kind = drifts[0] if len(drifts) == 1 else DriftKind.MULTIPLE
    return IdentityMismatch(
        drift_kind=kind,
        found_parse_state=parse_state,
        expected_model=expected.model_name,
        expected_version=expected.embedding_version,
        expected_dims=expected.embedding_dimensions,
        found_model=found_model if isinstance(found_model, str) else None,
        found_version=found_version if isinstance(found_version, str) else None,
        found_dims=found_dims if isinstance(found_dims, int) else None,
        vector_length_observed=vec_len,
    )
