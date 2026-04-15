"""Phase 2a §3.5 IC1–IC2 cache path + identity validation tests.

Covers:
- `cache_path_for` slug format + `/` → `--` safe replacement (IC1)
- `validate_cache_identity` returns None on match (IC2)
- Individual DriftKind detection (model / version / dims / vector_length)
- MULTIPLE when more than one category diverges
- FoundParseState tracks metadata completeness independently of drift
"""

from __future__ import annotations

from pathlib import Path

from src.pipelines.embedding_identity import EmbeddingIdentity
from src.pipelines.embedding_paths import (
    DriftKind,
    FoundParseState,
    IdentityMismatch,
    cache_path_for,
    validate_cache_identity,
)


def _expected() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_version="text-v1",
        embedding_dimensions=384,
    )


def _matching_entry(dims: int = 384) -> dict:
    return {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_version": "text-v1",
        "embedding_dimensions": dims,
        "episodes": {"ep:1": [0.0] * dims},
    }


# ── cache_path_for (IC1) ────────────────────────────────────────────────────

def test_cache_path_for_slug_format() -> None:
    """IC1: `<cache_dir>/<model>__<version>__dim<N>/<show_id>.json`."""
    ident = _expected()
    path = cache_path_for(Path("/cache"), ident, "show:1")

    assert path == Path(
        "/cache/"
        "paraphrase-multilingual-MiniLM-L12-v2__text-v1__dim384/"
        "show:1.json"
    )


def test_cache_path_for_replaces_slash_in_fields() -> None:
    """`/` in model or version must become `--` so the path remains flat."""
    ident = EmbeddingIdentity(
        model_name="org/model",
        embedding_version="v/1",
        embedding_dimensions=384,
    )
    path = cache_path_for(Path("/cache"), ident, "show:1")

    assert path == Path("/cache/org--model__v--1__dim384/show:1.json")


def test_cache_path_for_is_show_level_not_episode_level() -> None:
    """Compatibility compromise: filename is `<show_id>.json`, not per episode."""
    ident = _expected()
    assert cache_path_for(Path("/c"), ident, "show:abc").name == "show:abc.json"


# ── validate_cache_identity — no mismatch (IC2) ─────────────────────────────

def test_validate_returns_none_when_all_match() -> None:
    assert validate_cache_identity(_matching_entry(), _expected()) is None


def test_validate_returns_none_when_entry_has_no_episodes_but_metadata_matches() -> None:
    """Empty episodes is not a drift — just a FoundParseState annotation; but
    since we can't verify vector length, we cannot flag VECTOR_LENGTH_MISMATCH
    either. No drift ⇒ None."""
    entry = _matching_entry()
    entry["episodes"] = {}
    assert validate_cache_identity(entry, _expected()) is None


# ── Individual DriftKind ────────────────────────────────────────────────────

def test_validate_detects_model_mismatch() -> None:
    entry = _matching_entry()
    entry["model_name"] = "other-model"
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.drift_kind == DriftKind.MODEL_MISMATCH
    assert m.found_model == "other-model"
    assert m.expected_model == "paraphrase-multilingual-MiniLM-L12-v2"


def test_validate_detects_version_mismatch() -> None:
    entry = _matching_entry()
    entry["embedding_version"] = "text-v2"
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.drift_kind == DriftKind.VERSION_MISMATCH


def test_validate_detects_dimension_mismatch() -> None:
    """Metadata dims != expected, but internal (meta=vec_len) still consistent."""
    entry = _matching_entry(dims=768)
    entry["episodes"] = {"ep:1": [0.0] * 768}
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.drift_kind == DriftKind.DIMENSION_MISMATCH
    assert m.found_dims == 768
    assert m.vector_length_observed == 768


def test_validate_detects_vector_length_mismatch() -> None:
    """Metadata says 384 but payload has 768 — internal corruption."""
    entry = _matching_entry()
    entry["episodes"] = {"ep:1": [0.0] * 768}
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.drift_kind == DriftKind.VECTOR_LENGTH_MISMATCH
    assert m.found_dims == 384
    assert m.vector_length_observed == 768


# ── MULTIPLE drift ──────────────────────────────────────────────────────────

def test_validate_reports_multiple_when_more_than_one_category_diverges() -> None:
    entry = _matching_entry()
    entry["model_name"] = "other-model"
    entry["embedding_version"] = "text-v2"
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.drift_kind == DriftKind.MULTIPLE
    # Individual fields recoverable for SUM5 log
    assert m.found_model == "other-model"
    assert m.found_version == "text-v2"


# ── FoundParseState ─────────────────────────────────────────────────────────

def test_found_parse_state_missing_model() -> None:
    entry = _matching_entry()
    entry.pop("model_name")
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.found_parse_state == FoundParseState.MISSING_MODEL
    assert m.found_model is None


def test_found_parse_state_missing_version() -> None:
    entry = _matching_entry()
    entry.pop("embedding_version")
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.found_parse_state == FoundParseState.MISSING_VERSION


def test_found_parse_state_missing_dims() -> None:
    """Legacy flat cache — the shape we observed in Step 0b."""
    entry = _matching_entry()
    entry.pop("embedding_dimensions")
    m = validate_cache_identity(entry, _expected())
    assert m is not None
    assert m.found_parse_state == FoundParseState.MISSING_DIMS
    assert m.found_dims is None
    # Missing dims counts as a dimension mismatch (None != 384), so drift_kind
    # will be DIMENSION_MISMATCH (single category, since vec_len check is
    # skipped when found_dims is not an int).
    assert m.drift_kind == DriftKind.DIMENSION_MISMATCH


def test_found_parse_state_empty_episodes() -> None:
    entry = _matching_entry()
    entry["episodes"] = {}
    m = validate_cache_identity(entry, _expected())
    assert m is None  # empty episodes is not a drift by itself


# ── IdentityMismatch shape ──────────────────────────────────────────────────

def test_identity_mismatch_carries_sum5_fields() -> None:
    """SUM5 log must have: drift_kind, expected_*, found_*, vector_length_observed."""
    entry = _matching_entry()
    entry["model_name"] = "other-model"
    m = validate_cache_identity(entry, _expected())
    assert isinstance(m, IdentityMismatch)
    assert m.drift_kind is DriftKind.MODEL_MISMATCH
    assert m.expected_model and m.expected_version and m.expected_dims
    assert m.vector_length_observed == 384
