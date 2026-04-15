"""Phase 2a show_rebuild primitive tests.

Covers:
- ShowRebuildResult invariants on both "ok" and "failed" status
- status is only "ok" or "failed"
- Responsibilities the primitive owns:
    · cache written at versioned path (slug from cache_path_for)
    · cache payload carries three identity fields + embedded_at + episodes
    · atomic write (no tmp file left behind on success)
- Responsibilities the primitive must NOT do:
    · no sync_state / episode_status / ES bulk writes (by construction —
      the primitive has no client, verified via AST + backend mocks)
- Error taxonomy:
    · zero inputs → ZERO_EPISODE_IN_CANDIDATE
    · input-load exception → STORAGE_READ_FAILURE
    · embed_texts non-systemic exception → EMBEDDING_RUNTIME_ERROR
    · cache write exception → CACHE_WRITE_FAILURE
    · EmbeddingDimensionContractViolation bubbles (systemic halt)
- Result shape has no duplicate identity fields (`new_embedding_model` /
  `new_embedding_version` forbidden; always read from `identity_used`)
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines import show_rebuild as sr_mod
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
)
from src.pipelines.embedding_paths import cache_path_for
from src.pipelines.show_rebuild import (
    RebuildErrorCode,
    ShowRebuildResult,
    rebuild_show_cache,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _identity(dims: int = 4) -> EmbeddingIdentity:
    # Use a tiny dim so tests encode quickly with a mock backend.
    return EmbeddingIdentity(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_version="text-v1",
        embedding_dimensions=dims,
    )


def _mk_input(show_id: str, episode_id: str, text: str) -> dict:
    return {
        "show_id": show_id,
        "episode_id": episode_id,
        "embedding_input": {"text": text},
    }


def _mk_backend(dims: int = 4) -> MagicMock:
    backend = MagicMock()
    backend.embed_batch.side_effect = lambda texts, lang: [[0.1] * dims for _ in texts]
    return backend


def _happy_rebuild(tmp_path: Path) -> ShowRebuildResult:
    return rebuild_show_cache(
        show_id="show:1",
        identity=_identity(),
        language="zh-tw",
        cache_dir=tmp_path,
        episode_inputs=[
            _mk_input("show:1", "ep:1", "hello"),
            _mk_input("show:1", "ep:2", "world"),
        ],
        backend=_mk_backend(),
    )


# ── OK / failed result invariants ───────────────────────────────────────────

def test_ok_result_has_all_invariants(tmp_path: Path) -> None:
    r = _happy_rebuild(tmp_path)
    assert r.status == "ok"
    assert r.cache_written is True
    assert r.new_last_embedded_at is not None
    assert r.error_code is None
    assert r.error_message is None
    assert r.episode_count == 2
    assert r.identity_used == _identity()


def test_failed_result_has_all_invariants(tmp_path: Path) -> None:
    r = rebuild_show_cache(
        show_id="show:empty",
        identity=_identity(),
        language="zh-tw",
        cache_dir=tmp_path,
        episode_inputs=[],  # triggers ZERO_EPISODE_IN_CANDIDATE
        backend=_mk_backend(),
    )
    assert r.status == "failed"
    assert r.cache_written is False
    assert r.new_last_embedded_at is None
    assert r.error_code == RebuildErrorCode.ZERO_EPISODE_IN_CANDIDATE.value
    assert r.error_message is not None


def test_status_is_only_ok_or_failed() -> None:
    """Literal annotation is the contract — check it hasn't widened."""
    hints = get_type_hints(ShowRebuildResult)
    assert get_args(hints["status"]) == ("ok", "failed")


def test_result_has_no_duplicate_identity_fields() -> None:
    """Three identity fields live in `identity_used`; no shadow fields."""
    field_names = {f.name for f in fields(ShowRebuildResult)}
    assert "new_embedding_model" not in field_names
    assert "new_embedding_version" not in field_names
    assert "new_embedding_dimensions" not in field_names
    assert "identity_used" in field_names


# ── ZERO_EPISODE cases ──────────────────────────────────────────────────────

def test_zero_inputs_returns_zero_episode_code(tmp_path: Path) -> None:
    r = rebuild_show_cache(
        show_id="show:x",
        identity=_identity(),
        language="en",
        cache_dir=tmp_path,
        episode_inputs=[],
        backend=_mk_backend(),
    )
    assert r.error_code == RebuildErrorCode.ZERO_EPISODE_IN_CANDIDATE.value


def test_inputs_all_dropped_by_chunker_returns_zero_episode_code(tmp_path: Path) -> None:
    """If every input lacks episode_id / text, chunker drops them all."""
    r = rebuild_show_cache(
        show_id="show:x",
        identity=_identity(),
        language="en",
        cache_dir=tmp_path,
        episode_inputs=[
            {"show_id": "show:x"},  # missing episode_id
            {"show_id": "show:x", "episode_id": "ep:1"},  # missing text
        ],
        backend=_mk_backend(),
    )
    assert r.status == "failed"
    assert r.error_code == RebuildErrorCode.ZERO_EPISODE_IN_CANDIDATE.value


# ── Cache file output ───────────────────────────────────────────────────────

def test_cache_written_at_versioned_path(tmp_path: Path) -> None:
    _happy_rebuild(tmp_path)
    expected = cache_path_for(tmp_path, _identity(), "show:1")
    assert expected.exists()
    assert expected.parent.name == (
        f"paraphrase-multilingual-MiniLM-L12-v2__text-v1__dim{_identity().embedding_dimensions}"
    )


def test_cache_payload_carries_three_identity_fields_and_episodes(tmp_path: Path) -> None:
    _happy_rebuild(tmp_path)
    entry = json.loads(cache_path_for(tmp_path, _identity(), "show:1").read_text())
    assert entry["model_name"] == _identity().model_name
    assert entry["embedding_version"] == _identity().embedding_version
    assert entry["embedding_dimensions"] == _identity().embedding_dimensions
    assert entry["show_id"] == "show:1"
    assert "embedded_at" in entry
    assert set(entry["episodes"].keys()) == {"ep:1", "ep:2"}
    for vec in entry["episodes"].values():
        assert len(vec) == _identity().embedding_dimensions


def test_atomic_write_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    _happy_rebuild(tmp_path)
    slug_dir = cache_path_for(tmp_path, _identity(), "show:1").parent
    tmp_leftovers = [p for p in slug_dir.iterdir() if p.name.endswith(".tmp")]
    assert tmp_leftovers == []


def test_atomic_write_leaves_no_tmp_file_on_rename_failure(tmp_path: Path) -> None:
    """If replace() fails mid-flight, the tmp file is still cleaned up."""
    with patch("src.pipelines.show_rebuild.Path.replace", side_effect=OSError("boom")):
        r = rebuild_show_cache(
            show_id="show:1",
            identity=_identity(),
            language="zh-tw",
            cache_dir=tmp_path,
            episode_inputs=[_mk_input("show:1", "ep:1", "hello")],
            backend=_mk_backend(),
        )
    assert r.status == "failed"
    assert r.error_code == RebuildErrorCode.CACHE_WRITE_FAILURE.value

    slug_dir = cache_path_for(tmp_path, _identity(), "show:1").parent
    if slug_dir.exists():
        tmp_leftovers = [p for p in slug_dir.iterdir() if ".tmp" in p.name]
        assert tmp_leftovers == []


# ── Error taxonomy ──────────────────────────────────────────────────────────

def test_storage_read_failure_maps_to_error_code(tmp_path: Path) -> None:
    """Any exception while loading inputs from disk → STORAGE_READ_FAILURE."""
    with patch(
        "src.pipelines.show_rebuild._load_episode_inputs_for_show",
        side_effect=OSError("disk down"),
    ):
        r = rebuild_show_cache(
            show_id="show:1",
            identity=_identity(),
            language="zh-tw",
            cache_dir=tmp_path,
            episode_inputs=None,  # force the load path
            embedding_input_dir=tmp_path / "inputs",
            backend=_mk_backend(),
        )
    assert r.status == "failed"
    assert r.error_code == RebuildErrorCode.STORAGE_READ_FAILURE.value
    assert "disk down" in r.error_message


def test_embedding_runtime_error_maps_to_error_code(tmp_path: Path) -> None:
    """Non-systemic backend exception → EMBEDDING_RUNTIME_ERROR (ET2)."""
    backend = MagicMock()
    backend.embed_batch.side_effect = RuntimeError("model crashed")
    r = rebuild_show_cache(
        show_id="show:1",
        identity=_identity(),
        language="zh-tw",
        cache_dir=tmp_path,
        episode_inputs=[_mk_input("show:1", "ep:1", "hello")],
        backend=backend,
    )
    assert r.status == "failed"
    assert r.error_code == RebuildErrorCode.EMBEDDING_RUNTIME_ERROR.value
    assert "model crashed" in r.error_message


def test_cache_write_failure_maps_to_error_code(tmp_path: Path) -> None:
    """Any exception during cache write → CACHE_WRITE_FAILURE."""
    with patch(
        "src.pipelines.show_rebuild._atomic_write_json",
        side_effect=OSError("no space"),
    ):
        r = rebuild_show_cache(
            show_id="show:1",
            identity=_identity(),
            language="zh-tw",
            cache_dir=tmp_path,
            episode_inputs=[_mk_input("show:1", "ep:1", "hello")],
            backend=_mk_backend(),
        )
    assert r.status == "failed"
    assert r.error_code == RebuildErrorCode.CACHE_WRITE_FAILURE.value


def test_embedding_dimension_contract_violation_bubbles_as_systemic(tmp_path: Path) -> None:
    """ET1 must propagate past per-show boundary, never map to ET2."""
    backend = MagicMock()
    # Return a wrong-dim vector; embed_texts will raise from inside rebuild.
    backend.embed_batch.side_effect = lambda texts, lang: [[0.1] * 8 for _ in texts]  # wrong (expected 4)
    with pytest.raises(EmbeddingDimensionContractViolation):
        rebuild_show_cache(
            show_id="show:1",
            identity=_identity(dims=4),
            language="zh-tw",
            cache_dir=tmp_path,
            episode_inputs=[_mk_input("show:1", "ep:1", "hello")],
            backend=backend,
        )


# ── Does-not-do (per RP doc) ────────────────────────────────────────────────
#
# These checks walk the AST so docstrings and comments naming the forbidden
# symbols in a "must NOT" context don't trip the test. We only care about
# imports, name references, attribute accesses, and string constants that
# could be passed to runtime APIs.

def _symbols_referenced_by(module) -> set[str]:
    """Return every name / attribute the module's source references at code level.

    Excludes docstrings and comments — imports, identifier usages, and
    attribute access names are all included so `EpisodeStatusRepository` (as
    a class reference) is caught but a docstring mentioning its absence is not.
    """
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
    return names


def test_primitive_source_does_not_reference_sync_state() -> None:
    """No sync_state / SyncStateRepository at code level."""
    refs = _symbols_referenced_by(sr_mod)
    assert "SyncStateRepository" not in refs
    assert "sync_state" not in refs


def test_primitive_source_does_not_reference_es_bulk() -> None:
    refs = _symbols_referenced_by(sr_mod)
    assert "streaming_bulk" not in refs
    assert "ElasticsearchService" not in refs


def test_primitive_source_does_not_reference_embedding_status_done() -> None:
    """Phase 2a: primitive must not set `embedding_status='done'`."""
    refs = _symbols_referenced_by(sr_mod)
    assert "embedding_status" not in refs
    assert "EpisodeStatusRepository" not in refs
    assert "mark_embedded_batch" not in refs


def test_primitive_does_not_import_embed_episodes() -> None:
    """Isolation: the rebuild primitive must not pull in the batch pipeline.

    Checks imports specifically — not just name references — since the doc
    may legitimately mention `embed_episodes` in its narrative.
    """
    tree = ast.parse(inspect.getsource(sr_mod))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
    forbidden = {m for m in imported_modules if m.endswith("embed_episodes")}
    assert not forbidden, f"show_rebuild imports forbidden modules: {forbidden}"


# ── Loading from filesystem (episode_inputs=None) ───────────────────────────

def test_loads_inputs_from_embedding_input_dir_when_not_preloaded(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    cache_dir = tmp_path / "cache"
    # Write two JSON files for show:1 and one for a different show
    (input_dir / "ep_1.json").write_text(json.dumps(_mk_input("show:1", "ep:1", "hello")))
    (input_dir / "ep_2.json").write_text(json.dumps(_mk_input("show:1", "ep:2", "world")))
    (input_dir / "ep_other.json").write_text(json.dumps(_mk_input("show:2", "ep:99", "other")))

    r = rebuild_show_cache(
        show_id="show:1",
        identity=_identity(),
        language="zh-tw",
        cache_dir=cache_dir,
        embedding_input_dir=input_dir,
        episode_inputs=None,
        backend=_mk_backend(),
    )

    assert r.status == "ok"
    assert r.episode_count == 2
    entry = json.loads(cache_path_for(cache_dir, _identity(), "show:1").read_text())
    assert set(entry["episodes"].keys()) == {"ep:1", "ep:2"}
