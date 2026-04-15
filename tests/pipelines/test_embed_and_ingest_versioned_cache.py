"""Phase 2a cache-lookup + fallback tests for `embed_and_ingest`.

Covers the `from_cache=True` loading path after the Phase 2a rewire:

- Per-show buckets: cache_hit / cache_miss / cache_identity_mismatch —
  each show lands in exactly one, and `fallback_rebuild_count` equals
  `miss + mismatch` for rebuild-attempted shows.
- Cache-hit prohibitions: no cache rewrite, no rebuild call, no encoder
  call — the cache file is read verbatim into the in-memory vector cache.
- Fallback rebuild: cache_miss or identity_mismatch triggers
  `rebuild_show_cache`, the freshly-written cache is re-read to populate
  the in-memory vector cache.
- Rebuild failure is recorded in `_rebuild_failures` and does not raise;
  the load continues for other shows.
- Systemic halt: `EmbeddingDimensionContractViolation` propagates out of
  `_load_vector_cache`.
- BM25-only (no backend): fallback is skipped gracefully; no rebuild
  attempt, no counter increment on `fallback_rebuild_count`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
    resolve_expected_identity,
)
from src.pipelines.embedding_paths import cache_path_for


# ── Fixtures / helpers ─────────────────────────────────────────────────────

EXPECTED_DIMS = 384  # matches current MODEL_MAP / _DIM_TABLE seed


def _make_pipeline(
    tmp_path: Path,
    *,
    backend: object | None = "default",
    allowed: Iterable[str] | None = None,
) -> EmbedAndIngestPipeline:
    """Build a pipeline with from_cache=True and paths under tmp_path.

    Uses a mock backend whose `embed_batch` returns deterministic 384-dim
    vectors so rebuild flows end-to-end without the real encoder.
    """
    embedding_input_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    cache_dir = tmp_path / "embeddings"
    embedding_input_dir.mkdir(parents=True)
    cleaned_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    if backend == "default":
        backend = MagicMock()
        backend.embed_batch.side_effect = lambda texts, lang: [[0.1] * EXPECTED_DIMS for _ in texts]

    pipeline = EmbedAndIngestPipeline(
        environment="default",
        es_service=MagicMock(),
        embedding_backend=backend,
        storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda _: []),
        sync_repo=None,
        from_cache=True,
        cache_dir=cache_dir,
        allowed_show_ids=set(allowed) if allowed else None,
    )
    pipeline.EMBEDDING_INPUT_DIR = embedding_input_dir
    pipeline.CLEANED_EPISODES_DIR = cleaned_dir
    return pipeline


def _register_show(pipeline: EmbedAndIngestPipeline, show_id: str, language: str = "zh-tw") -> None:
    """Make `_resolve_identity_for_show` able to resolve this show."""
    target_index = {
        "zh-tw": "podcast-episodes-zh-tw",
        "zh-cn": "podcast-episodes-zh-cn",
        "en": "podcast-episodes-en",
    }[language]
    pipeline._show_cache[show_id] = {
        "show_id": show_id, "title": "t", "author": "a",
        "image_url": None, "external_urls": {}, "target_index": target_index,
    }


def _register_episode_input(pipeline: EmbedAndIngestPipeline, show_id: str, episode_id: str, text: str = "hello") -> None:
    """Drop an embedding-input JSON so rebuild can load it."""
    (pipeline.EMBEDDING_INPUT_DIR / f"{episode_id.replace(':', '_')}.json").write_text(
        json.dumps({"show_id": show_id, "episode_id": episode_id, "embedding_input": {"text": text}}),
        encoding="utf-8",
    )


def _write_cache(
    pipeline: EmbedAndIngestPipeline,
    *,
    show_id: str,
    identity: EmbeddingIdentity,
    episodes: dict[str, list[float]] | None = None,
) -> Path:
    """Pre-create a versioned cache file with the given identity."""
    path = cache_path_for(pipeline._cache_dir, identity, show_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "show_id": show_id,
        "model_name": identity.model_name,
        "embedding_version": identity.embedding_version,
        "embedding_dimensions": identity.embedding_dimensions,
        "embedded_at": "2026-04-14T00:00:00+00:00",
        "episodes": episodes or {"ep:1": [0.42] * identity.embedding_dimensions},
    }
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _expected_identity(language: str = "zh-tw") -> EmbeddingIdentity:
    return resolve_expected_identity(language=language)


# ── Cache hit ──────────────────────────────────────────────────────────────

class TestCacheHit:
    def test_populates_vector_cache_and_increments_counter(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _write_cache(
            pipeline, show_id="show:1", identity=_expected_identity(),
            episodes={"ep:a": [0.1] * EXPECTED_DIMS, "ep:b": [0.2] * EXPECTED_DIMS},
        )

        pipeline._load_vector_cache()

        assert pipeline._cache_hit_count == 1
        assert pipeline._cache_miss_count == 0
        assert pipeline._cache_identity_mismatch_count == 0
        assert pipeline._fallback_rebuild_count == 0
        assert pipeline._vector_cache["ep:a"] == [0.1] * EXPECTED_DIMS
        assert pipeline._vector_cache["ep:b"] == [0.2] * EXPECTED_DIMS

    def test_does_not_call_rebuild_on_hit(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _write_cache(pipeline, show_id="show:1", identity=_expected_identity())

        with patch("src.pipelines.embed_and_ingest.rebuild_show_cache") as mock_rebuild:
            pipeline._load_vector_cache()

        mock_rebuild.assert_not_called()

    def test_does_not_rewrite_cache_file_on_hit(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        path = _write_cache(pipeline, show_id="show:1", identity=_expected_identity())
        mtime_before = path.stat().st_mtime_ns
        content_before = path.read_bytes()

        pipeline._load_vector_cache()

        assert path.stat().st_mtime_ns == mtime_before
        assert path.read_bytes() == content_before

    def test_does_not_call_encoder_on_hit(self, tmp_path):
        backend = MagicMock()
        backend.embed_batch.side_effect = AssertionError("backend must not be called on cache hit")
        pipeline = _make_pipeline(tmp_path, backend=backend, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _write_cache(pipeline, show_id="show:1", identity=_expected_identity())

        pipeline._load_vector_cache()  # must not call backend

        backend.embed_batch.assert_not_called()


# ── Cache miss → fallback rebuild ──────────────────────────────────────────

class TestCacheMiss:
    def test_triggers_rebuild_and_populates_vector_cache(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _register_episode_input(pipeline, "show:1", "ep:1")

        pipeline._load_vector_cache()

        assert pipeline._cache_miss_count == 1
        assert pipeline._cache_hit_count == 0
        assert pipeline._cache_identity_mismatch_count == 0
        assert pipeline._fallback_rebuild_count == 1
        assert "ep:1" in pipeline._vector_cache
        assert len(pipeline._vector_cache["ep:1"]) == EXPECTED_DIMS
        # Rebuild result tracked (for upcoming per-show DB commit)
        assert "show:1" in pipeline._rebuild_results
        assert pipeline._rebuild_results["show:1"].status == "ok"

    def test_writes_versioned_cache_file(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _register_episode_input(pipeline, "show:1", "ep:1")

        pipeline._load_vector_cache()

        written = cache_path_for(pipeline._cache_dir, _expected_identity(), "show:1")
        assert written.exists()
        entry = json.loads(written.read_text())
        assert entry["model_name"] == _expected_identity().model_name
        assert entry["embedding_dimensions"] == EXPECTED_DIMS


# ── Identity mismatch → fallback rebuild ───────────────────────────────────

class TestCacheIdentityMismatch:
    def test_triggers_rebuild_and_counter(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _register_episode_input(pipeline, "show:1", "ep:1")
        # Pre-write a cache under the EXPECTED path but with WRONG metadata.
        # cache_path_for uses expected identity; content's model_name is rogue.
        rogue = EmbeddingIdentity(
            model_name="some-other-model",
            embedding_version=_expected_identity().embedding_version,
            embedding_dimensions=EXPECTED_DIMS,
        )
        path = cache_path_for(pipeline._cache_dir, _expected_identity(), "show:1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "show_id": "show:1",
            "model_name": rogue.model_name,  # mismatch!
            "embedding_version": rogue.embedding_version,
            "embedding_dimensions": rogue.embedding_dimensions,
            "embedded_at": "x",
            "episodes": {"ep:old": [0.0] * EXPECTED_DIMS},
        }), encoding="utf-8")

        pipeline._load_vector_cache()

        assert pipeline._cache_identity_mismatch_count == 1
        assert pipeline._cache_miss_count == 0
        assert pipeline._cache_hit_count == 0
        assert pipeline._fallback_rebuild_count == 1
        # Vector cache should have the freshly-rebuilt episode, not the rogue one
        assert "ep:1" in pipeline._vector_cache
        assert "ep:old" not in pipeline._vector_cache

    def test_emits_structured_mismatch_log(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="src.pipelines.embed_and_ingest")

        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _register_episode_input(pipeline, "show:1", "ep:1")
        path = cache_path_for(pipeline._cache_dir, _expected_identity(), "show:1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "show_id": "show:1",
            "model_name": "rogue",
            "embedding_version": _expected_identity().embedding_version,
            "embedding_dimensions": EXPECTED_DIMS,
            "embedded_at": "x",
            "episodes": {"ep:old": [0.0] * EXPECTED_DIMS},
        }), encoding="utf-8")

        pipeline._load_vector_cache()

        records = [r for r in caplog.records if r.message == "cache_identity_mismatch_detected"]
        assert len(records) == 1
        extra = records[0]
        assert getattr(extra, "show_id") == "show:1"
        assert getattr(extra, "drift_kind") == "model_mismatch"
        assert getattr(extra, "found_model") == "rogue"


# ── Counter partition ──────────────────────────────────────────────────────

class TestCounterPartition:
    def test_four_shows_sum_to_expected_buckets(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed={"hit:a", "hit:b", "miss:1", "mismatch:1"})

        # 2 hits
        for sid in ("hit:a", "hit:b"):
            _register_show(pipeline, sid)
            _write_cache(pipeline, show_id=sid, identity=_expected_identity())

        # 1 miss
        _register_show(pipeline, "miss:1")
        _register_episode_input(pipeline, "miss:1", "ep:m1")

        # 1 mismatch
        _register_show(pipeline, "mismatch:1")
        _register_episode_input(pipeline, "mismatch:1", "ep:mm1")
        mismatch_path = cache_path_for(pipeline._cache_dir, _expected_identity(), "mismatch:1")
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        mismatch_path.write_text(json.dumps({
            "show_id": "mismatch:1",
            "model_name": "rogue",
            "embedding_version": _expected_identity().embedding_version,
            "embedding_dimensions": EXPECTED_DIMS,
            "embedded_at": "x",
            "episodes": {"ep:old": [0.0] * EXPECTED_DIMS},
        }), encoding="utf-8")

        pipeline._load_vector_cache()

        assert pipeline._cache_hit_count == 2
        assert pipeline._cache_miss_count == 1
        assert pipeline._cache_identity_mismatch_count == 1
        assert pipeline._fallback_rebuild_count == 2  # miss + mismatch


# ── Rebuild failure is recorded, not raised ────────────────────────────────

class TestRebuildFailure:
    def test_failed_rebuild_recorded_and_load_continues(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed={"bad:1", "good:1"})
        _register_show(pipeline, "bad:1")
        _register_show(pipeline, "good:1")
        _register_episode_input(pipeline, "good:1", "ep:good")
        # bad:1 has no embedding_input → rebuild returns ZERO_EPISODE_IN_CANDIDATE

        pipeline._load_vector_cache()

        assert pipeline._cache_miss_count == 2
        assert pipeline._fallback_rebuild_count == 2
        assert len(pipeline._rebuild_failures) == 1
        assert pipeline._rebuild_failures[0]["show_id"] == "bad:1"
        assert pipeline._rebuild_failures[0]["error_code"] == "zero_episode_in_candidate"
        # Good show still populated
        assert "ep:good" in pipeline._vector_cache


# ── BM25-only (no backend) ─────────────────────────────────────────────────

class TestNoBackend:
    def test_cache_miss_without_backend_skips_rebuild(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, backend=None, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        # No cache file, no backend — cannot rebuild.

        pipeline._load_vector_cache()

        assert pipeline._cache_miss_count == 1
        assert pipeline._fallback_rebuild_count == 0  # rebuild skipped
        assert "show:1" not in pipeline._rebuild_results

    def test_cache_hit_without_backend_still_populates(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, backend=None, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _write_cache(
            pipeline, show_id="show:1", identity=_expected_identity(),
            episodes={"ep:1": [0.9] * EXPECTED_DIMS},
        )

        pipeline._load_vector_cache()

        assert pipeline._cache_hit_count == 1
        assert pipeline._vector_cache["ep:1"] == [0.9] * EXPECTED_DIMS


# ── Systemic halt: ET1 bubbles out ─────────────────────────────────────────

class TestEt1Bubbles:
    def test_dimension_contract_violation_propagates(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, allowed=["show:1"])
        _register_show(pipeline, "show:1")
        _register_episode_input(pipeline, "show:1", "ep:1")

        # Force rebuild to raise ET1 by making backend return wrong-dim vectors.
        pipeline._embedding_backend.embed_batch.side_effect = (
            lambda texts, lang: [[0.0] * (EXPECTED_DIMS * 2) for _ in texts]
        )

        with pytest.raises(EmbeddingDimensionContractViolation):
            pipeline._load_vector_cache()
