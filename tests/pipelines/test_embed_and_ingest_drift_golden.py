"""End-to-end golden tests for Phase 2a drift non-interference.

These tests run the full `EmbedAndIngestPipeline.run()` from a stale
cache file on disk through `_load_vector_cache` → `rebuild_show_cache`
→ bulk ingest → per-show metadata commit, without short-circuiting
any step. The existing per-show commit and versioned-cache tests each
check one layer in isolation; this file exists specifically to prove
the layers are wired together correctly.

Concretely:

1. `test_end_to_end_mismatch_fallback_commits_fresh_identity`
   Puts a cache file on disk with the legacy flat shape and a stale
   `embedding_version`, runs the pipeline, and captures the actions
   fed into `streaming_bulk`. Asserts that:
     - Mismatch was detected (`cache_identity_mismatch_count == 1`).
     - Fallback fired (`fallback_rebuild_count == 1`).
     - Every bulk action carries a vector whose length matches the
       expected `embedding_dimensions` (not whatever was on disk).
     - The per-show DB commit ran with the expected identity's
       cleaned `embedding_version` — not the stale version from the
       on-disk file.

2. `test_mixed_hit_and_mismatch_only_rebuilt_show_commits`
   Three shows: two with a matching cache file (hit) and one with a
   stale file (mismatch → fallback). Asserts that hit shows do NOT
   trigger the encoder and do NOT get a DB metadata commit, while
   the rebuilt show does both. This is the "per-show commit precision"
   check from the design doc's §4.5 (e).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline
from src.pipelines.embedding_identity import resolve_expected_identity
from src.pipelines.embedding_paths import cache_path_for
from src.storage.episode_status import EpisodeStatusRepository


EXPECTED_DIMS = 384
EXPECTED_IDENTITY = resolve_expected_identity(language="zh-tw")


# ── Setup helpers ───────────────────────────────────────────────────────────


def _make_pipeline(
    tmp_path: Path,
    *,
    status_repo: object,
    allowed: set[str],
    backend: object | None = None,
) -> EmbedAndIngestPipeline:
    """Build a from_cache pipeline whose encoder returns deterministic vectors.

    Anything that has to hit ES is mocked; anything that has to hit disk
    is routed under `tmp_path`. The encoder returns 384-dim vectors so
    the rebuild primitive's write-side dim check succeeds.
    """
    emb_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    cache_dir = tmp_path / "embeddings"
    emb_dir.mkdir(parents=True)
    cleaned_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    if backend is None:
        backend = MagicMock()
        backend.embed_batch.side_effect = (
            lambda texts, lang: [[0.5] * EXPECTED_DIMS for _ in texts]
        )

    pipeline = EmbedAndIngestPipeline(
        environment="default",
        es_service=MagicMock(),
        embedding_backend=backend,
        storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda _: []),
        sync_repo=None,
        episode_status_repo=status_repo,
        from_cache=True,
        cache_dir=cache_dir,
        allowed_show_ids=allowed,
    )
    pipeline.EMBEDDING_INPUT_DIR = emb_dir
    pipeline.CLEANED_EPISODES_DIR = cleaned_dir
    return pipeline


def _seed_show_and_episode(
    pipeline: EmbedAndIngestPipeline,
    *,
    show_id: str,
    episode_id: str,
) -> None:
    """Populate the in-memory caches plus the embedding_input JSON file
    that `rebuild_show_cache` reads when it has to rebuild."""
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": "podcast-episodes-zh-tw",
        "cleaned": {"normalized": {"title": f"T-{episode_id}", "description": "desc"}},
        "original_meta": {
            "pub_date": None,
            "duration": None,
            "audio_url": None,
            "language": "zh-tw",
        },
    }
    pipeline._show_cache[show_id] = {
        "show_id": show_id,
        "title": "Show",
        "author": "A",
        "image_url": None,
        "external_urls": {},
        "target_index": "podcast-episodes-zh-tw",
    }
    (pipeline.EMBEDDING_INPUT_DIR / f"{episode_id}.json").write_text(
        json.dumps({
            "episode_id": episode_id,
            "show_id": show_id,
            "embedding_input": {"text": f"text for {episode_id}"},
        }),
        encoding="utf-8",
    )


def _write_stale_cache(pipeline: EmbedAndIngestPipeline, show_id: str) -> Path:
    """Materialize a cache file with the right path slug but a stale
    embedding_version so `validate_cache_identity` flags a mismatch."""
    path = cache_path_for(pipeline._cache_dir, EXPECTED_IDENTITY, show_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "show_id": show_id,
            "model_name": EXPECTED_IDENTITY.model_name,
            "embedding_version": "text-v0",  # stale — triggers mismatch
            "embedding_dimensions": EXPECTED_DIMS,
            "embedded_at": "2026-01-01T00:00:00Z",
            "episodes": {
                # Put a wrong-length placeholder so test fails loudly
                # if the stale vectors ever reached the bulk stage.
                "ep-stale": [0.0] * 10,
            },
        }),
        encoding="utf-8",
    )
    return path


def _write_matching_cache(
    pipeline: EmbedAndIngestPipeline,
    *,
    show_id: str,
    episode_ids: list[str],
) -> Path:
    """Materialize a cache file whose identity matches expected — a true
    cache hit."""
    path = cache_path_for(pipeline._cache_dir, EXPECTED_IDENTITY, show_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "show_id": show_id,
            "model_name": EXPECTED_IDENTITY.model_name,
            "embedding_version": EXPECTED_IDENTITY.embedding_version,
            "embedding_dimensions": EXPECTED_DIMS,
            "embedded_at": "2026-03-01T00:00:00Z",
            "episodes": {eid: [0.9] * EXPECTED_DIMS for eid in episode_ids},
        }),
        encoding="utf-8",
    )
    return path


def _capture_bulk(captured: list) -> callable:
    """streaming_bulk side_effect that records the fully-materialized
    actions list and returns one ok result per action."""

    def _side_effect(client, actions, **kwargs):
        materialized = list(actions)
        captured.extend(materialized)
        return iter(
            (True, {"index": {"_id": a["_id"], "_index": a.get("_index", "x"), "_result": "created"}})
            for a in materialized
        )

    return _side_effect


# ── Golden (a)+(c): end-to-end identity preservation ───────────────────────


class TestMismatchFallbackEndToEnd:
    def test_end_to_end_mismatch_fallback_commits_fresh_identity(self, tmp_path: Path) -> None:
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()

        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed={"show:drift"})
        _seed_show_and_episode(pipeline, show_id="show:drift", episode_id="ep:1")
        _seed_show_and_episode(pipeline, show_id="show:drift", episode_id="ep:2")

        _write_stale_cache(pipeline, "show:drift")

        captured: list = []
        with patch(
            "src.pipelines.embed_and_ingest.streaming_bulk",
            side_effect=_capture_bulk(captured),
        ):
            stats = pipeline.run()

        # Mismatch detected and fallback fired.
        assert stats["cache_identity_mismatch_count"] == 1
        assert stats["fallback_rebuild_count"] == 1
        assert stats["cache_hit_count"] == 0

        # Every bulk action carries a fresh 384-dim vector. If the stale
        # 10-dim placeholder had leaked through, this would trip.
        assert len(captured) == 2
        for action in captured:
            assert "_source" in action
            vec = action["_source"].get("embedding")
            assert isinstance(vec, list)
            assert len(vec) == EXPECTED_DIMS

        # Per-show commit ran with the expected identity — not the stale
        # `text-v0` that was on disk.
        assert repo.mark_embedding_metadata_only.call_count == 1
        assert repo.mark_embedded_batch.call_count == 0
        commit = repo.mark_embedding_metadata_only.call_args
        assert commit.kwargs["model"] == EXPECTED_IDENTITY.model_name
        assert commit.kwargs["version"] == EXPECTED_IDENTITY.embedding_version
        assert commit.kwargs["version"] != "text-v0"
        assert sorted(commit.kwargs["episode_ids"]) == ["ep:1", "ep:2"]
        assert stats["committed_shows"] == 1


# ── Golden (e): mixed cache states — per-show commit precision ─────────────


class TestMixedCacheStatesPerShowCommit:
    def test_mixed_hit_and_mismatch_only_rebuilt_show_commits(self, tmp_path: Path) -> None:
        """3 shows: hit + hit + mismatch-fallback.

        Hit shows must not commit DB metadata, must not rewrite their
        cache, and must not call the encoder. The rebuilt show must
        do all three.
        """
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()

        pipeline = _make_pipeline(
            tmp_path,
            status_repo=repo,
            allowed={"show:hit-a", "show:hit-b", "show:drift"},
        )

        # Seed episodes and embedding_input files for all three shows.
        _seed_show_and_episode(pipeline, show_id="show:hit-a", episode_id="ep:a1")
        _seed_show_and_episode(pipeline, show_id="show:hit-b", episode_id="ep:b1")
        _seed_show_and_episode(pipeline, show_id="show:drift", episode_id="ep:d1")

        # Two matching caches (hits) and one stale cache (mismatch).
        hit_a_path = _write_matching_cache(
            pipeline, show_id="show:hit-a", episode_ids=["ep:a1"],
        )
        hit_b_path = _write_matching_cache(
            pipeline, show_id="show:hit-b", episode_ids=["ep:b1"],
        )
        _write_stale_cache(pipeline, "show:drift")

        hit_a_mtime = hit_a_path.stat().st_mtime_ns
        hit_b_mtime = hit_b_path.stat().st_mtime_ns
        hit_a_content = hit_a_path.read_text()
        hit_b_content = hit_b_path.read_text()

        captured: list = []
        with patch(
            "src.pipelines.embed_and_ingest.streaming_bulk",
            side_effect=_capture_bulk(captured),
        ):
            stats = pipeline.run()

        # Counters partition cleanly: 2 hits, 0 misses, 1 mismatch.
        assert stats["cache_hit_count"] == 2
        assert stats["cache_miss_count"] == 0
        assert stats["cache_identity_mismatch_count"] == 1
        assert stats["fallback_rebuild_count"] == 1

        # Hit shows: cache files untouched (mtime + content stable).
        assert hit_a_path.stat().st_mtime_ns == hit_a_mtime
        assert hit_b_path.stat().st_mtime_ns == hit_b_mtime
        assert hit_a_path.read_text() == hit_a_content
        assert hit_b_path.read_text() == hit_b_content

        # Only the rebuilt show's episodes triggered the encoder.
        # `show:drift` has one episode → one encode call via rebuild.
        # Hit shows never reach `embed_batch`.
        calls = pipeline._embedding_backend.embed_batch.call_args_list
        all_texts = [t for call in calls for t in call.args[0]]
        assert any("ep:d1" in t or "text for ep:d1" in t for t in all_texts)
        assert not any("text for ep:a1" in t for t in all_texts)
        assert not any("text for ep:b1" in t for t in all_texts)

        # Exactly one DB commit — for the rebuilt show only.
        assert repo.mark_embedding_metadata_only.call_count == 1
        commit = repo.mark_embedding_metadata_only.call_args
        assert sorted(commit.kwargs["episode_ids"]) == ["ep:d1"]
        assert stats["committed_shows"] == 1

        # All three shows processed successfully (each got exactly one
        # bulk action, all ok).
        assert len(captured) == 3
        assert stats["processed_shows"] == 3
        assert stats["failed_shows"] == 0
