"""Phase 2a per-show commit (CB1 / CB2) + bucket partition (OB1 / OB2) tests.

Covers the run-time outcome of `EmbedAndIngestPipeline.run()` after
Batch 6 wired in per-show bulk tallying, DB metadata commit, and
OB1/OB2 counters:

- CB1: a show with rebuild_ok AND show_bulk_ok commits its metadata
  through `mark_embedded_daily` (the Phase 2b-A V1e-A daily-path
  artifact-ready writer). `mark_embedded_batch` remains for legacy
  standalone `embed_episodes`; `mark_embedding_metadata_only` stays
  on force_embed / fallback-rebuild paths.
- CB2: cache-hit shows, rebuild-failed shows, and shows with any
  per-doc bulk error do NOT receive a metadata commit.
- show_bulk_ok hard rule: a single per-doc error for a show makes
  its show_bulk_ok false, regardless of how many other docs succeeded.
- OB1: `processed_shows` counts shows where (cache_hit OR rebuild_ok)
  AND show_bulk_ok. Per-doc partial failure excludes the show.
- OB2: `failed_shows` = candidate_shows − processed_shows.
- Missing repo: CB1 skipped gracefully without affecting the rest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock, patch

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline
from src.pipelines.embedding_identity import EmbeddingIdentity
from src.pipelines.show_rebuild import ShowRebuildResult
from src.storage.episode_status import EpisodeStatusRepository


# ── Helpers ─────────────────────────────────────────────────────────────────

EXPECTED_DIMS = 384


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_version="text-v1",
        embedding_dimensions=EXPECTED_DIMS,
    )


def _ok_rebuild(show_id: str) -> ShowRebuildResult:
    return ShowRebuildResult(
        show_id=show_id,
        status="ok",
        cache_written=True,
        episode_count=1,
        identity_used=_identity(),
        new_last_embedded_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
        error_code=None,
        error_message=None,
    )


def _make_pipeline(tmp_path: Path, *, status_repo=None, allowed: Iterable[str] | None = None) -> EmbedAndIngestPipeline:
    emb_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    emb_dir.mkdir(parents=True)
    cleaned_dir.mkdir(parents=True)

    pipeline = EmbedAndIngestPipeline(
        environment="default",
        es_service=MagicMock(),
        embedding_backend=MagicMock(embed_batch=lambda texts, lang: [[0.1] * EXPECTED_DIMS] * len(texts)),
        storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda _: []),
        sync_repo=None,
        episode_status_repo=status_repo,
        allowed_show_ids=set(allowed) if allowed else None,
    )
    pipeline.EMBEDDING_INPUT_DIR = emb_dir
    pipeline.CLEANED_EPISODES_DIR = cleaned_dir
    return pipeline


def _add_episode(pipeline: EmbedAndIngestPipeline, episode_id: str, show_id: str) -> None:
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": "podcast-episodes-zh-tw",
        "cleaned": {"normalized": {"title": f"T {episode_id}", "description": "d"}},
        "original_meta": {"pub_date": None, "duration": None, "audio_url": None, "language": "zh-tw"},
    }
    pipeline._show_cache[show_id] = {
        "show_id": show_id, "title": "Show", "author": "A",
        "image_url": None, "external_urls": {}, "target_index": "podcast-episodes-zh-tw",
    }
    (pipeline.EMBEDDING_INPUT_DIR / f"{episode_id}.json").write_text(
        json.dumps({"episode_id": episode_id, "show_id": show_id, "embedding_input": {"text": "t"}}),
        encoding="utf-8",
    )


def _fake_bulk(items):
    """Consume the actions generator first so to_es_doc side-effects run."""
    def _side_effect(client, actions, **kwargs):
        list(actions)
        return iter(items)
    return _side_effect


def _index_ok(ep_id: str):
    return (True, {"index": {"_id": ep_id, "_index": "episodes-zh-tw-v3", "_result": "created"}})


def _index_fail(ep_id: str):
    return (False, {"index": {"_id": ep_id, "error": {"type": "mapper_error"}}})


# ── CB1: happy path per-show commit ────────────────────────────────────────

class TestCB1:
    def test_rebuild_ok_plus_bulk_ok_commits_metadata_only(self, tmp_path):
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")
        # Simulate the rebuild happened during _load_vector_cache.
        pipeline._rebuild_results["show:1"] = _ok_rebuild("show:1")

        items = [_index_ok("ep:001"), _index_ok("ep:002")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            stats = pipeline.run()

        # CB1 now uses mark_embedded_daily per V1e-A: writes
        # embedding_status='done' together with the embedding metadata.
        # Legacy mark_embedded_batch and metadata-only are both quiet.
        assert repo.mark_embedded_daily.call_count == 1
        assert repo.mark_embedded_batch.call_count == 0
        call = repo.mark_embedded_daily.call_args
        assert sorted(call.kwargs["episode_ids"]) == ["ep:001", "ep:002"]
        assert call.kwargs["model"] == _identity().model_name
        assert call.kwargs["version"] == _identity().embedding_version
        assert stats["committed_shows"] == 1


# ── CB2: commit is skipped in three distinct cases ─────────────────────────

class TestCB2:
    def test_rebuild_ok_but_any_per_doc_failure_blocks_commit(self, tmp_path):
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")
        pipeline._rebuild_results["show:1"] = _ok_rebuild("show:1")

        mixed = [_index_ok("ep:001"), _index_fail("ep:002")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(mixed)):
            stats = pipeline.run()

        assert repo.mark_embedded_daily.call_count == 0
        assert stats["committed_shows"] == 0

    def test_rebuild_failed_show_is_not_committed(self, tmp_path):
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        # Rebuild failed → NOT inserted into _rebuild_results, but recorded
        # in _rebuild_failures.
        pipeline._rebuild_failures.append(
            {"show_id": "show:1", "error_code": "zero_episode_in_candidate",
             "error_message": "e", "rebuild_reason": "cache_miss"}
        )

        items = [_index_ok("ep:001")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            pipeline.run()

        assert repo.mark_embedded_daily.call_count == 0

    def test_cache_hit_show_is_not_committed(self, tmp_path):
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        # Cache-hit shows never enter _rebuild_results — matches the Phase 2a
        # three-prohibitions contract (no cache rewrite, no DB write, no
        # encoder call).
        pipeline._cache_hit_count = 1
        pipeline._cache_hit_show_ids.add("show:1")

        items = [_index_ok("ep:001")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            pipeline.run()

        assert repo.mark_embedded_daily.call_count == 0


# ── show_bulk_ok hard rule ─────────────────────────────────────────────────

class TestShowBulkOk:
    def test_single_per_doc_error_flips_show_bulk_ok_to_false(self, tmp_path):
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        # Four episodes; one fails — that alone disqualifies the whole show.
        for n in range(1, 5):
            _add_episode(pipeline, f"ep:00{n}", "show:1")
        pipeline._rebuild_results["show:1"] = _ok_rebuild("show:1")

        items = [
            _index_ok("ep:001"),
            _index_ok("ep:002"),
            _index_fail("ep:003"),  # one fail is enough
            _index_ok("ep:004"),
        ]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            stats = pipeline.run()

        assert repo.mark_embedded_daily.call_count == 0
        assert stats["committed_shows"] == 0


# ── OB1 / OB2 partition ────────────────────────────────────────────────────

class TestOB1OB2:
    def test_processed_plus_failed_equals_candidate_count(self, tmp_path):
        """Candidate set: 3 shows with distinct outcomes."""
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo,
                                  allowed={"hit:1", "rebuild:1", "mixed:1"})
        _add_episode(pipeline, "ep:hit", "hit:1")
        _add_episode(pipeline, "ep:rb", "rebuild:1")
        _add_episode(pipeline, "ep:mx:ok", "mixed:1")
        _add_episode(pipeline, "ep:mx:bad", "mixed:1")

        pipeline._cache_hit_count = 1  # hit:1 counted as cache hit
        pipeline._cache_hit_show_ids.add("hit:1")
        pipeline._rebuild_results["rebuild:1"] = _ok_rebuild("rebuild:1")
        pipeline._rebuild_results["mixed:1"] = _ok_rebuild("mixed:1")

        items = [
            _index_ok("ep:hit"),      # hit:1 bulk ok
            _index_ok("ep:rb"),       # rebuild:1 bulk ok
            _index_ok("ep:mx:ok"),    # mixed:1 partial
            _index_fail("ep:mx:bad"),
        ]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            stats = pipeline.run()

        # hit:1 and rebuild:1 processed; mixed:1 failed (per-doc error)
        assert stats["processed_shows"] == 2
        assert stats["failed_shows"] == 1
        # rebuild:1 commits; hit:1 does NOT (cache hit three-prohibitions);
        # mixed:1 does NOT (show_bulk_ok=false).
        assert stats["committed_shows"] == 1

    def test_per_doc_partial_failure_excludes_show_from_processed(self, tmp_path):
        """OB1 tightened: even if most docs succeed, any failure → OB2."""
        repo = MagicMock(spec=EpisodeStatusRepository)
        repo._db = MagicMock()
        pipeline = _make_pipeline(tmp_path, status_repo=repo, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")
        pipeline._rebuild_results["show:1"] = _ok_rebuild("show:1")

        items = [_index_ok("ep:001"), _index_fail("ep:002")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            stats = pipeline.run()

        assert stats["processed_shows"] == 0
        assert stats["failed_shows"] == 1


# ── Missing repo: graceful skip ────────────────────────────────────────────

class TestNoRepo:
    def test_commit_skipped_when_episode_status_repo_is_none(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, status_repo=None, allowed=["show:1"])
        _add_episode(pipeline, "ep:001", "show:1")
        pipeline._rebuild_results["show:1"] = _ok_rebuild("show:1")

        items = [_index_ok("ep:001")]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(items)):
            stats = pipeline.run()

        # Run completes successfully; no commit attempted.
        assert stats["success"] == 1
        assert stats["errors"] == 0
        assert stats["committed_shows"] == 0
