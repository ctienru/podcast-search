"""Tests for embed_and_ingest writeback logic (search_sync_state)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline
from src.storage.sync_state import SyncStateRepository


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline(tmp_path: Path, sync_repo=None) -> EmbedAndIngestPipeline:
    """Return a pipeline wired with mock ES and in-memory caches."""
    embedding_input_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    embedding_input_dir.mkdir(parents=True)
    cleaned_dir.mkdir(parents=True)

    pipeline = EmbedAndIngestPipeline(
        environment="default",
        es_service=MagicMock(),
        embedding_backend=MagicMock(embed_batch=lambda texts, lang: [[0.1] * 384] * len(texts)),
        storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda s: []),
        sync_repo=sync_repo,
    )
    pipeline.EMBEDDING_INPUT_DIR = embedding_input_dir
    pipeline.CLEANED_EPISODES_DIR = cleaned_dir
    return pipeline


def _add_episode(pipeline: EmbedAndIngestPipeline, episode_id: str, show_id: str) -> None:
    """Inject a pre-built episode into the pipeline's in-memory caches."""
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": "podcast-episodes-zh-tw",
        "cleaned": {"normalized": {"title": f"Title {episode_id}", "description": "desc"}},
        "original_meta": {"pub_date": None, "duration": None, "audio_url": None, "language": "zh-tw"},
    }
    pipeline._show_cache[show_id] = {
        "show_id": show_id,
        "title": "Test Show",
        "author": "Author",
        "image_url": None,
        "external_urls": {},
        "target_index": "podcast-episodes-zh-tw",
    }
    (pipeline.EMBEDDING_INPUT_DIR / f"{episode_id}.json").write_text(
        json.dumps({"episode_id": episode_id, "show_id": show_id, "embedding_input": {"text": "text"}}),
        encoding="utf-8",
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def _fake_bulk(ok_items):
    """Return a streaming_bulk side_effect that consumes the actions generator first.

    This ensures build_actions() → to_es_doc() runs so _episode_aliases is populated,
    matching production behaviour where streaming_bulk iterates the generator.
    """
    def _side_effect(client, actions, **kwargs):
        list(actions)  # consume the generator so to_es_doc() populates _episode_aliases
        return iter(ok_items)
    return _side_effect


class TestWritebackSyncState:
    def test_mark_done_called_for_successful_episodes(self, tmp_path):
        sync_repo = MagicMock(spec=SyncStateRepository)
        sync_repo._db = MagicMock()

        pipeline = _make_pipeline(tmp_path, sync_repo=sync_repo)
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")

        ok_items = [
            (True, {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3", "_result": "created"}}),
            (True, {"index": {"_id": "ep:002", "_index": "episodes-zh-tw-v3", "_result": "created"}}),
        ]

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(ok_items)):
            pipeline.run()

        assert sync_repo.mark_done.call_count == 2
        calls = sync_repo.mark_done.call_args_list
        episode_ids = {c.kwargs["entity_id"] for c in calls}
        assert episode_ids == {"ep:001", "ep:002"}
        for c in calls:
            assert c.kwargs["entity_type"] == "episode"
            assert c.kwargs["index_alias"] == "episodes-zh-tw"
            assert c.kwargs["environment"] == "default"

    def test_mark_done_all_or_nothing_on_partial_failure(self, tmp_path):
        """Phase 1 R3: any doc error → zero mark_done calls (all-or-nothing).

        Replaces the pre-Phase-1 `test_mark_done_not_called_for_failed_episodes`
        which asserted partial flush (succeeded docs still flushed). Phase 1
        tightens this to all-or-nothing: if ANY doc fails, NO rows are flushed.
        """
        sync_repo = MagicMock(spec=SyncStateRepository)
        sync_repo._db = MagicMock()

        pipeline = _make_pipeline(tmp_path, sync_repo=sync_repo)
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")

        mixed_items = [
            (True,  {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}}),
            (False, {"index": {"_id": "ep:002", "error": {"type": "mapper_error"}}}),
        ]

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(mixed_items)):
            pipeline.run()

        assert sync_repo.mark_done.call_count == 0
        sync_repo.commit.assert_not_called()

    def test_no_sync_repo_does_not_raise(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, sync_repo=None)
        _add_episode(pipeline, "ep:001", "show:1")

        ok_items = [(True, {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}})]

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(ok_items)):
            stats = pipeline.run()

        assert stats["success"] == 1

    def test_environment_passed_from_constructor(self, tmp_path):
        """Phase 1: environment is caller-injected via constructor, not read from settings.

        Replaces the pre-Phase-1 `test_environment_passed_from_settings` which
        monkeypatched `settings.ES_ENV` — that path is removed in Phase 1.
        """
        sync_repo = MagicMock(spec=SyncStateRepository)
        sync_repo._db = MagicMock()

        embedding_input_dir = tmp_path / "embedding_input" / "episodes"
        embedding_input_dir.mkdir(parents=True)
        (tmp_path / "cleaned" / "episodes").mkdir(parents=True)

        pipeline = EmbedAndIngestPipeline(
            environment="local",
            es_service=MagicMock(),
            embedding_backend=MagicMock(embed_batch=lambda texts, lang: [[0.1] * 384] * len(texts)),
            storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda s: []),
            sync_repo=sync_repo,
        )
        pipeline.EMBEDDING_INPUT_DIR = embedding_input_dir
        pipeline.CLEANED_EPISODES_DIR = tmp_path / "cleaned" / "episodes"
        _add_episode(pipeline, "ep:001", "show:1")

        ok_items = [(True, {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}})]

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(ok_items)):
            pipeline.run()

        assert sync_repo.mark_done.call_args.kwargs["environment"] == "local"

    def test_update_operation_episode_id_captured(self, tmp_path):
        """BM25-only mode uses 'update' op type; episode_id must still be collected."""
        sync_repo = MagicMock(spec=SyncStateRepository)
        sync_repo._db = MagicMock()

        pipeline = _make_pipeline(tmp_path, sync_repo=sync_repo)
        _add_episode(pipeline, "ep:bm25", "show:1")

        update_item = [(True, {"update": {"_id": "ep:bm25", "_index": "episodes-zh-tw-v3"}})]

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(update_item)):
            pipeline.run()

        assert sync_repo.mark_done.called
        assert sync_repo.mark_done.call_args.kwargs["entity_id"] == "ep:bm25"
