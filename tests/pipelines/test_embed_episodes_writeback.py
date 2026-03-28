"""Tests for embed_episodes DB writeback (embedding_status)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.pipelines.embed_episodes import run
from src.storage.episode_status import EpisodeStatusRepository


def _write_input(tmp_dir: Path, episode_id: str, show_id: str) -> None:
    (tmp_dir / f"{episode_id}.json").write_text(
        json.dumps({"episode_id": episode_id, "show_id": show_id, "embedding_input": {"text": "test text"}}),
        encoding="utf-8",
    )


def _write_cleaned(tmp_dir: Path, episode_id: str, show_id: str) -> None:
    (tmp_dir / f"{episode_id}.json").write_text(
        json.dumps({
            "episode_id": episode_id,
            "show_id": show_id,
            "target_index": "podcast-episodes-zh-tw",
            "cleaned": {"normalized": {"title": "Title", "description": "desc"}},
            "original_meta": {},
        }),
        encoding="utf-8",
    )


@pytest.fixture
def dirs(tmp_path):
    input_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    cache_dir = tmp_path / "embeddings"
    input_dir.mkdir(parents=True)
    cleaned_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    return input_dir, cleaned_dir, cache_dir


class TestEmbedEpisodesWriteback:
    def test_mark_embedded_batch_called_on_success(self, dirs, monkeypatch):
        input_dir, cleaned_dir, cache_dir = dirs
        _write_input(input_dir, "ep:001", "show:1")
        _write_input(input_dir, "ep:002", "show:1")
        _write_cleaned(cleaned_dir, "ep:001", "show:1")
        _write_cleaned(cleaned_dir, "ep:002", "show:1")

        # Patch module-level path constants
        monkeypatch.setattr("src.pipelines.embed_episodes.EMBEDDING_INPUT_DIR", input_dir)
        monkeypatch.setattr("src.pipelines.embed_episodes.CLEANED_EPISODES_DIR", cleaned_dir)

        mock_backend = MagicMock()
        mock_backend.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]

        mock_repo = MagicMock(spec=EpisodeStatusRepository)
        mock_db = MagicMock()

        with patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend), \
             patch("src.pipelines.embed_episodes.EpisodeStatusRepository", return_value=mock_repo):
            run(cache_dir=cache_dir, db=mock_db)

        assert mock_repo.mark_embedded_batch.called
        call_args = mock_repo.mark_embedded_batch.call_args
        # episode_ids is the first positional argument
        episode_ids = set(call_args[0][0])
        assert episode_ids == {"ep:001", "ep:002"}

    def test_no_db_does_not_raise(self, dirs, monkeypatch):
        input_dir, cleaned_dir, cache_dir = dirs
        _write_input(input_dir, "ep:001", "show:1")
        _write_cleaned(cleaned_dir, "ep:001", "show:1")

        monkeypatch.setattr("src.pipelines.embed_episodes.EMBEDDING_INPUT_DIR", input_dir)
        monkeypatch.setattr("src.pipelines.embed_episodes.CLEANED_EPISODES_DIR", cleaned_dir)

        mock_backend = MagicMock()
        mock_backend.embed_batch.return_value = [[0.1] * 768]

        with patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
            stats = run(cache_dir=cache_dir, db=None)

        assert stats["written"] == 1

    def test_mark_embedded_batch_not_called_on_write_failure(self, dirs, monkeypatch):
        input_dir, cleaned_dir, cache_dir = dirs
        _write_input(input_dir, "ep:001", "show:1")
        _write_cleaned(cleaned_dir, "ep:001", "show:1")

        monkeypatch.setattr("src.pipelines.embed_episodes.EMBEDDING_INPUT_DIR", input_dir)
        monkeypatch.setattr("src.pipelines.embed_episodes.CLEANED_EPISODES_DIR", cleaned_dir)

        mock_backend = MagicMock()
        mock_backend.embed_batch.return_value = [[0.1] * 768]

        mock_repo = MagicMock(spec=EpisodeStatusRepository)
        mock_db = MagicMock()

        # Patch json.dump (used for cache write) to raise; reads use json.load which is unaffected
        with patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend), \
             patch("src.pipelines.embed_episodes.EpisodeStatusRepository", return_value=mock_repo), \
             patch("src.pipelines.embed_episodes.json.dump", side_effect=OSError("disk full")):
            run(cache_dir=cache_dir, db=mock_db)

        mock_repo.mark_embedded_batch.assert_not_called()
