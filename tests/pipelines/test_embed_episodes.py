"""Tests for embed_episodes pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.pipelines.embed_episodes as embed_ep
from src.pipelines.embed_episodes import run


# ── Helpers ───────────────────────────────────────────────────────────────────

ZH_MODEL = "BAAI/bge-base-zh-v1.5"
FAKE_VEC = [0.1] * 768


def _write_embedding_input(directory: Path, episode_id: str, show_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "episode_id": episode_id,
        "show_id": show_id,
        "embedding_input": {"text": f"Title of {episode_id}"},
    }
    (directory / f"{episode_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_cleaned_episode(
    directory: Path,
    episode_id: str,
    show_id: str,
    target_index: str = "podcast-episodes-zh-tw",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": target_index,
    }
    (directory / f"{episode_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_cache(cache_dir: Path, show_id: str, episode_ids: list[str], model_name: str = ZH_MODEL) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "show_id": show_id,
        "model_key": "zh",
        "model_name": model_name,
        "embedded_at": "2026-03-26T00:00:00+00:00",
        "episodes": {ep_id: FAKE_VEC for ep_id in episode_ids},
    }
    (cache_dir / f"{show_id}.json").write_text(json.dumps(entry), encoding="utf-8")


def _read_cache(cache_dir: Path, show_id: str) -> dict:
    return json.loads((cache_dir / f"{show_id}.json").read_text(encoding="utf-8"))


def _patch_dirs(tmp_path: Path):
    """Return a context manager that patches EMBEDDING_INPUT_DIR and CLEANED_EPISODES_DIR."""
    embedding_input_dir = tmp_path / "embedding_input"
    cleaned_dir = tmp_path / "cleaned"
    return (
        patch.object(embed_ep, "EMBEDDING_INPUT_DIR", embedding_input_dir),
        patch.object(embed_ep, "CLEANED_EPISODES_DIR", cleaned_dir),
        embedding_input_dir,
        cleaned_dir,
    )


def _mock_backend(num_texts: int = 1):
    mock = MagicMock()
    mock.embed_batch.side_effect = lambda texts, lang: [[0.2] * 768 for _ in texts]
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_fresh_run(tmp_path: Path) -> None:
    """No existing cache → all episodes get embedded and cache is written."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-1", "show-A")
        _write_embedding_input(ei_dir, "ep-2", "show-A")
        _write_cleaned_episode(cl_dir, "ep-1", "show-A")
        _write_cleaned_episode(cl_dir, "ep-2", "show-A")

        stats = run(cache_dir=cache_dir)

    assert stats["written"] == 2
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    cached = _read_cache(cache_dir, "show-A")
    assert set(cached["episodes"].keys()) == {"ep-1", "ep-2"}
    assert cached["model_name"] == ZH_MODEL


def test_full_cache_hit(tmp_path: Path) -> None:
    """All episodes already in cache with same model → nothing re-embedded."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-1", "show-A")
        _write_embedding_input(ei_dir, "ep-2", "show-A")
        _write_cleaned_episode(cl_dir, "ep-1", "show-A")
        _write_cleaned_episode(cl_dir, "ep-2", "show-A")
        _write_cache(cache_dir, "show-A", ["ep-1", "ep-2"])

        stats = run(cache_dir=cache_dir)

    assert stats["written"] == 0
    assert stats["skipped"] == 2
    mock_backend.embed_batch.assert_not_called()


def test_partial_cache_miss(tmp_path: Path) -> None:
    """2 episodes cached, 1 new → only new episode embedded; cache contains all 3."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-1", "show-A")
        _write_embedding_input(ei_dir, "ep-2", "show-A")
        _write_embedding_input(ei_dir, "ep-3", "show-A")  # new episode
        _write_cleaned_episode(cl_dir, "ep-1", "show-A")
        _write_cleaned_episode(cl_dir, "ep-2", "show-A")
        _write_cleaned_episode(cl_dir, "ep-3", "show-A")
        _write_cache(cache_dir, "show-A", ["ep-1", "ep-2"])  # ep-3 not yet cached

        stats = run(cache_dir=cache_dir)

    assert stats["written"] == 1   # only ep-3
    assert stats["skipped"] == 2  # ep-1 and ep-2
    assert stats["failed"] == 0

    # embed_batch was called exactly once, with just ep-3
    assert mock_backend.embed_batch.call_count == 1
    texts_sent = mock_backend.embed_batch.call_args[0][0]
    assert len(texts_sent) == 1

    # Final cache must contain all 3 episodes
    cached = _read_cache(cache_dir, "show-A")
    assert set(cached["episodes"].keys()) == {"ep-1", "ep-2", "ep-3"}


def test_model_mismatch_triggers_full_reembed(tmp_path: Path) -> None:
    """Cache exists but model_name differs → all episodes re-embedded, skipped=0."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-1", "show-A")
        _write_cleaned_episode(cl_dir, "ep-1", "show-A")
        _write_cache(cache_dir, "show-A", ["ep-1"], model_name="old-model-v1")

        stats = run(cache_dir=cache_dir)

    assert stats["written"] == 1
    assert stats["skipped"] == 0
    mock_backend.embed_batch.assert_called_once()


def test_force_flag_bypasses_cache(tmp_path: Path) -> None:
    """force=True re-embeds even when valid cache exists."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-1", "show-A")
        _write_cleaned_episode(cl_dir, "ep-1", "show-A")
        _write_cache(cache_dir, "show-A", ["ep-1"])

        stats = run(force=True, cache_dir=cache_dir)

    assert stats["written"] == 1
    assert stats["skipped"] == 0
    mock_backend.embed_batch.assert_called_once()


def test_show_id_filter(tmp_path: Path) -> None:
    """allowed_show_ids restricts which shows are processed."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-A", "show-A")
        _write_embedding_input(ei_dir, "ep-B", "show-B")
        _write_cleaned_episode(cl_dir, "ep-A", "show-A")
        _write_cleaned_episode(cl_dir, "ep-B", "show-B")

        stats = run(allowed_show_ids={"show-A"}, cache_dir=cache_dir)

    assert stats["written"] == 1
    assert (cache_dir / "show-A.json").exists()
    assert not (cache_dir / "show-B.json").exists()


def test_missing_language_counts_as_failed(tmp_path: Path) -> None:
    """Episode with no cleaned data (unknown language) increments stats['failed']."""
    p_input, p_cleaned, ei_dir, cl_dir = _patch_dirs(tmp_path)
    cache_dir = tmp_path / "embeddings"
    mock_backend = _mock_backend()

    with p_input, p_cleaned, patch("src.pipelines.embed_episodes.LocalEmbeddingBackend", return_value=mock_backend):
        _write_embedding_input(ei_dir, "ep-orphan", "show-Z")
        # No cleaned episode file → language detection returns None for all eps
        # → entire show has no valid language → stats["failed"] incremented

        stats = run(cache_dir=cache_dir)

    assert stats["failed"] > 0
    assert stats["written"] == 0
    mock_backend.embed_batch.assert_not_called()
