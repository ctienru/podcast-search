"""Tests for prepare_embedding_input pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import src.pipelines.prepare_embedding_input as prep
from src.pipelines.prepare_embedding_input import run, _build_embedding_text


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_cleaned(
    directory: Path,
    episode_id: str,
    show_id: str,
    title: str = "Test Title",
    paragraphs: list | None = None,
    target_index: str = "podcast-episodes-zh-tw",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if paragraphs is None:
        paragraphs = [
            {"index": 0, "text": "First paragraph.", "char_count": 16, "flags": [], "kept": True, "rules_hit": []},
        ]
    data = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": target_index,
        "cleaned": {
            "normalized": {"title": title, "description": ""},
            "paragraphs": paragraphs,
            "stats": {},
        },
        "cleaning_meta": {},
        "original_meta": {},
    }
    safe = episode_id.replace(":", "_") + ".json"
    (directory / safe).write_text(json.dumps(data), encoding="utf-8")


def _read_output(output_dir: Path, episode_id: str) -> dict:
    safe = episode_id.replace(":", "_") + ".json"
    return json.loads((output_dir / safe).read_text(encoding="utf-8"))


def _patch_dirs(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    output_dir = tmp_path / "embedding_input"
    return cleaned_dir, output_dir


# ── Unit tests for _build_embedding_text ──────────────────────────────────────


def test_build_text_title_and_kept_paragraphs() -> None:
    cleaned = {
        "cleaned": {
            "normalized": {"title": "My Title"},
            "paragraphs": [
                {"text": "Para one.", "kept": True},
                {"text": "Para two.", "kept": True},
                {"text": "Removed.", "kept": False},
            ],
        }
    }
    result = _build_embedding_text(cleaned)
    assert result == "My Title\n\nPara one.\n\nPara two."


def test_build_text_no_kept_paragraphs() -> None:
    cleaned = {
        "cleaned": {
            "normalized": {"title": "Only Title"},
            "paragraphs": [
                {"text": "Removed.", "kept": False},
            ],
        }
    }
    result = _build_embedding_text(cleaned)
    assert result == "Only Title"


def test_build_text_empty_paragraphs() -> None:
    cleaned = {
        "cleaned": {
            "normalized": {"title": "Solo Title"},
            "paragraphs": [],
        }
    }
    assert _build_embedding_text(cleaned) == "Solo Title"


def test_build_text_missing_cleaned_section() -> None:
    result = _build_embedding_text({})
    assert result == ""


# ── Integration tests for run() ───────────────────────────────────────────────


def test_fresh_run_writes_files(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(cleaned_dir, "ep-1", "show-A")
    _write_cleaned(cleaned_dir, "ep-2", "show-A")

    stats = run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["written"] == 2
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert stats["total"] == 2

    out = _read_output(output_dir, "ep-1")
    assert out["episode_id"] == "ep-1"
    assert out["show_id"] == "show-A"
    assert "text" in out["embedding_input"]
    assert "Test Title" in out["embedding_input"]["text"]


def test_output_format(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(
        cleaned_dir, "ep-1", "show-A",
        title="Episode One",
        paragraphs=[
            {"text": "Kept para.", "kept": True, "index": 0, "char_count": 10, "flags": [], "rules_hit": []},
            {"text": "Removed.", "kept": False, "index": 1, "char_count": 8, "flags": [], "rules_hit": []},
        ],
    )
    run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    out = _read_output(output_dir, "ep-1")
    assert out["embedding_input"]["text"] == "Episode One\n\nKept para."


def test_incremental_skips_existing(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(cleaned_dir, "ep-1", "show-A")

    run(cleaned_dir=cleaned_dir, output_dir=output_dir)
    first_mtime = (output_dir / "ep-1.json").stat().st_mtime

    stats = run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["written"] == 0
    assert stats["skipped"] == 1
    assert (output_dir / "ep-1.json").stat().st_mtime == first_mtime


def test_force_overwrites_existing(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(cleaned_dir, "ep-1", "show-A")

    run(cleaned_dir=cleaned_dir, output_dir=output_dir)
    stats = run(force=True, cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["written"] == 1
    assert stats["skipped"] == 0


def test_show_id_filter(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(cleaned_dir, "ep-A1", "show-A")
    _write_cleaned(cleaned_dir, "ep-B1", "show-B")

    stats = run(allowed_show_ids={"show-A"}, cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["written"] == 1
    assert stats["total"] == 1
    assert (output_dir / "ep-A1.json").exists()
    assert not (output_dir / "ep-B1.json").exists()


def test_empty_cleaned_dir(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    cleaned_dir.mkdir(parents=True)

    stats = run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["total"] == 0
    assert stats["written"] == 0


def test_missing_cleaned_dir(tmp_path: Path) -> None:
    cleaned_dir = tmp_path / "nonexistent"
    output_dir = tmp_path / "output"

    stats = run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["total"] == 0
    assert stats["written"] == 0


def test_episode_with_colon_in_id(tmp_path: Path) -> None:
    cleaned_dir, output_dir = _patch_dirs(tmp_path)
    _write_cleaned(cleaned_dir, "episode:apple:123", "show-A")

    stats = run(cleaned_dir=cleaned_dir, output_dir=output_dir)

    assert stats["written"] == 1
    assert (output_dir / "episode_apple_123.json").exists()
    out = _read_output(output_dir, "episode:apple:123")
    assert out["episode_id"] == "episode:apple:123"
