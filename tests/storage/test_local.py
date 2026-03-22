"""Tests for LocalStorage."""

import json
import tempfile
from pathlib import Path

import pytest

from src.storage.base import StorageBase
from src.storage.local import LocalStorage
from src.types import Show


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "normalized" / "shows").mkdir(parents=True)
        (data_dir / "normalized" / "episodes").mkdir(parents=True)
        yield data_dir


@pytest.fixture
def storage(temp_data_dir):
    return LocalStorage(temp_data_dir)


# ── StorageBase interface ────────────────────────────────────────────────────

class TestStorageBaseContract:
    def test_is_storage_base_subclass(self, storage):
        assert isinstance(storage, StorageBase)

    def test_get_shows_returns_empty_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = LocalStorage(Path(tmpdir))
            assert list(s.get_shows()) == []

    def test_get_shows_yields_show_dataclasses(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        (shows_dir / "s1.json").write_text(
            json.dumps({
                "show_id": "s1", "title": "T1", "author": "A",
                "language_detected": "zh-tw", "language_confidence": 0.95,
                "language_uncertain": False, "target_index": "podcast-episodes-zh-tw",
                "rss_feed_url": "http://x", "updated_at": "2026-01-01T00:00:00Z",
            }),
            encoding="utf-8",
        )
        shows = list(storage.get_shows())
        assert len(shows) == 1
        assert isinstance(shows[0], Show)
        assert shows[0].show_id == "s1"
        assert shows[0].language_detected == "zh-tw"
        assert shows[0].target_index == "podcast-episodes-zh-tw"

    def test_get_shows_filters_by_language(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        for show_id, lang in [("s1", "zh-tw"), ("s2", "en"), ("s3", "zh-tw")]:
            (shows_dir / f"{show_id}.json").write_text(
                json.dumps({"show_id": show_id, "language_detected": lang}),
                encoding="utf-8",
            )
        result = list(storage.get_shows(language="zh-tw"))
        assert len(result) == 2
        assert all(s.language_detected == "zh-tw" for s in result)

    def test_get_shows_updated_since_filters_by_timestamp(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        (shows_dir / "old.json").write_text(
            json.dumps({"show_id": "old", "updated_at": "2026-01-01T00:00:00Z"}),
            encoding="utf-8",
        )
        (shows_dir / "new.json").write_text(
            json.dumps({"show_id": "new", "updated_at": "2026-03-20T00:00:00Z"}),
            encoding="utf-8",
        )
        result = list(storage.get_shows_updated_since(since="2026-02-01T00:00:00Z"))
        assert len(result) == 1
        assert result[0].show_id == "new"

    def test_get_shows_updated_since_empty_returns_all(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        for i in range(3):
            (shows_dir / f"s{i}.json").write_text(
                json.dumps({"show_id": f"s{i}"}), encoding="utf-8"
            )
        assert len(list(storage.get_shows_updated_since(since=""))) == 3


# ── Legacy helpers ───────────────────────────────────────────────────────────

class TestLegacyHelpers:
    def test_list_show_ids_returns_empty_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = LocalStorage(Path(tmpdir))
            assert s.list_show_ids() == []

    def test_list_show_ids_returns_sorted_ids(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        (shows_dir / "show_003.json").write_text("{}")
        (shows_dir / "show_001.json").write_text("{}")
        (shows_dir / "show_002.json").write_text("{}")
        (shows_dir / "not_json.txt").write_text("ignore")
        assert storage.list_show_ids() == ["show_001", "show_002", "show_003"]

    def test_load_show_success(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        data = {"show_id": "s1", "title": "Test", "author": "A"}
        (shows_dir / "s1.json").write_text(json.dumps(data), encoding="utf-8")
        assert storage.load_show("s1") == data

    def test_load_show_not_found(self, storage):
        with pytest.raises(FileNotFoundError, match="show_not_found"):
            storage.load_show("nonexistent")

    def test_load_show_with_unicode(self, storage, temp_data_dir):
        shows_dir = temp_data_dir / "normalized" / "shows"
        data = {"show_id": "s1", "title": "測試 Podcast 🎙️"}
        (shows_dir / "s1.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        assert storage.load_show("s1")["title"] == "測試 Podcast 🎙️"

    def test_list_episode_ids_returns_empty_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = LocalStorage(Path(tmpdir))
            assert list(s.list_episode_ids()) == []

    def test_load_episode_success(self, storage, temp_data_dir):
        episodes_dir = temp_data_dir / "normalized" / "episodes"
        data = {"episode_id": "ep1", "title": "Ep 1"}
        (episodes_dir / "ep1.json").write_text(json.dumps(data), encoding="utf-8")
        assert storage.load_episode("ep1") == data

    def test_load_episode_not_found(self, storage):
        with pytest.raises(FileNotFoundError, match="episode_not_found"):
            storage.load_episode("nonexistent")

    def test_initializes_correct_paths(self, temp_data_dir):
        s = LocalStorage(temp_data_dir)
        assert s.shows_dir == temp_data_dir / "normalized" / "shows"
        assert s.episodes_dir == temp_data_dir / "normalized" / "episodes"
