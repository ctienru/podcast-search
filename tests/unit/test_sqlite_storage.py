"""Unit tests for SQLiteStorage."""

from pathlib import Path

import pytest
from sqlite_utils import Database

from src.storage.base import StorageBase
from src.storage.sqlite import SQLiteStorage
from src.types import Show


def _seed_db(db_path: Path, rows: list[dict]) -> None:
    """Create and populate the shows table in a temp SQLite DB."""
    db = Database(db_path)
    db["shows"].insert_all(rows)


_BASE_ROW = {
    "show_id": "s1",
    "title": "T1",
    "author": "A",
    "language_detected": "zh-tw",
    "language_confidence": 0.95,
    "language_uncertain": 0,
    "target_index": "podcast-episodes-zh-tw",
    "rss_feed_url": "http://x",
    "updated_at": "2026-01-01T00:00:00Z",
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "crawler.db"


class TestStorageBaseContract:
    def test_is_storage_base_subclass(self, db_path):
        _seed_db(db_path, [])
        assert isinstance(SQLiteStorage(db_path), StorageBase)


class TestGetShows:
    def test_returns_all_shows(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1"},
            {**_BASE_ROW, "show_id": "s2", "language_detected": "en",
             "target_index": "podcast-episodes-en"},
        ])
        shows = list(SQLiteStorage(db_path).get_shows())
        assert len(shows) == 2

    def test_yields_show_dataclasses(self, db_path):
        _seed_db(db_path, [_BASE_ROW])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert isinstance(show, Show)
        assert show.show_id == "s1"
        assert show.language_detected == "zh-tw"
        assert show.language_uncertain is False

    def test_excludes_null_target_index(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1"},
            {**_BASE_ROW, "show_id": "s2", "target_index": None},
        ])
        shows = list(SQLiteStorage(db_path).get_shows())
        assert len(shows) == 1
        assert shows[0].show_id == "s1"

    def test_excludes_null_language_detected(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1"},
            {**_BASE_ROW, "show_id": "s2", "language_detected": None},
        ])
        shows = list(SQLiteStorage(db_path).get_shows())
        assert len(shows) == 1

    def test_filters_by_language(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1", "language_detected": "zh-tw"},
            {**_BASE_ROW, "show_id": "s2", "language_detected": "en",
             "target_index": "podcast-episodes-en"},
        ])
        result = list(SQLiteStorage(db_path).get_shows(language="zh-tw"))
        assert len(result) == 1
        assert result[0].show_id == "s1"

    def test_returns_empty_for_empty_db(self, db_path):
        _seed_db(db_path, [])
        assert list(SQLiteStorage(db_path).get_shows()) == []


class TestGetShowsUpdatedSince:
    def test_filters_by_timestamp(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "old", "updated_at": "2026-01-01T00:00:00Z"},
            {**_BASE_ROW, "show_id": "new", "updated_at": "2026-03-20T00:00:00Z"},
        ])
        result = list(SQLiteStorage(db_path).get_shows_updated_since(
            since="2026-02-01T00:00:00Z"
        ))
        assert len(result) == 1
        assert result[0].show_id == "new"

    def test_empty_since_returns_all(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1"},
            {**_BASE_ROW, "show_id": "s2"},
        ])
        storage = SQLiteStorage(db_path)
        assert list(storage.get_shows_updated_since(since="")) == list(storage.get_shows())

    def test_filters_by_language_and_timestamp(self, db_path):
        _seed_db(db_path, [
            {**_BASE_ROW, "show_id": "s1", "language_detected": "zh-tw",
             "updated_at": "2026-03-20T00:00:00Z"},
            {**_BASE_ROW, "show_id": "s2", "language_detected": "en",
             "target_index": "podcast-episodes-en", "updated_at": "2026-03-20T00:00:00Z"},
        ])
        result = list(SQLiteStorage(db_path).get_shows_updated_since(
            since="2026-02-01T00:00:00Z", language="zh-tw"
        ))
        assert len(result) == 1
        assert result[0].show_id == "s1"
