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


class TestShowOptionalFields:
    def test_image_url_parsed_from_json_column(self, db_path):
        """image_url should be extracted from the image JSON column."""
        _seed_db(db_path, [{
            **_BASE_ROW,
            "image": '{"url": "https://example.com/cover.jpg", "width": 3000}',
        }])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.image_url == "https://example.com/cover.jpg"

    def test_external_urls_parsed_from_json_column(self, db_path):
        """external_urls should be parsed from the external_urls JSON column."""
        _seed_db(db_path, [{
            **_BASE_ROW,
            "external_urls": '{"apple_podcasts": "https://podcasts.apple.com/tw/podcast/123"}',
        }])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.external_urls == {"apple_podcasts": "https://podcasts.apple.com/tw/podcast/123"}

    def test_categories_parsed_from_json_column(self, db_path):
        """categories should be parsed from the categories JSON column."""
        _seed_db(db_path, [{
            **_BASE_ROW,
            "categories": '["Technology", "Technology > AI"]',
        }])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.categories == ("Technology", "Technology > AI")

    def test_null_image_returns_none(self, db_path):
        """NULL image column should yield image_url=None."""
        _seed_db(db_path, [{**_BASE_ROW, "image": None}])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.image_url is None

    def test_null_external_urls_returns_empty_dict(self, db_path):
        """NULL external_urls column should yield external_urls={}."""
        _seed_db(db_path, [{**_BASE_ROW, "external_urls": None}])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.external_urls == {}

    def test_null_categories_returns_empty_tuple(self, db_path):
        """NULL categories column should yield categories=()."""
        _seed_db(db_path, [{**_BASE_ROW, "categories": None}])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.categories == ()

    def test_malformed_json_returns_defaults(self, db_path):
        """Malformed JSON in image/external_urls/categories should not raise."""
        _seed_db(db_path, [{
            **_BASE_ROW,
            "image": "not-json",
            "external_urls": "{broken",
            "categories": "[bad",
        }])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.image_url is None
        assert show.external_urls == {}
        assert show.categories == ()

    def test_scalar_fields_populated(self, db_path):
        """provider, external_id, description, episode_count, last_episode_at populated."""
        _seed_db(db_path, [{
            **_BASE_ROW,
            "provider": "apple",
            "external_id": "77001367",
            "description": "A great podcast.",
            "episode_count": 42,
            "last_episode_at": "2026-03-01T00:00:00Z",
        }])
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.provider == "apple"
        assert show.external_id == "77001367"
        assert show.description == "A great podcast."
        assert show.episode_count == 42
        assert show.last_episode_at == "2026-03-01T00:00:00Z"

    def test_missing_optional_columns_return_defaults(self, db_path):
        """Rows without optional columns (e.g. older crawler versions) use defaults."""
        _seed_db(db_path, [_BASE_ROW])  # _BASE_ROW has no optional columns
        show = list(SQLiteStorage(db_path).get_shows())[0]
        assert show.provider == ""
        assert show.external_id == ""
        assert show.description is None
        assert show.image_url is None
        assert show.external_urls == {}
        assert show.episode_count is None
        assert show.last_episode_at is None
        assert show.categories == ()
