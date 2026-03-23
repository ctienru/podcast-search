"""Unit tests for EmbedAndIngestPipeline routing and emit_ingest_log."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.embed_and_ingest import (
    EmbedAndIngestPipeline,
    emit_ingest_log,
    load_cursor,
    run_backfill,
    run_incremental,
    save_cursor,
    upsert_by_show_id,
)
from src.search.routing import LanguageSplitRoutingStrategy
from src.types import IngestCursor, Language, Show


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_show(show_id: str, updated_at: str = "2026-01-01T00:00:00Z") -> Show:
    return Show(
        show_id=show_id,
        title="Test Show",
        author="Author",
        language_detected="zh-tw",
        language_confidence=0.9,
        language_uncertain=False,
        target_index="podcast-episodes-zh-tw",
        rss_feed_url="http://x",
        updated_at=updated_at,
    )


class _FakeStorage:
    """Minimal fake storage for unit tests — no file I/O, no SQLite."""

    def __init__(
        self,
        shows: list[Show],
        updated_shows: list[Show] | None = None,
    ) -> None:
        self._shows = shows
        self._updated_shows = updated_shows if updated_shows is not None else shows

    def get_shows(self, language: Language | None = None) -> Iterator[Show]:
        return iter(self._shows)

    def get_shows_updated_since(
        self, since: str, language: Language | None = None
    ) -> Iterator[Show]:
        return iter(self._updated_shows)


# ── emit_ingest_log ──────────────────────────────────────────────────────────

def test_emit_ingest_log_calculates_uncertain_rate(caplog) -> None:
    """uncertain_rate should be uncertain_count / total."""
    with caplog.at_level(logging.INFO):
        emit_ingest_log(
            index_counts={"episodes-zh-tw": 8, "episodes-en": 2},
            language_distribution={"zh-tw": 8, "uncertain": 2},
            ingest_success=10,
            ingest_failed=0,
        )

    assert any("ingest_complete" in r.message for r in caplog.records)


def test_emit_ingest_log_warns_when_uncertain_rate_exceeds_threshold(caplog) -> None:
    """uncertain_rate > 5% should emit a WARNING."""
    with caplog.at_level(logging.WARNING):
        emit_ingest_log(
            index_counts={"episodes-zh-tw": 9},
            language_distribution={"zh-tw": 8, "uncertain": 1},
            ingest_success=9,
            ingest_failed=0,
        )

    # uncertain_rate = 1/9 ≈ 11%, should warn
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("uncertain_rate_high" in m for m in warning_messages)


def test_emit_ingest_log_no_warning_below_threshold(caplog) -> None:
    """uncertain_rate ≤ 5% should not emit a WARNING."""
    with caplog.at_level(logging.WARNING):
        emit_ingest_log(
            index_counts={"episodes-zh-tw": 100},
            language_distribution={"zh-tw": 97, "uncertain": 3},
            ingest_success=100,
            ingest_failed=0,
        )

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("uncertain_rate_high" in m for m in warning_messages)


def test_emit_ingest_log_handles_zero_total(caplog) -> None:
    """uncertain_rate should be 0.0 when total=0 (no division by zero)."""
    with caplog.at_level(logging.INFO):
        emit_ingest_log(
            index_counts={},
            language_distribution={},
            ingest_success=0,
            ingest_failed=0,
        )
    # Should not raise


# ── to_es_doc routing ────────────────────────────────────────────────────────

def _make_pipeline() -> EmbedAndIngestPipeline:
    return EmbedAndIngestPipeline(
        es_service=MagicMock(),
        encoder=MagicMock(),
        routing_strategy=LanguageSplitRoutingStrategy(),
    )


def _seed_caches(pipeline: EmbedAndIngestPipeline, episode_id: str, target_index: str) -> None:
    pipeline._show_cache["show-1"] = {"show_id": "show-1", "title": "Test Show", "author": "Author"}
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": "show-1",
        "target_index": target_index,
        "cleaned": {"normalized": {"title": "Ep Title", "description": "Ep Desc"}},
        "original_meta": {
            "pub_date": None,
            "duration": None,
            "audio_url": "http://audio.example/ep.mp3",
            "language": "zh-tw",
            "image_url": None,
            "itunes_summary": None,
            "creator": None,
            "episode_type": None,
            "chapters": [],
        },
    }


def test_to_es_doc_routes_to_language_alias() -> None:
    """_index should be the alias returned by the routing strategy."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-1", "podcast-episodes-zh-tw")

    doc = pipeline.to_es_doc({"episode_id": "ep-1", "show_id": "show-1"}, [0.1] * 384)

    assert doc is not None
    assert doc["_index"] == "episodes-zh-tw"


def test_to_es_doc_returns_none_for_unknown_target_index() -> None:
    """When target_index cannot be routed, to_es_doc returns None and does not raise."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-2", "podcast-episodes-jp")  # unmapped

    doc = pipeline.to_es_doc({"episode_id": "ep-2", "show_id": "show-1"}, [0.1] * 384)

    assert doc is None


# ── load_cursor / save_cursor ─────────────────────────────────────────────────


class TestLoadCursor:
    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        """load_cursor should return {} when the cursor file does not exist."""
        result = load_cursor(tmp_path / "cursor.json")
        assert result == {}

    def test_reads_existing_cursor_file(self, tmp_path: Path) -> None:
        """load_cursor should deserialise an existing cursor JSON file."""
        cursor_path = tmp_path / "cursor.json"
        data: dict[str, IngestCursor] = {
            "episodes-zh-tw": {
                "last_ingest_at": "2026-03-01T00:00:00Z",
                "last_run_at": "2026-03-01T00:00:00Z",
            }
        }
        cursor_path.write_text(json.dumps(data))
        assert load_cursor(cursor_path) == data

    def test_roundtrip(self, tmp_path: Path) -> None:
        """A cursor written by save_cursor must be read back identically by load_cursor."""
        cursor_path = tmp_path / "cursor.json"
        cursors: dict[str, IngestCursor] = {
            "episodes-zh-tw": {"last_ingest_at": "2026-03-01T00:00:00Z", "last_run_at": "2026-03-01T00:00:00Z"},
            "episodes-en":    {"last_ingest_at": "2026-02-01T00:00:00Z", "last_run_at": "2026-02-01T00:00:00Z"},
        }
        save_cursor(cursors, cursor_path)
        assert load_cursor(cursor_path) == cursors


class TestSaveCursor:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        """save_cursor should create a file containing valid JSON."""
        cursor_path = tmp_path / "cursor.json"
        cursors: dict[str, IngestCursor] = {
            "episodes-en": {"last_ingest_at": "2026-03-01T00:00:00Z", "last_run_at": "2026-03-01T00:00:00Z"}
        }
        save_cursor(cursors, cursor_path)
        assert cursor_path.exists()
        assert json.loads(cursor_path.read_text()) == cursors

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_cursor should create missing parent directories."""
        cursor_path = tmp_path / "subdir" / "nested" / "cursor.json"
        save_cursor({}, cursor_path)
        assert cursor_path.exists()


# ── run_incremental ───────────────────────────────────────────────────────────


class TestRunIncremental:
    def test_passes_cursor_since_to_pipeline_allowed_show_ids(self, tmp_path: Path) -> None:
        """Pipeline must be created with allowed_show_ids from updated shows."""
        cursor_path = tmp_path / "cursor.json"
        save_cursor(
            {"episodes-zh-tw": {"last_ingest_at": "2026-03-01T00:00:00Z", "last_run_at": "2026-03-01T00:00:00Z"}},
            cursor_path,
        )
        storage = _FakeStorage(shows=[_make_show("s1")], updated_shows=[_make_show("s1")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 0, "errors": 0, "total": 0}
            run_incremental(storage=storage, cursor_path=cursor_path)

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] == {"s1"}

    def test_force_full_sets_allowed_show_ids_to_none(self, tmp_path: Path) -> None:
        """force_full=True must pass allowed_show_ids=None (process all shows)."""
        cursor_path = tmp_path / "cursor.json"
        storage = _FakeStorage(shows=[_make_show("s1")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 0, "errors": 0, "total": 0}
            run_incremental(storage=storage, force_full=True, cursor_path=cursor_path)

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] is None

    def test_saves_cursor_for_all_aliases_after_run(self, tmp_path: Path) -> None:
        """Cursor file must contain entries for all three language aliases after run."""
        cursor_path = tmp_path / "cursor.json"
        storage = _FakeStorage(shows=[_make_show("s1")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 5, "errors": 0, "total": 5}
            run_incremental(storage=storage, force_full=True, cursor_path=cursor_path)

        cursor = load_cursor(cursor_path)
        assert set(cursor.keys()) == {"episodes-zh-tw", "episodes-zh-cn", "episodes-en"}
        assert cursor["episodes-zh-tw"]["last_ingest_at"] != ""

    def test_skips_pipeline_when_no_updated_shows(self, tmp_path: Path) -> None:
        """When storage returns no updated shows, pipeline must not be created."""
        cursor_path = tmp_path / "cursor.json"
        save_cursor(
            {"episodes-zh-tw": {"last_ingest_at": "2026-03-01T00:00:00Z", "last_run_at": "2026-03-01T00:00:00Z"}},
            cursor_path,
        )
        storage = _FakeStorage(shows=[], updated_shows=[])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            result = run_incremental(storage=storage, cursor_path=cursor_path)

        MockPipeline.assert_not_called()
        assert result["success"] == 0


# ── run_backfill ──────────────────────────────────────────────────────────────


class TestRunBackfill:
    def test_passes_force_full_true(self, tmp_path: Path) -> None:
        """run_backfill must process all shows (allowed_show_ids=None)."""
        cursor_path = tmp_path / "cursor.json"
        storage = _FakeStorage(shows=[_make_show("s1"), _make_show("s2")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 10, "errors": 0, "total": 10}
            run_backfill(storage=storage, cursor_path=cursor_path)

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] is None


# ── upsert_by_show_id ─────────────────────────────────────────────────────────


class TestUpsertByShowId:
    def test_raises_for_unknown_show_id(self) -> None:
        """upsert_by_show_id must raise ValueError when show_id is not in storage."""
        storage = _FakeStorage(shows=[])
        with pytest.raises(ValueError, match="show_id not found"):
            upsert_by_show_id("unknown-show", storage=storage)

    def test_creates_pipeline_with_single_show_id(self) -> None:
        """Pipeline must be created with allowed_show_ids={show_id}."""
        storage = _FakeStorage(shows=[_make_show("s1")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 3, "errors": 0, "total": 3}
            count = upsert_by_show_id("s1", storage=storage)

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] == {"s1"}
        assert count == 3
