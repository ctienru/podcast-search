"""Unit tests for EmbedAndIngestPipeline routing and emit_ingest_log."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.embedding.backend import EmbeddingBackend
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
            language_distribution={"zh-tw": 8, "unknown": 2},
            ingest_success=10,
            ingest_failed=0,
        )

    assert any("ingest_complete" in r.message for r in caplog.records)


def test_emit_ingest_log_warns_when_uncertain_rate_exceeds_threshold(caplog) -> None:
    """uncertain_rate > 5% should emit a WARNING."""
    with caplog.at_level(logging.WARNING):
        emit_ingest_log(
            index_counts={"episodes-zh-tw": 9},
            language_distribution={"zh-tw": 8, "unknown": 1},
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
            language_distribution={"zh-tw": 97, "unknown": 3},
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
        environment="default",
        es_service=MagicMock(),
        embedding_backend=MagicMock(spec=EmbeddingBackend),
        routing_strategy=LanguageSplitRoutingStrategy(),
        storage=MagicMock(),
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

    with patch("src.pipelines.embed_and_ingest.settings") as mock_settings:
        mock_settings.ENABLE_LANGUAGE_SPLIT = True
        doc = pipeline.to_es_doc({"episode_id": "ep-1", "show_id": "show-1"}, [0.1] * 384)

    assert doc is not None
    assert doc["_index"] == "episodes-zh-tw"


def test_to_es_doc_returns_none_for_unknown_target_index() -> None:
    """When target_index cannot be routed, to_es_doc returns None and does not raise."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-2", "podcast-episodes-jp")  # unmapped

    with patch("src.pipelines.embed_and_ingest.settings") as mock_settings:
        mock_settings.ENABLE_LANGUAGE_SPLIT = True
        doc = pipeline.to_es_doc({"episode_id": "ep-2", "show_id": "show-1"}, [0.1] * 384)

    assert doc is None


def test_to_es_doc_returns_none_and_no_warning_when_no_target_index(caplog) -> None:
    """Episode with no target_index (show excluded by SQLiteStorage._WHERE_BASE) is
    silently skipped at DEBUG level — not WARNING."""
    pipeline = _make_pipeline()
    # Seed episode without target_index; show absent from cache (simulates NULL
    # target_index show being filtered out by _WHERE_BASE in SQLiteStorage)
    pipeline._cleaned_episode_cache["ep-noroute"] = {
        "episode_id": "ep-noroute",
        "show_id": "show-missing",
        "cleaned": {"normalized": {"title": "Test", "description": "Desc"}},
        "original_meta": {
            "pub_date": None, "duration": None, "audio_url": None,
            "language": "ja", "image_url": None, "itunes_summary": None,
            "creator": None, "episode_type": None, "chapters": [],
        },
    }

    with patch("src.pipelines.embed_and_ingest.settings") as mock_settings:
        mock_settings.ENABLE_LANGUAGE_SPLIT = True
        with caplog.at_level(logging.DEBUG, logger="src.pipelines.embed_and_ingest"):
            doc = pipeline.to_es_doc(
                {"episode_id": "ep-noroute", "show_id": "show-missing"}, [0.1] * 384
            )

    assert doc is None
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_to_es_doc_includes_embedding_when_vector_provided() -> None:
    """A non-empty embedding vector must appear in _source['embedding']."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-vec", "podcast-episodes-zh-tw")

    embedding = [0.1] * 384
    doc = pipeline.to_es_doc({"episode_id": "ep-vec", "show_id": "show-1"}, embedding)

    assert doc is not None
    assert doc["_source"]["embedding"] == embedding


def test_to_es_doc_omits_embedding_when_vector_is_empty() -> None:
    """BM25-only mode: uses ES update semantics (doc key, not _source)."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-bm25", "podcast-episodes-zh-tw")

    doc = pipeline.to_es_doc({"episode_id": "ep-bm25", "show_id": "show-1"}, [])

    assert doc is not None
    assert doc["_op_type"] == "update"
    assert "upsert" in doc          # first-insert path carries created_at
    assert "doc_as_upsert" not in doc
    assert "created_at" not in doc["doc"]   # partial update must not overwrite created_at
    assert "embedding" not in doc["doc"]


# ── show subobject fields ─────────────────────────────────────────────────────


def test_to_es_doc_includes_show_image_url() -> None:
    """show.image_url from the cache must appear in the episode's show subobject."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-img", "podcast-episodes-zh-tw")
    pipeline._show_cache["show-1"]["image_url"] = "https://example.com/cover.jpg"

    doc = pipeline.to_es_doc({"episode_id": "ep-img", "show_id": "show-1"}, [])

    # BM25-only path uses update semantics → content is under "doc", not "_source"
    assert doc is not None
    assert doc["doc"]["show"]["image_url"] == "https://example.com/cover.jpg"


def test_to_es_doc_includes_show_external_urls() -> None:
    """show.external_urls from the cache must appear in the episode's show subobject."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-url", "podcast-episodes-zh-tw")
    pipeline._show_cache["show-1"]["external_urls"] = {
        "apple_podcasts": "https://podcasts.apple.com/tw/podcast/123"
    }

    doc = pipeline.to_es_doc({"episode_id": "ep-url", "show_id": "show-1"}, [])

    assert doc is not None
    assert doc["doc"]["show"]["external_urls"] == {
        "apple_podcasts": "https://podcasts.apple.com/tw/podcast/123"
    }


def test_to_es_doc_show_image_url_defaults_to_none() -> None:
    """When image_url is absent from the cache, show subobject image_url is None."""
    pipeline = _make_pipeline()
    _seed_caches(pipeline, "ep-noimg", "podcast-episodes-zh-tw")
    # _seed_caches sets cache without image_url

    doc = pipeline.to_es_doc({"episode_id": "ep-noimg", "show_id": "show-1"}, [])

    assert doc is not None
    assert doc["doc"]["show"].get("image_url") is None


# ── batch_encode ──────────────────────────────────────────────────────────────


def _make_embedding_input(episode_id: str, show_id: str, text: str) -> dict:
    return {"episode_id": episode_id, "show_id": show_id, "embedding_input": {"text": text}}


def _seed_episode(
    pipeline: EmbedAndIngestPipeline,
    episode_id: str,
    target_index: str,
) -> None:
    pipeline._show_cache["show-1"] = {"show_id": "show-1", "title": "T", "author": "A"}
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": "show-1",
        "target_index": target_index,
        "cleaned": {"normalized": {"title": "T", "description": "D"}},
        "original_meta": {
            "pub_date": None,
            "duration": None,
            "audio_url": None,
            "language": "zh-tw",
            "image_url": None,
            "itunes_summary": None,
            "creator": None,
            "episode_type": None,
            "chapters": [],
        },
    }


class TestBatchEncode:
    def test_groups_same_language_into_single_embed_batch_call(self) -> None:
        """Two zh-tw episodes must be passed to embed_batch in one call (not two)."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        mock_backend.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]

        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=mock_backend,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
        )
        _seed_episode(pipeline, "ep-tw-1", "podcast-episodes-zh-tw")
        _seed_episode(pipeline, "ep-tw-2", "podcast-episodes-zh-tw")

        inputs = [
            _make_embedding_input("ep-tw-1", "show-1", "text one"),
            _make_embedding_input("ep-tw-2", "show-1", "text two"),
        ]
        pipeline.batch_encode(inputs)

        zh_tw_calls = [
            c for c in mock_backend.embed_batch.call_args_list
            if c.args[1] == "zh-tw"
        ]
        assert len(zh_tw_calls) == 1, "both zh-tw texts must be in one embed_batch call"
        assert len(zh_tw_calls[0].args[0]) == 2

    def test_preserves_input_order_across_language_groups(self) -> None:
        """Results must be in the same order as inputs, even when languages interleave."""
        mock_backend = MagicMock(spec=EmbeddingBackend)

        def embed_batch_side_effect(texts: list[str], language: str) -> list[list[float]]:
            if language in ("zh-tw", "zh-cn"):
                return [[0.9] * 384 for _ in texts]
            return [[0.1] * 384 for _ in texts]

        mock_backend.embed_batch.side_effect = embed_batch_side_effect

        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=mock_backend,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
        )
        _seed_episode(pipeline, "ep-zh", "podcast-episodes-zh-tw")
        _seed_episode(pipeline, "ep-en", "podcast-episodes-en")

        inputs = [
            _make_embedding_input("ep-zh", "show-1", "zh text"),
            _make_embedding_input("ep-en", "show-1", "en text"),
        ]
        results = pipeline.batch_encode(inputs)

        assert len(results) == 2
        inp0, vec0 = results[0]
        inp1, vec1 = results[1]
        assert inp0["episode_id"] == "ep-zh"
        assert inp1["episode_id"] == "ep-en"
        assert len(vec0) == 384   # zh vector
        assert len(vec1) == 384   # en vector

    def test_returns_empty_vectors_when_backend_is_none(self) -> None:
        """BM25-only mode: batch_encode returns (input, []) for every item."""
        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=None,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
        )
        inputs = [_make_embedding_input("ep-x", "show-1", "some text")]
        results = pipeline.batch_encode(inputs)

        assert len(results) == 1
        _, vec = results[0]
        assert vec == []


class TestBatchEncodeFromCache:
    def test_returns_vector_from_cache(self) -> None:
        """from_cache=True: episode in _vector_cache → cached vector returned, backend not called."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=mock_backend,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
            from_cache=True,
        )
        pipeline._vector_cache["ep-cached"] = [0.5] * 384

        inputs = [_make_embedding_input("ep-cached", "show-1", "some text")]
        results = pipeline.batch_encode(inputs)

        assert len(results) == 1
        inp, vec = results[0]
        assert inp["episode_id"] == "ep-cached"
        assert vec == [0.5] * 384
        mock_backend.embed_batch.assert_not_called()

    def test_cache_miss_returns_empty_vector(self) -> None:
        """from_cache=True: episode not in _vector_cache → empty vector (BM25-only update path)."""
        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=None,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
            from_cache=True,
        )
        # _vector_cache is empty — simulate forgetting to run embed_episodes

        inputs = [_make_embedding_input("ep-missing", "show-1", "some text")]
        results = pipeline.batch_encode(inputs)

        assert len(results) == 1
        _, vec = results[0]
        assert vec == []


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


    def test_allowed_show_ids_restricts_force_full_to_specified_shows(self, tmp_path: Path) -> None:
        """force_full=True + allowed_show_ids must only process the specified shows."""
        cursor_path = tmp_path / "cursor.json"
        storage = _FakeStorage(shows=[_make_show("s1"), _make_show("s2")])

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 3, "errors": 0, "total": 3}
            run_incremental(
                storage=storage,
                force_full=True,
                cursor_path=cursor_path,
                allowed_show_ids={"s1"},
            )

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] == {"s1"}

    def test_allowed_show_ids_intersects_with_cursor_updated_shows(self, tmp_path: Path) -> None:
        """allowed_show_ids must be intersected with cursor-updated shows in incremental mode."""
        cursor_path = tmp_path / "cursor.json"
        save_cursor(
            {"episodes-zh-tw": {"last_ingest_at": "2026-03-01T00:00:00Z", "last_run_at": "2026-03-01T00:00:00Z"}},
            cursor_path,
        )
        # Only s1 is returned as updated by cursor; s2 is in allowed but not updated
        storage = _FakeStorage(
            shows=[_make_show("s1"), _make_show("s2")],
            updated_shows=[_make_show("s1")],
        )

        with patch("src.pipelines.embed_and_ingest.EmbedAndIngestPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = {"success": 1, "errors": 0, "total": 1}
            run_incremental(
                storage=storage,
                cursor_path=cursor_path,
                allowed_show_ids={"s1", "s2"},
            )

        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["allowed_show_ids"] == {"s1"}  # intersection: s2 not updated


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


# ── strict_cache mode ─────────────────────────────────────────────────────────


class TestStrictCache:
    def _make_pipeline(self, strict_cache: bool = True) -> EmbedAndIngestPipeline:
        return EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=None,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
            from_cache=True,
            strict_cache=strict_cache,
        )

    def test_strict_cache_raises_on_cache_miss(self) -> None:
        """strict_cache=True: if any episode has a cache miss, raise CacheMissError."""
        from src.pipelines.exceptions import CacheMissError
        pipeline = self._make_pipeline(strict_cache=True)
        # _vector_cache is empty — every lookup will be a miss

        inputs = [
            _make_embedding_input("ep-1", "show-1", "text one"),
            _make_embedding_input("ep-2", "show-1", "text two"),
        ]

        with pytest.raises(CacheMissError, match="2"):
            pipeline.batch_encode(inputs)

    def test_strict_cache_raises_with_partial_miss(self) -> None:
        """strict_cache=True: even a single miss should raise."""
        from src.pipelines.exceptions import CacheMissError
        pipeline = self._make_pipeline(strict_cache=True)
        pipeline._vector_cache["ep-hit"] = [0.1] * 384
        # ep-miss has no cache entry

        inputs = [
            _make_embedding_input("ep-hit", "show-1", "text one"),
            _make_embedding_input("ep-miss", "show-1", "text two"),
        ]

        with pytest.raises(CacheMissError, match="1"):
            pipeline.batch_encode(inputs)

    def test_non_strict_cache_miss_returns_empty_vector(self) -> None:
        """strict_cache=False (default): cache miss returns empty vector, no exception."""
        pipeline = self._make_pipeline(strict_cache=False)
        # _vector_cache is empty

        inputs = [_make_embedding_input("ep-missing", "show-1", "text")]
        results = pipeline.batch_encode(inputs)

        assert len(results) == 1
        _, vec = results[0]
        assert vec == []

    def test_strict_cache_no_raise_when_all_hits(self) -> None:
        """strict_cache=True: no exception when all episodes have cache entries."""
        from src.pipelines.exceptions import CacheMissError
        pipeline = self._make_pipeline(strict_cache=True)
        pipeline._vector_cache["ep-1"] = [0.5] * 384
        pipeline._vector_cache["ep-2"] = [0.3] * 384

        inputs = [
            _make_embedding_input("ep-1", "show-1", "text one"),
            _make_embedding_input("ep-2", "show-1", "text two"),
        ]

        results = pipeline.batch_encode(inputs)
        assert len(results) == 2


# ── run summary ───────────────────────────────────────────────────────────────


class TestRunSummary:
    def test_batch_encode_tracks_cache_hit_and_miss_counts(self) -> None:
        """batch_encode() increments _cache_hits/_cache_misses when from_cache=True."""
        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=None,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
            from_cache=True,
        )
        pipeline._vector_cache["ep-1"] = [0.5] * 384
        # ep-2 will be a miss

        inputs = [
            _make_embedding_input("ep-1", "show-1", "text one"),
            _make_embedding_input("ep-2", "show-1", "text two"),
        ]
        pipeline.batch_encode(inputs)

        assert pipeline._cache_hits == 1
        assert pipeline._cache_misses == 1

    def test_run_stats_include_cache_counters(self) -> None:
        """run() stats dict must include cache_hits and cache_misses keys."""
        pipeline = EmbedAndIngestPipeline(
            environment="default",
            es_service=MagicMock(),
            embedding_backend=None,
            routing_strategy=LanguageSplitRoutingStrategy(),
            storage=MagicMock(),
            from_cache=True,
        )
        # Pre-seed counters; counter accumulation is tested by test_batch_encode_tracks_cache_hit_and_miss_counts
        pipeline._cache_hits = 3
        pipeline._cache_misses = 1

        with (
            patch.object(pipeline, "_load_show_cache"),
            patch.object(pipeline, "_load_cleaned_episode_cache"),
            patch.object(pipeline, "_load_vector_cache"),
            patch.object(pipeline, "list_embedding_input_files", return_value=["f1"]),
            patch.object(pipeline, "load_embedding_inputs", return_value=iter([])),
            patch("src.pipelines.embed_and_ingest.streaming_bulk", return_value=iter([])),
        ):
            stats = pipeline.run()

        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 1
