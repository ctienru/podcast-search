"""Unit tests for EmbedAndIngestPipeline routing and emit_ingest_log."""

import logging
from unittest.mock import MagicMock

import pytest

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline, emit_ingest_log
from src.search.routing import LanguageSplitRoutingStrategy


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

def _make_pipeline(enable_language_split: bool) -> EmbedAndIngestPipeline:
    pipeline = EmbedAndIngestPipeline(
        es_service=MagicMock(),
        encoder=MagicMock(),
        enable_language_split=enable_language_split,
        routing_strategy=LanguageSplitRoutingStrategy(),
    )
    return pipeline


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


def test_to_es_doc_v2_routes_to_language_alias() -> None:
    """When enable_language_split=True, _index should be the routing strategy's alias."""
    pipeline = _make_pipeline(enable_language_split=True)
    _seed_caches(pipeline, "ep-1", "podcast-episodes-zh-tw")

    doc = pipeline.to_es_doc({"episode_id": "ep-1", "show_id": "show-1"}, [0.1] * 384)

    assert doc is not None
    assert doc["_index"] == "episodes-zh-tw"


def test_to_es_doc_v2_returns_none_for_unknown_target_index() -> None:
    """When target_index cannot be routed, to_es_doc returns None and does not raise."""
    pipeline = _make_pipeline(enable_language_split=True)
    _seed_caches(pipeline, "ep-2", "podcast-episodes-jp")  # unmapped

    doc = pipeline.to_es_doc({"episode_id": "ep-2", "show_id": "show-1"}, [0.1] * 384)

    assert doc is None


def test_to_es_doc_v1_uses_default_alias() -> None:
    """When enable_language_split=False, _index should be the legacy INDEX_ALIAS."""
    pipeline = _make_pipeline(enable_language_split=False)
    _seed_caches(pipeline, "ep-3", "podcast-episodes-zh-tw")

    doc = pipeline.to_es_doc({"episode_id": "ep-3", "show_id": "show-1"}, [0.1] * 384)

    assert doc is not None
    assert doc["_index"] == EmbedAndIngestPipeline.INDEX_ALIAS
