"""Unit tests for QueryLogEntry and QueryLogger."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.query_logger import QueryLogEntry, QueryLogger


def _make_entry(**kwargs) -> QueryLogEntry:
    defaults = dict(
        query="test",
        query_lang="zh-tw",
        selected_lang="zh-tw",
        mode="hybrid",
        target_index=["episodes-zh-tw"],
        is_cross_lang=False,
        result_count=5,
        result_ids=[],
        result_languages=[],
        page=1,
        latency_ms=30,
    )
    defaults.update(kwargs)
    return QueryLogEntry(**defaults)


def test_log_entry_has_auto_generated_request_id() -> None:
    """QueryLogEntry should auto-generate a non-empty UUID for request_id."""
    entry = _make_entry()
    assert len(entry.request_id) == 36  # UUID4 format


def test_log_entry_request_ids_are_unique() -> None:
    """Each QueryLogEntry should get a distinct request_id."""
    a = _make_entry()
    b = _make_entry()
    assert a.request_id != b.request_id


def test_log_entry_zh_both_has_two_target_indices() -> None:
    """zh-both search should record two target indices in a single log entry."""
    entry = QueryLogEntry(
        query="人工智慧",
        query_lang="zh-tw",
        selected_lang="zh-both",
        mode="hybrid",
        target_index=["episodes-zh-tw", "episodes-zh-cn"],
        is_cross_lang=True,
        result_count=10,
        result_ids=[],
        result_languages=[],
        page=1,
        latency_ms=55,
    )
    assert entry.target_index == ["episodes-zh-tw", "episodes-zh-cn"]
    assert entry.is_cross_lang is True


def test_logger_appends_valid_json_line(tmp_path: Path) -> None:
    """Each log() call must append exactly one parseable JSON line."""
    log_file = tmp_path / "query_log.jsonl"
    ql = QueryLogger(log_file)
    entry = QueryLogEntry(
        query="AI podcast",
        query_lang="en",
        selected_lang="en",
        mode="hybrid",
        target_index=["episodes-en"],
        is_cross_lang=False,
        result_count=10,
        result_ids=["ep:1"],
        result_languages=["en"],
        page=1,
        latency_ms=42,
    )
    ql.log(entry)

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["query"] == "AI podcast"
    assert parsed["target_index"] == ["episodes-en"]


def test_logger_appends_multiple_lines(tmp_path: Path) -> None:
    """Multiple log() calls should each append a separate line."""
    log_file = tmp_path / "query_log.jsonl"
    ql = QueryLogger(log_file)

    ql.log(_make_entry(query="first"))
    ql.log(_make_entry(query="second"))
    ql.log(_make_entry(query="third"))

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 3
    queries = [json.loads(ln)["query"] for ln in lines]
    assert queries == ["first", "second", "third"]


def test_logger_creates_parent_directory(tmp_path: Path) -> None:
    """QueryLogger should create the parent directory if it doesn't exist."""
    log_file = tmp_path / "logs" / "subdir" / "query_log.jsonl"
    ql = QueryLogger(log_file)
    ql.log(_make_entry())
    assert log_file.exists()


def test_logger_does_not_raise_on_write_failure(tmp_path: Path) -> None:
    """If the log file cannot be written, the error must be caught (search must not fail)."""
    log_file = tmp_path / "query_log.jsonl"
    ql = QueryLogger(log_file)

    with patch.object(Path, "open", side_effect=OSError("disk full")):
        ql.log(_make_entry())  # must not raise
