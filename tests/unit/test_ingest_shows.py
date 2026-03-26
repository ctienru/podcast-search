"""Unit tests for IngestShowsPipeline sync state integration."""

from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.ingest_shows import IngestShowsPipeline


def _make_show_dict(show_id: str = "show:apple:123") -> dict:
    return {
        "show_id": show_id,
        "provider": "apple",
        "external_id": show_id.split(":")[-1],
        "title": "Test Show",
        "author": "Author",
        "description": None,
        "language": "zh-tw",
        "image": {},
        "external_urls": {},
        "episode_stats": {"episode_count": 10, "last_episode_at": None},
        "categories": [],
        "content_hash": "abc123",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def es_service():
    svc = MagicMock()
    svc.client = MagicMock()
    return svc


class TestIngestShowsSyncState:
    def test_mark_done_called_for_successful_show(self, es_service):
        show = _make_show_dict()
        sync_repo = MagicMock()
        pipeline = IngestShowsPipeline(es_service=es_service, sync_repo=sync_repo)

        with patch("src.pipelines.ingest_shows.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (1, [])
            pipeline.run(shows=[show])

        sync_repo.mark_done.assert_called_once_with(
            entity_type="show",
            entity_id="show:apple:123",
            index_alias=IngestShowsPipeline.INDEX_ALIAS,
            content_hash="abc123",
            source_updated_at="2026-01-01T00:00:00Z",
        )

    def test_mark_done_not_called_for_errored_show(self, es_service):
        show1 = _make_show_dict("show:apple:1")
        show2 = _make_show_dict("show:apple:2")
        sync_repo = MagicMock()
        pipeline = IngestShowsPipeline(es_service=es_service, sync_repo=sync_repo)

        error = {"index": {"_id": "show:apple:2", "error": {"type": "mapper_parsing_exception"}}}
        with patch("src.pipelines.ingest_shows.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (1, [error])
            pipeline.run(shows=[show1, show2])

        call_ids = {c.kwargs["entity_id"] for c in sync_repo.mark_done.call_args_list}
        assert "show:apple:1" in call_ids
        assert "show:apple:2" not in call_ids

    def test_no_sync_repo_runs_without_error(self, es_service):
        show = _make_show_dict()
        pipeline = IngestShowsPipeline(es_service=es_service, sync_repo=None)

        with patch("src.pipelines.ingest_shows.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (1, [])
            pipeline.run(shows=[show])

    def test_mark_done_not_called_when_no_sync_repo(self, es_service):
        show = _make_show_dict()
        pipeline = IngestShowsPipeline(es_service=es_service, sync_repo=None)

        with patch("src.pipelines.ingest_shows.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (1, [])
            pipeline.run(shows=[show])
        # No assertion needed — test just ensures no AttributeError raised
