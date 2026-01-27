"""Tests for IngestEpisodesPipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.pipelines.ingest_episodes import IngestEpisodesPipeline


class TestCleanHtml:
    """Test the _clean_html() static method."""

    def test_basic_html_removal(self):
        """Test basic HTML tag removal."""
        input_html = "<p>This is a test</p>"
        expected = "This is a test"
        assert IngestEpisodesPipeline._clean_html(input_html) == expected

    def test_paragraph_to_newlines(self):
        """Test that paragraph tags are converted to newlines."""
        input_html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = IngestEpisodesPipeline._clean_html(input_html)
        assert "\n" in result
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_br_to_newlines(self):
        """Test that <br> tags are converted to newlines."""
        input_html = "Line 1<br>Line 2<br/>Line 3"
        result = IngestEpisodesPipeline._clean_html(input_html)
        assert "\n" in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_html_entities_decoded(self):
        """Test that HTML entities are decoded."""
        input_html = "Tom &amp; Jerry &lt;test&gt; &quot;quoted&quot;"
        expected = 'Tom & Jerry <test> "quoted"'
        assert IngestEpisodesPipeline._clean_html(input_html) == expected

    def test_complex_html(self):
        """Test complex HTML with multiple tags."""
        input_html = "<p>This is a <strong>test</strong> description with <a href='https://example.com'>link</a>.</p>"
        result = IngestEpisodesPipeline._clean_html(input_html)
        assert result == "This is a test description with link."

    def test_empty_string_returns_none(self):
        """Test that empty string after cleaning returns None."""
        input_html = "<p></p>"
        assert IngestEpisodesPipeline._clean_html(input_html) is None

    def test_none_input_returns_none(self):
        """Test that None input returns None."""
        assert IngestEpisodesPipeline._clean_html(None) is None

    def test_whitespace_normalization(self):
        """Test that multiple spaces and newlines are normalized."""
        input_html = "<p>Too    many     spaces</p><p></p><p>New paragraph</p>"
        result = IngestEpisodesPipeline._clean_html(input_html)
        assert "Too many spaces" in result
        assert result.count("\n") <= 2  # Should consolidate multiple newlines

    def test_preserves_special_characters(self):
        """Test that non-HTML special characters are preserved."""
        input_html = "<p>Price: $100.00 — 50% off!</p>"
        result = IngestEpisodesPipeline._clean_html(input_html)
        assert "$100.00" in result
        assert "—" in result
        assert "%" in result


class TestToEsDoc:
    """Test the to_es_doc() transformation method."""

    @patch('src.pipelines.ingest_episodes.storage')
    def test_complete_episode_transformation(self, mock_storage):
        """Test transformation of complete episode data."""
        # Setup
        show_data = {
            "title": "Test Podcast",
            "author": "Test Publisher",
            "image": {"url": "https://example.com/show.jpg"},
            "external_urls": {"apple_podcasts": "https://podcasts.apple.com/show"}
        }

        mock_storage.list_show_ids.return_value = ["show_123"]
        mock_storage.load_show.return_value = show_data

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        episode = {
            "episode_id": "ep_123",
            "show_id": "show_123",
            "title": "Episode 1",
            "description": "<p>Test description</p>",
            "language": "en",
            "published_at": "2025-01-15T10:00:00Z",
            "duration_sec": 1800,
            "audio": {
                "url": "https://example.com/ep1.mp3",
                "type": "audio/mpeg",
                "length_bytes": 1024000
            },
            "image": {"url": "https://example.com/ep1.jpg"},
            "external_ids": {"apple": "apple_ep_123"},
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-01-17T00:00:00Z"
        }

        # Execute
        result = pipeline.to_es_doc(episode)

        # Verify structure
        assert result["_index"] == "episodes"
        assert result["_id"] == "ep_123"

        source = result["_source"]
        assert source["episode_id"] == "ep_123"
        assert source["title"] == "Episode 1"
        assert source["description"] == "Test description"
        assert source["language"] == "en"
        assert source["published_at"] == "2025-01-15T10:00:00Z"
        assert source["duration_sec"] == 1800
        assert source["image_url"] == "https://example.com/ep1.jpg"

        # Verify audio object
        assert source["audio"]["url"] == "https://example.com/ep1.mp3"
        assert source["audio"]["type"] == "audio/mpeg"
        assert source["audio"]["length_bytes"] == 1024000

        # Verify show embedding
        assert source["show"]["show_id"] == "show_123"
        assert source["show"]["title"] == "Test Podcast"
        assert source["show"]["publisher"] == "Test Publisher"
        assert source["show"]["image_url"] == "https://example.com/show.jpg"
        assert source["show"]["external_urls"]["apple_podcasts"] == "https://podcasts.apple.com/show"

    @patch('src.pipelines.ingest_episodes.storage')
    def test_episode_without_show_data(self, mock_storage):
        """Test transformation when show data is not found."""
        mock_storage.list_show_ids.return_value = []

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        episode = {
            "episode_id": "ep_123",
            "show_id": "show_unknown",
            "title": "Episode 1",
            "description": "Test description"
        }

        result = pipeline.to_es_doc(episode)
        source = result["_source"]

        # Show object should still exist with minimal data
        assert source["show"]["show_id"] == "show_unknown"
        assert "title" not in source["show"] or source["show"].get("title") is None
        assert "publisher" not in source["show"] or source["show"].get("publisher") is None

    @patch('src.pipelines.ingest_episodes.storage')
    def test_episode_with_missing_image(self, mock_storage):
        """Test transformation when episode image is missing."""
        mock_storage.list_show_ids.return_value = []

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        episode = {
            "episode_id": "ep_123",
            "title": "Episode 1"
        }

        result = pipeline.to_es_doc(episode)
        source = result["_source"]

        # image_url should be None when image is missing
        assert source["image_url"] is None

    @patch('src.pipelines.ingest_episodes.storage')
    def test_description_html_cleaning(self, mock_storage):
        """Test that description HTML is cleaned during transformation."""
        mock_storage.list_show_ids.return_value = []

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        episode = {
            "episode_id": "ep_123",
            "title": "Episode 1",
            "description": "<p>Test <strong>bold</strong> &amp; entities</p>"
        }

        result = pipeline.to_es_doc(episode)
        source = result["_source"]

        # HTML should be cleaned
        assert "<p>" not in source["description"]
        assert "<strong>" not in source["description"]
        assert "&amp;" not in source["description"]
        assert "Test bold & entities" == source["description"]


class TestShowCache:
    """Test the show cache mechanism."""

    @patch('src.pipelines.ingest_episodes.storage')
    def test_show_cache_loads_on_init(self, mock_storage):
        """Test that show cache is loaded during initialization."""
        show_data = {
            "title": "Test Podcast",
            "author": "Test Publisher"
        }

        mock_storage.list_show_ids.return_value = ["show_123"]
        mock_storage.load_show.return_value = show_data

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        # Verify cache is populated
        assert "show_123" in pipeline._show_cache
        assert pipeline._show_cache["show_123"]["title"] == "Test Podcast"

    @patch('src.pipelines.ingest_episodes.storage')
    def test_get_show_data_from_cache(self, mock_storage):
        """Test retrieving show data from cache."""
        show_data = {"title": "Test Podcast"}

        mock_storage.list_show_ids.return_value = ["show_123"]
        mock_storage.load_show.return_value = show_data

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        # Get from cache
        result = pipeline._get_show_data("show_123")
        assert result == show_data

        # Non-existent show returns None
        result = pipeline._get_show_data("show_unknown")
        assert result is None


class TestBuildActions:
    """Test the build_actions() generator."""

    @patch('src.pipelines.ingest_episodes.storage')
    def test_build_actions_generates_docs(self, mock_storage):
        """Test that build_actions generates ES documents."""
        mock_storage.list_show_ids.return_value = []

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        episodes = [
            {"episode_id": "ep_1", "title": "Episode 1"},
            {"episode_id": "ep_2", "title": "Episode 2"}
        ]

        actions = list(pipeline.build_actions(episodes))

        assert len(actions) == 2
        assert actions[0]["_id"] == "ep_1"
        assert actions[1]["_id"] == "ep_2"

    @patch('src.pipelines.ingest_episodes.storage')
    def test_build_actions_handles_errors(self, mock_storage, caplog):
        """Test that build_actions handles transformation errors gracefully."""
        mock_storage.list_show_ids.return_value = []

        pipeline = IngestEpisodesPipeline(es_service=MagicMock())

        # Create invalid episode that will cause error
        episodes = [
            None,  # This will cause an error
        ]

        actions = list(pipeline.build_actions(episodes))

        # Should return empty list without crashing
        assert len(actions) == 0
