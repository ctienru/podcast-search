"""Tests for IngestShowsPipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.pipelines.ingest_shows import IngestShowsPipeline


class TestToEsDoc:
    """Test the to_es_doc() transformation method."""

    def test_complete_show_transformation(self):
        """Test transformation of complete show data."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast",
            "author": "Test Publisher",
            "language": "en",
            "external_id": "apple_123",
            "provider": "apple_podcasts",
            "image": {
                "url": "https://example.com/show.jpg"
            },
            "external_urls": {
                "apple_podcasts": "https://podcasts.apple.com/show_123"
            },
            "episode_stats": {
                "episode_count": 100,
                "last_episode_at": "2025-01-15T10:00:00Z"
            },
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-17T00:00:00Z"
        }

        result = pipeline.to_es_doc(show)

        # Verify structure
        assert result["_index"] == "shows"
        assert result["_id"] == "show_123"

        source = result["_source"]
        assert source["show_id"] == "show_123"
        assert source["title"] == "Test Podcast"
        assert source["publisher"] == "Test Publisher"
        assert source["language"] == "en"
        assert source["image_url"] == "https://example.com/show.jpg"

        # Verify external_ids with dynamic provider
        assert source["external_ids"]["apple_podcasts"] == "apple_123"

        # Verify external_urls with dynamic provider
        assert source["external_urls"]["apple_podcasts"] == "https://podcasts.apple.com/show_123"

        # Verify episode stats
        assert source["episode_count"] == 100
        assert source["last_episode_at"] == "2025-01-15T10:00:00Z"

        # Verify timestamps
        assert source["created_at"] == "2025-01-01T00:00:00Z"
        assert source["updated_at"] == "2025-01-17T00:00:00Z"

        # Verify popularity_score is None by default
        assert source["popularity_score"] is None

    def test_show_without_image(self):
        """Test transformation when show image is missing."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast"
        }

        result = pipeline.to_es_doc(show)
        source = result["_source"]

        # image_url should be None when image is missing
        assert source["image_url"] is None

    def test_show_without_episode_stats(self):
        """Test transformation when episode_stats is missing."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast"
        }

        result = pipeline.to_es_doc(show)
        source = result["_source"]

        # Episode stats should be None when missing
        assert source["episode_count"] is None
        assert source["last_episode_at"] is None

    def test_default_provider_backward_compatibility(self):
        """Test that provider defaults to 'apple_podcasts' for backward compatibility."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast",
            "external_id": "apple_123",
            # No provider field
            "external_urls": {
                "apple_podcasts": "https://podcasts.apple.com/show_123"
            }
        }

        result = pipeline.to_es_doc(show)
        source = result["_source"]

        # Should default to apple_podcasts
        assert "apple_podcasts" in source["external_ids"]
        assert source["external_ids"]["apple_podcasts"] == "apple_123"
        assert source["external_urls"]["apple_podcasts"] == "https://podcasts.apple.com/show_123"

    def test_custom_provider(self):
        """Test that custom provider is used when specified."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast",
            "external_id": "spotify_123",
            "provider": "spotify",
            "external_urls": {
                "spotify": "https://open.spotify.com/show_123"
            }
        }

        result = pipeline.to_es_doc(show)
        source = result["_source"]

        # Should use custom provider
        assert "spotify" in source["external_ids"]
        assert source["external_ids"]["spotify"] == "spotify_123"
        assert source["external_urls"]["spotify"] == "https://open.spotify.com/show_123"

    def test_external_urls_handling(self):
        """Test that external_urls are correctly preserved."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        show = {
            "show_id": "show_123",
            "title": "Test Podcast",
            "provider": "apple_podcasts",
            "external_urls": {
                "apple_podcasts": "https://podcasts.apple.com/show_123",
                "web": "https://example.com/podcast"
            }
        }

        result = pipeline.to_es_doc(show)
        source = result["_source"]

        # Should only include the provider's URL
        assert source["external_urls"]["apple_podcasts"] == "https://podcasts.apple.com/show_123"
        # Other URLs from external_urls should not be in the output
        assert "web" not in source["external_urls"] or source["external_urls"].get("web") is None


class TestLoadShows:
    """Test the load_shows() method."""

    @patch('src.pipelines.ingest_shows.storage')
    def test_load_shows_success(self, mock_storage):
        """Test successful loading of shows."""
        show_data = {
            "show_id": "show_123",
            "title": "Test Podcast"
        }

        mock_storage.list_show_ids.return_value = ["show_123"]
        mock_storage.load_show.return_value = show_data

        pipeline = IngestShowsPipeline(es_service=MagicMock())

        shows = list(pipeline.load_shows())

        assert len(shows) == 1
        assert shows[0]["show_id"] == "show_123"

    @patch('src.pipelines.ingest_shows.storage')
    def test_load_shows_with_missing_show(self, mock_storage, caplog):
        """Test that missing shows are skipped with warning."""
        mock_storage.list_show_ids.return_value = ["show_123", "show_456"]
        mock_storage.load_show.side_effect = [
            {"show_id": "show_123", "title": "Test 1"},
            None  # Missing show
        ]

        pipeline = IngestShowsPipeline(es_service=MagicMock())

        shows = list(pipeline.load_shows())

        # Should only return the valid show
        assert len(shows) == 1
        assert shows[0]["show_id"] == "show_123"


class TestBuildActions:
    """Test the build_actions() generator."""

    def test_build_actions_generates_docs(self):
        """Test that build_actions generates ES documents."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        shows = [
            {"show_id": "show_1", "title": "Show 1"},
            {"show_id": "show_2", "title": "Show 2"}
        ]

        actions = list(pipeline.build_actions(shows))

        assert len(actions) == 2
        assert actions[0]["_id"] == "show_1"
        assert actions[1]["_id"] == "show_2"

    def test_build_actions_handles_errors(self, caplog):
        """Test that build_actions handles transformation errors gracefully."""
        pipeline = IngestShowsPipeline(es_service=MagicMock())

        # Create invalid show that will cause error (missing show_id)
        shows = [
            {"title": "Valid Show", "show_id": "show_1"},
            {},  # Missing show_id will cause KeyError
        ]

        actions = list(pipeline.build_actions(shows))

        # Should return only the valid action
        assert len(actions) == 1
        assert actions[0]["_id"] == "show_1"
