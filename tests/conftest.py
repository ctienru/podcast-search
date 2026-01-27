"""Pytest fixtures for podcast-search tests."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from datetime import datetime, timezone


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.exists = Mock(return_value=False)
    client.indices.create = Mock()
    client.indices.delete = Mock()
    client.indices.exists_alias = Mock(return_value=False)
    client.indices.update_aliases = Mock()
    client.indices.get_alias = Mock(return_value={})
    client.reindex = Mock()
    return client


@pytest.fixture
def mock_storage():
    """Mock LocalSearchDataStorage."""
    storage = MagicMock()
    storage.list_show_ids = Mock(return_value=[])
    storage.load_show = Mock(return_value={})
    storage.list_episode_ids = Mock(return_value=[])
    storage.load_episode = Mock(return_value={})
    return storage


@pytest.fixture
def sample_show():
    """Sample show data."""
    return {
        "show_id": "show_123",
        "title": "Test Podcast",
        "author": "Test Publisher",
        "external_id": "apple_123",
        "provider": "apple_podcasts",
        "image": {
            "url": "https://example.com/show.jpg"
        },
        "external_urls": {
            "apple_podcasts": "https://podcasts.apple.com/show_123"
        },
        "episode_stats": {
            "total_count": 100
        },
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-17T00:00:00Z"
    }


@pytest.fixture
def sample_episode():
    """Sample episode data."""
    return {
        "episode_id": "ep_123",
        "show_id": "show_123",
        "title": "Episode 1: Test",
        "description": "<p>This is a <strong>test</strong> description with <a href='https://example.com'>link</a>.</p>",
        "audio": {
            "url": "https://example.com/ep1.mp3",
            "duration_seconds": 1800
        },
        "image": {
            "url": "https://example.com/ep1.jpg"
        },
        "publish_date": "2025-01-15T10:00:00Z",
        "created_at": "2025-01-15T10:00:00Z",
        "updated_at": "2025-01-17T00:00:00Z"
    }


@pytest.fixture
def sample_show_for_episode():
    """Sample show data for episode embedding."""
    return {
        "show_id": "show_123",
        "title": "Test Podcast",
        "publisher": "Test Publisher",
        "image_url": "https://example.com/show.jpg"
    }
