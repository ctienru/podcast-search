"""Tests for LocalSearchDataStorage."""

import pytest
import json
import tempfile
from pathlib import Path
from src.storage.local import LocalSearchDataStorage


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create expected directory structure
        (data_dir / "normalized" / "shows").mkdir(parents=True)
        (data_dir / "normalized" / "episodes").mkdir(parents=True)
        yield data_dir


@pytest.fixture
def storage(temp_data_dir):
    """Create a LocalSearchDataStorage instance for testing."""
    return LocalSearchDataStorage(temp_data_dir)


class TestShowOperations:
    """Test show-related operations."""

    def test_list_show_ids_returns_empty_when_dir_missing(self):
        """Test that list_show_ids returns empty list when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalSearchDataStorage(Path(tmpdir))
            result = storage.list_show_ids()
            assert result == []

    def test_list_show_ids_returns_sorted_ids(self, storage, temp_data_dir):
        """Test that list_show_ids returns sorted show IDs."""
        shows_dir = temp_data_dir / "normalized" / "shows"

        # Create test show files
        (shows_dir / "show_003.json").write_text("{}")
        (shows_dir / "show_001.json").write_text("{}")
        (shows_dir / "show_002.json").write_text("{}")
        (shows_dir / "not_json.txt").write_text("ignore")

        result = storage.list_show_ids()

        assert result == ["show_001", "show_002", "show_003"]
        assert "not_json" not in result

    def test_load_show_success(self, storage, temp_data_dir):
        """Test successfully loading a show."""
        shows_dir = temp_data_dir / "normalized" / "shows"

        show_data = {
            "show_id": "show_123",
            "title": "Test Podcast",
            "author": "Test Publisher"
        }

        (shows_dir / "show_123.json").write_text(
            json.dumps(show_data),
            encoding="utf-8"
        )

        result = storage.load_show("show_123")

        assert result == show_data
        assert result["show_id"] == "show_123"
        assert result["title"] == "Test Podcast"

    def test_load_show_file_not_found(self, storage):
        """Test loading a show that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="show_not_found"):
            storage.load_show("nonexistent_show")

    def test_load_show_with_unicode(self, storage, temp_data_dir):
        """Test loading a show with unicode characters."""
        shows_dir = temp_data_dir / "normalized" / "shows"

        show_data = {
            "show_id": "show_123",
            "title": "測試 Podcast 🎙️",
            "author": "Test Publisher"
        }

        (shows_dir / "show_123.json").write_text(
            json.dumps(show_data, ensure_ascii=False),
            encoding="utf-8"
        )

        result = storage.load_show("show_123")

        assert result["title"] == "測試 Podcast 🎙️"


class TestEpisodeOperations:
    """Test episode-related operations."""

    def test_list_episode_ids_returns_empty_when_dir_missing(self):
        """Test that list_episode_ids returns empty list when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalSearchDataStorage(Path(tmpdir))
            result = list(storage.list_episode_ids())
            assert result == []

    def test_list_episode_ids_returns_generator(self, storage, temp_data_dir):
        """Test that list_episode_ids returns a generator."""
        episodes_dir = temp_data_dir / "normalized" / "episodes"

        # Create test episode files
        (episodes_dir / "ep_001.json").write_text("{}")
        (episodes_dir / "ep_002.json").write_text("{}")
        (episodes_dir / "not_json.txt").write_text("ignore")

        result = storage.list_episode_ids()

        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

        # Convert to list to check content
        episode_ids = list(result)
        assert len(episode_ids) == 2
        assert "ep_001" in episode_ids
        assert "ep_002" in episode_ids
        assert "not_json" not in episode_ids

    def test_load_episode_success(self, storage, temp_data_dir):
        """Test successfully loading an episode."""
        episodes_dir = temp_data_dir / "normalized" / "episodes"

        episode_data = {
            "episode_id": "ep_123",
            "show_id": "show_123",
            "title": "Episode 1",
            "description": "Test description"
        }

        (episodes_dir / "ep_123.json").write_text(
            json.dumps(episode_data),
            encoding="utf-8"
        )

        result = storage.load_episode("ep_123")

        assert result == episode_data
        assert result["episode_id"] == "ep_123"
        assert result["title"] == "Episode 1"

    def test_load_episode_file_not_found(self, storage):
        """Test loading an episode that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="episode_not_found"):
            storage.load_episode("nonexistent_episode")

    def test_load_episode_with_complex_data(self, storage, temp_data_dir):
        """Test loading an episode with complex nested data."""
        episodes_dir = temp_data_dir / "normalized" / "episodes"

        episode_data = {
            "episode_id": "ep_123",
            "show_id": "show_123",
            "title": "Episode 1",
            "audio": {
                "url": "https://example.com/ep1.mp3",
                "duration_seconds": 1800
            },
            "image": {
                "url": "https://example.com/ep1.jpg"
            }
        }

        (episodes_dir / "ep_123.json").write_text(
            json.dumps(episode_data),
            encoding="utf-8"
        )

        result = storage.load_episode("ep_123")

        assert result["audio"]["url"] == "https://example.com/ep1.mp3"
        assert result["audio"]["duration_seconds"] == 1800
        assert result["image"]["url"] == "https://example.com/ep1.jpg"


class TestDirectoryStructure:
    """Test directory structure initialization."""

    def test_initializes_correct_paths(self, temp_data_dir):
        """Test that storage initializes with correct directory paths."""
        storage = LocalSearchDataStorage(temp_data_dir)

        assert storage.data_dir == temp_data_dir
        assert storage.shows_dir == temp_data_dir / "normalized" / "shows"
        assert storage.episodes_dir == temp_data_dir / "normalized" / "episodes"

    def test_handles_non_existent_data_dir(self):
        """Test that storage can be initialized with non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "nonexistent"
            storage = LocalSearchDataStorage(data_dir)

            # Should not raise error on initialization
            assert storage.data_dir == data_dir

            # Should return empty lists when directories don't exist
            assert storage.list_show_ids() == []
            assert list(storage.list_episode_ids()) == []
