"""Tests for RSSParser."""

import pytest
import tempfile
from pathlib import Path
from src.cleaning.rss_parser import RSSParser, RawEpisode, RawShow, NAMESPACES


@pytest.fixture
def parser():
    """Create an RSSParser instance."""
    return RSSParser()


@pytest.fixture
def sample_rss_xml():
    """Sample RSS XML content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Test Podcast</title>
    <description>A test podcast description</description>
    <language>en-US</language>
    <itunes:author>Test Author</itunes:author>
    <itunes:image href="https://example.com/image.jpg"/>
    <link>https://example.com</link>
    <item>
      <title>Episode 1: Pilot</title>
      <guid>guid-episode-1</guid>
      <description>First episode description</description>
      <content:encoded><![CDATA[<p>Rich <strong>content</strong> for episode 1</p>]]></content:encoded>
      <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
      <itunes:duration>30:00</itunes:duration>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" length="12345678"/>
      <link>https://example.com/ep1</link>
    </item>
    <item>
      <title>Episode 2: Second</title>
      <guid>guid-episode-2</guid>
      <description>Second episode description</description>
      <pubDate>Mon, 08 Jan 2024 10:00:00 +0000</pubDate>
      <itunes:duration>45:30</itunes:duration>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg" length="23456789"/>
    </item>
  </channel>
</rss>"""


@pytest.fixture
def temp_rss_file(sample_rss_xml):
    """Create a temporary RSS XML file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix="show_apple_123456_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(sample_rss_xml)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


class TestParseFile:
    """Test parsing RSS XML files."""

    def test_parses_show_data(self, parser, temp_rss_file):
        """Test parsing show-level data."""
        show, episodes = parser.parse_file(temp_rss_file)

        assert isinstance(show, RawShow)
        assert show.title == "Test Podcast"
        assert show.description == "A test podcast description"
        assert show.language == "en-US"
        assert show.author == "Test Author"
        assert show.image_url == "https://example.com/image.jpg"
        assert show.link == "https://example.com"

    def test_parses_episodes(self, parser, temp_rss_file):
        """Test parsing episode data."""
        show, episodes = parser.parse_file(temp_rss_file)

        assert len(episodes) == 2

        ep1 = episodes[0]
        assert isinstance(ep1, RawEpisode)
        assert ep1.title == "Episode 1: Pilot"
        assert ep1.guid == "guid-episode-1"
        assert ep1.description == "First episode description"
        assert "Rich" in ep1.content_encoded
        assert "strong" in ep1.content_encoded
        assert ep1.pub_date == "Mon, 01 Jan 2024 10:00:00 +0000"
        assert ep1.duration == "30:00"
        assert ep1.audio_url == "https://example.com/ep1.mp3"
        assert ep1.audio_type == "audio/mpeg"
        assert ep1.audio_length == 12345678
        assert ep1.link == "https://example.com/ep1"

    def test_extracts_show_id_from_filename(self, parser, temp_rss_file):
        """Test that show_id is extracted from filename."""
        show, episodes = parser.parse_file(temp_rss_file)

        # show_id should be the filename stem
        expected_show_id = temp_rss_file.stem
        assert show.show_id == expected_show_id

    def test_generates_episode_ids(self, parser, temp_rss_file):
        """Test that episode IDs are generated correctly."""
        show, episodes = parser.parse_file(temp_rss_file)

        for ep in episodes:
            assert ep.episode_id.startswith("episode:")
            assert ep.show_id == show.show_id


class TestParseShow:
    """Test show-level parsing."""

    def test_handles_missing_optional_fields(self, parser):
        """Test handling of missing optional fields."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Minimal Podcast</title>
  </channel>
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            show, episodes = parser.parse_file(temp_path)

            assert show.title == "Minimal Podcast"
            assert show.description is None
            assert show.language is None
            assert show.author is None
            assert show.image_url is None
            assert show.link is None
        finally:
            temp_path.unlink()

    def test_handles_image_url_fallback(self, parser):
        """Test fallback to <image><url> when itunes:image is missing."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <image>
      <url>https://example.com/fallback.jpg</url>
    </image>
  </channel>
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            show, episodes = parser.parse_file(temp_path)
            assert show.image_url == "https://example.com/fallback.jpg"
        finally:
            temp_path.unlink()


class TestParseEpisode:
    """Test episode-level parsing."""

    def test_handles_missing_optional_fields(self, parser):
        """Test handling of episodes with missing optional fields."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Minimal Episode</title>
      <guid>guid-minimal</guid>
    </item>
  </channel>
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            show, episodes = parser.parse_file(temp_path)

            assert len(episodes) == 1
            ep = episodes[0]
            assert ep.title == "Minimal Episode"
            assert ep.guid == "guid-minimal"
            assert ep.description is None
            assert ep.content_encoded is None
            assert ep.pub_date is None
            assert ep.duration is None
            assert ep.audio_url is None
            assert ep.audio_type is None
            assert ep.audio_length is None
        finally:
            temp_path.unlink()

    def test_handles_enclosure_without_length(self, parser):
        """Test handling of enclosure without length attribute."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Episode</title>
      <guid>guid-1</guid>
      <enclosure url="https://example.com/ep.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            show, episodes = parser.parse_file(temp_path)
            ep = episodes[0]

            assert ep.audio_url == "https://example.com/ep.mp3"
            assert ep.audio_type == "audio/mpeg"
            assert ep.audio_length is None
        finally:
            temp_path.unlink()


class TestGenerateEpisodeId:
    """Test episode ID generation."""

    def test_generates_correct_format(self, parser):
        """Test that episode ID format is correct."""
        show_id = "show:apple:1234567890"
        guid = "test-guid-12345"

        episode_id = parser._generate_episode_id(show_id, guid)

        assert episode_id.startswith("episode:apple:")
        assert "1234567890" in episode_id
        # Should have hash suffix
        assert len(episode_id.split(":")) == 4

    def test_different_guids_produce_different_ids(self, parser):
        """Test that different GUIDs produce different episode IDs."""
        show_id = "show:apple:123"

        id1 = parser._generate_episode_id(show_id, "guid-1")
        id2 = parser._generate_episode_id(show_id, "guid-2")

        assert id1 != id2

    def test_same_guid_produces_same_id(self, parser):
        """Test that same GUID produces same episode ID."""
        show_id = "show:apple:123"
        guid = "same-guid"

        id1 = parser._generate_episode_id(show_id, guid)
        id2 = parser._generate_episode_id(show_id, guid)

        assert id1 == id2

    def test_handles_non_standard_show_id(self, parser):
        """Test handling of non-standard show ID format."""
        show_id = "custom_show_id"
        guid = "test-guid"

        episode_id = parser._generate_episode_id(show_id, guid)

        # Should use entire show_id as apple_id
        assert "custom_show_id" in episode_id


class TestParseAll:
    """Test parsing all RSS files in a directory."""

    def test_parses_multiple_files(self, parser, sample_rss_xml):
        """Test parsing multiple RSS files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rss_dir = Path(tmpdir)

            # Create multiple RSS files
            (rss_dir / "show1.xml").write_text(sample_rss_xml, encoding="utf-8")
            (rss_dir / "show2.xml").write_text(sample_rss_xml, encoding="utf-8")

            results = list(parser.parse_all(rss_dir))

            assert len(results) == 2
            for show_id, show, episodes in results:
                assert show_id in ["show1", "show2"]
                assert isinstance(show, RawShow)
                assert len(episodes) == 2

    def test_ignores_non_xml_files(self, parser, sample_rss_xml):
        """Test that non-XML files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rss_dir = Path(tmpdir)

            # Create XML and non-XML files
            (rss_dir / "show.xml").write_text(sample_rss_xml, encoding="utf-8")
            (rss_dir / "readme.txt").write_text("Not XML", encoding="utf-8")
            (rss_dir / "data.json").write_text("{}", encoding="utf-8")

            results = list(parser.parse_all(rss_dir))

            assert len(results) == 1
            assert results[0][0] == "show"

    def test_handles_empty_directory(self, parser):
        """Test handling of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(parser.parse_all(Path(tmpdir)))
            assert results == []

    def test_skips_invalid_xml(self, parser, sample_rss_xml):
        """Test that invalid XML files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rss_dir = Path(tmpdir)

            # Create valid and invalid XML files
            (rss_dir / "valid.xml").write_text(sample_rss_xml, encoding="utf-8")
            (rss_dir / "invalid.xml").write_text("Not valid XML <><>", encoding="utf-8")

            results = list(parser.parse_all(rss_dir))

            # Should only return valid file
            assert len(results) == 1
            assert results[0][0] == "valid"


class TestErrorHandling:
    """Test error handling."""

    def test_raises_error_for_missing_channel(self, parser):
        """Test that missing channel raises error."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="No <channel> found"):
                parser.parse_file(temp_path)
        finally:
            temp_path.unlink()

    def test_handles_unicode_content(self, parser):
        """Test handling of Unicode content."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>測試 Podcast 🎙️</title>
    <description>中文描述 with émojis</description>
    <itunes:author>作者名</itunes:author>
    <item>
      <title>第一集：開播</title>
      <guid>unicode-guid</guid>
      <description>中英混合 content</description>
    </item>
  </channel>
</rss>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            show, episodes = parser.parse_file(temp_path)

            assert show.title == "測試 Podcast 🎙️"
            assert "中文描述" in show.description
            assert show.author == "作者名"
            assert episodes[0].title == "第一集：開播"
        finally:
            temp_path.unlink()


class TestNamespaces:
    """Test namespace handling."""

    def test_namespaces_defined(self):
        """Test that required namespaces are defined."""
        assert "itunes" in NAMESPACES
        assert "content" in NAMESPACES
        assert "atom" in NAMESPACES

    def test_parses_content_encoded(self, parser, temp_rss_file):
        """Test parsing content:encoded element."""
        show, episodes = parser.parse_file(temp_rss_file)

        ep1 = episodes[0]
        assert ep1.content_encoded is not None
        assert "Rich" in ep1.content_encoded

    def test_parses_itunes_duration(self, parser, temp_rss_file):
        """Test parsing itunes:duration element."""
        show, episodes = parser.parse_file(temp_rss_file)

        assert episodes[0].duration == "30:00"
        assert episodes[1].duration == "45:30"
