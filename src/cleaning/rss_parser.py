"""
RSS XML Parser

Parse show and episode data from raw RSS XML files.
"""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


# RSS 2.0 namespaces
NAMESPACES = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "soundon": "http://soundon.fm/spec/podcast-1.0",
    "googleplay": "http://www.google.com/schemas/play-podcasts/1.0",
}


@dataclass
class RawEpisode:
    """Raw episode data from RSS XML"""

    episode_id: str
    show_id: str
    guid: str
    title: str
    description: Optional[str]  # <description> tag
    content_encoded: Optional[str]  # <content:encoded> tag (richer HTML)
    pub_date: Optional[str]
    duration: Optional[str]
    audio_url: Optional[str]
    audio_type: Optional[str]
    audio_length: Optional[int]
    link: Optional[str]


@dataclass
class RawShow:
    """Raw show data from RSS XML"""

    show_id: str
    title: str
    description: Optional[str]
    language: Optional[str]
    author: Optional[str]
    image_url: Optional[str]
    link: Optional[str]


class RSSParser:
    """
    Parse RSS XML files to extract show and episode data.

    Usage:
        parser = RSSParser()
        for show_id, show, episodes in parser.parse_all(raw_rss_dir):
            # show: RawShow
            # episodes: list[RawEpisode]
    """

    def __init__(self):
        # Register namespaces for proper parsing
        for prefix, uri in NAMESPACES.items():
            ET.register_namespace(prefix, uri)

    def parse_file(self, xml_path: Path) -> tuple[RawShow, list[RawEpisode]]:
        """
        Parse a single RSS XML file.

        Args:
            xml_path: Path to RSS XML file (e.g., show:apple:1234567890.xml)

        Returns:
            (RawShow, list[RawEpisode])
        """
        # Extract show_id from filename
        show_id = xml_path.stem  # e.g., "show:apple:1234567890"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        channel = root.find("channel")
        if channel is None:
            raise ValueError(f"No <channel> found in {xml_path}")

        # Parse show
        show = self._parse_show(channel, show_id)

        # Parse episodes
        episodes = list(self._parse_episodes(channel, show_id))

        return show, episodes

    def _parse_show(self, channel: ET.Element, show_id: str) -> RawShow:
        """Parse show-level data from <channel>"""
        return RawShow(
            show_id=show_id,
            title=self._get_text(channel, "title") or "",
            description=self._get_text(channel, "description"),
            language=self._get_text(channel, "language"),
            author=self._get_text(channel, "itunes:author", NAMESPACES),
            image_url=self._get_itunes_image(channel),
            link=self._get_text(channel, "link"),
        )

    def _parse_episodes(
        self, channel: ET.Element, show_id: str
    ) -> Iterator[RawEpisode]:
        """Parse all <item> elements as episodes"""
        for item in channel.findall("item"):
            yield self._parse_episode(item, show_id)

    def _parse_episode(self, item: ET.Element, show_id: str) -> RawEpisode:
        """Parse a single <item> element"""
        # Get GUID for unique identification
        guid = self._get_text(item, "guid") or ""

        # Generate episode_id from show_id + guid hash
        episode_id = self._generate_episode_id(show_id, guid)

        # Get enclosure (audio) info
        enclosure = item.find("enclosure")
        audio_url = None
        audio_type = None
        audio_length = None
        if enclosure is not None:
            audio_url = enclosure.get("url")
            audio_type = enclosure.get("type")
            length_str = enclosure.get("length")
            if length_str and length_str.isdigit():
                audio_length = int(length_str)

        # Get duration (could be in various formats)
        duration = self._get_text(item, "itunes:duration", NAMESPACES)

        return RawEpisode(
            episode_id=episode_id,
            show_id=show_id,
            guid=guid,
            title=self._get_text(item, "title") or "",
            description=self._get_text(item, "description"),
            content_encoded=self._get_text(item, "content:encoded", NAMESPACES),
            pub_date=self._get_text(item, "pubDate"),
            duration=duration,
            audio_url=audio_url,
            audio_type=audio_type,
            audio_length=audio_length,
            link=self._get_text(item, "link"),
        )

    def _generate_episode_id(self, show_id: str, guid: str) -> str:
        """
        Generate episode_id from show_id and guid.

        Format: episode:apple:{apple_id}:{hash8}
        """
        # Extract apple_id from show_id (e.g., "show:apple:1234567890" -> "1234567890")
        parts = show_id.split(":")
        if len(parts) >= 3:
            apple_id = parts[2]
        else:
            apple_id = show_id

        # Hash the guid to create a short unique suffix
        guid_hash = hashlib.md5(guid.encode()).hexdigest()[:8]

        return f"episode:apple:{apple_id}:{guid_hash}"

    def _get_text(
        self,
        element: ET.Element,
        tag: str,
        namespaces: Optional[dict] = None,
    ) -> Optional[str]:
        """Get text content of a child element"""
        if namespaces and ":" in tag:
            child = element.find(tag, namespaces)
        else:
            child = element.find(tag)

        if child is not None and child.text:
            return child.text.strip()
        return None

    def _get_itunes_image(self, channel: ET.Element) -> Optional[str]:
        """Get iTunes image URL"""
        # Try itunes:image first
        itunes_image = channel.find("itunes:image", NAMESPACES)
        if itunes_image is not None:
            href = itunes_image.get("href")
            if href:
                return href

        # Fallback to <image><url>
        image = channel.find("image")
        if image is not None:
            url = image.find("url")
            if url is not None and url.text:
                return url.text.strip()

        return None

    def parse_all(
        self, raw_rss_dir: Path
    ) -> Iterator[tuple[str, RawShow, list[RawEpisode]]]:
        """
        Parse all RSS XML files in a directory.

        Args:
            raw_rss_dir: Directory containing RSS XML files

        Yields:
            (show_id, RawShow, list[RawEpisode])
        """
        for xml_path in sorted(raw_rss_dir.glob("*.xml")):
            try:
                show, episodes = self.parse_file(xml_path)
                yield show.show_id, show, episodes
            except Exception as e:
                print(f"Error parsing {xml_path}: {e}")
                continue
