"""
Podcast Data Cleaning Module

Reference: Entity and Event Topic Extraction from Podcast Episode Title and Description
           Using Entity Linking (WWW '23, Amazon)

Three-layer architecture:
- Layer 1: RAW XML (podcast-crawler/data/raw/rss/)
- Layer 2: CLEANED JSON (podcast-search/data/cleaned/)
- Layer 3: EMBEDDING_INPUT (podcast-search/data/embedding_input/)
"""

from src.cleaning.text_cleaner import PodcastTextCleaner
from src.cleaning.rss_parser import RSSParser

__all__ = ["PodcastTextCleaner", "RSSParser"]
