"""
Podcast Text Cleaner

Reference: Entity and Event Topic Extraction from Podcast Episode Title and Description
           Using Entity Linking (WWW '23, Amazon)

Core principles:
1. Metadata is a "hint" not full text -> aim for topic representativeness
2. Extraneous text should be removed, not learned -> rule-based cleaning
3. Structure matters more than linguistic variety -> paragraph-level processing (not sentence-level)
4. Make data "comparable" -> consistent length, content, and structure
"""

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class ParagraphInfo:
    """Paragraph cleaning result"""

    index: int
    text: str
    char_count: int
    flags: list[str] = field(default_factory=list)
    kept: bool = True
    rules_hit: list[str] = field(default_factory=list)


@dataclass
class CleaningStats:
    """Cleaning statistics"""

    raw_char_count: int = 0
    kept_char_count: int = 0
    total_paragraphs: int = 0
    kept_paragraphs: int = 0
    removed_paragraphs: int = 0
    removal_breakdown: dict = field(default_factory=dict)


@dataclass
class CleanedEpisode:
    """Layer 2: CLEANED structure"""

    episode_id: str
    show_id: str
    normalized_title: str
    normalized_description: str
    paragraphs: list[ParagraphInfo]
    stats: CleaningStats
    pipeline_version: str = "clean-v1.0"
    rules_version: str = "rules-v1.0"


class PodcastTextCleaner:
    """
    Podcast metadata cleaner

    Supports cleaning rules for multiple sources (SoundOn, Flightcast, NPR, NYT, etc.)
    """

    # ========== Chinese Boilerplate Rules ==========
    ZH_SPONSOR_PATTERNS = [
        r"(本集|本節目|本期|這集).*?(贊助|支持|呈現|提供)",
        r"(折扣碼|優惠碼|promo\s*code)",
        r"輸入.*?(折扣|優惠|code)",
    ]

    ZH_CTA_PATTERNS = [
        r"(訂閱|追蹤|關注|按讚).*(頻道|節目|podcast|youtube|ig|fb)",
        r"(留言|評論|review).*(apple|spotify|itunes|kkbox)",
        r"(每週|每月|weekly|monthly).*(更新|上架)",
        r"(歡迎|記得|別忘了).*(分享|轉發|訂閱)",
        r"(加入|追蹤|follow).*(@|ig|instagram|facebook|fb)",
        r"訂閱方案",
        r"更新頻率",
        r"訂閱平台",
        r"合作聯繫",
    ]

    ZH_HOSTING_PATTERNS = [
        r"Hosting\s+provided\s+by",
        r"Powered\s+by\s+(SoundOn|Firstory|KKBOX)",
    ]

    # ========== English Boilerplate Rules ==========
    EN_SPONSOR_PATTERNS = [
        r"(sponsored|brought to you|powered)\s+by",
        r"use\s+code\s+.{1,20}\s+(for|to\s+get)",
        r"(discount|promo)\s+code",
        r"\d+%\s+off",
        r"Sponsors?:",
    ]

    EN_CTA_PATTERNS = [
        r"(subscribe|follow|like)\s+(to\s+)?(the\s+)?(channel|podcast|show|newsletter)",
        r"(leave\s+a\s+)?(review|rating)\s+(on\s+)?(apple|spotify|itunes)",
        r"sign\s+up\s+(for|at)",
        r"join\s+(our|the)\s+(newsletter|community|circle)",
        r"follow\s+(us\s+)?(on\s+)?(instagram|twitter|x|facebook)",
        r"get\s+email\s+updates",
        r"Learn\s+more\s+about\s+sponsor\s+message\s+choices",
        r"Privacy\s+Policy",
    ]

    EN_PRODUCTION_PATTERNS = [
        r"(produced|edited)\s+by",
        r"(executive\s+)?producer\s+is",
        r"This\s+(podcast|episode)\s+was\s+(produced|edited)",
        r"This\s+episode:",  # NPR style "This episode: ..."
    ]

    EN_SUBSCRIBE_PATTERNS = [
        r"Subscribe\s+today\s+at",
        r"For\s+more\s+information\s+on\s+today's\s+episode",
        r"Transcripts?\s+(of\s+each\s+episode\s+)?will\s+be\s+made\s+available",
        r"download\s+.*?\s+app\s+at",
    ]

    # ========== General Rules ==========
    URL_PATTERN = r"https?://\S+"
    EMAIL_PATTERN = r"[\w.-]+@[\w.-]+\.\w+"
    TIMESTAMP_PATTERN = r"\(\d{1,2}:\d{2}(:\d{2})?\)\s*"  # (00:00) or (01:23:45)

    def __init__(self):
        self.show_paragraph_freq: dict[str, dict] = {}

    def clean_html(self, text: str) -> str:
        """
        Step 1: HTML -> plain text
        Key: preserve paragraph boundaries (separated by \\n)
        """
        if not text:
            return ""

        # Decode HTML entities first
        text = html.unescape(text)

        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Get text with newlines as separator (preserves paragraph boundaries)
        text = soup.get_text(separator="\n")

        return text

    def normalize_text(self, text: str) -> str:
        """
        Step 1b: Normalize whitespace and special characters
        Key: preserve paragraph boundaries (\\n\\n)
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters (but keep \n)
        text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]", "", text)

        # Compress inline whitespace (but preserve \n)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Multiple newlines → double newline (paragraph separator)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def split_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.
        Paragraph = text block separated by blank lines
        """
        if not text:
            return []

        paragraphs = re.split(r"\n\s*\n", text)
        # Filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def detect_language(self, text: str) -> str:
        """
        Simple language detection (Chinese vs English)
        """
        if not text:
            return "unknown"

        # Count Chinese characters
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        return "en"

    def is_boilerplate_paragraph(
        self, paragraph: str, language: str
    ) -> tuple[bool, list[str], list[str]]:
        """
        Determine if paragraph is boilerplate (functional text)

        Returns:
            (is_boilerplate, flags, rules_hit)
        """
        flags = []
        rules_hit = []
        paragraph_lower = paragraph.lower()

        # Choose patterns based on language
        if language == "zh":
            sponsor_patterns = self.ZH_SPONSOR_PATTERNS
            cta_patterns = self.ZH_CTA_PATTERNS
            hosting_patterns = self.ZH_HOSTING_PATTERNS
            production_patterns = []
            subscribe_patterns = []
        else:
            sponsor_patterns = self.EN_SPONSOR_PATTERNS
            cta_patterns = self.EN_CTA_PATTERNS
            hosting_patterns = []
            production_patterns = self.EN_PRODUCTION_PATTERNS
            subscribe_patterns = self.EN_SUBSCRIBE_PATTERNS

        # Check sponsor patterns
        for pattern in sponsor_patterns:
            if re.search(pattern, paragraph_lower, re.IGNORECASE):
                flags.append("sponsor")
                rules_hit.append(f"sponsor:{pattern[:30]}")
                break

        # Check CTA patterns
        for pattern in cta_patterns:
            if re.search(pattern, paragraph_lower, re.IGNORECASE):
                flags.append("cta")
                rules_hit.append(f"cta:{pattern[:30]}")
                break

        # Check hosting patterns (both languages)
        for pattern in self.ZH_HOSTING_PATTERNS + hosting_patterns:
            if re.search(pattern, paragraph_lower, re.IGNORECASE):
                flags.append("hosting")
                rules_hit.append(f"hosting:{pattern[:30]}")
                break

        # Check production patterns (English)
        for pattern in production_patterns:
            if re.search(pattern, paragraph_lower, re.IGNORECASE):
                flags.append("production")
                rules_hit.append(f"production:{pattern[:30]}")
                break

        # Check subscribe patterns (English)
        for pattern in subscribe_patterns:
            if re.search(pattern, paragraph_lower, re.IGNORECASE):
                flags.append("subscribe")
                rules_hit.append(f"subscribe:{pattern[:30]}")
                break

        # Check URL density (>2 URLs in a short paragraph = likely boilerplate)
        urls = re.findall(self.URL_PATTERN, paragraph)
        if len(urls) >= 2 and len(paragraph) < 500:
            flags.append("url_heavy")
            rules_hit.append("url_heavy:>=2_urls")

        # Check if paragraph is mostly timestamps
        timestamp_removed = re.sub(self.TIMESTAMP_PATTERN, "", paragraph)
        if len(timestamp_removed.strip()) < len(paragraph) * 0.3:
            flags.append("timestamps")
            rules_hit.append("timestamps:>70%")

        # Check short CTA-like text
        if len(paragraph) < 50:
            cta_keywords = [
                "訂閱",
                "追蹤",
                "subscribe",
                "follow",
                "更多",
                "more",
            ]
            if any(kw in paragraph_lower for kw in cta_keywords):
                flags.append("short_cta")
                rules_hit.append("short_cta:<50chars")

        is_boilerplate = len(flags) > 0
        return is_boilerplate, flags, rules_hit

    def build_frequency_table(
        self, show_id: str, episodes: list[dict]
    ) -> None:
        """
        Build paragraph frequency table for a single show.

        Used to detect cross-episode repeated boilerplate (e.g., fixed intro/outro)
        """
        paragraph_counter: Counter = Counter()

        for episode in episodes:
            desc = episode.get("description") or episode.get("content_encoded") or ""
            desc = self.clean_html(desc)
            desc = self.normalize_text(desc)

            paragraphs = self.split_paragraphs(desc)
            for para in paragraphs:
                normalized = self._normalize_for_comparison(para)
                if len(normalized) > 20:  # Ignore very short paragraphs
                    paragraph_counter[normalized] += 1

        self.show_paragraph_freq[show_id] = {
            "total_episodes": len(episodes),
            "counts": paragraph_counter,
        }

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_frequent_paragraph(
        self, paragraph: str, show_id: str, threshold: float = 0.25
    ) -> bool:
        """
        Check if paragraph appears frequently in the show (> threshold of episodes have it)
        """
        show_data = self.show_paragraph_freq.get(show_id)
        if not show_data:
            return False

        total_episodes = show_data["total_episodes"]
        if total_episodes < 4:  # Need at least 4 episodes to detect frequency
            return False

        normalized = self._normalize_for_comparison(paragraph)
        count = show_data["counts"].get(normalized, 0)

        return count > total_episodes * threshold

    def clean_episode(
        self,
        episode_id: str,
        show_id: str,
        title: str,
        description: str,
        content_encoded: Optional[str] = None,
    ) -> CleanedEpisode:
        """
        Clean a single episode and return Layer 2 structure

        Args:
            episode_id: Episode ID
            show_id: Show ID
            title: Episode title
            description: <description> tag content
            content_encoded: <content:encoded> tag content (optional, richer HTML)

        Returns:
            CleanedEpisode with full debug info
        """
        # Use content:encoded if available (richer content), else description
        raw_text = content_encoded or description or ""

        # Step 1: HTML → plain text
        text = self.clean_html(raw_text)
        text = self.normalize_text(text)

        # Detect language
        language = self.detect_language(text)

        # Split into paragraphs
        paragraphs = self.split_paragraphs(text)

        # Process each paragraph
        paragraph_infos: list[ParagraphInfo] = []
        stats = CleaningStats(
            raw_char_count=len(raw_text),
            total_paragraphs=len(paragraphs),
        )
        removal_breakdown: dict[str, int] = {}

        for i, para in enumerate(paragraphs):
            is_boilerplate, flags, rules_hit = self.is_boilerplate_paragraph(
                para, language
            )

            # Check frequency-based removal
            if not is_boilerplate and self.is_frequent_paragraph(para, show_id):
                is_boilerplate = True
                flags.append("frequent")
                rules_hit.append("frequent:>25%_episodes")

            kept = not is_boilerplate

            para_info = ParagraphInfo(
                index=i,
                text=para,
                char_count=len(para),
                flags=flags if flags else ["content"],
                kept=kept,
                rules_hit=rules_hit,
            )
            paragraph_infos.append(para_info)

            # Update stats
            if kept:
                stats.kept_paragraphs += 1
                stats.kept_char_count += len(para)
            else:
                stats.removed_paragraphs += 1
                for flag in flags:
                    removal_breakdown[flag] = removal_breakdown.get(flag, 0) + 1

        stats.removal_breakdown = removal_breakdown

        # Build normalized description from kept paragraphs
        kept_paragraphs = [p.text for p in paragraph_infos if p.kept]
        normalized_description = "\n\n".join(kept_paragraphs)

        # Clean title (usually minimal cleaning needed)
        normalized_title = self.normalize_text(self.clean_html(title))

        return CleanedEpisode(
            episode_id=episode_id,
            show_id=show_id,
            normalized_title=normalized_title,
            normalized_description=normalized_description,
            paragraphs=paragraph_infos,
            stats=stats,
        )

    def to_layer2_dict(self, cleaned: CleanedEpisode) -> dict:
        """
        Convert CleanedEpisode to Layer 2 JSON structure
        """
        return {
            "episode_id": cleaned.episode_id,
            "show_id": cleaned.show_id,
            "cleaned": {
                "normalized": {
                    "title": cleaned.normalized_title,
                    "description": cleaned.normalized_description,
                },
                "paragraphs": [
                    {
                        "index": p.index,
                        "text": p.text,
                        "char_count": p.char_count,
                        "flags": p.flags,
                        "kept": p.kept,
                        "rules_hit": p.rules_hit,
                    }
                    for p in cleaned.paragraphs
                ],
                "stats": {
                    "raw_char_count": cleaned.stats.raw_char_count,
                    "kept_char_count": cleaned.stats.kept_char_count,
                    "total_paragraphs": cleaned.stats.total_paragraphs,
                    "kept_paragraphs": cleaned.stats.kept_paragraphs,
                    "removed_paragraphs": cleaned.stats.removed_paragraphs,
                    "removal_breakdown": cleaned.stats.removal_breakdown,
                },
            },
            "cleaning_meta": {
                "pipeline_version": cleaned.pipeline_version,
                "rules_version": cleaned.rules_version,
            },
        }
