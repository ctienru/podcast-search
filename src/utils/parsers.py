"""
Common parsing utilities for podcast data.
"""

from email.utils import parsedate_to_datetime
from typing import Optional


def normalize_language(lang: Optional[str]) -> Optional[str]:
    """
    Normalize language code.

    Rules:
        - English variants (en, en-US, en-gb) -> "en"
        - Traditional Chinese (zh-Hant, zh-TW, zh-tw, zh) -> "zh-hant"
        - Simplified Chinese (zh-Hans) -> "zh-hans"

    Args:
        lang: Language code from RSS (e.g., "en-US", "zh-Hant", "zh-tw")

    Returns:
        Normalized language code, or None if input is None/empty
    """
    if not lang:
        return None

    lower = lang.lower()

    # English variants -> "en"
    if lower.startswith("en"):
        return "en"

    # Simplified Chinese
    if lower in ("zh-hans", "zh-cn"):
        return "zh-hans"

    # Traditional Chinese (zh-Hant, zh-TW, zh-tw, or plain zh)
    if lower.startswith("zh"):
        return "zh-hant"

    # Other languages: return primary tag
    primary = lower.split("-")[0]
    return primary if len(primary) == 2 else None


def parse_pub_date(pub_date_str: Optional[str]) -> Optional[str]:
    """
    Parse RFC 2822 date to ISO 8601 format for ES.

    Args:
        pub_date_str: Date string in RFC 2822 format (e.g., "Sat, 13 Aug 2022 09:00:56 +0000")

    Returns:
        ISO 8601 formatted date string, or None if parsing fails
    """
    if not pub_date_str:
        return None
    try:
        dt = parsedate_to_datetime(pub_date_str)
        return dt.isoformat()
    except Exception:
        return None


def parse_duration(duration: Optional[str]) -> Optional[int]:
    """
    Parse duration string to seconds.

    Supports:
    - Integer strings (e.g., "3600")
    - HH:MM:SS format (e.g., "1:30:00")
    - MM:SS format (e.g., "45:30")

    Args:
        duration: Duration string in various formats

    Returns:
        Duration in seconds, or None if parsing fails
    """
    if not duration:
        return None

    # Already an integer
    if isinstance(duration, int):
        return duration

    # Try to parse as integer string
    try:
        return int(duration)
    except ValueError:
        pass

    # Parse HH:MM:SS or MM:SS format
    parts = duration.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass

    return None
