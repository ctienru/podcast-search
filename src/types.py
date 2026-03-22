from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict


# ── Primitive type aliases ───────────────────────────────────────────────────

Language = Literal["zh-tw", "zh-cn", "en"]

LangParam = Literal["zh-tw", "zh-cn", "en", "zh-both"]
"""UI language selection. zh-both triggers cross-index search (zh-tw + zh-cn)."""

IndexAlias = Literal["episodes-zh-tw", "episodes-zh-cn", "episodes-en"]


# ── Domain dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Show:
    """A podcast show as stored in the crawler's SQLite database."""

    show_id: str
    title: str
    author: str
    language_detected: Language
    language_confidence: float
    language_uncertain: bool
    target_index: str        # raw value from SQLite, e.g. "podcast-episodes-zh-tw"
    rss_feed_url: str
    updated_at: str


@dataclass(frozen=True)
class Episode:
    """A podcast episode ready for ingestion into Elasticsearch."""

    episode_id: str
    show_id: str
    title: str
    description: str
    target_index: str        # inherited from Show.target_index
    language: Language


# ── Structured log schemas ───────────────────────────────────────────────────

class IngestStats(TypedDict):
    """Structured log emitted at the end of each ingest run."""

    event: str
    timestamp: str
    index_counts: dict[IndexAlias, int]
    language_distribution: dict[str, int]
    ingest_success: int
    ingest_failed: int
    uncertain_rate: float


class IngestCursor(TypedDict):
    """Per-index cursor tracking the last successful incremental ingest.

    Persisted to data/ingest_cursor.json between runs.
    On the next run, only shows with updated_at > last_ingest_at are processed.
    """

    last_ingest_at: str   # ISO 8601 UTC — passed to get_shows_updated_since()
    last_run_at: str      # when this cursor was written (for debugging)
