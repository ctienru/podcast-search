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
    """A podcast show as stored in the crawler's SQLite database.

    Required fields are the routing-critical columns guaranteed to be non-null
    (enforced by SQLiteStorage._WHERE_BASE).

    Optional fields are populated from SQLite columns that may be NULL in older
    rows or not yet crawled. Callers must handle None / empty defaults.
    """

    # ── Required (routing-critical, always non-null) ─────────────────────────
    show_id:             str
    title:               str
    author:              str
    language_detected:   Language
    language_confidence: float
    language_uncertain:  bool
    target_index:        str      # raw value from SQLite, e.g. "podcast-episodes-zh-tw"
    rss_feed_url:        str
    updated_at:          str

    # ── Optional (media / display metadata) ──────────────────────────────────
    provider:        str                   = ""    # e.g. "apple"
    external_id:     str                   = ""    # provider-specific ID, e.g. "77001367"
    description:     Optional[str]         = None
    image_url:       Optional[str]         = None  # parsed from SQLite image JSON column
    external_urls:   dict[str, str]        = field(default_factory=dict)   # e.g. {"apple_podcasts": "https://..."}
    episode_count:   Optional[int]         = None
    last_episode_at: Optional[str]         = None
    categories:      tuple[str, ...]       = field(default_factory=tuple)  # tuple for hashability (frozen dataclass)


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
