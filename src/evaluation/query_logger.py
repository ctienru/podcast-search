"""QueryLogger: append-only JSONL log for search requests.

Pattern: Middleware — wrap the search pipeline; call log() after the
search response is built. Failures are caught and logged; search is
never blocked by a logging error.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.types import IndexAlias, Language, LangParam

logger = logging.getLogger(__name__)


@dataclass
class QueryLogEntry:
    """Represents one search request, used for offline analysis.

    request_id and timestamp are auto-generated on creation.
    The request_id is returned in the search response so the frontend
    can attach it to subsequent click events.

    Attributes:
        query:            Raw query string from the user.
        query_lang:       Language detected by langdetect (never "zh-both").
        selected_lang:    Language selected in the UI (may be "zh-both").
        mode:             Search mode, e.g. "hybrid", "bm25", "knn".
        target_index:     ES alias(es) queried. Usually one; two for zh-both.
        is_cross_lang:    True when selected_lang="zh-both" (two indices searched).
        result_count:     Number of hits returned to the user.
        result_ids:       Ordered list of episode_id values in the response.
        result_languages: Language field of each result (same order as result_ids).
        page:             Page number (1-based).
        latency_ms:       Total server-side latency in milliseconds.
        request_id:       Auto-generated UUID4; links query log to click log.
        timestamp:        Auto-generated ISO 8601 UTC timestamp.
    """

    query:            str
    query_lang:       Language
    selected_lang:    LangParam
    mode:             str
    target_index:     list[IndexAlias]
    is_cross_lang:    bool
    result_count:     int
    result_ids:       list[str]
    result_languages: list[Language]
    page:             int
    latency_ms:       int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QueryLogger:
    """Appends QueryLogEntry records to a JSONL file.

    Args:
        log_path: Path to the JSONL log file. Parent directory will be created
            if needed; the file is created on first write.
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def log(self, entry: QueryLogEntry) -> None:
        """Append one log entry as a JSON line.

        Catches OSError so that a logging failure never interrupts search.

        Args:
            entry: The QueryLogEntry to persist.
        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError as exc:
            logger.error("query_log_write_failed", extra={"error": str(exc)})
