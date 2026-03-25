"""ClickTracker: append-only JSONL log for episode click events.

Pattern: Middleware — receives POST /api/log/click payloads forwarded
from the backend and persists them for offline metric computation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.types import Language, LangParam

logger = logging.getLogger(__name__)


@dataclass
class ClickLogEntry:
    """Represents one episode click event.

    request_id links this click back to its QueryLogEntry,
    enabling First Click Rank and Same-Language Click Rate calculation.

    Attributes:
        request_id:         UUID from the corresponding QueryLogEntry.
        timestamp:          ISO 8601 UTC timestamp of the click.
        query:              The query string that produced this result page.
        selected_lang:      Language selected in the UI at the time of the search.
        clicked_episode_id: episode_id of the clicked result.
        clicked_rank:       1-based position of the clicked result in the list.
        clicked_language:   Language field of the clicked episode document.
        time_to_click_sec:  Seconds from search response to click (optional).
    """

    request_id:         str
    timestamp:          str
    query:              str
    selected_lang:      LangParam
    clicked_episode_id: str
    clicked_rank:       int
    clicked_language:   Language
    time_to_click_sec:  Optional[float] = None


class ClickTracker:
    """Appends ClickLogEntry records to a JSONL file.

    Args:
        log_path: Path to the JSONL log file. Parent directory must exist.
            File is created on first write.
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def log(self, entry: ClickLogEntry) -> None:
        """Append one click entry as a JSON line.

        Catches OSError so that a logging failure never interrupts the response.

        Args:
            entry: The ClickLogEntry to persist.
        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError as exc:
            logger.error("click_log_write_failed", extra={"error": str(exc)})
