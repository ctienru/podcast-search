"""Compute online behavioral metrics from query and click logs.

Metrics
-------
- Search Success Rate    : % of queries with a click within SESSION_TIMEOUT seconds
- Same-Language Click Rate: % of clicks where clicked_language == query_lang
- First Click Rank (mean): mean rank of the first click per query
- Reformulation Rate     : % of queries re-issued within SESSION_TIMEOUT in the same session

Usage
-----
    python scripts/compute_online_metrics.py \\
        --query-log logs/query_log.jsonl \\
        --click-log logs/click_log.jsonl \\
        --session-timeout 30
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


SESSION_TIMEOUT_SECONDS: int = 30


@dataclass
class OnlineMetrics:
    """Container for all computed online behavioral metrics."""

    search_success_rate:      float  # % of queries with ≥1 click within session timeout
    same_language_click_rate: float  # % of clicks where clicked_language == query_lang
    mean_first_click_rank:    float  # mean rank of the first click per query
    reformulation_rate:       float  # % of queries followed by a re-query in the same session
    query_count:              int    # total queries in the log
    click_count:              int    # total clicks in the log


def _load_jsonl(path: Path) -> List[Dict]:
    """Load all lines from a JSONL file. Skip unparseable lines."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _parse_ts(ts: str) -> float:
    """Parse ISO 8601 UTC timestamp to Unix seconds."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def compute(
    query_log_path: Path,
    click_log_path: Path,
    session_timeout: int = SESSION_TIMEOUT_SECONDS,
) -> OnlineMetrics:
    """Load logs and compute all online metrics.

    Session definition: consecutive queries where the gap between
    successive queries is ≤ session_timeout seconds.

    Args:
        query_log_path: Path to query_log.jsonl.
        click_log_path: Path to click_log.jsonl.
        session_timeout: Seconds within which a re-query counts as a reformulation.

    Returns:
        OnlineMetrics dataclass with all computed values.
    """
    queries = _load_jsonl(query_log_path)
    clicks  = _load_jsonl(click_log_path)

    if not queries:
        return OnlineMetrics(
            search_success_rate=0.0,
            same_language_click_rate=0.0,
            mean_first_click_rank=0.0,
            reformulation_rate=0.0,
            query_count=0,
            click_count=len(clicks),
        )

    # Index clicks by request_id for O(1) lookup
    clicks_by_request: Dict[str, List[Dict]] = {}
    for click in clicks:
        rid = click.get("request_id", "")
        clicks_by_request.setdefault(rid, []).append(click)

    # Index query_lang by request_id for same-language calculation
    query_lang_by_request: Dict[str, str] = {
        q["request_id"]: q.get("query_lang", "") for q in queries if "request_id" in q
    }

    # ── Search Success Rate ───────────────────────────────────────────────────
    # A query is "successful" if it has at least one click whose timestamp
    # is within session_timeout seconds after the query timestamp.
    successful = 0
    for q in queries:
        rid = q.get("request_id", "")
        q_ts = _parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00"))
        for click in clicks_by_request.get(rid, []):
            c_ts = _parse_ts(click.get("timestamp", "1970-01-01T00:00:00+00:00"))
            if 0 <= c_ts - q_ts <= session_timeout:
                successful += 1
                break

    search_success_rate = successful / len(queries)

    # ── Same-Language Click Rate ─────────────────────────────────────────────
    same_lang = 0
    for click in clicks:
        rid = click.get("request_id", "")
        query_lang = query_lang_by_request.get(rid, "")
        if click.get("clicked_language") == query_lang:
            same_lang += 1

    same_language_click_rate = same_lang / len(clicks) if clicks else 0.0

    # ── Mean First Click Rank ────────────────────────────────────────────────
    first_ranks: List[int] = []
    for q in queries:
        rid = q.get("request_id", "")
        q_ts = _parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00"))
        valid = [
            c for c in clicks_by_request.get(rid, [])
            if 0 <= _parse_ts(c.get("timestamp", "1970-01-01T00:00:00+00:00")) - q_ts <= session_timeout
        ]
        if valid:
            # First click = earliest by timestamp
            first = min(valid, key=lambda c: _parse_ts(c.get("timestamp", "1970-01-01T00:00:00+00:00")))
            first_ranks.append(first.get("clicked_rank", 0))

    mean_first_click_rank = sum(first_ranks) / len(first_ranks) if first_ranks else 0.0

    # ── Reformulation Rate ───────────────────────────────────────────────────
    # Sort queries by timestamp, then find consecutive pairs within session_timeout
    # where the query text changed.
    sorted_queries = sorted(queries, key=lambda q: _parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00")))
    reformulations = 0
    for i in range(1, len(sorted_queries)):
        prev = sorted_queries[i - 1]
        curr = sorted_queries[i]
        prev_ts = _parse_ts(prev.get("timestamp", "1970-01-01T00:00:00+00:00"))
        curr_ts = _parse_ts(curr.get("timestamp", "1970-01-01T00:00:00+00:00"))
        gap = curr_ts - prev_ts
        if gap <= session_timeout and curr.get("query") != prev.get("query"):
            reformulations += 1

    reformulation_rate = reformulations / len(queries)

    return OnlineMetrics(
        search_success_rate=round(search_success_rate, 4),
        same_language_click_rate=round(same_language_click_rate, 4),
        mean_first_click_rank=round(mean_first_click_rank, 2),
        reformulation_rate=round(reformulation_rate, 4),
        query_count=len(queries),
        click_count=len(clicks),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute online behavioral metrics")
    parser.add_argument(
        "--query-log",
        type=Path,
        default=Path("logs/query_log.jsonl"),
        help="Path to query_log.jsonl (default: logs/query_log.jsonl)",
    )
    parser.add_argument(
        "--click-log",
        type=Path,
        default=Path("logs/click_log.jsonl"),
        help="Path to click_log.jsonl (default: logs/click_log.jsonl)",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=SESSION_TIMEOUT_SECONDS,
        help=f"Session timeout in seconds (default: {SESSION_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    metrics = compute(
        query_log_path=args.query_log,
        click_log_path=args.click_log,
        session_timeout=args.session_timeout,
    )

    print(f"Queries:                 {metrics.query_count}")
    print(f"Clicks:                  {metrics.click_count}")
    print(f"Search Success Rate:     {metrics.search_success_rate:.1%}")
    print(f"Same-Language Click Rate:{metrics.same_language_click_rate:.1%}")
    print(f"Mean First Click Rank:   {metrics.mean_first_click_rank:.2f}")
    print(f"Reformulation Rate:      {metrics.reformulation_rate:.1%}")


if __name__ == "__main__":
    main()
