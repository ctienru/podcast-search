"""
Weekly Evaluation Report Generator

Produces a Markdown report combining three data sources:

  1. Offline Quality — latest NDCG/MRR report (method comparison: BM25 vs embedding vs hybrid)
  2. Online Behavior — query + click logs (SSR, same-lang click rate,
                       reformulation rate, mean first-click rank)
  3. Regression   — per-query NDCG delta vs previous report (top movers)

Usage:
    # Use today's suite report
    python scripts/generate_weekly_report.py

    # Specify report date and compare against a previous run
    python scripts/generate_weekly_report.py \\
        --date 2026-03-25 \\
        --prev-date 2026-03-18 \\
        --query-log logs/query_log.jsonl \\
        --click-log logs/click_log.jsonl \\
        --output reports/weekly_2026-03-25.md

Online behavior thresholds (for PASS/FAIL annotation):
    Search Success Rate      >= 60%
    Same-Language Click Rate >= 80%
    Reformulation Rate       <= 20%
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Online behavior thresholds (same as Section 3.3 of evaluation doc)
ONLINE_THRESHOLDS = {
    "search_success_rate":      {"threshold": 0.60, "direction": ">="},
    "same_language_click_rate": {"threshold": 0.80, "direction": ">="},
    "reformulation_rate":       {"threshold": 0.20, "direction": "<="},
}

NDCG_THRESHOLDS = {
    "zh-tw": 0.897,
    "zh-cn": 0.897,
    "en":    0.853,
}

METHODS = ("bm25", "embedding", "hybrid")

REPORTS_DIR = Path("data/evaluation/reports")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def find_latest_report_date() -> Optional[str]:
    """Return the most recent date directory under REPORTS_DIR."""
    if not REPORTS_DIR.exists():
        return None
    dirs = sorted(
        [d.name for d in REPORTS_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )
    return dirs[0] if dirs else None


def load_ndcg_report(report_date: str) -> Optional[Dict[str, Any]]:
    """Load ndcg_mrr_report.json from a dated reports directory."""
    path = REPORTS_DIR / report_date / "ndcg_mrr_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    if not path or not path.exists():
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


# ---------------------------------------------------------------------------
# Offline A/B section
# ---------------------------------------------------------------------------

def _best_method(lang_stats: Dict) -> Tuple[str, float]:
    best_m, best_v = None, -1.0
    for m in METHODS:
        if m in lang_stats:
            v = lang_stats[m]["ndcg@10"]
            if v > best_v:
                best_m, best_v = m, v
    return best_m or "—", best_v


def format_offline_quality(report: Dict) -> str:
    lines = []
    by_lang = report.get("by_language", {})
    overall = report.get("overall", {})

    lines.append("### Offline Quality — NDCG@10")
    lines.append("")
    lines.append(f"Report date: `{report['meta']['timestamp'][:10]}`  ")
    lines.append(f"Total queries: {report['meta']['total_queries']}  ")
    lines.append(f"Methods: {', '.join(report['meta']['methods'])}")
    lines.append("")

    # Overall table
    lines.append("#### Overall")
    lines.append("")
    lines.append("| Method | NDCG@10 | MRR |")
    lines.append("|--------|---------|-----|")
    for m in METHODS:
        if m in overall:
            ci = overall[m].get("ci", [])
            ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            lines.append(f"| {m} | {overall[m]['ndcg@10']:.4f}{ci_str} | {overall[m]['mrr']:.4f} |")
    lines.append("")

    # Per-language table
    lines.append("#### By Language")
    lines.append("")
    lines.append("| Language | n | Best Method | NDCG@10 | Threshold | Gate |")
    lines.append("|----------|---|-------------|---------|-----------|------|")
    for lang, stats in sorted(by_lang.items()):
        best_m, best_v = _best_method(stats)
        threshold = NDCG_THRESHOLDS.get(lang)
        if threshold is None:
            gate_str = "N/A"
        elif best_v >= threshold:
            gate_str = "PASS"
        else:
            gate_str = "FAIL"
        thr_str = str(threshold) if threshold else "—"
        lines.append(f"| {lang} | {stats['queries']} | {best_m} | {best_v:.4f} | {thr_str} | {gate_str} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression query top list
# ---------------------------------------------------------------------------

def format_regression_list(
    curr_report: Dict,
    prev_report: Dict,
    top_n: int = 10,
) -> str:
    lines = []
    lines.append("### Regression Analysis — Per-Query NDCG Delta")
    lines.append("")

    curr_pq = curr_report.get("per_query", {})
    prev_pq = prev_report.get("per_query", {})

    if not curr_pq or not prev_pq:
        lines.append("_No per-query data available in one or both reports._")
        return "\n".join(lines)

    deltas = []
    for query, curr_methods in curr_pq.items():
        if query not in prev_pq:
            continue
        prev_methods = prev_pq[query]

        # Best NDCG in current vs previous
        curr_best = max(
            (curr_methods[m]["ndcg@10"] for m in METHODS if m in curr_methods),
            default=0.0,
        )
        prev_best = max(
            (prev_methods[m]["ndcg@10"] for m in METHODS if m in prev_methods),
            default=0.0,
        )
        delta = curr_best - prev_best
        deltas.append((query, delta, prev_best, curr_best))

    if not deltas:
        lines.append("_No overlapping queries between current and previous report._")
        return "\n".join(lines)

    prev_date = prev_report["meta"]["timestamp"][:10]
    curr_date = curr_report["meta"]["timestamp"][:10]
    lines.append(f"Comparing `{curr_date}` vs `{prev_date}`. Best-method NDCG@10 delta.")
    lines.append("")

    # Biggest drops
    drops = sorted(deltas, key=lambda x: x[1])[:top_n]
    lines.append(f"#### Top {len(drops)} Regressions (biggest drops)")
    lines.append("")
    lines.append("| Query | Prev | Curr | Delta |")
    lines.append("|-------|------|------|-------|")
    for q, delta, prev_v, curr_v in drops:
        lines.append(f"| {q} | {prev_v:.4f} | {curr_v:.4f} | {delta:+.4f} |")
    lines.append("")

    # Biggest gains
    gains = sorted(deltas, key=lambda x: x[1], reverse=True)[:top_n]
    lines.append(f"#### Top {len(gains)} Improvements")
    lines.append("")
    lines.append("| Query | Prev | Curr | Delta |")
    lines.append("|-------|------|------|-------|")
    for q, delta, prev_v, curr_v in gains:
        lines.append(f"| {q} | {prev_v:.4f} | {curr_v:.4f} | {delta:+.4f} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Online C section
# ---------------------------------------------------------------------------

@dataclass
class OnlineSection:
    query_count: int
    click_count: int
    search_success_rate: Optional[float]
    same_language_click_rate: Optional[float]
    reformulation_rate: Optional[float]
    mean_first_click_rank: Optional[float]


def compute_online_section(
    query_log: Path,
    click_log: Path,
    session_timeout: int = 30,
) -> OnlineSection:
    """Compute online metrics inline (mirrors compute_online_metrics.py logic)."""
    from datetime import timezone as tz

    def parse_ts(ts: str) -> float:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.utc)
        return dt.timestamp()

    queries = load_jsonl(query_log)
    clicks  = load_jsonl(click_log)

    if not queries:
        return OnlineSection(
            query_count=0, click_count=len(clicks),
            search_success_rate=None, same_language_click_rate=None,
            reformulation_rate=None, mean_first_click_rank=None,
        )

    clicks_by_request: Dict[str, List[Dict]] = {}
    for c in clicks:
        rid = c.get("request_id", "")
        clicks_by_request.setdefault(rid, []).append(c)

    query_lang_by_request = {
        q["request_id"]: q.get("query_lang", "")
        for q in queries if "request_id" in q
    }

    # Search success rate
    successful = sum(
        1 for q in queries
        if any(
            0 <= parse_ts(c.get("timestamp", "1970-01-01T00:00:00+00:00"))
               - parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00"))
               <= session_timeout
            for c in clicks_by_request.get(q.get("request_id", ""), [])
        )
    )
    ssr = successful / len(queries)

    # Same-language click rate
    same_lang = sum(
        1 for c in clicks
        if c.get("clicked_language") == query_lang_by_request.get(c.get("request_id", ""), "")
    )
    slcr = same_lang / len(clicks) if clicks else 0.0

    # Mean first click rank
    first_ranks = []
    for q in queries:
        rid = q.get("request_id", "")
        q_ts = parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00"))
        valid = [
            c for c in clicks_by_request.get(rid, [])
            if 0 <= parse_ts(c.get("timestamp", "1970-01-01T00:00:00+00:00")) - q_ts <= session_timeout
        ]
        if valid:
            first = min(valid, key=lambda c: parse_ts(c.get("timestamp", "1970-01-01T00:00:00+00:00")))
            first_ranks.append(first.get("clicked_rank", 0))
    mean_rank = sum(first_ranks) / len(first_ranks) if first_ranks else None

    # Reformulation rate
    sorted_q = sorted(queries, key=lambda q: parse_ts(q.get("timestamp", "1970-01-01T00:00:00+00:00")))
    reformulations = sum(
        1 for i in range(1, len(sorted_q))
        if parse_ts(sorted_q[i].get("timestamp", "1970-01-01T00:00:00+00:00"))
           - parse_ts(sorted_q[i-1].get("timestamp", "1970-01-01T00:00:00+00:00"))
           <= session_timeout
        and sorted_q[i].get("query") != sorted_q[i-1].get("query")
    )
    rr = reformulations / len(queries)

    return OnlineSection(
        query_count=len(queries),
        click_count=len(clicks),
        search_success_rate=round(ssr, 4),
        same_language_click_rate=round(slcr, 4),
        reformulation_rate=round(rr, 4),
        mean_first_click_rank=round(mean_rank, 2) if mean_rank is not None else None,
    )


def _online_threshold_status(metric: str, value: float) -> str:
    cfg = ONLINE_THRESHOLDS.get(metric)
    if cfg is None:
        return ""
    if cfg["direction"] == ">=":
        return "PASS" if value >= cfg["threshold"] else "FAIL"
    return "PASS" if value <= cfg["threshold"] else "FAIL"


def format_online_behavior(section: OnlineSection) -> str:
    lines = []
    lines.append("### Online Behavioral Metrics")
    lines.append("")

    if section.query_count == 0:
        lines.append("_No query log data available. Online behavior is N/A (pre-launch or logs not provided)._")
        lines.append("")
        lines.append("Run with `--query-log logs/query_log.jsonl --click-log logs/click_log.jsonl`")
        lines.append("once sufficient traffic has been collected (≥ 500 sessions recommended).")
        return "\n".join(lines)

    lines.append(f"Sessions analyzed: {section.query_count} queries, {section.click_count} clicks")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Gate |")
    lines.append("|--------|-------|-----------|------|")

    def row(label: str, metric_key: str, value: Optional[float], fmt: str = ".1%") -> str:
        if value is None:
            return f"| {label} | N/A | — | N/A |"
        thr = ONLINE_THRESHOLDS.get(metric_key, {}).get("threshold", "—")
        direction = ONLINE_THRESHOLDS.get(metric_key, {}).get("direction", "")
        gate = _online_threshold_status(metric_key, value)
        thr_str = f"{direction} {thr:.0%}" if isinstance(thr, float) else str(thr)
        return f"| {label} | {value:{fmt}} | {thr_str} | {gate} |"

    lines.append(row("Search Success Rate",      "search_success_rate",      section.search_success_rate))
    lines.append(row("Same-Language Click Rate",  "same_language_click_rate", section.same_language_click_rate))
    lines.append(row("Reformulation Rate",        "reformulation_rate",       section.reformulation_rate))
    if section.mean_first_click_rank is not None:
        lines.append(f"| Mean First Click Rank | {section.mean_first_click_rank:.2f} | — | (reference) |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report(
    curr_report: Dict,
    prev_report: Optional[Dict],
    online: OnlineSection,
    report_date: str,
) -> str:
    sections = []

    sections.append(f"# Weekly Evaluation Report — {report_date}")
    sections.append("")
    sections.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    sections.append("Evaluation framework: Correctness (index health + language routing) · Offline Quality (NDCG + latency) · Online Behavior")
    sections.append("")
    sections.append("---")
    sections.append("")

    sections.append(format_offline_quality(curr_report))

    if prev_report:
        sections.append(format_regression_list(curr_report, prev_report))
    else:
        sections.append("### Regression Analysis")
        sections.append("")
        sections.append("_No previous report provided. Use `--prev-date YYYY-MM-DD` to enable delta comparison._")
        sections.append("")

    sections.append(format_online_behavior(online))

    sections.append("---")
    sections.append("")
    sections.append("## How to update this report")
    sections.append("")
    sections.append("```bash")
    sections.append("# 1. Run evaluation suite (requires local ES)")
    sections.append("python scripts/run_evaluation_suite.py")
    sections.append("")
    sections.append("# 2. Generate weekly report with delta comparison")
    sections.append(f"python scripts/generate_weekly_report.py \\")
    sections.append(f"    --date {report_date} \\")
    sections.append(f"    --prev-date YYYY-MM-DD \\")
    sections.append(f"    --query-log logs/query_log.jsonl \\")
    sections.append(f"    --click-log logs/click_log.jsonl \\")
    sections.append(f"    --output reports/weekly_{report_date}.md")
    sections.append("```")
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly evaluation report")
    parser.add_argument(
        "--date",
        type=str,
        help="Date of the NDCG report to use (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--prev-date",
        type=str,
        help="Date of the previous NDCG report for regression comparison (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--query-log",
        type=Path,
        default=Path("logs/query_log.jsonl"),
        help="Path to query log JSONL file (for online behavior metrics)",
    )
    parser.add_argument(
        "--click-log",
        type=Path,
        default=Path("logs/click_log.jsonl"),
        help="Path to click log JSONL file (for online behavior metrics)",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=30,
        help="Session timeout in seconds for online metrics (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the Markdown report. Default: print to stdout.",
    )
    args = parser.parse_args()

    # Resolve date
    report_date = args.date or find_latest_report_date()
    if not report_date:
        print(
            "ERROR: No dated report directories found under data/evaluation/reports/.\n"
            "Run 'python scripts/run_evaluation_suite.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load current NDCG report
    curr_report = load_ndcg_report(report_date)
    if not curr_report:
        print(
            f"ERROR: ndcg_mrr_report.json not found in data/evaluation/reports/{report_date}/.\n"
            f"Run 'python scripts/run_evaluation_suite.py' to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Current report:  data/evaluation/reports/{report_date}/ndcg_mrr_report.json")

    # Load previous NDCG report (optional)
    prev_report = None
    if args.prev_date:
        prev_report = load_ndcg_report(args.prev_date)
        if prev_report:
            print(f"Previous report: data/evaluation/reports/{args.prev_date}/ndcg_mrr_report.json")
        else:
            print(f"WARNING: previous report not found for {args.prev_date}, skipping delta comparison.")

    # Compute online metrics
    online = compute_online_section(args.query_log, args.click_log, args.session_timeout)
    if online.query_count > 0:
        print(f"Online logs: {online.query_count} queries, {online.click_count} clicks")
    else:
        print("Online logs: not found or empty — Online Behavior will show N/A")

    # Generate report
    report_md = generate_report(curr_report, prev_report, online, report_date)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_md, encoding="utf-8")
        print(f"Report saved to: {args.output}")
    else:
        print()
        print(report_md)


if __name__ == "__main__":
    main()
