"""
Tests for generate_weekly_report.py

Pure logic tests — no file I/O required.
Tests cover: _online_threshold_status(), format_offline_quality(), format_regression_list().
"""

import pytest

from scripts.generate_weekly_report import (
    _online_threshold_status,
    format_offline_quality,
    format_regression_list,
    ONLINE_THRESHOLDS,
    NDCG_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# _online_threshold_status
# ---------------------------------------------------------------------------

class TestOnlineThresholdStatus:
    def test_search_success_rate_pass(self):
        # SSR >= 0.60
        assert _online_threshold_status("search_success_rate", 0.70) == "PASS"

    def test_search_success_rate_fail(self):
        assert _online_threshold_status("search_success_rate", 0.50) == "FAIL"

    def test_search_success_rate_exactly_at_threshold(self):
        assert _online_threshold_status("search_success_rate", 0.60) == "PASS"

    def test_same_language_click_rate_pass(self):
        # >= 0.80
        assert _online_threshold_status("same_language_click_rate", 0.85) == "PASS"

    def test_same_language_click_rate_fail(self):
        assert _online_threshold_status("same_language_click_rate", 0.75) == "FAIL"

    def test_reformulation_rate_pass(self):
        # <= 0.20 (lower is better)
        assert _online_threshold_status("reformulation_rate", 0.15) == "PASS"

    def test_reformulation_rate_fail(self):
        assert _online_threshold_status("reformulation_rate", 0.25) == "FAIL"

    def test_reformulation_rate_exactly_at_threshold(self):
        assert _online_threshold_status("reformulation_rate", 0.20) == "PASS"

    def test_unknown_metric_returns_not_pass_or_fail(self):
        # _online_threshold_status has no explicit fallback for unknown metrics;
        # verify it does not incorrectly return PASS or FAIL
        result = _online_threshold_status("nonexistent_metric", 0.5)
        assert result not in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# NDCG_THRESHOLDS sanity check
# ---------------------------------------------------------------------------

class TestNdcgThresholds:
    def test_zh_tw_threshold(self):
        assert NDCG_THRESHOLDS["zh-tw"] == 0.897

    def test_en_threshold(self):
        assert NDCG_THRESHOLDS["en"] == 0.853

    def test_zh_cn_has_threshold(self):
        # zh-cn has a provisional threshold
        assert "zh-cn" in NDCG_THRESHOLDS


# ---------------------------------------------------------------------------
# format_offline_quality
# ---------------------------------------------------------------------------

def _make_ndcg_report(langs: dict) -> dict:
    """Build a minimal NDCG report dict for testing format_offline_quality()."""
    by_language = {}
    for lang, ndcg_val in langs.items():
        by_language[lang] = {
            "bm25": {"ndcg@10": ndcg_val, "mrr": 0.9},
            "queries": 12,
        }
    overall_ndcg = sum(langs.values()) / len(langs) if langs else 0.0
    return {
        "meta": {
            "timestamp": "2026-03-25T00:00:00+00:00",
            "total_queries": len(langs) * 12,
            "methods": ["bm25"],
        },
        "overall": {
            "bm25": {"ndcg@10": overall_ndcg, "mrr": 0.9},
            "queries": len(langs) * 12,
        },
        "by_language": by_language,
        "by_category": {},
    }


class TestFormatOfflineQuality:
    def test_output_contains_ndcg_header(self):
        report = _make_ndcg_report({"zh-tw": 0.91, "en": 0.88})
        output = format_offline_quality(report)
        assert "NDCG@10" in output

    def test_output_contains_language_names(self):
        report = _make_ndcg_report({"zh-tw": 0.91, "en": 0.88})
        output = format_offline_quality(report)
        assert "zh-tw" in output
        assert "en" in output

    def test_pass_annotation_when_above_threshold(self):
        # zh-tw 0.91 >= 0.897 → PASS marker expected
        report = _make_ndcg_report({"zh-tw": 0.91})
        output = format_offline_quality(report)
        assert "PASS" in output

    def test_fail_annotation_when_below_threshold(self):
        # en 0.80 < 0.853 → FAIL marker expected
        report = _make_ndcg_report({"en": 0.80})
        output = format_offline_quality(report)
        assert "FAIL" in output

    def test_empty_by_language_does_not_crash(self):
        report = _make_ndcg_report({})
        output = format_offline_quality(report)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# format_regression_list (NDCG delta)
# ---------------------------------------------------------------------------

def _make_pq_report(per_query: dict, timestamp: str = "2026-03-25T00:00:00+00:00") -> dict:
    """Build a minimal report with per_query section."""
    return {
        "meta": {"timestamp": timestamp, "total_queries": len(per_query), "methods": ["bm25"]},
        "per_query": per_query,
    }


class TestFormatRegressionList:
    def test_improvement_shows_positive_delta(self):
        curr = _make_pq_report({"tech podcast": {"bm25": {"ndcg@10": 0.91, "mrr": 0.9}}})
        prev = _make_pq_report(
            {"tech podcast": {"bm25": {"ndcg@10": 0.89, "mrr": 0.88}}},
            timestamp="2026-03-18T00:00:00+00:00",
        )
        output = format_regression_list(curr, prev)
        assert "+0.02" in output or "+0.0200" in output

    def test_regression_shows_negative_delta(self):
        curr = _make_pq_report({"tech podcast": {"bm25": {"ndcg@10": 0.85, "mrr": 0.88}}})
        prev = _make_pq_report(
            {"tech podcast": {"bm25": {"ndcg@10": 0.90, "mrr": 0.90}}},
            timestamp="2026-03-18T00:00:00+00:00",
        )
        output = format_regression_list(curr, prev)
        assert "-0.05" in output or "-0.0500" in output

    def test_no_overlap_shows_na_message(self):
        curr = _make_pq_report({"query A": {"bm25": {"ndcg@10": 0.91, "mrr": 0.9}}})
        prev = _make_pq_report(
            {"query B": {"bm25": {"ndcg@10": 0.89, "mrr": 0.88}}},
            timestamp="2026-03-18T00:00:00+00:00",
        )
        output = format_regression_list(curr, prev)
        assert "No overlapping" in output

    def test_empty_per_query_shows_na_message(self):
        curr = _make_pq_report({})
        prev = _make_pq_report({}, timestamp="2026-03-18T00:00:00+00:00")
        output = format_regression_list(curr, prev)
        assert "No per-query" in output

    def test_multiple_queries_sorted_by_delta(self):
        curr = _make_pq_report({
            "q1": {"bm25": {"ndcg@10": 0.70, "mrr": 0.7}},  # big drop
            "q2": {"bm25": {"ndcg@10": 0.95, "mrr": 0.9}},  # big gain
            "q3": {"bm25": {"ndcg@10": 0.85, "mrr": 0.8}},  # neutral
        })
        prev = _make_pq_report(
            {
                "q1": {"bm25": {"ndcg@10": 0.90, "mrr": 0.9}},
                "q2": {"bm25": {"ndcg@10": 0.80, "mrr": 0.8}},
                "q3": {"bm25": {"ndcg@10": 0.85, "mrr": 0.8}},
            },
            timestamp="2026-03-18T00:00:00+00:00",
        )
        output = format_regression_list(curr, prev)
        # q1 should appear in regressions, q2 in improvements
        assert "q1" in output
        assert "q2" in output
