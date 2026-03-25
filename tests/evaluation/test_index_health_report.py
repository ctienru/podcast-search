"""
Tests for index_health_report.py

Uses unittest.mock to simulate Elasticsearch responses.
No real ES connection required.
"""

import pytest
from unittest.mock import MagicMock, patch

from scripts.index_health_report import (
    check_field_coverage,
    check_zero_result_rate,
    THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_es(total: int, present: int):
    """Return a mock ES client where count() reflects total or present based on call."""
    es = MagicMock()
    # First call (total): body=None → total count
    # Second call (present): body has 'query' → present count
    es.count.side_effect = [
        {"count": total},
        {"count": present},
    ]
    return es


def _mock_es_search(results_per_query: list):
    """Return a mock ES client whose search() returns hits in order."""
    es = MagicMock()
    es.search.side_effect = [
        {"hits": {"total": {"value": n}}}
        for n in results_per_query
    ]
    return es


# ---------------------------------------------------------------------------
# check_field_coverage
# ---------------------------------------------------------------------------

class TestCheckFieldCoverage:
    def test_full_coverage(self):
        es = _mock_es(total=1000, present=1000)
        coverage = check_field_coverage(es, "episodes-zh-tw", "embedding")
        assert coverage == 1.0

    def test_partial_coverage(self):
        es = _mock_es(total=1000, present=992)
        coverage = check_field_coverage(es, "episodes-zh-tw", "embedding")
        assert abs(coverage - 0.992) < 1e-6

    def test_coverage_below_embedding_threshold(self):
        # 0.985 < 0.99 → would fail the threshold
        es = _mock_es(total=1000, present=985)
        coverage = check_field_coverage(es, "episodes-zh-tw", "embedding")
        assert coverage < THRESHOLDS["embedding_coverage"]

    def test_coverage_above_image_url_threshold(self):
        # 0.95 >= 0.90 → passes
        es = _mock_es(total=1000, present=950)
        coverage = check_field_coverage(es, "episodes-zh-tw", "show.image_url")
        assert coverage >= THRESHOLDS["image_url_coverage"]

    def test_empty_index_returns_zero(self):
        es = MagicMock()
        es.count.return_value = {"count": 0}
        coverage = check_field_coverage(es, "episodes-zh-tw", "embedding")
        assert coverage == 0.0


# ---------------------------------------------------------------------------
# check_zero_result_rate
# ---------------------------------------------------------------------------

class TestCheckZeroResultRate:
    def test_all_queries_have_results(self):
        es = _mock_es_search([5, 3, 10])
        result = check_zero_result_rate(es, "episodes-zh-tw", ["q1", "q2", "q3"])
        assert result["rate"] == 0.0
        assert result["zero_result_queries"] == []
        assert result["total_tested"] == 3

    def test_some_queries_return_zero_results(self):
        # query q2 returns 0 results
        es = _mock_es_search([5, 0, 10])
        result = check_zero_result_rate(es, "episodes-zh-tw", ["q1", "q2", "q3"])
        assert abs(result["rate"] - 1 / 3) < 1e-6
        assert "q2" in result["zero_result_queries"]

    def test_zero_result_rate_above_threshold(self):
        # 2/10 = 0.20 > 0.05 threshold → would fail
        results = [0, 0, 5, 5, 5, 5, 5, 5, 5, 5]
        es = _mock_es_search(results)
        queries = [f"q{i}" for i in range(10)]
        result = check_zero_result_rate(es, "episodes-zh-tw", queries)
        assert result["rate"] > THRESHOLDS["zero_result_rate"]

    def test_empty_query_list_returns_none_rate(self):
        es = MagicMock()
        result = check_zero_result_rate(es, "episodes-zh-tw", [])
        assert result["rate"] is None
        assert result["total_tested"] == 0

    def test_all_zero_results(self):
        es = _mock_es_search([0, 0, 0])
        result = check_zero_result_rate(es, "episodes-zh-tw", ["q1", "q2", "q3"])
        assert result["rate"] == 1.0
        assert len(result["zero_result_queries"]) == 3


# ---------------------------------------------------------------------------
# THRESHOLDS sanity check
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_embedding_threshold_value(self):
        assert THRESHOLDS["embedding_coverage"] == 0.99

    def test_image_url_threshold_value(self):
        assert THRESHOLDS["image_url_coverage"] == 0.90

    def test_zero_result_threshold_value(self):
        assert THRESHOLDS["zero_result_rate"] == 0.05
