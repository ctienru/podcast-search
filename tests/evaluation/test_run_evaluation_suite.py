"""
Tests for run_evaluation_suite.py

Pure logic tests — no Elasticsearch, no subprocess calls.
Tests cover: compute_gate_status(), language_routing_na_result().
"""

import pytest

from scripts.run_evaluation_suite import (
    compute_gate_status,
    language_routing_na_result,
)


# ---------------------------------------------------------------------------
# compute_gate_status
# ---------------------------------------------------------------------------

class TestComputeGateStatus:
    def test_all_pass_returns_pass(self):
        items = {
            "index_health": {"status": "PASS"},
            "language_routing": {"status": "PASS"},
        }
        assert compute_gate_status(items) == "PASS"

    def test_any_fail_returns_fail(self):
        items = {
            "index_health": {"status": "FAIL"},
            "language_routing": {"status": "PASS"},
        }
        assert compute_gate_status(items) == "FAIL"

    def test_all_fail_returns_fail(self):
        items = {
            "index_health": {"status": "FAIL"},
            "language_routing": {"status": "FAIL"},
        }
        assert compute_gate_status(items) == "FAIL"

    def test_na_does_not_block_pass(self):
        # N/A is non-blocking — PASS + N/A should still be PASS
        items = {
            "index_health": {"status": "PASS"},
            "language_routing": {"status": "N/A"},
        }
        assert compute_gate_status(items) == "PASS"

    def test_fail_and_na_returns_fail(self):
        items = {
            "index_health": {"status": "FAIL"},
            "language_routing": {"status": "N/A"},
        }
        assert compute_gate_status(items) == "FAIL"

    def test_all_na_returns_na(self):
        items = {
            "language_routing": {"status": "N/A"},
        }
        assert compute_gate_status(items) == "N/A"

    def test_three_items_one_fail(self):
        items = {
            "ndcg": {"status": "PASS"},
            "latency": {"status": "FAIL"},
            "extra": {"status": "PASS"},
        }
        assert compute_gate_status(items) == "FAIL"

    def test_pass_and_na_and_pass_returns_pass(self):
        items = {
            "ndcg": {"status": "PASS"},
            "latency": {"status": "N/A"},
            "index_health": {"status": "PASS"},
        }
        assert compute_gate_status(items) == "PASS"


# ---------------------------------------------------------------------------
# language_routing_na_result
# ---------------------------------------------------------------------------

class TestLanguageRoutingNaResult:
    def test_returns_na_status(self):
        result = language_routing_na_result()
        assert result["status"] == "N/A"

    def test_includes_reason(self):
        result = language_routing_na_result()
        assert "reason" in result
        assert len(result["reason"]) > 0

    def test_reason_mentions_rss_or_crawler(self):
        result = language_routing_na_result()
        reason_lower = result["reason"].lower()
        assert "rss" in reason_lower or "crawler" in reason_lower or "ingest" in reason_lower

    def test_consistent_across_calls(self):
        # Same result every time (not stateful)
        r1 = language_routing_na_result()
        r2 = language_routing_na_result()
        assert r1 == r2
