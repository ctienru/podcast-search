"""
Tests for evaluation mode query construction.

Ensures BM25 evaluation mode queries don't contain:
- gauss (time decay)
- language filter
- match_phrase boost
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestBM25EvaluationModeQuery:
    """Tests for BM25 evaluation mode query construction."""

    def test_evaluation_mode_no_gauss(self):
        """Evaluation mode query should not contain gauss (time decay)."""
        from src.services.search_service import SearchService

        # Create a mock SearchService to test _build_bm25_query
        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        query = service._build_bm25_query("test query", evaluation_mode=True)
        query_str = json.dumps(query)

        assert "gauss" not in query_str, "Evaluation mode should not have gauss decay"

    def test_evaluation_mode_no_language_filter(self):
        """Evaluation mode query should not contain language filter."""
        from src.services.search_service import SearchService

        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        query = service._build_bm25_query("test query", evaluation_mode=True)
        query_str = json.dumps(query)

        # Check for language filter patterns
        assert "language" not in query_str.lower() or "filter" not in query_str.lower(), \
            "Evaluation mode should not have language filter"

    def test_evaluation_mode_no_match_phrase(self):
        """Evaluation mode query should not contain match_phrase boost."""
        from src.services.search_service import SearchService

        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        query = service._build_bm25_query("test query", evaluation_mode=True)
        query_str = json.dumps(query)

        assert "match_phrase" not in query_str, \
            "Evaluation mode should not have match_phrase boost"

    def test_evaluation_mode_has_multi_match(self):
        """Evaluation mode query should still use multi_match."""
        from src.services.search_service import SearchService

        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        query = service._build_bm25_query("test query", evaluation_mode=True)
        query_str = json.dumps(query)

        assert "multi_match" in query_str, \
            "Evaluation mode should still use multi_match"

    def test_production_mode_may_have_boost(self):
        """Production mode (evaluation_mode=False) may have additional boosts."""
        from src.services.search_service import SearchService

        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        eval_query = service._build_bm25_query("test", evaluation_mode=True)
        prod_query = service._build_bm25_query("test", evaluation_mode=False)

        # Production query should be different (more complex)
        assert eval_query != prod_query, \
            "Production and evaluation queries should differ"

    def test_evaluation_mode_query_structure(self):
        """Evaluation mode query should have expected structure."""
        from src.services.search_service import SearchService

        service = SearchService.__new__(SearchService)
        service.client = MagicMock()

        query = service._build_bm25_query("投資理財", evaluation_mode=True)

        # Should be a bool query with should clause
        assert "bool" in query
        assert "should" in query["bool"]
        assert "minimum_should_match" in query["bool"]

        # Should have multi_match in should
        should = query["bool"]["should"]
        assert len(should) >= 1
        assert "multi_match" in should[0]

        # Multi_match should have expected fields
        mm = should[0]["multi_match"]
        assert mm["query"] == "投資理財"
        assert "title" in str(mm.get("fields", []))
