"""
Tests for build_annotation_pool.py

Tests verify correct pooling and de-duplication behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    episode_id: str
    title: str
    description: str
    score: float = 1.0


@dataclass
class MockSearchResponse:
    """Mock search response for testing."""
    results: List[MockSearchResult]
    total: int = 0
    took_ms: int = 0


class TestAnnotationPoolDedup:
    """Tests for de-duplication logic."""

    def test_dedup_same_episode_different_methods(self):
        """Same episode from BM25 and embedding should appear once."""
        from scripts.build_annotation_pool import build_annotation_pool

        # Mock search service
        mock_service = MagicMock()

        # Both methods return the same episode
        same_episode = MockSearchResult(
            episode_id="ep_001",
            title="Test Episode",
            description="Test description",
        )

        mock_service.search_bm25.return_value = MockSearchResponse(
            results=[same_episode]
        )
        mock_service.search_knn.return_value = MockSearchResponse(
            results=[same_episode]
        )

        queries = [{"query": "test query"}]
        pool = build_annotation_pool(
            queries=queries,
            search_service=mock_service,
            k=10,
            include_hybrid=False,
        )

        # Should have exactly 1 episode
        assert len(pool["test query"]) == 1
        assert "ep_001" in pool["test query"]

        # Should have both sources
        ep_info = pool["test query"]["ep_001"]
        assert "bm25" in ep_info["sources"]
        assert "embedding" in ep_info["sources"]

        # Should have ranks from both
        assert ep_info["ranks"]["bm25"] == 1
        assert ep_info["ranks"]["embedding"] == 1

    def test_dedup_different_episodes(self):
        """Different episodes should all appear in pool."""
        from scripts.build_annotation_pool import build_annotation_pool

        mock_service = MagicMock()

        bm25_episode = MockSearchResult(
            episode_id="ep_bm25",
            title="BM25 Episode",
            description="From BM25",
        )
        knn_episode = MockSearchResult(
            episode_id="ep_knn",
            title="KNN Episode",
            description="From KNN",
        )

        mock_service.search_bm25.return_value = MockSearchResponse(
            results=[bm25_episode]
        )
        mock_service.search_knn.return_value = MockSearchResponse(
            results=[knn_episode]
        )

        queries = [{"query": "test"}]
        pool = build_annotation_pool(
            queries=queries,
            search_service=mock_service,
            k=10,
            include_hybrid=False,
        )

        # Should have both episodes
        assert len(pool["test"]) == 2
        assert "ep_bm25" in pool["test"]
        assert "ep_knn" in pool["test"]

        # Each should have only one source
        assert pool["test"]["ep_bm25"]["sources"] == ["bm25"]
        assert pool["test"]["ep_knn"]["sources"] == ["embedding"]


class TestNoHybridInPhase2:
    """Tests to ensure hybrid is not included in Phase 2."""

    def test_no_hybrid_in_phase2(self):
        """Phase 2 pool should not contain hybrid sources."""
        from scripts.build_annotation_pool import build_annotation_pool

        mock_service = MagicMock()

        episode = MockSearchResult(
            episode_id="ep_001",
            title="Test",
            description="Test",
        )

        mock_service.search_bm25.return_value = MockSearchResponse(results=[episode])
        mock_service.search_knn.return_value = MockSearchResponse(results=[episode])

        queries = [{"query": "test"}]
        pool = build_annotation_pool(
            queries=queries,
            search_service=mock_service,
            k=10,
            include_hybrid=False,  # Phase 2
        )

        # Hybrid search should not be called
        mock_service.search_hybrid.assert_not_called()

        # No episode should have hybrid source
        for query_episodes in pool.values():
            for ep_info in query_episodes.values():
                assert "hybrid" not in ep_info["sources"]
                assert "hybrid" not in ep_info["ranks"]

    def test_hybrid_included_when_requested(self):
        """Phase 3 pool should contain hybrid sources when requested."""
        from scripts.build_annotation_pool import build_annotation_pool

        mock_service = MagicMock()

        episode = MockSearchResult(
            episode_id="ep_001",
            title="Test",
            description="Test",
        )

        mock_service.search_bm25.return_value = MockSearchResponse(results=[episode])
        mock_service.search_knn.return_value = MockSearchResponse(results=[episode])
        mock_service.search_hybrid.return_value = MockSearchResponse(results=[episode])

        queries = [{"query": "test"}]
        pool = build_annotation_pool(
            queries=queries,
            search_service=mock_service,
            k=10,
            include_hybrid=True,  # Phase 3
        )

        # Hybrid search should be called
        mock_service.search_hybrid.assert_called_once()

        # Episode should have hybrid source
        ep_info = pool["test"]["ep_001"]
        assert "hybrid" in ep_info["sources"]
        assert "hybrid" in ep_info["ranks"]


class TestRankRecording:
    """Tests for correct rank recording."""

    def test_ranks_are_1_indexed(self):
        """Ranks should be 1-indexed (first result = rank 1)."""
        from scripts.build_annotation_pool import build_annotation_pool

        mock_service = MagicMock()

        episodes = [
            MockSearchResult(episode_id=f"ep_{i}", title=f"Title {i}", description="")
            for i in range(3)
        ]

        mock_service.search_bm25.return_value = MockSearchResponse(results=episodes)
        mock_service.search_knn.return_value = MockSearchResponse(results=[])

        queries = [{"query": "test"}]
        pool = build_annotation_pool(
            queries=queries,
            search_service=mock_service,
            k=10,
            include_hybrid=False,
        )

        # Check ranks are 1-indexed
        assert pool["test"]["ep_0"]["ranks"]["bm25"] == 1
        assert pool["test"]["ep_1"]["ranks"]["bm25"] == 2
        assert pool["test"]["ep_2"]["ranks"]["bm25"] == 3
