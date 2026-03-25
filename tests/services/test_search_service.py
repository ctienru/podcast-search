"""Tests for SearchService."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.services.search_service import (
    SearchService,
    SearchMode,
    SearchResult,
    SearchResponse,
)


@pytest.fixture
def mock_es_client():
    """Create a mock Elasticsearch client."""
    with patch('src.services.search_service.get_es_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_encoder():
    """Create a mock embedding encoder."""
    encoder = MagicMock()
    encoder.encode.return_value = np.zeros(768)
    return encoder


def create_mock_es_response(hits: list[dict], total: int = 10, took: int = 5):
    """Create a mock Elasticsearch response."""
    return {
        "hits": {
            "total": {"value": total},
            "hits": hits,
        },
        "took": took,
    }


def create_mock_hit(episode_id: str, title: str, score: float = 1.0, show_id: str = "show_123"):
    """Create a mock Elasticsearch hit."""
    return {
        "_id": episode_id,
        "_score": score,
        "_source": {
            "episode_id": episode_id,
            "title": title,
            "description": f"Description for {title}",
            "show": {
                "show_id": show_id,
                "title": "Test Show",
            },
            "published_at": "2024-01-01T00:00:00Z",
            "duration_sec": 3600,
        },
    }


class TestSearchMode:
    """Test SearchMode enum."""

    def test_bm25_mode_value(self):
        """Test BM25 mode has correct value."""
        assert SearchMode.BM25.value == "bm25"

    def test_knn_mode_value(self):
        """Test KNN mode has correct value."""
        assert SearchMode.KNN.value == "knn"

    def test_hybrid_mode_value(self):
        """Test HYBRID mode has correct value."""
        assert SearchMode.HYBRID.value == "hybrid"

    def test_exact_mode_value(self):
        """Test EXACT mode has correct value."""
        assert SearchMode.EXACT.value == "exact"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            episode_id="ep_123",
            title="Test Episode",
            description="Test description",
            score=0.95,
            show_title="Test Show",
            show_id="show_123",
        )

        assert result.episode_id == "ep_123"
        assert result.title == "Test Episode"
        assert result.description == "Test description"
        assert result.score == 0.95
        assert result.show_title == "Test Show"
        assert result.show_id == "show_123"

    def test_search_result_optional_fields(self):
        """Test SearchResult with optional fields as None."""
        result = SearchResult(
            episode_id="ep_123",
            title="Test Episode",
            description=None,
            score=0.5,
        )

        assert result.description is None
        assert result.show_title is None
        assert result.show_id is None
        assert result.published_at is None
        assert result.duration_sec is None


class TestSearchBm25:
    """Test BM25 search functionality."""

    def test_search_bm25_returns_results(self, mock_es_client):
        """Test BM25 search returns results."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[
                create_mock_hit("ep_1", "AI Podcast Episode"),
                create_mock_hit("ep_2", "Machine Learning Talk"),
            ],
            total=2,
        )

        service = SearchService()
        response = service.search_bm25("AI podcast", size=10)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2
        assert response.mode == SearchMode.BM25
        assert response.total == 2

    def test_search_bm25_parses_hits_correctly(self, mock_es_client):
        """Test BM25 search parses ES hits correctly."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "AI Podcast", score=2.5)],
        )

        service = SearchService()
        response = service.search_bm25("AI", size=10)

        result = response.results[0]
        assert result.episode_id == "ep_1"
        assert result.title == "AI Podcast"
        assert result.score == 2.5
        assert result.show_title == "Test Show"

    def test_search_bm25_builds_correct_query(self, mock_es_client):
        """Test BM25 search builds correct ES query."""
        mock_es_client.search.return_value = create_mock_es_response(hits=[])

        service = SearchService()
        service.search_bm25("test query", size=5)

        # Verify search was called with correct body structure
        call_args = mock_es_client.search.call_args
        body = call_args.kwargs.get("body", call_args[1].get("body"))

        assert "query" in body
        assert "bool" in body["query"]
        assert "should" in body["query"]["bool"]
        assert body["size"] == 5


class TestSearchExact:
    """Test exact phrase match search functionality."""

    def test_search_exact_returns_results(self, mock_es_client):
        """Test exact search returns results."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "AI Podcast Episode")],
            total=1,
        )

        service = SearchService()
        response = service.search_exact("AI Podcast", size=10)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.mode == SearchMode.EXACT
        assert response.total == 1

    def test_search_exact_builds_match_phrase_query(self, mock_es_client):
        """Test exact search uses match_phrase queries."""
        mock_es_client.search.return_value = create_mock_es_response(hits=[])

        service = SearchService()
        service.search_exact("exact phrase", size=5)

        call_args = mock_es_client.search.call_args
        body = call_args.kwargs.get("body", call_args[1].get("body"))

        assert "query" in body
        assert "bool" in body["query"]

        # Check that match_phrase is used
        should_clauses = body["query"]["bool"]["should"]
        has_match_phrase = any(
            "match_phrase" in clause for clause in should_clauses
        )
        assert has_match_phrase, "Exact query should use match_phrase"

    def test_search_exact_includes_chinese_fields(self, mock_es_client):
        """Test exact search includes Chinese text fields."""
        mock_es_client.search.return_value = create_mock_es_response(hits=[])

        service = SearchService()
        service.search_exact("人工智慧", size=5)

        call_args = mock_es_client.search.call_args
        body = call_args.kwargs.get("body", call_args[1].get("body"))

        should_clauses = body["query"]["bool"]["should"]

        # Convert to string to check for Chinese field references
        query_str = str(should_clauses)
        assert "title.chinese" in query_str, "Should include title.chinese"
        assert "description.chinese" in query_str, "Should include description.chinese"


class TestSearchKnn:
    """Test kNN semantic search functionality."""

    def test_search_knn_returns_results(self, mock_es_client, mock_encoder):
        """Test kNN search returns results."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "AI Podcast Episode")],
            total=1,
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search_knn("AI podcast", size=10)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.mode == SearchMode.KNN

    def test_search_knn_calls_encoder(self, mock_es_client, mock_encoder):
        """Test kNN search encodes query to vector."""
        mock_es_client.search.return_value = create_mock_es_response(hits=[])

        service = SearchService(encoder=mock_encoder)
        service.search_knn("test query", size=5)

        mock_encoder.embed.assert_called_once_with("test query", language="zh-tw")

    def test_search_knn_builds_knn_clause(self, mock_es_client, mock_encoder):
        """Test kNN search builds correct kNN clause."""
        mock_es_client.search.return_value = create_mock_es_response(hits=[])

        service = SearchService(encoder=mock_encoder)
        service.search_knn("test", size=5)

        call_args = mock_es_client.search.call_args
        body = call_args.kwargs.get("body", call_args[1].get("body"))

        assert "knn" in body
        assert "field" in body["knn"]
        assert body["knn"]["field"] == "embedding"
        assert "query_vector" in body["knn"]


class TestSearchHybrid:
    """Test hybrid search functionality with RRF fusion."""

    def test_search_hybrid_returns_results(self, mock_es_client, mock_encoder):
        """Test hybrid search returns results."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "AI Podcast")],
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search_hybrid("AI podcast", size=5)

        assert isinstance(response, SearchResponse)
        assert response.mode == SearchMode.HYBRID

    def test_search_hybrid_deduplicates_results(self, mock_es_client, mock_encoder):
        """Test hybrid search deduplicates results from BM25 and kNN."""
        # Both BM25 and kNN return the same episode
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "AI Podcast")],
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search_hybrid("AI podcast", size=5)

        # Should have only 1 result despite appearing in both BM25 and kNN
        assert len(response.results) == 1
        assert response.results[0].episode_id == "ep_1"


class TestComputeRrfScores:
    """Test RRF score computation."""

    def test_rrf_scores_single_list(self, mock_es_client):
        """Test RRF scores with results from single list."""
        service = SearchService()

        bm25_results = [
            SearchResult("ep_1", "A", None, 1.0),
            SearchResult("ep_2", "B", None, 0.9),
        ]
        knn_results = []

        scores = service._compute_rrf_scores(bm25_results, knn_results)

        # ep_1 at rank 1: 1/(60+1) = 0.01639
        # ep_2 at rank 2: 1/(60+2) = 0.01613
        assert "ep_1" in scores
        assert "ep_2" in scores
        assert scores["ep_1"] > scores["ep_2"]

    def test_rrf_scores_overlapping_results(self, mock_es_client):
        """Test RRF scores boost overlapping results."""
        service = SearchService()

        bm25_results = [
            SearchResult("ep_1", "A", None, 1.0),
            SearchResult("ep_2", "B", None, 0.9),
        ]
        knn_results = [
            SearchResult("ep_1", "A", None, 0.95),  # ep_1 appears in both
            SearchResult("ep_3", "C", None, 0.8),
        ]

        scores = service._compute_rrf_scores(bm25_results, knn_results)

        # ep_1 should have highest score (appears in both lists)
        assert scores["ep_1"] > scores["ep_2"]
        assert scores["ep_1"] > scores["ep_3"]


class TestUnifiedSearch:
    """Test unified search interface."""

    def test_search_with_bm25_mode(self, mock_es_client):
        """Test unified search with BM25 mode."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "Test")],
        )

        service = SearchService()
        response = service.search("query", mode=SearchMode.BM25)

        assert response.mode == SearchMode.BM25

    def test_search_with_exact_mode(self, mock_es_client):
        """Test unified search with EXACT mode."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "Test")],
        )

        service = SearchService()
        response = service.search("query", mode=SearchMode.EXACT)

        assert response.mode == SearchMode.EXACT

    def test_search_with_knn_mode(self, mock_es_client, mock_encoder):
        """Test unified search with KNN mode."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "Test")],
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search("query", mode=SearchMode.KNN)

        assert response.mode == SearchMode.KNN

    def test_search_default_mode_is_hybrid(self, mock_es_client, mock_encoder):
        """Test unified search defaults to HYBRID mode."""
        mock_es_client.search.return_value = create_mock_es_response(
            hits=[create_mock_hit("ep_1", "Test")],
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search("query")

        assert response.mode == SearchMode.HYBRID


class TestSearchHybridDiversity:
    """Test per-show diversity cap in hybrid search."""

    def test_per_show_cap_is_enforced(self, mock_es_client, mock_encoder):
        """Test that no single show appears more than MAX_PER_SHOW times in results."""
        # One dominant show has 8 episodes, one other show has 2 episodes
        dominant_hits = [
            create_mock_hit(f"dom_{i}", f"Dominant Ep {i}", score=1.0 - i * 0.01, show_id="dominant_show")
            for i in range(8)
        ]
        other_hits = [
            create_mock_hit(f"other_{i}", f"Other Ep {i}", score=0.5, show_id="other_show")
            for i in range(2)
        ]
        mock_es_client.search.return_value = create_mock_es_response(
            hits=dominant_hits + other_hits,
            total=10,
        )

        service = SearchService(encoder=mock_encoder)
        response = service.search_hybrid("test query", size=10)

        show_counts: dict[str, int] = {}
        for r in response.results:
            show_counts[r.show_id or "unknown"] = show_counts.get(r.show_id or "unknown", 0) + 1

        assert all(count <= SearchService.MAX_PER_SHOW for count in show_counts.values())
        assert show_counts.get("dominant_show", 0) <= SearchService.MAX_PER_SHOW

    def test_diversity_cap_does_not_drop_results_when_enough_shows(self, mock_es_client, mock_encoder):
        """Test that size results are still returned when enough diverse shows exist."""
        # 4 shows, each with 5 episodes
        hits = [
            create_mock_hit(f"show{s}_ep{e}", f"Show {s} Ep {e}", show_id=f"show_{s}")
            for s in range(4)
            for e in range(5)
        ]
        mock_es_client.search.return_value = create_mock_es_response(hits=hits, total=20)

        service = SearchService(encoder=mock_encoder)
        response = service.search_hybrid("test query", size=10)

        assert len(response.results) == 10


class TestBuildExactQuery:
    """Test exact query building."""

    def test_build_exact_query_structure(self, mock_es_client):
        """Test exact query has correct structure."""
        service = SearchService()
        query = service._build_exact_query("test phrase")

        assert "bool" in query
        assert "should" in query["bool"]
        assert "minimum_should_match" in query["bool"]
        assert query["bool"]["minimum_should_match"] == 1

    def test_build_exact_query_uses_match_phrase(self, mock_es_client):
        """Test exact query uses match_phrase for all fields."""
        service = SearchService()
        query = service._build_exact_query("test phrase")

        should_clauses = query["bool"]["should"]

        # All clauses should use match_phrase
        for clause in should_clauses:
            assert "match_phrase" in clause, f"Expected match_phrase in {clause}"

    def test_build_exact_query_title_boost(self, mock_es_client):
        """Test exact query has higher boost for title fields."""
        service = SearchService()
        query = service._build_exact_query("test")

        should_clauses = query["bool"]["should"]

        # Find title clauses and check boost
        title_boosts = []
        desc_boosts = []

        for clause in should_clauses:
            match_phrase = clause.get("match_phrase", {})
            if "title" in match_phrase and "chinese" not in list(match_phrase.keys())[0]:
                title_boosts.append(match_phrase["title"].get("boost", 1))
            if "title.chinese" in match_phrase:
                title_boosts.append(match_phrase["title.chinese"].get("boost", 1))
            if "description" in match_phrase and "chinese" not in list(match_phrase.keys())[0]:
                desc_boosts.append(match_phrase["description"].get("boost", 1))

        # Title boost should be higher
        assert all(tb > 1 for tb in title_boosts), "Title fields should have boost > 1"
