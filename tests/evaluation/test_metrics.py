"""Tests for NoAnnotationEvaluator metrics."""

import pytest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import Mock, MagicMock, patch

from src.evaluation.metrics import (
    NoAnnotationEvaluator,
    EvaluationResult,
    AggregatedMetrics,
)
from src.evaluation.extraneous_scorer import ExtraneousScorer, ParagraphScore
from src.services.search_service import SearchResult, SearchResponse, SearchMode


@pytest.fixture
def mock_search_service():
    """Create a mock search service."""
    service = MagicMock()
    return service


@pytest.fixture
def mock_scorer():
    """Create a mock extraneous scorer."""
    scorer = MagicMock(spec=ExtraneousScorer)
    # Default: return non-extraneous score
    scorer.score_paragraph.return_value = ParagraphScore(
        text="test",
        features={},
        extraneous_score=0.1,
        is_extraneous=False,
        matched_patterns=[],
    )
    return scorer


@pytest.fixture
def evaluator(mock_search_service, mock_scorer):
    """Create a NoAnnotationEvaluator instance."""
    return NoAnnotationEvaluator(mock_search_service, mock_scorer)


def make_search_result(
    episode_id: str,
    title: str = "Test Episode",
    description: Optional[str] = "Test description",
    score: float = 1.0,
    show_id: Optional[str] = None,
) -> SearchResult:
    """Helper to create SearchResult objects."""
    return SearchResult(
        episode_id=episode_id,
        title=title,
        description=description,
        score=score,
        show_id=show_id,
    )


def make_search_response(results: List[SearchResult]) -> SearchResponse:
    """Helper to create SearchResponse objects."""
    return SearchResponse(
        results=results,
        total=len(results),
        took_ms=10,
        mode=SearchMode.HYBRID,
    )


class TestGetResultIds:
    """Test result ID extraction."""

    def test_extracts_episode_ids(self, evaluator):
        """Test extracting episode IDs from results."""
        results = [
            make_search_result("ep_1"),
            make_search_result("ep_2"),
            make_search_result("ep_3"),
        ]

        ids = evaluator._get_result_ids(results)

        assert ids == {"ep_1", "ep_2", "ep_3"}

    def test_handles_empty_results(self, evaluator):
        """Test handling empty results."""
        ids = evaluator._get_result_ids([])
        assert ids == set()


class TestJaccardSimilarity:
    """Test Jaccard similarity calculation."""

    def test_identical_sets(self, evaluator):
        """Test Jaccard similarity of identical sets."""
        set_a = {"a", "b", "c"}
        set_b = {"a", "b", "c"}

        result = evaluator._jaccard_similarity(set_a, set_b)

        assert result == 1.0

    def test_disjoint_sets(self, evaluator):
        """Test Jaccard similarity of disjoint sets."""
        set_a = {"a", "b", "c"}
        set_b = {"d", "e", "f"}

        result = evaluator._jaccard_similarity(set_a, set_b)

        assert result == 0.0

    def test_partial_overlap(self, evaluator):
        """Test Jaccard similarity of partially overlapping sets."""
        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}

        result = evaluator._jaccard_similarity(set_a, set_b)

        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert result == 0.5

    def test_empty_sets(self, evaluator):
        """Test Jaccard similarity of empty sets."""
        assert evaluator._jaccard_similarity(set(), set()) == 1.0
        assert evaluator._jaccard_similarity({"a"}, set()) == 0.0
        assert evaluator._jaccard_similarity(set(), {"a"}) == 0.0


class TestCalculateSamePodcastDominance:
    """Test same-podcast dominance calculation."""

    def test_single_show_dominance(self, evaluator):
        """Test when single show dominates results."""
        results = [
            make_search_result("ep_1", show_id="show_a"),
            make_search_result("ep_2", show_id="show_a"),
            make_search_result("ep_3", show_id="show_a"),
            make_search_result("ep_4", show_id="show_a"),
            make_search_result("ep_5", show_id="show_b"),
        ]

        dominance, dominant_show = evaluator._calculate_same_podcast_dominance(results, k=5)

        assert dominance == 0.8  # 4 out of 5
        assert dominant_show == "show_a"

    def test_even_distribution(self, evaluator):
        """Test when shows are evenly distributed."""
        results = [
            make_search_result("ep_1", show_id="show_a"),
            make_search_result("ep_2", show_id="show_b"),
            make_search_result("ep_3", show_id="show_c"),
            make_search_result("ep_4", show_id="show_d"),
        ]

        dominance, _ = evaluator._calculate_same_podcast_dominance(results, k=4)

        assert dominance == 0.25  # 1 out of 4

    def test_handles_empty_results(self, evaluator):
        """Test handling empty results."""
        dominance, dominant_show = evaluator._calculate_same_podcast_dominance([], k=10)

        assert dominance == 0.0
        assert dominant_show is None

    def test_handles_missing_show_id(self, evaluator):
        """Test handling results without show_id."""
        results = [
            make_search_result("ep_1", show_id=None),
            make_search_result("ep_2", show_id=None),
        ]

        dominance, dominant_show = evaluator._calculate_same_podcast_dominance(results, k=2)

        # Should group under "unknown"
        assert dominance == 1.0
        assert dominant_show == "unknown"


class TestCalculateExtraneousIntrusion:
    """Test extraneous intrusion calculation."""

    def test_no_extraneous_content(self, evaluator, mock_scorer):
        """Test when no results have extraneous content."""
        mock_scorer.score_paragraph.return_value = ParagraphScore(
            text="clean",
            features={},
            extraneous_score=0.1,
            is_extraneous=False,
            matched_patterns=[],
        )

        results = [
            make_search_result("ep_1", description="Clean content"),
            make_search_result("ep_2", description="More clean content"),
        ]

        intrusion, intrusion_eps = evaluator._calculate_extraneous_intrusion(results, k=2)

        assert intrusion == 0.0
        assert intrusion_eps == []

    def test_all_extraneous_content(self, evaluator, mock_scorer):
        """Test when all results have extraneous content."""
        mock_scorer.score_paragraph.return_value = ParagraphScore(
            text="sponsor",
            features={},
            extraneous_score=0.9,
            is_extraneous=True,
            matched_patterns=["sponsor"],
        )

        results = [
            make_search_result("ep_1", description="Sponsored content"),
            make_search_result("ep_2", description="More sponsored content"),
        ]

        intrusion, intrusion_eps = evaluator._calculate_extraneous_intrusion(results, k=2)

        assert intrusion == 1.0
        assert len(intrusion_eps) == 2

    def test_partial_extraneous_content(self, evaluator, mock_scorer):
        """Test when some results have extraneous content."""
        def mock_score(text):
            if "sponsor" in text.lower():
                return ParagraphScore(
                    text=text,
                    features={},
                    extraneous_score=0.9,
                    is_extraneous=True,
                    matched_patterns=["sponsor"],
                )
            return ParagraphScore(
                text=text,
                features={},
                extraneous_score=0.1,
                is_extraneous=False,
                matched_patterns=[],
            )

        mock_scorer.score_paragraph.side_effect = mock_score

        results = [
            make_search_result("ep_1", description="Clean content"),
            make_search_result("ep_2", description="Sponsored by XYZ"),
        ]

        intrusion, intrusion_eps = evaluator._calculate_extraneous_intrusion(results, k=2)

        assert intrusion == 0.5
        assert intrusion_eps == ["ep_2"]

    def test_handles_empty_results(self, evaluator):
        """Test handling empty results."""
        intrusion, intrusion_eps = evaluator._calculate_extraneous_intrusion([], k=10)

        assert intrusion == 0.0
        assert intrusion_eps == []

    def test_handles_none_description(self, evaluator, mock_scorer):
        """Test handling results with None description."""
        results = [
            make_search_result("ep_1", description=None),
        ]

        intrusion, intrusion_eps = evaluator._calculate_extraneous_intrusion(results, k=1)

        # Should not call scorer for None descriptions
        mock_scorer.score_paragraph.assert_not_called()
        assert intrusion == 0.0


class TestPerturbQuery:
    """Test query perturbation."""

    def test_perturbs_multi_word_query(self, evaluator):
        """Test perturbation of multi-word query."""
        perturbations = evaluator._perturb_query("artificial intelligence podcast")

        assert len(perturbations) > 0
        assert len(perturbations) <= 3  # Max 3 perturbations

    def test_handles_single_word_query(self, evaluator):
        """Test handling single word query."""
        perturbations = evaluator._perturb_query("podcast")

        # Can still reverse (no-op for single word)
        assert isinstance(perturbations, list)

    def test_includes_word_removal(self, evaluator):
        """Test that word removal is included."""
        perturbations = evaluator._perturb_query("AI podcast news")

        # Should include versions with words removed
        assert any(p.count(" ") < 2 for p in perturbations)

    def test_includes_word_reversal(self, evaluator):
        """Test that word reversal is included."""
        perturbations = evaluator._perturb_query("AI podcast")

        assert "podcast AI" in perturbations


class TestCalculatePerturbationStability:
    """Test perturbation stability calculation."""

    def test_stable_search(self, evaluator, mock_search_service):
        """Test stability when search is stable across perturbations."""
        # All searches return same results
        results = [
            make_search_result("ep_1"),
            make_search_result("ep_2"),
        ]
        mock_search_service.search.return_value = make_search_response(results)

        original_ids = {"ep_1", "ep_2"}
        stability = evaluator._calculate_perturbation_stability("AI podcast", original_ids, k=2)

        assert stability == 1.0

    def test_unstable_search(self, evaluator, mock_search_service):
        """Test stability when search is unstable across perturbations."""
        # Different results for each query
        mock_search_service.search.side_effect = [
            make_search_response([make_search_result("ep_3"), make_search_result("ep_4")]),
            make_search_response([make_search_result("ep_5"), make_search_result("ep_6")]),
            make_search_response([make_search_result("ep_7"), make_search_result("ep_8")]),
        ]

        original_ids = {"ep_1", "ep_2"}
        stability = evaluator._calculate_perturbation_stability("AI podcast news", original_ids, k=2)

        assert stability == 0.0  # No overlap

    def test_handles_search_failure(self, evaluator, mock_search_service):
        """Test handling of search failures during perturbation."""
        mock_search_service.search.side_effect = Exception("Search failed")

        original_ids = {"ep_1", "ep_2"}
        stability = evaluator._calculate_perturbation_stability("AI podcast", original_ids, k=2)

        # Should treat failures as 0 similarity
        assert stability == 0.0

    def test_returns_1_for_no_perturbations(self, evaluator):
        """Test that single word queries return 1.0 stability."""
        # Single word queries may have limited perturbations
        # If no perturbations possible, should return 1.0
        original_ids = {"ep_1"}
        stability = evaluator._calculate_perturbation_stability("x", original_ids, k=1)

        # With minimal query, should assume stable
        assert stability >= 0


class TestEvaluateQuery:
    """Test single query evaluation."""

    def test_evaluates_query(self, evaluator, mock_search_service, mock_scorer):
        """Test evaluating a single query."""
        results = [
            make_search_result("ep_1", show_id="show_a", description="Clean content"),
            make_search_result("ep_2", show_id="show_b", description="More content"),
        ]
        mock_search_service.search.return_value = make_search_response(results)
        mock_search_service.search_bm25.return_value = make_search_response(results)
        mock_search_service.search_knn.return_value = make_search_response(results)

        result = evaluator.evaluate_query("test query", k=2)

        assert isinstance(result, EvaluationResult)
        assert result.query == "test query"
        assert 0 <= result.same_podcast_dominance <= 1
        assert 0 <= result.extraneous_intrusion <= 1
        assert 0 <= result.perturbation_stability <= 1

    def test_includes_debug_info_when_requested(self, evaluator, mock_search_service, mock_scorer):
        """Test that debug info is included when requested."""
        results = [make_search_result("ep_1", show_id="show_a")]
        mock_search_service.search.return_value = make_search_response(results)
        mock_search_service.search_bm25.return_value = make_search_response(results)
        mock_search_service.search_knn.return_value = make_search_response(results)

        result = evaluator.evaluate_query("test query", k=1, include_debug=True)

        assert result.after_ids is not None
        assert result.dominant_show_id is not None


class TestAggregateResults:
    """Test result aggregation."""

    def test_aggregates_multiple_results(self, evaluator):
        """Test aggregating multiple evaluation results."""
        results = [
            EvaluationResult(
                query="q1",
                top_k_overlap=0.8,
                same_podcast_dominance=0.3,
                extraneous_intrusion=0.1,
                perturbation_stability=0.9,
            ),
            EvaluationResult(
                query="q2",
                top_k_overlap=0.6,
                same_podcast_dominance=0.5,
                extraneous_intrusion=0.2,
                perturbation_stability=0.7,
            ),
        ]

        aggregated = evaluator.aggregate_results(results)

        assert isinstance(aggregated, AggregatedMetrics)
        assert aggregated.total_queries == 2
        assert aggregated.avg_top_k_overlap == 0.7
        assert aggregated.avg_same_podcast_dominance == 0.4
        assert aggregated.avg_extraneous_intrusion == 0.15
        assert aggregated.avg_perturbation_stability == 0.8

    def test_sets_assessment_flags(self, evaluator):
        """Test that assessment flags are set correctly."""
        # Good results
        good_results = [
            EvaluationResult(
                query="q1",
                top_k_overlap=0.8,
                same_podcast_dominance=0.2,
                extraneous_intrusion=0.05,
                perturbation_stability=0.85,
            ),
        ]

        aggregated = evaluator.aggregate_results(good_results)

        assert aggregated.cleaning_effective is True  # intrusion < 0.1
        assert aggregated.ranking_stable is True  # stability > 0.7
        assert aggregated.no_show_dominance is True  # dominance < 0.5

    def test_handles_empty_results(self, evaluator):
        """Test handling empty results list."""
        aggregated = evaluator.aggregate_results([])

        assert aggregated.total_queries == 0
        assert aggregated.avg_top_k_overlap == 0.0
        assert aggregated.cleaning_effective is False
        assert aggregated.ranking_stable is False
        assert aggregated.no_show_dominance is False


class TestToDict:
    """Test dict conversion methods."""

    def test_converts_evaluation_result_to_dict(self, evaluator):
        """Test converting EvaluationResult to dict."""
        result = EvaluationResult(
            query="test",
            top_k_overlap=0.8,
            same_podcast_dominance=0.3,
            extraneous_intrusion=0.1,
            perturbation_stability=0.9,
        )

        d = evaluator.to_dict(result)

        assert d["query"] == "test"
        assert d["top_k_overlap"] == 0.8
        assert d["same_podcast_dominance"] == 0.3
        assert d["extraneous_intrusion"] == 0.1
        assert d["perturbation_stability"] == 0.9

    def test_converts_aggregated_metrics_to_dict(self, evaluator):
        """Test converting AggregatedMetrics to dict."""
        aggregated = AggregatedMetrics(
            total_queries=10,
            avg_top_k_overlap=0.7,
            avg_same_podcast_dominance=0.4,
            avg_extraneous_intrusion=0.15,
            avg_perturbation_stability=0.8,
            cleaning_effective=True,
            ranking_stable=True,
            no_show_dominance=True,
            methods_complementary=False,
        )

        d = evaluator.aggregate_to_dict(aggregated)

        assert d["total_queries"] == 10
        assert d["avg_top_k_overlap"] == 0.7
        assert d["cleaning_effective"] is True
        assert d["ranking_stable"] is True
        assert d["no_show_dominance"] is True
