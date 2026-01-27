"""
No-Annotation Evaluation Metrics

Annotation-free search quality evaluation metrics:
1. Top-K Overlap Rate: Before/After search result overlap
2. Same-Podcast Dominance Rate: Single podcast domination rate
3. Extraneous Intrusion Rate: Extraneous content intrusion rate
4. Query Perturbation Stability: Query perturbation stability

Usage:
    from src.evaluation.metrics import NoAnnotationEvaluator

    evaluator = NoAnnotationEvaluator(search_service, extraneous_scorer)
    result = evaluator.evaluate_query("AI podcast", k=10)
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Set

from src.evaluation.extraneous_scorer import ExtraneousScorer
from src.services.search_service import SearchService, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result for a single query"""

    query: str
    top_k_overlap: float  # Before/After Top-K overlap (Jaccard)
    same_podcast_dominance: float  # Single podcast domination rate
    extraneous_intrusion: float  # Extraneous content intrusion rate
    perturbation_stability: float  # Query perturbation stability

    # Debug info
    before_ids: Optional[List[str]] = None
    after_ids: Optional[List[str]] = None
    dominant_show_id: Optional[str] = None
    intrusion_episodes: Optional[List[str]] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for multiple queries"""

    total_queries: int
    avg_top_k_overlap: float
    avg_same_podcast_dominance: float
    avg_extraneous_intrusion: float
    avg_perturbation_stability: float

    # Assessment
    cleaning_effective: bool  # extraneous_intrusion < 0.1
    ranking_stable: bool  # perturbation_stability > 0.7
    no_show_dominance: bool  # same_podcast_dominance < 0.5


class NoAnnotationEvaluator:
    """
    Annotation-free search quality evaluator

    Used to verify whether data cleaning improves search quality
    """

    def __init__(
        self,
        search_service: SearchService,
        extraneous_scorer: ExtraneousScorer,
    ):
        self.search = search_service
        self.scorer = extraneous_scorer

    def _get_result_ids(self, results: List[SearchResult]) -> Set[str]:
        """Extract episode IDs from search results"""
        return set(r.episode_id for r in results)

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _calculate_same_podcast_dominance(
        self, results: List[SearchResult], k: int
    ) -> tuple[float, Optional[str]]:
        """
        Calculate same-podcast dominance rate

        If top-k has multiple results from the same podcast, it may indicate a problem
        """
        if not results:
            return 0.0, None

        show_counts: dict[str, int] = {}
        for r in results[:k]:
            show_id = r.show_id or "unknown"
            show_counts[show_id] = show_counts.get(show_id, 0) + 1

        if not show_counts:
            return 0.0, None

        max_count = max(show_counts.values())
        dominant_show = max(show_counts, key=show_counts.get)

        return max_count / k, dominant_show

    def _calculate_extraneous_intrusion(
        self, results: List[SearchResult], k: int
    ) -> tuple[float, List[str]]:
        """
        Calculate extraneous content intrusion rate

        Check if top-k result descriptions have high extraneous scores
        """
        if not results:
            return 0.0, []

        intrusion_episodes = []
        for r in results[:k]:
            if r.description:
                score = self.scorer.score_paragraph(r.description)
                if score.is_extraneous:
                    intrusion_episodes.append(r.episode_id)

        return len(intrusion_episodes) / k, intrusion_episodes

    def _perturb_query(self, query: str) -> List[str]:
        """
        Generate small variations of the query

        Strategies:
        1. Remove one word (if >1 word)
        2. Swap order (if >=2 words)
        3. Add synonym prefix
        """
        words = query.split()
        perturbations = []

        # Remove one word
        if len(words) > 1:
            for i in range(min(len(words), 2)):  # At most 2 variations by removing words
                perturbations.append(" ".join(words[:i] + words[i + 1 :]))

        # Swap order
        if len(words) >= 2:
            perturbations.append(" ".join(reversed(words)))

        return perturbations[:3]  # At most 3 variations

    def _calculate_perturbation_stability(
        self,
        query: str,
        original_ids: Set[str],
        k: int,
    ) -> float:
        """
        Calculate query perturbation stability

        Make small changes to query and measure result stability (Jaccard similarity)
        """
        perturbed_queries = self._perturb_query(query)

        if not perturbed_queries:
            return 1.0  # No perturbations possible, assume stable

        stability_scores = []
        for pq in perturbed_queries:
            try:
                pq_response = self.search.search_hybrid(pq, size=k)
                pq_ids = self._get_result_ids(pq_response.results)
                sim = self._jaccard_similarity(original_ids, pq_ids)
                stability_scores.append(sim)
            except Exception as e:
                logger.warning(
                    "perturbation_search_failed",
                    extra={"query": pq, "error": str(e)},
                )
                # Treat failed search as completely different results
                stability_scores.append(0.0)

        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0

    def evaluate_query(
        self,
        query: str,
        k: int = 10,
        include_debug: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate search quality for a single query

        Args:
            query: Search query
            k: Top-K result count
            include_debug: Whether to include debug info

        Returns:
            EvaluationResult with 4 metrics
        """
        # Execute search
        response = self.search.search_hybrid(query, size=k)
        results = response.results
        result_ids = self._get_result_ids(results)

        # 1. Top-K Overlap (placeholder - needs before/after comparison)
        # For now, we'll set this to 1.0 (same index)
        # In actual use, you'd compare against a "before" index
        top_k_overlap = 1.0

        # 2. Same-Podcast Dominance
        dominance, dominant_show = self._calculate_same_podcast_dominance(results, k)

        # 3. Extraneous Intrusion
        intrusion, intrusion_episodes = self._calculate_extraneous_intrusion(results, k)

        # 4. Perturbation Stability
        stability = self._calculate_perturbation_stability(query, result_ids, k)

        result = EvaluationResult(
            query=query,
            top_k_overlap=top_k_overlap,
            same_podcast_dominance=dominance,
            extraneous_intrusion=intrusion,
            perturbation_stability=stability,
        )

        if include_debug:
            result.after_ids = list(result_ids)
            result.dominant_show_id = dominant_show
            result.intrusion_episodes = intrusion_episodes

        logger.info(
            "query_evaluated",
            extra={
                "query": query,
                "dominance": round(dominance, 3),
                "intrusion": round(intrusion, 3),
                "stability": round(stability, 3),
            },
        )

        return result

    def evaluate_query_with_before_after(
        self,
        query: str,
        before_results: List[SearchResult],
        after_results: List[SearchResult],
        k: int = 10,
        include_debug: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate before/after search quality difference for a single query

        Args:
            query: Search query
            before_results: Search results before cleaning
            after_results: Search results after cleaning
            k: Top-K result count
            include_debug: Whether to include debug info

        Returns:
            EvaluationResult with 4 metrics
        """
        before_ids = self._get_result_ids(before_results[:k])
        after_ids = self._get_result_ids(after_results[:k])

        # 1. Top-K Overlap (Jaccard)
        top_k_overlap = self._jaccard_similarity(before_ids, after_ids)

        # 2. Same-Podcast Dominance (use after results)
        dominance, dominant_show = self._calculate_same_podcast_dominance(
            after_results, k
        )

        # 3. Extraneous Intrusion (use after results)
        intrusion, intrusion_episodes = self._calculate_extraneous_intrusion(
            after_results, k
        )

        # 4. Perturbation Stability (use after results)
        stability = self._calculate_perturbation_stability(query, after_ids, k)

        result = EvaluationResult(
            query=query,
            top_k_overlap=top_k_overlap,
            same_podcast_dominance=dominance,
            extraneous_intrusion=intrusion,
            perturbation_stability=stability,
        )

        if include_debug:
            result.before_ids = list(before_ids)
            result.after_ids = list(after_ids)
            result.dominant_show_id = dominant_show
            result.intrusion_episodes = intrusion_episodes

        return result

    def aggregate_results(
        self, results: List[EvaluationResult]
    ) -> AggregatedMetrics:
        """
        Aggregate evaluation results from multiple queries

        Args:
            results: List of EvaluationResult

        Returns:
            AggregatedMetrics with averages and assessment
        """
        if not results:
            return AggregatedMetrics(
                total_queries=0,
                avg_top_k_overlap=0.0,
                avg_same_podcast_dominance=0.0,
                avg_extraneous_intrusion=0.0,
                avg_perturbation_stability=0.0,
                cleaning_effective=False,
                ranking_stable=False,
                no_show_dominance=False,
            )

        n = len(results)
        avg_overlap = sum(r.top_k_overlap for r in results) / n
        avg_dominance = sum(r.same_podcast_dominance for r in results) / n
        avg_intrusion = sum(r.extraneous_intrusion for r in results) / n
        avg_stability = sum(r.perturbation_stability for r in results) / n

        return AggregatedMetrics(
            total_queries=n,
            avg_top_k_overlap=round(avg_overlap, 4),
            avg_same_podcast_dominance=round(avg_dominance, 4),
            avg_extraneous_intrusion=round(avg_intrusion, 4),
            avg_perturbation_stability=round(avg_stability, 4),
            cleaning_effective=avg_intrusion < 0.1,
            ranking_stable=avg_stability > 0.7,
            no_show_dominance=avg_dominance < 0.5,
        )

    def to_dict(self, result: EvaluationResult) -> dict:
        """Convert EvaluationResult to dict"""
        return asdict(result)

    def aggregate_to_dict(self, agg: AggregatedMetrics) -> dict:
        """Convert AggregatedMetrics to dict"""
        return asdict(agg)
