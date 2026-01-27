"""
Search Evaluation Pipeline

Execute search quality evaluation and generate reports.

Usage:
    python -m src.pipelines.evaluate_search
    python -m src.pipelines.evaluate_search --queries data/test_queries.txt
    python -m src.pipelines.evaluate_search --output data/evaluation_report.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.evaluation.extraneous_scorer import ExtraneousScorer
from src.evaluation.metrics import NoAnnotationEvaluator, EvaluationResult, AggregatedMetrics
from src.services.search_service import SearchService
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Default test queries (various types)
DEFAULT_TEST_QUERIES = [
    # Chinese - Topic queries
    "科技新聞",
    "投資理財",
    "學英文",
    "創業故事",
    "心理健康",
    # Chinese - Entity queries
    "台灣政治",
    "AI 人工智慧",
    "股票分析",
    # English - Topic queries
    "technology podcast",
    "business news",
    "self improvement",
    "true crime",
    "comedy podcast",
    # English - Entity queries
    "artificial intelligence",
    "startup founders",
    "book recommendations",
    # Mixed
    "podcast 推薦",
    "最新科技",
    "programming",
    "machine learning",
    # Long-tail queries
    "如何開始投資",
    "英文學習方法",
    "創業經驗分享",
    "how to start a business",
    "best podcasts for learning",
    # Edge cases
    "2024",
    "interview",
    "story",
    "news",
    "AI",
]


class EvaluationPipeline:
    """Search quality evaluation pipeline"""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        extraneous_scorer: Optional[ExtraneousScorer] = None,
        k: int = 10,
    ):
        self.search = search_service or SearchService()
        self.scorer = extraneous_scorer or ExtraneousScorer()
        self.evaluator = NoAnnotationEvaluator(self.search, self.scorer)
        self.k = k

    def load_queries(self, query_file: Optional[Path] = None) -> List[str]:
        """
        Load test queries

        Args:
            query_file: File containing queries (one query per line)

        Returns:
            List of queries
        """
        if query_file and query_file.exists():
            with open(query_file, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
            logger.info(
                "queries_loaded_from_file",
                extra={"file": str(query_file), "count": len(queries)},
            )
            return queries

        logger.info(
            "using_default_queries",
            extra={"count": len(DEFAULT_TEST_QUERIES)},
        )
        return DEFAULT_TEST_QUERIES

    def run(
        self,
        queries: Optional[List[str]] = None,
        query_file: Optional[Path] = None,
        include_debug: bool = False,
    ) -> dict:
        """
        Execute evaluation

        Args:
            queries: List of queries (if None, load from query_file or use defaults)
            query_file: Query file path
            include_debug: Whether to include debug info

        Returns:
            Evaluation report dict
        """
        start_time = datetime.now(timezone.utc)

        # Load queries
        if queries is None:
            queries = self.load_queries(query_file)

        logger.info(
            "evaluation_start",
            extra={"total_queries": len(queries), "k": self.k},
        )

        # Evaluate each query
        results: List[EvaluationResult] = []
        failed_queries: List[str] = []

        for i, query in enumerate(queries):
            try:
                result = self.evaluator.evaluate_query(
                    query, k=self.k, include_debug=include_debug
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(
                        "evaluation_progress",
                        extra={"completed": i + 1, "total": len(queries)},
                    )

            except Exception as e:
                logger.warning(
                    "query_evaluation_failed",
                    extra={"query": query, "error": str(e)},
                )
                failed_queries.append(query)

        # Aggregate results
        aggregated = self.evaluator.aggregate_results(results)

        end_time = datetime.now(timezone.utc)
        duration_sec = (end_time - start_time).total_seconds()

        # Build report
        report = {
            "meta": {
                "timestamp": end_time.isoformat(),
                "duration_sec": round(duration_sec, 2),
                "k": self.k,
                "total_queries": len(queries),
                "successful_queries": len(results),
                "failed_queries": len(failed_queries),
            },
            "summary": self.evaluator.aggregate_to_dict(aggregated),
            "assessment": {
                "cleaning_effective": aggregated.cleaning_effective,
                "ranking_stable": aggregated.ranking_stable,
                "no_show_dominance": aggregated.no_show_dominance,
                "overall_pass": (
                    aggregated.cleaning_effective
                    and aggregated.ranking_stable
                    and aggregated.no_show_dominance
                ),
            },
            "per_query": [self.evaluator.to_dict(r) for r in results],
        }

        if failed_queries:
            report["failed_queries"] = failed_queries

        logger.info(
            "evaluation_complete",
            extra={
                "successful": len(results),
                "failed": len(failed_queries),
                "duration_sec": round(duration_sec, 2),
                "cleaning_effective": aggregated.cleaning_effective,
                "ranking_stable": aggregated.ranking_stable,
            },
        )

        return report

    def save_report(self, report: dict, output_path: Path) -> None:
        """Save report to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(
            "report_saved",
            extra={"path": str(output_path)},
        )

    def print_summary(self, report: dict) -> None:
        """Print summary"""
        summary = report["summary"]
        assessment = report["assessment"]

        print("\n" + "=" * 60)
        print("Search Quality Evaluation Report")
        print("=" * 60)
        print(f"Total Queries: {report['meta']['total_queries']}")
        print(f"Successful: {report['meta']['successful_queries']}")
        print(f"Duration: {report['meta']['duration_sec']} sec")
        print()
        print("Metrics Summary:")
        print(f"  Top-K Overlap:           {summary['avg_top_k_overlap']:.2%}")
        print(f"  Same-Podcast Dominance:  {summary['avg_same_podcast_dominance']:.2%}")
        print(f"  Extraneous Intrusion:    {summary['avg_extraneous_intrusion']:.2%}")
        print(f"  Perturbation Stability:  {summary['avg_perturbation_stability']:.2%}")
        print()
        print("Assessment:")
        print(f"  Cleaning effective (intrusion < 10%): {'PASS' if assessment['cleaning_effective'] else 'FAIL'}")
        print(f"  Ranking stable (stability > 70%):     {'PASS' if assessment['ranking_stable'] else 'FAIL'}")
        print(f"  No dominance (dominance < 50%):       {'PASS' if assessment['no_show_dominance'] else 'FAIL'}")
        print()
        print(f"Overall: {'PASS' if assessment['overall_pass'] else 'FAIL'}")
        print("=" * 60 + "\n")


def run() -> None:
    """CLI entry point"""
    setup_logging()

    parser = argparse.ArgumentParser(description="Search Quality Evaluation")
    parser.add_argument(
        "--queries",
        type=Path,
        help="Path to query file (one query per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/report.json"),
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K results to evaluate",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include debug info in report",
    )

    args = parser.parse_args()

    pipeline = EvaluationPipeline(k=args.k)
    report = pipeline.run(query_file=args.queries, include_debug=args.debug)

    # Save report
    pipeline.save_report(report, args.output)

    # Print summary
    pipeline.print_summary(report)


if __name__ == "__main__":
    run()
