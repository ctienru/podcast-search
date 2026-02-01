"""
Search Evaluation Module

Provides search quality evaluation tools:
1. ExtraneousScorer - Paragraph-level extraneous content scoring
2. NoAnnotationEvaluator - Annotation-free evaluation metrics
3. EvaluationPipeline - Evaluation pipeline executor
4. Ranking Metrics - NDCG@k, MRR for search quality evaluation
"""

from src.evaluation.extraneous_scorer import ExtraneousScorer, ParagraphScore
from src.evaluation.ranking_metrics import dcg_at_k, mrr, ndcg_at_k

# Lazy import for metrics (requires search_service which needs sentence_transformers)
def __getattr__(name):
    if name == "NoAnnotationEvaluator":
        from src.evaluation.metrics import NoAnnotationEvaluator
        return NoAnnotationEvaluator
    if name == "EvaluationResult":
        from src.evaluation.metrics import EvaluationResult
        return EvaluationResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExtraneousScorer",
    "ParagraphScore",
    "NoAnnotationEvaluator",
    "EvaluationResult",
    "dcg_at_k",
    "ndcg_at_k",
    "mrr",
]
