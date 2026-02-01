"""
Ranking Metrics for Search Evaluation

Implements NDCG@k and MRR for evaluating search result quality.

Usage:
    from src.evaluation.ranking_metrics import ndcg_at_k, mrr

    relevances = [3, 2, 1, 0, 0]  # relevance scores for top-k results
    print(ndcg_at_k(relevances, k=5))  # 0.0 - 1.0
    print(mrr(relevances, threshold=2))  # 0.0 - 1.0
"""

import math
from typing import List


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at position k.

    DCG = sum(rel_i / log2(i + 2)) for i in 0..k-1

    Args:
        relevances: List of relevance scores (0-3) in rank order
        k: Number of positions to consider

    Returns:
        DCG score (unbounded, depends on relevance scale)
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at position k.

    NDCG = DCG / IDCG (Ideal DCG)

    Args:
        relevances: List of relevance scores (0-3) in rank order
        k: Number of positions to consider

    Returns:
        NDCG score between 0.0 and 1.0
        - 1.0 = perfect ranking (best items at top)
        - 0.0 = no relevant items or empty list
    """
    dcg = dcg_at_k(relevances, k)
    # Ideal DCG: sort relevances descending (best possible ranking)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def mrr(relevances: List[int], threshold: int = 2) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant_item

    Args:
        relevances: List of relevance scores in rank order
        threshold: Minimum relevance score to count as "relevant"
            Default 2 means rel=0,1 are not relevant, rel=2,3 are relevant

    Returns:
        MRR score between 0.0 and 1.0
        - 1.0 = first result is relevant
        - 0.5 = second result is first relevant
        - 0.0 = no relevant results
    """
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0
