"""
Tests for ranking_metrics module.

Tests verify correct computation of NDCG@k and MRR metrics.
"""

import pytest

from src.evaluation.ranking_metrics import dcg_at_k, mrr, ndcg_at_k


class TestDCG:
    """Tests for DCG computation."""

    def test_dcg_basic(self):
        """DCG with known values."""
        # [3, 2, 1] at positions 1, 2, 3
        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        # DCG = 3/1 + 2/1.585 + 1/2 = 3 + 1.262 + 0.5 = 4.762
        relevances = [3, 2, 1]
        dcg = dcg_at_k(relevances, k=3)
        assert 4.7 < dcg < 4.8

    def test_dcg_empty(self):
        """DCG of empty list is 0."""
        assert dcg_at_k([], k=10) == 0.0

    def test_dcg_k_larger_than_list(self):
        """DCG with k larger than list uses all items."""
        relevances = [3, 2]
        dcg_k2 = dcg_at_k(relevances, k=2)
        dcg_k10 = dcg_at_k(relevances, k=10)
        assert dcg_k2 == dcg_k10


class TestNDCG:
    """Tests for NDCG computation."""

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        # Already sorted descending = perfect
        relevances = [3, 2, 1, 0]
        assert ndcg_at_k(relevances, k=4) == 1.0

    def test_ndcg_worst_ranking(self):
        """Worst ranking (reversed) should give NDCG < 1.0."""
        # Worst case: best item at the end
        relevances = [0, 1, 2, 3]
        ndcg = ndcg_at_k(relevances, k=4)
        assert ndcg < 1.0
        assert ndcg > 0.0  # Still has relevant items

    def test_ndcg_no_relevant_items(self):
        """NDCG with no relevant items should be 0.0."""
        relevances = [0, 0, 0, 0]
        assert ndcg_at_k(relevances, k=4) == 0.0

    def test_ndcg_empty_list(self):
        """NDCG of empty list should be 0.0."""
        assert ndcg_at_k([], k=10) == 0.0

    def test_ndcg_single_relevant_at_top(self):
        """Single relevant item at top should give NDCG = 1.0."""
        relevances = [3, 0, 0, 0]
        assert ndcg_at_k(relevances, k=4) == 1.0

    def test_ndcg_bounded(self):
        """NDCG should always be between 0 and 1."""
        import random

        for _ in range(100):
            relevances = [random.randint(0, 3) for _ in range(10)]
            ndcg = ndcg_at_k(relevances, k=10)
            assert 0.0 <= ndcg <= 1.0


class TestMRR:
    """Tests for MRR computation."""

    def test_mrr_first_relevant(self):
        """Relevant item at position 1 gives MRR = 1.0."""
        relevances = [3, 0, 0, 0]
        assert mrr(relevances, threshold=2) == 1.0

    def test_mrr_second_relevant(self):
        """Relevant item at position 2 gives MRR = 0.5."""
        relevances = [1, 2, 0, 0]  # rel=1 doesn't count with threshold=2
        assert mrr(relevances, threshold=2) == 0.5

    def test_mrr_third_relevant(self):
        """Relevant item at position 3 gives MRR = 1/3."""
        relevances = [1, 1, 2]
        assert mrr(relevances, threshold=2) == pytest.approx(1 / 3)

    def test_mrr_no_relevant(self):
        """No relevant items gives MRR = 0.0."""
        relevances = [1, 1, 1, 0]  # None meets threshold=2
        assert mrr(relevances, threshold=2) == 0.0

    def test_mrr_threshold_effect(self):
        """Threshold affects what counts as relevant."""
        relevances = [1, 2, 3]

        # threshold=1: position 1 is relevant
        assert mrr(relevances, threshold=1) == 1.0

        # threshold=2: position 2 is first relevant
        assert mrr(relevances, threshold=2) == 0.5

        # threshold=3: position 3 is first relevant
        assert mrr(relevances, threshold=3) == pytest.approx(1 / 3)

    def test_mrr_empty_list(self):
        """MRR of empty list is 0.0."""
        assert mrr([], threshold=2) == 0.0
