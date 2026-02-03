"""
Tests for cross_encoder_judge module.

Note: These tests require loading the cross-encoder model,
which may be slow on first run.
"""

import pytest

from src.evaluation.cross_encoder_judge import CrossEncoderJudge


@pytest.fixture(scope="module")
def judge():
    """Shared judge instance to avoid reloading model."""
    return CrossEncoderJudge(seed=42)


class TestScoresToLabels:
    """Tests for label conversion (no model needed)."""

    def test_labels_have_variance(self):
        """Labels should have variance for diverse scores."""
        judge = CrossEncoderJudge.__new__(CrossEncoderJudge)
        # Don't call __init__, just test the method

        scores = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0]
        labels = judge.scores_to_labels(scores)

        # Should have at least 3 different labels
        unique_labels = set(labels)
        assert len(unique_labels) >= 3, f"Expected variance, got {labels}"

    def test_labels_bounded(self):
        """Labels should be between 0 and 3."""
        judge = CrossEncoderJudge.__new__(CrossEncoderJudge)

        scores = [10.0, 5.0, 0.0, -5.0, -10.0, 3.0, 2.0, 1.0, -1.0, -2.0]
        labels = judge.scores_to_labels(scores)

        for label in labels:
            assert 0 <= label <= 3

    def test_small_pool_rank_based(self):
        """Small pools should use rank-based labeling."""
        judge = CrossEncoderJudge.__new__(CrossEncoderJudge)

        # 5 items < 10, should use rank-based
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        labels = judge.scores_to_labels(scores)

        # Top 1 should be 3, 2-3 should be 2, 4-5 should be 1
        assert labels[0] == 3
        assert labels[1] == 2
        assert labels[2] == 2
        assert labels[3] == 1
        assert labels[4] == 1

    def test_empty_scores(self):
        """Empty scores should return empty labels."""
        judge = CrossEncoderJudge.__new__(CrossEncoderJudge)
        assert judge.scores_to_labels([]) == []


class TestCrossEncoderIntegration:
    """Integration tests that require loading the model."""

    @pytest.mark.slow
    def test_score_batch_basic(self, judge):
        """Basic scoring should work."""
        query = "投資理財"
        texts = ["這是一個投資理財的節目", "今天天氣很好"]

        scores = judge.score_batch(query, texts)

        assert len(scores) == 2
        # First text should be more relevant
        assert scores[0] > scores[1]

    @pytest.mark.slow
    def test_deterministic(self, judge):
        """Same input should give same output."""
        judge2 = CrossEncoderJudge(seed=42)

        query = "test query"
        texts = ["document one", "document two"]

        scores1 = judge.score_batch(query, texts)
        scores2 = judge2.score_batch(query, texts)

        # Should be identical with same seed
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 0.01

    @pytest.mark.slow
    def test_judge_returns_both(self, judge):
        """Judge method should return scores and labels."""
        query = "machine learning"
        texts = ["deep learning tutorial", "cooking recipe", "neural networks"]

        results = judge.judge(query, texts)

        assert len(results) == 3
        for r in results:
            assert "score" in r
            assert "relevance" in r
            assert isinstance(r["score"], float)
            assert isinstance(r["relevance"], int)
            assert 0 <= r["relevance"] <= 3
