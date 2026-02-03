"""
Cross-Encoder Judge for Pseudo Relevance Labeling

Uses a cross-encoder model to score (query, document) pairs and convert
scores to relevance labels for evaluation.

Usage:
    from src.evaluation.cross_encoder_judge import CrossEncoderJudge

    judge = CrossEncoderJudge()
    scores = judge.score_batch("投資理財", ["doc1 text", "doc2 text", ...])
    labels = judge.scores_to_labels(scores)  # [3, 2, 1, 0, ...]

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - Outputs relevance scores (higher = more relevant)
    - Good correlation with human judgments
"""

import logging
import random
from typing import List

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default model for relevance judgment
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class CrossEncoderJudge:
    """
    Cross-Encoder based relevance judge.

    Uses a cross-encoder model to score (query, document) pairs
    and convert raw scores to discrete relevance labels (0-3).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        seed: int = 42,
        device: str | None = None,
    ):
        """
        Initialize the cross-encoder judge.

        Args:
            model_name: HuggingFace model name for cross-encoder
            seed: Random seed for reproducibility
            device: Device to use (None for auto-detect)
        """
        set_seed(seed)
        self.model_name = model_name
        self.seed = seed

        logger.info(
            "loading_cross_encoder",
            extra={"model": model_name, "seed": seed},
        )

        self.model = CrossEncoder(model_name, device=device)
        self._score_cache: dict[tuple[str, str], float] = {}

        logger.info(
            "cross_encoder_loaded",
            extra={"device": str(self.model.model.device)},
        )

    def score_batch(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Score a batch of (query, text) pairs.

        Args:
            query: Search query
            texts: List of document texts to score
            batch_size: Batch size for inference

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not texts:
            return []

        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=batch_size)
        return scores.tolist()

    def scores_to_labels(self, scores: List[float]) -> List[int]:
        """
        Convert raw scores to discrete relevance labels (0-3).

        Uses query-normalized percentile-based labeling:
        - 3: >= 80th percentile (highly relevant)
        - 2: >= 50th percentile (relevant)
        - 1: >= 20th percentile (marginally relevant)
        - 0: < 20th percentile (not relevant)

        For small pools (< 10 items), falls back to rank-based assignment
        to ensure label variance.

        Args:
            scores: List of raw relevance scores

        Returns:
            List of integer labels (0-3)
        """
        if not scores:
            return []

        arr = np.array(scores)

        # Small pool (< 10): use rank-based labeling
        if len(arr) < 10:
            order = arr.argsort()[::-1]  # Descending order
            labels = np.zeros(len(arr), dtype=int)
            labels[order[:1]] = 3  # Top 1
            labels[order[1:3]] = 2  # Top 2-3
            labels[order[3:6]] = 1  # Top 4-6
            # Rest stays 0
            return labels.tolist()

        # Normal: percentile-based labeling
        p80, p50, p20 = np.percentile(arr, [80, 50, 20])

        labels = []
        for s in arr:
            if s >= p80:
                labels.append(3)
            elif s >= p50:
                labels.append(2)
            elif s >= p20:
                labels.append(1)
            else:
                labels.append(0)

        return labels

    def judge(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[dict]:
        """
        Score texts and return both scores and labels.

        Args:
            query: Search query
            texts: List of document texts
            batch_size: Batch size for inference

        Returns:
            List of dicts with 'score' and 'relevance' keys
        """
        scores = self.score_batch(query, texts, batch_size=batch_size)
        labels = self.scores_to_labels(scores)

        return [
            {"score": score, "relevance": label}
            for score, label in zip(scores, labels)
        ]
