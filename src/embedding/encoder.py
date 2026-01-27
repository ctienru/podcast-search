"""
Embedding Encoder

Convert text to vectors using sentence-transformers.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Sentence Transformer Embedding Encoder

    Usage:
        encoder = EmbeddingEncoder()
        embedding = encoder.encode("Hello world")
        # embedding.shape = (768,)

        # Batch encoding
        embeddings = encoder.encode_batch(["Hello", "World"])
        # embeddings.shape = (2, 768)
    """

    # Model options:
    # - paraphrase-multilingual-MiniLM-L12-v2: 384 dim, ~120MB (recommended for cloud)
    # - paraphrase-multilingual-mpnet-base-v2: 768 dim, ~400MB (higher quality)
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'mps', 'cpu', or None for auto-detect
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize

        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dim: {self.embedding_dim}, Device: {self.model.device}")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding vector.

        Args:
            text: Input text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embedding

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode multiple texts to embedding vectors.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return embeddings

    def to_list(self, embedding: np.ndarray) -> list[float]:
        """Convert numpy array to list for JSON/ES serialization."""
        return embedding.tolist()
