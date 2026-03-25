"""Embedding backend factory — selects backend from EMBEDDING_STRATEGY env var."""
import os

from src.embedding.backend import APIEmbeddingBackend, EmbeddingBackend, LocalEmbeddingBackend


def create_backend() -> EmbeddingBackend:
    """Create an EmbeddingBackend instance based on EMBEDDING_STRATEGY env var.

    EMBEDDING_STRATEGY=local (default): load local SentenceTransformer model.
    EMBEDDING_STRATEGY=api: call external embedding API (requires EMBEDDING_API_URL
    and EMBEDDING_API_KEY env vars).
    """
    strategy = os.getenv("EMBEDDING_STRATEGY", "local")
    if strategy == "api":
        return APIEmbeddingBackend(
            api_url=os.environ["EMBEDDING_API_URL"],
            api_key=os.environ["EMBEDDING_API_KEY"],
            model_zh=os.getenv("EMBEDDING_MODEL_ZH", "BAAI/bge-base-zh-v1.5"),
            model_en=os.getenv("EMBEDDING_MODEL_EN", "paraphrase-multilingual-MiniLM-L12-v2"),
            timeout=float(os.getenv("EMBEDDING_TIMEOUT_MS", "2000")) / 1000,
        )
    return LocalEmbeddingBackend()
