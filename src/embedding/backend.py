"""Embedding backend abstraction.

Pattern: Strategy — swap between local model (index-time batch) and external
API (query-time) without changing the ingest pipeline or search service.

IMPORTANT: Index-time and query-time MUST use the same backend per language.
Mixing backends produces incomparable vector spaces and silently degrades kNN
quality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx
from sentence_transformers import SentenceTransformer

from src.types import Language

logger = logging.getLogger(__name__)

_MODEL_MAP: dict[str, str] = {
    "zh": "BAAI/bge-base-zh-v1.5",                    # 768 dim, zh-tw + zh-cn
    "en": "paraphrase-multilingual-MiniLM-L12-v2",    # 384 dim
}


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model (once per process).

    Uses lru_cache so repeated calls with the same model_name return the
    already-loaded instance without re-downloading or re-initialising.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer instance.
    """
    logger.info("loading_model", extra={"model": model_name})
    return SentenceTransformer(model_name)


class EmbeddingBackend(ABC):
    """Abstract base for all embedding backends.

    Pattern: Strategy — swap between local model and external API without
    changing the ingest pipeline or search service.
    """

    @abstractmethod
    def embed(self, text: str, language: Language) -> list[float]:
        """Encode a single text into an embedding vector.

        Args:
            text:     The text to encode (query string or document field).
            language: Language of the text, used to select the correct model.

        Returns:
            Embedding vector as a list of floats.
        """
        ...

    def embed_batch(self, texts: list[str], language: Language) -> list[list[float]]:
        """Encode multiple texts of the same language.

        Default implementation calls embed() for each text.
        Override this method for batch-efficient backends (e.g. LocalEmbeddingBackend).

        Args:
            texts:    List of texts to encode. All texts must share the same language.
            language: Language of all texts in this batch.

        Returns:
            List of embedding vectors, one per input text, in the same order.
        """
        return [self.embed(t, language) for t in texts]


# ── Local Backend (index-time / offline batch) ────────────────────────────────


class LocalEmbeddingBackend(EmbeddingBackend):
    """Runs embedding locally using SentenceTransformer.

    Suitable for index-time batch processing (latency-insensitive).
    Models are loaded lazily on first use and cached for the process lifetime
    via _load_model's lru_cache.

    Language routing:
        zh-tw, zh-cn  →  BAAI/bge-base-zh-v1.5   (768 dim)
        en            →  paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
    """

    @staticmethod
    def _model_key(language: Language) -> str:
        return "zh" if language in ("zh-tw", "zh-cn") else "en"

    def embed(self, text: str, language: Language) -> list[float]:
        """Encode a single text.

        Args:
            text:     Text to encode.
            language: Controls model selection.

        Returns:
            Normalised embedding vector as list[float].
        """
        model = _load_model(_MODEL_MAP[self._model_key(language)])
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str], language: Language) -> list[list[float]]:
        """Encode a batch of texts of the same language.

        Passes the full list to model.encode() for efficient batching
        (single forward pass vs. N separate calls).

        Args:
            texts:    Texts to encode. Must all share the same language.
            language: Controls model selection.

        Returns:
            List of embedding vectors in the same order as texts.
        """
        model = _load_model(_MODEL_MAP[self._model_key(language)])
        return model.encode(texts, normalize_embeddings=True).tolist()


# ── API Backend (query-time / external service) ───────────────────────────────


class EmbeddingFallbackError(Exception):
    """Raised when the external embedding API is unavailable.

    Caller should degrade gracefully to BM25-only search.
    """


class APIEmbeddingBackend(EmbeddingBackend):
    """Calls an external embedding API for query-time embedding.

    Suitable for query-time use — removes the need for a resident model in the
    VM. Caller must handle EmbeddingFallbackError and degrade to BM25-only
    search when the API is unavailable.

    Args:
        api_url: External embedding endpoint URL.
        api_key: API authentication key (sent as Bearer token).
        timeout: Request timeout in seconds (default 2.0).
    """

    def __init__(self, api_url: str, api_key: str, timeout: float = 2.0) -> None:
        self._api_url = api_url
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def __del__(self) -> None:
        self._client.close()

    def embed(self, text: str, language: Language) -> list[float]:
        """Call the external API to embed a single text.

        Args:
            text:     Text to encode.
            language: Passed to the API so it can select the right model.

        Returns:
            Embedding vector from the API response.

        Raises:
            EmbeddingFallbackError: When the API is unreachable, times out,
                or returns a response without an 'embedding' key.
        """
        try:
            resp = self._client.post(
                self._api_url,
                json={"text": text, "language": language},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (httpx.HTTPError, KeyError) as exc:
            logger.error("embedding_api_failed", extra={"error": str(exc)})
            raise EmbeddingFallbackError(str(exc)) from exc

    def embed_batch(self, texts: list[str], language: Language) -> list[list[float]]:
        # TODO: implement true batch request once external API format is confirmed.
        # OpenAI-compatible: {"input": ["t1", "t2", ...]} → {"data": [{"embedding": [...]}]}
        return super().embed_batch(texts, language)


# ── Query-time LRU cache (works with any backend) ─────────────────────────────


@lru_cache(maxsize=1000)
def embed_query_cached(
    backend: EmbeddingBackend,
    query: str,
    language: Language,
) -> tuple[float, ...]:
    """Embed a search query with a process-level LRU cache.

    Returns a tuple (hashable, required for lru_cache).
    Convert to list before passing to ES: list(embed_query_cached(...))

    Cache is process-scoped and cleared on restart.
    Hit rate is observable via embed_query_cached.cache_info().

    Args:
        backend:  Embedding backend to use. Cache key includes the backend
                  instance, so switching backends bypasses cached entries.
        query:    Search query string.
        language: Query language, passed to backend.embed().

    Returns:
        Embedding vector as an immutable tuple of floats.
    """
    return tuple(backend.embed(query, language))
