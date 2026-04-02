"""Embedding backend abstraction.

Pattern: Strategy — swap between local model (index-time batch) and external
API (query-time) without changing the ingest pipeline or search service.

IMPORTANT: Index-time and query-time MUST use the same backend per language.
Mixing backends produces incomparable vector spaces and silently degrades kNN
quality.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx
from sentence_transformers import SentenceTransformer

from src.types import Language

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "zh": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim, zh-tw + zh-cn
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
        zh-tw, zh-cn  →  paraphrase-multilingual-MiniLM-L12-v2   (384 dim)
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
        model = _load_model(MODEL_MAP[self._model_key(language)])
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
        model = _load_model(MODEL_MAP[self._model_key(language)])
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


# ── API Backend (query-time / external service) ───────────────────────────────


class EmbeddingFallbackError(Exception):
    """Raised when the external embedding API is unavailable.

    Caller should degrade gracefully to BM25-only search.
    """


class APIEmbeddingBackend(EmbeddingBackend):
    """Calls an OpenAI-compatible external embedding API for query-time embedding.

    Suitable for query-time use — removes the need for a resident model in the
    VM. Caller must handle EmbeddingFallbackError and degrade to BM25-only
    search when the API is unavailable.

    embed() retries up to 3 times with exponential backoff on transient errors
    (5xx, network errors). 401/403 and timeouts fail immediately.

    embed_batch() is fail-fast — no retry, because retrying a large batch is
    too expensive. Language consistency across texts is the caller's responsibility
    (batch_encode() in the ingest pipeline groups by language before calling this).

    Args:
        api_url:  External embedding endpoint URL (must accept POST /v1/embeddings).
        api_key:  API authentication key (sent as Bearer token).
        model_zh: Model name for Chinese text (zh-tw / zh-cn).
        model_en: Model name for English text.
        timeout:  Request timeout in seconds (default 2.0).
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_zh: str,
        model_en: str,
        timeout: float = 2.0,
    ) -> None:
        self._api_url = api_url
        self._model_map = {"zh": model_zh, "en": model_en}
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def __del__(self) -> None:
        self._client.close()

    @staticmethod
    def _model_key(language: Language) -> str:
        return "zh" if language in ("zh-tw", "zh-cn") else "en"

    def embed(self, text: str, language: Language) -> list[float]:
        """Call the external API to embed a single text, with retry.

        Retries up to 3 times on transient failures (5xx, network errors).
        Fails immediately on 401/403 or timeout.

        Args:
            text:     Text to encode.
            language: Controls model selection (zh-tw/zh-cn → model_zh, en → model_en).

        Returns:
            Embedding vector from the API response.

        Raises:
            EmbeddingFallbackError: On auth failure, timeout, or exhausted retries.
        """
        model = self._model_map[self._model_key(language)]
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.post(self._api_url, json={"model": model, "input": text})
                if resp.status_code in (401, 403):
                    raise EmbeddingFallbackError(f"Auth failed: HTTP {resp.status_code}")
                if resp.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                data = sorted(resp.json()["data"], key=lambda x: x["index"])
                return data[0]["embedding"]
            except EmbeddingFallbackError:
                raise
            except httpx.TimeoutException as exc:
                raise EmbeddingFallbackError("Timeout") from exc
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
        raise EmbeddingFallbackError(f"Failed after 3 attempts: {last_exc}") from last_exc

    def embed_batch(self, texts: list[str], language: Language) -> list[list[float]]:
        """Call the external API to embed a batch of texts (fail-fast, no retry).

        Args:
            texts:    Texts to encode. Must all share the same language.
            language: Controls model selection.

        Returns:
            List of embedding vectors sorted by index, in the same order as texts.

        Raises:
            EmbeddingFallbackError: On any error (auth failure, timeout, HTTP error).
        """
        model = self._model_map[self._model_key(language)]
        try:
            resp = self._client.post(self._api_url, json={"model": model, "input": texts})
            if resp.status_code in (401, 403):
                raise EmbeddingFallbackError(f"Auth failed: HTTP {resp.status_code}")
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except EmbeddingFallbackError:
            raise
        except httpx.TimeoutException as exc:
            raise EmbeddingFallbackError("Batch timeout") from exc
        except (httpx.HTTPError, KeyError) as exc:
            raise EmbeddingFallbackError(f"Batch failed: {exc}") from exc


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
