"""Unit tests for EmbeddingBackend implementations.

Covers:
- LocalEmbeddingBackend: model selection by language, single/batch embedding
- APIEmbeddingBackend: happy path, fallback errors, request body format
- embed_query_cached: return type (tuple), LRU caching behaviour
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from src.embedding.backend import (
    APIEmbeddingBackend,
    EmbeddingBackend,
    EmbeddingFallbackError,
    LocalEmbeddingBackend,
    MODEL_MAP,
    embed_query_cached,
)


# ── LocalEmbeddingBackend ─────────────────────────────────────────────────────


class TestLocalEmbeddingBackend:
    def _make_mock_model(self, vector: list[float] | None = None) -> MagicMock:
        mock = MagicMock()
        vec = vector or [0.1, 0.2, 0.3]
        # encode() returns a mock whose .tolist() gives the vector
        mock.encode.return_value = MagicMock(tolist=lambda: vec)
        return mock

    def test_zh_tw_uses_zh_model(self) -> None:
        """zh-tw should select the zh model (bge-zh), not the en model."""
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_load.return_value = self._make_mock_model()
            backend = LocalEmbeddingBackend()
            backend.embed("podcast title", "zh-tw")

        mock_load.assert_called_once_with(MODEL_MAP["zh"])

    def test_zh_cn_uses_same_model_as_zh_tw(self) -> None:
        """zh-cn and zh-tw must share the same underlying model."""
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_load.return_value = self._make_mock_model()
            backend = LocalEmbeddingBackend()
            backend.embed("tw text", "zh-tw")
            backend.embed("cn text", "zh-cn")

        calls = mock_load.call_args_list
        assert calls[0] == calls[1], "zh-tw and zh-cn must call the same model"

    def test_en_uses_different_model_from_zh(self) -> None:
        """en language must use a different model than zh-tw/zh-cn."""
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_load.return_value = self._make_mock_model()
            backend = LocalEmbeddingBackend()
            backend.embed("zh text", "zh-tw")
            backend.embed("en text", "en")

        zh_model = mock_load.call_args_list[0].args[0]
        en_model = mock_load.call_args_list[1].args[0]
        assert zh_model != en_model

    def test_embed_returns_list_of_floats(self) -> None:
        """embed() must return list[float], not np.ndarray."""
        expected = [0.1, 0.2, 0.3]
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_load.return_value = self._make_mock_model(expected)
            backend = LocalEmbeddingBackend()
            result = backend.embed("test text", "en")

        assert isinstance(result, list)
        assert result == expected

    def test_embed_batch_returns_one_vector_per_text(self) -> None:
        """embed_batch() must return a vector for every input text."""
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(
                tolist=lambda: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            )
            mock_load.return_value = mock_model
            backend = LocalEmbeddingBackend()
            results = backend.embed_batch(["text a", "text b", "text c"], "en")

        assert len(results) == 3

    def test_embed_batch_uses_same_model_as_embed_for_zh(self) -> None:
        """embed_batch with zh-tw must load the same model as embed with zh-tw."""
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]])
            mock_load.return_value = mock_model
            backend = LocalEmbeddingBackend()
            backend.embed_batch(["text"], "zh-tw")

        mock_load.assert_called_once_with(MODEL_MAP["zh"])

    def test_embed_batch_passes_list_to_model_encode(self) -> None:
        """embed_batch must call model.encode with the full list (batch efficiency)."""
        texts = ["first", "second", "third"]
        with patch("src.embedding.backend._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]] * 3)
            mock_load.return_value = mock_model
            backend = LocalEmbeddingBackend()
            backend.embed_batch(texts, "en")

        # encode() must be called with the full list, not one text at a time
        mock_model.encode.assert_called_once()
        first_arg = mock_model.encode.call_args.args[0]
        assert first_arg == texts


# ── APIEmbeddingBackend ───────────────────────────────────────────────────────


_MODEL_ZH = "BAAI/bge-base-zh-v1.5"
_MODEL_EN = "paraphrase-multilingual-MiniLM-L12-v2"


class TestAPIEmbeddingBackend:
    _URL = "http://api.example/v1/embeddings"
    _KEY = "test-key"

    def _make_backend(self) -> APIEmbeddingBackend:
        """Return an APIEmbeddingBackend with a mocked httpx.Client."""
        with patch("src.embedding.backend.httpx.Client"):
            return APIEmbeddingBackend(
                api_url=self._URL, api_key=self._KEY,
                model_zh=_MODEL_ZH, model_en=_MODEL_EN,
            )

    def _mock_response(self, body: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = body
        resp.raise_for_status.return_value = None
        return resp

    def _openai_body(self, embeddings: list) -> dict:
        return {
            "data": [{"index": i, "embedding": e} for i, e in enumerate(embeddings)]
        }

    # -- connection pooling --

    def test_has_client_attribute(self) -> None:
        """APIEmbeddingBackend must create and store an httpx.Client instance."""
        with patch("src.embedding.backend.httpx.Client") as mock_client_cls:
            backend = APIEmbeddingBackend(
                api_url=self._URL, api_key=self._KEY,
                model_zh=_MODEL_ZH, model_en=_MODEL_EN,
            )
        assert backend._client is mock_client_cls.return_value

    def test_client_initialized_with_bearer_auth_header(self) -> None:
        """httpx.Client must be created with Authorization: Bearer <key> header."""
        with patch("src.embedding.backend.httpx.Client") as mock_client_cls:
            APIEmbeddingBackend(
                api_url=self._URL, api_key=self._KEY,
                model_zh=_MODEL_ZH, model_en=_MODEL_EN,
            )
        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == f"Bearer {self._KEY}"

    def test_client_initialized_with_configured_timeout(self) -> None:
        """httpx.Client must be created with the configured timeout."""
        with patch("src.embedding.backend.httpx.Client") as mock_client_cls:
            APIEmbeddingBackend(
                api_url=self._URL, api_key=self._KEY,
                model_zh=_MODEL_ZH, model_en=_MODEL_EN, timeout=5.0,
            )
        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["timeout"] == 5.0

    def test_embed_uses_client_post(self) -> None:
        """embed() must call self._client.post, not httpx.post directly."""
        backend = self._make_backend()
        backend._client.post.return_value = self._mock_response(self._openai_body([[0.1]]))
        backend.embed("text", "en")
        backend._client.post.assert_called_once()

    # -- functional correctness --

    def test_returns_embedding_on_success(self) -> None:
        """A 200 response must return the embedding vector from data[0]."""
        backend = self._make_backend()
        expected = [0.1, 0.2, 0.3]
        backend._client.post.return_value = self._mock_response(self._openai_body([expected]))
        result = backend.embed("search query", "zh-tw")
        assert result == expected

    def test_sends_model_and_input_in_body(self) -> None:
        """POST body must use OpenAI format: 'model' and 'input' (no 'language' field)."""
        backend = self._make_backend()
        backend._client.post.return_value = self._mock_response(self._openai_body([[0.1]]))
        backend.embed("query text", "zh-cn")
        body = backend._client.post.call_args.kwargs["json"]
        assert body["model"] == _MODEL_ZH
        assert body["input"] == "query text"
        assert "language" not in body

    def test_raises_fallback_error_on_timeout(self) -> None:
        """httpx.TimeoutException must be wrapped in EmbeddingFallbackError immediately."""
        backend = self._make_backend()
        backend._client.post.side_effect = httpx.TimeoutException("timed out")
        with pytest.raises(EmbeddingFallbackError):
            backend.embed("test", "en")

    def test_raises_fallback_error_when_data_key_missing(self) -> None:
        """Missing 'data' key in response must raise EmbeddingFallbackError."""
        backend = self._make_backend()
        backend._client.post.return_value = self._mock_response({"result": "ok"})
        with pytest.raises(EmbeddingFallbackError):
            backend.embed("test", "en")


# ── embed_query_cached ────────────────────────────────────────────────────────


class TestEmbedQueryCached:
    def setup_method(self) -> None:
        embed_query_cached.cache_clear()

    def test_returns_tuple(self) -> None:
        """embed_query_cached must return a tuple (required for lru_cache hashability)."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [0.1, 0.2, 0.3]

        result = embed_query_cached(mock_backend, "search query", "en")

        assert isinstance(result, tuple)
        assert result == (0.1, 0.2, 0.3)

    def test_caches_repeated_calls_with_same_args(self) -> None:
        """Same backend + query + language should call backend.embed() only once."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        mock_backend.embed.return_value = [0.1, 0.2]

        embed_query_cached(mock_backend, "same query", "en")
        embed_query_cached(mock_backend, "same query", "en")

        assert mock_backend.embed.call_count == 1

    def test_different_language_produces_separate_cache_entry(self) -> None:
        """Same text with different language must not share a cache entry."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        mock_backend.embed.side_effect = [[0.1], [0.9]]

        embed_query_cached(mock_backend, "query", "zh-tw")
        embed_query_cached(mock_backend, "query", "en")

        assert mock_backend.embed.call_count == 2

    def test_different_query_produces_separate_cache_entry(self) -> None:
        """Different queries must not share a cache entry."""
        mock_backend = MagicMock(spec=EmbeddingBackend)
        mock_backend.embed.side_effect = [[0.1], [0.2]]

        embed_query_cached(mock_backend, "query one", "en")
        embed_query_cached(mock_backend, "query two", "en")

        assert mock_backend.embed.call_count == 2
