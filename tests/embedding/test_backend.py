"""Tests for APIEmbeddingBackend."""

import httpx
import pytest

from src.embedding.backend import APIEmbeddingBackend, EmbeddingFallbackError

MODEL_ZH = "BAAI/bge-base-zh-v1.5"
MODEL_EN = "paraphrase-multilingual-MiniLM-L12-v2"

API_URL = "http://test-api/v1/embeddings"


def make_backend() -> APIEmbeddingBackend:
    return APIEmbeddingBackend(
        api_url=API_URL,
        api_key="test-key",
        model_zh=MODEL_ZH,
        model_en=MODEL_EN,
    )


def make_response(mocker, status: int = 200, data: dict | None = None):
    resp = mocker.MagicMock()
    resp.status_code = status
    resp.request = mocker.MagicMock()
    if data is not None:
        resp.json.return_value = data
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=resp.request, response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def success_data(embeddings: list[list[float]], model: str = MODEL_ZH) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": e}
            for i, e in enumerate(embeddings)
        ],
        "model": model,
    }


# ── embed() ───────────────────────────────────────────────────────────────────

class TestEmbed:
    def test_zh_sends_correct_model_and_format(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, data=success_data([[0.1, 0.2, 0.3]]))
        mocker.patch.object(backend._client, "post", return_value=resp)

        result = backend.embed("人工智慧", "zh-tw")

        call_json = backend._client.post.call_args.kwargs["json"]
        assert call_json["model"] == MODEL_ZH
        assert call_json["input"] == "人工智慧"
        assert "language" not in call_json, "Must not send old 'language' field"
        assert result == [0.1, 0.2, 0.3]

    def test_zh_cn_uses_zh_model(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, data=success_data([[0.1]]))
        mocker.patch.object(backend._client, "post", return_value=resp)

        backend.embed("人工智慧", "zh-cn")

        assert backend._client.post.call_args.kwargs["json"]["model"] == MODEL_ZH

    def test_en_sends_correct_model(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, data=success_data([[0.5, 0.6]], model=MODEL_EN))
        mocker.patch.object(backend._client, "post", return_value=resp)

        backend.embed("machine learning", "en")

        assert backend._client.post.call_args.kwargs["json"]["model"] == MODEL_EN

    def test_sorts_by_index(self, mocker):
        backend = make_backend()
        out_of_order = {
            "data": [
                {"index": 0, "embedding": [1.0, 2.0, 3.0]},
            ]
        }
        resp = make_response(mocker, data=out_of_order)
        mocker.patch.object(backend._client, "post", return_value=resp)

        result = backend.embed("test", "en")
        assert result == [1.0, 2.0, 3.0]

    def test_401_raises_immediately_no_retry(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, status=401)
        mocker.patch.object(backend._client, "post", return_value=resp)

        with pytest.raises(EmbeddingFallbackError, match="Auth failed"):
            backend.embed("test", "zh-tw")

        assert backend._client.post.call_count == 1

    def test_403_raises_immediately_no_retry(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, status=403)
        mocker.patch.object(backend._client, "post", return_value=resp)

        with pytest.raises(EmbeddingFallbackError, match="Auth failed"):
            backend.embed("test", "zh-tw")

        assert backend._client.post.call_count == 1

    def test_500_retries_3_times_then_raises(self, mocker):
        mocker.patch("time.sleep")
        backend = make_backend()
        resp = make_response(mocker, status=500)
        mocker.patch.object(backend._client, "post", return_value=resp)

        with pytest.raises(EmbeddingFallbackError):
            backend.embed("test", "zh-tw")

        assert backend._client.post.call_count == 3

    def test_timeout_raises_immediately_no_retry(self, mocker):
        backend = make_backend()
        mocker.patch.object(
            backend._client, "post",
            side_effect=httpx.TimeoutException("timeout"),
        )

        with pytest.raises(EmbeddingFallbackError, match="Timeout"):
            backend.embed("test", "zh-tw")

        assert backend._client.post.call_count == 1


# ── embed_batch() ─────────────────────────────────────────────────────────────

class TestEmbedBatch:
    def test_sends_list_input(self, mocker):
        backend = make_backend()
        texts = ["text one", "text two"]
        resp = make_response(mocker, data=success_data([[0.1], [0.2]], model=MODEL_EN))
        mocker.patch.object(backend._client, "post", return_value=resp)

        backend.embed_batch(texts, "en")

        call_json = backend._client.post.call_args.kwargs["json"]
        assert call_json["input"] == texts
        assert call_json["model"] == MODEL_EN

    def test_sorts_results_by_index(self, mocker):
        backend = make_backend()
        reversed_data = {
            "data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]},
            ]
        }
        resp = make_response(mocker, data=reversed_data)
        mocker.patch.object(backend._client, "post", return_value=resp)

        result = backend.embed_batch(["a", "b"], "zh-tw")
        assert result == [[0.1], [0.2]]

    def test_500_fails_fast_no_retry(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, status=500)
        mocker.patch.object(backend._client, "post", return_value=resp)

        with pytest.raises(EmbeddingFallbackError):
            backend.embed_batch(["a", "b"], "zh-tw")

        assert backend._client.post.call_count == 1

    def test_401_raises_immediately(self, mocker):
        backend = make_backend()
        resp = make_response(mocker, status=401)
        mocker.patch.object(backend._client, "post", return_value=resp)

        with pytest.raises(EmbeddingFallbackError, match="Auth failed"):
            backend.embed_batch(["a"], "zh-tw")

    def test_timeout_raises_fallback_error(self, mocker):
        backend = make_backend()
        mocker.patch.object(
            backend._client, "post",
            side_effect=httpx.TimeoutException("timeout"),
        )

        with pytest.raises(EmbeddingFallbackError, match="Batch timeout"):
            backend.embed_batch(["a", "b"], "zh-tw")
