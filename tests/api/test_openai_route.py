"""Tests for the /v1/embeddings OpenAI-compatible route."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_ZH = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_EN = "paraphrase-multilingual-MiniLM-L12-v2"

client = TestClient(app)


def make_mock_backend(mocker, return_value: list[list[float]]):
    """Patch get_backend() to return a mock with embed_batch returning return_value."""
    mock_backend = mocker.MagicMock()
    mock_backend.embed_batch.return_value = return_value
    mocker.patch("src.api.routes.get_backend", return_value=mock_backend)
    return mock_backend


# ── /v1/embeddings ────────────────────────────────────────────────────────────

class TestOpenAIEmbed:
    def test_single_string_input(self, mocker):
        mock = make_mock_backend(mocker, [[0.1, 0.2, 0.3]])

        resp = client.post("/v1/embeddings", json={"model": MODEL_ZH, "input": "人工智慧"})

        assert resp.status_code == 200
        mock.embed_batch.assert_called_once_with(["人工智慧"], "en")

    def test_array_input(self, mocker):
        mock = make_mock_backend(mocker, [[0.1], [0.2]])

        resp = client.post("/v1/embeddings", json={"model": MODEL_EN, "input": ["hello", "world"]})

        assert resp.status_code == 200
        mock.embed_batch.assert_called_once_with(["hello", "world"], "en")

    def test_response_schema(self, mocker):
        make_mock_backend(mocker, [[0.5, 0.6]])

        resp = client.post("/v1/embeddings", json={"model": MODEL_EN, "input": "test"})
        body = resp.json()

        assert body["object"] == "list"
        assert body["model"] == MODEL_EN
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert body["data"][0]["embedding"] == [0.5, 0.6]

    def test_batch_response_indices(self, mocker):
        make_mock_backend(mocker, [[0.1], [0.2], [0.3]])

        resp = client.post("/v1/embeddings", json={"model": MODEL_ZH, "input": ["a", "b", "c"]})
        body = resp.json()

        indices = [item["index"] for item in body["data"]]
        assert indices == [0, 1, 2]

    def test_unknown_model_returns_422(self, mocker):
        mocker.patch("src.api.routes.get_backend")  # should not be called

        resp = client.post("/v1/embeddings", json={"model": "unknown/model", "input": "test"})

        assert resp.status_code == 422

    def test_backend_exception_returns_500(self, mocker):
        mock_backend = mocker.MagicMock()
        mock_backend.embed_batch.side_effect = RuntimeError("upstream failure")
        mocker.patch("src.api.routes.get_backend", return_value=mock_backend)

        resp = client.post("/v1/embeddings", json={"model": MODEL_EN, "input": "test"})

        assert resp.status_code == 500
