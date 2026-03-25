"""Unit tests for the embedding API routes.

Tests that /embed accepts a 'language' parameter, routes encoding to the
correct language model via EmbeddingBackend, and remains backward-compatible
when language is omitted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.embedding.backend import EmbeddingBackend

client = TestClient(app)


def _mock_backend(vectors: list[list[float]]) -> MagicMock:
    """Return a mock EmbeddingBackend whose embed_batch returns the given vectors."""
    backend = MagicMock(spec=EmbeddingBackend)
    backend.embed_batch.return_value = vectors
    return backend


# ── /embed endpoint ───────────────────────────────────────────────────────────


class TestEmbedEndpoint:
    def test_returns_200_with_language_parameter(self) -> None:
        """POST /embed with a 'language' field must return HTTP 200."""
        backend = _mock_backend([[0.1, 0.2, 0.3]])
        with patch("src.api.routes.get_backend", return_value=backend):
            resp = client.post("/embed", json={"texts": ["podcast title"], "language": "zh-tw"})
        assert resp.status_code == 200

    def test_passes_language_to_embed_batch(self) -> None:
        """The language from the request must be forwarded to backend.embed_batch."""
        backend = _mock_backend([[0.1] * 768])
        with patch("src.api.routes.get_backend", return_value=backend):
            client.post("/embed", json={"texts": ["some text"], "language": "zh-tw"})

        backend.embed_batch.assert_called_once_with(["some text"], "zh-tw")

    def test_passes_zh_cn_language_to_embed_batch(self) -> None:
        """zh-cn language must be forwarded unchanged to the backend."""
        backend = _mock_backend([[0.1] * 768])
        with patch("src.api.routes.get_backend", return_value=backend):
            client.post("/embed", json={"texts": ["简体中文"], "language": "zh-cn"})

        backend.embed_batch.assert_called_once_with(["简体中文"], "zh-cn")

    def test_defaults_to_en_when_language_omitted(self) -> None:
        """Backward compat: missing 'language' field must default to 'en'."""
        backend = _mock_backend([[0.1] * 384])
        with patch("src.api.routes.get_backend", return_value=backend):
            client.post("/embed", json={"texts": ["english text"]})

        call_args = backend.embed_batch.call_args
        assert call_args.args[1] == "en"

    def test_response_contains_embeddings(self) -> None:
        """Response body must include the embedding vectors returned by the backend."""
        expected = [[0.1, 0.2, 0.3]]
        backend = _mock_backend(expected)
        with patch("src.api.routes.get_backend", return_value=backend):
            resp = client.post("/embed", json={"texts": ["test"], "language": "en"})

        data = resp.json()
        assert data["embeddings"] == expected

    def test_response_includes_model_name(self) -> None:
        """Response must include a 'model' field identifying the backend model."""
        backend = _mock_backend([[0.1]])
        with patch("src.api.routes.get_backend", return_value=backend):
            resp = client.post("/embed", json={"texts": ["t"], "language": "en"})

        assert "model" in resp.json()

    def test_multiple_texts_encoded_together(self) -> None:
        """All texts in one request must be passed to embed_batch as a single list."""
        texts = ["first text", "second text", "third text"]
        backend = _mock_backend([[0.1]] * 3)
        with patch("src.api.routes.get_backend", return_value=backend):
            client.post("/embed", json={"texts": texts, "language": "en"})

        call_args = backend.embed_batch.call_args
        assert call_args.args[0] == texts

    def test_returns_500_on_backend_error(self) -> None:
        """When the backend raises an exception the endpoint must return HTTP 500."""
        backend = MagicMock(spec=EmbeddingBackend)
        backend.embed_batch.side_effect = RuntimeError("model crashed")
        with patch("src.api.routes.get_backend", return_value=backend):
            resp = client.post("/embed", json={"texts": ["t"], "language": "en"})

        assert resp.status_code == 500


# ── /health endpoint ──────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        """GET /health must return HTTP 200."""
        backend = _mock_backend([])
        with patch("src.api.routes.get_backend", return_value=backend):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self) -> None:
        """Health response body must include status: ok."""
        backend = _mock_backend([])
        with patch("src.api.routes.get_backend", return_value=backend):
            data = client.get("/health").json()
        assert data["status"] == "ok"
