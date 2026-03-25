"""Unit tests for embedding backend factory."""

from unittest.mock import patch

import pytest


def test_create_backend_default_is_local(monkeypatch):
    monkeypatch.delenv("EMBEDDING_STRATEGY", raising=False)
    # Re-import to pick up env changes
    import importlib
    import src.embedding.factory as factory_mod
    importlib.reload(factory_mod)

    from src.embedding.backend import LocalEmbeddingBackend
    backend = factory_mod.create_backend()
    assert isinstance(backend, LocalEmbeddingBackend)


def test_create_backend_api(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STRATEGY", "api")
    monkeypatch.setenv("EMBEDDING_API_URL", "http://test.api")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")

    import importlib
    import src.embedding.factory as factory_mod
    importlib.reload(factory_mod)

    with patch("src.embedding.backend.httpx.Client"):
        from src.embedding.backend import APIEmbeddingBackend
        backend = factory_mod.create_backend()
    assert isinstance(backend, APIEmbeddingBackend)


def test_create_backend_api_uses_timeout_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STRATEGY", "api")
    monkeypatch.setenv("EMBEDDING_API_URL", "http://test.api")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_MS", "5000")

    import importlib
    import src.embedding.factory as factory_mod
    importlib.reload(factory_mod)

    with patch("src.embedding.backend.httpx.Client") as mock_client_cls:
        factory_mod.create_backend()

    _, kwargs = mock_client_cls.call_args
    assert kwargs["timeout"] == 5.0


def test_create_backend_api_missing_url_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STRATEGY", "api")
    monkeypatch.delenv("EMBEDDING_API_URL", raising=False)
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")

    import importlib
    import src.embedding.factory as factory_mod
    importlib.reload(factory_mod)

    with pytest.raises(KeyError):
        factory_mod.create_backend()
