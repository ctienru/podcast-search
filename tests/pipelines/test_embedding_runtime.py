"""Phase 2a runtime primitive tests (WV1 systemic halt).

Covers:
- Pass-through: returns backend vectors in order when dims match
- Batching: respects batch_size, all output in order
- Dimension self-validation: raises on length mismatch (systemic halt)
- The raised exception carries context (expected/actual/model/batch info)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
)
from src.pipelines.embedding_runtime import embed_texts


def _identity(dims: int = 384) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_version="text-v1",
        embedding_dimensions=dims,
    )


def _mk_backend(side_effect) -> MagicMock:
    b = MagicMock()
    b.embed_batch.side_effect = side_effect
    return b


# ── Pass-through ────────────────────────────────────────────────────────────

def test_returns_vectors_in_input_order() -> None:
    backend = _mk_backend(lambda texts, lang: [[float(i)] * 384 for i in range(len(texts))])
    vectors = embed_texts(
        texts=["a", "b", "c"],
        language="zh-tw",
        identity=_identity(),
        backend=backend,
        batch_size=64,
    )
    assert len(vectors) == 3
    assert vectors[0][0] == 0.0
    assert vectors[1][0] == 1.0
    assert vectors[2][0] == 2.0


def test_splits_into_batches_of_batch_size() -> None:
    calls: list[int] = []

    def side_effect(texts, lang):
        calls.append(len(texts))
        return [[0.0] * 384 for _ in texts]

    backend = _mk_backend(side_effect)
    vectors = embed_texts(
        texts=["x"] * 5,
        language="en",
        identity=_identity(),
        backend=backend,
        batch_size=2,
    )
    assert calls == [2, 2, 1]
    assert len(vectors) == 5


def test_empty_texts_returns_empty_without_calling_backend() -> None:
    backend = _mk_backend(None)
    vectors = embed_texts(
        texts=[],
        language="en",
        identity=_identity(),
        backend=backend,
    )
    assert vectors == []
    assert backend.embed_batch.call_count == 0


# ── WV1: dimension self-validation (systemic halt) ──────────────────────────

def test_raises_when_returned_vector_is_too_short() -> None:
    # Backend gives 2 vectors: first ok, second has wrong length.
    backend = _mk_backend(lambda texts, lang: [[0.0] * 384, [0.0] * 256])
    with pytest.raises(EmbeddingDimensionContractViolation) as excinfo:
        embed_texts(
            texts=["a", "b"],
            language="zh-tw",
            identity=_identity(dims=384),
            backend=backend,
        )
    exc = excinfo.value
    assert exc.expected == 384
    assert exc.actual == 256
    assert exc.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    # Context points at the offending batch position
    assert "zh-tw" in exc.context
    assert "index_in_batch=1" in exc.context


def test_raises_when_returned_vector_is_too_long() -> None:
    backend = _mk_backend(lambda texts, lang: [[0.0] * 768 for _ in texts])
    with pytest.raises(EmbeddingDimensionContractViolation):
        embed_texts(
            texts=["a"],
            language="en",
            identity=_identity(dims=384),
            backend=backend,
        )


def test_raises_in_second_batch_when_mismatch_appears_later() -> None:
    """Validation happens per-batch, so even the Nth batch is caught."""
    call_count = {"n": 0}

    def side_effect(texts, lang):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [[0.0] * 384 for _ in texts]
        return [[0.0] * 200 for _ in texts]  # second batch wrong

    backend = _mk_backend(side_effect)
    with pytest.raises(EmbeddingDimensionContractViolation) as excinfo:
        embed_texts(
            texts=["a", "b", "c"],
            language="en",
            identity=_identity(dims=384),
            backend=backend,
            batch_size=2,
        )
    # batch_start should identify which batch failed
    assert "batch_start=2" in excinfo.value.context


def test_validation_runs_per_returned_vector_not_only_first() -> None:
    # First vector correct, last wrong — must still raise.
    backend = _mk_backend(
        lambda texts, lang: [[0.0] * 384] * (len(texts) - 1) + [[0.0] * 128]
    )
    with pytest.raises(EmbeddingDimensionContractViolation) as excinfo:
        embed_texts(
            texts=["x", "y", "z"],
            language="en",
            identity=_identity(dims=384),
            backend=backend,
            batch_size=64,
        )
    assert excinfo.value.actual == 128
