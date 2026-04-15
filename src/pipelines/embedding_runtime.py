"""Text → vector runtime primitive — Phase 2a.

The sole path through which the pipeline turns text into vectors. Every
vector produced by this module is length-checked against the supplied
`EmbeddingIdentity.embedding_dimensions`; any disagreement raises
`EmbeddingDimensionContractViolation`, which is a systemic-halt error
the caller must not catch at the per-show boundary.

Rationale: if the runtime ever returns a vector whose length differs
from the expected dimension, either the in-repo dims table is wrong or
the configured model has changed underneath us — in both cases every
subsequent cache write would be poisoned, so we stop the entire run
before producing any more output.

Callers (`embed_episodes`, upcoming `show_rebuild`) MUST route through
`embed_texts` and are forbidden from calling the lower-level backend's
`embed_batch` directly. This keeps the self-validation in one place.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.embedding.backend import EmbeddingBackend, LocalEmbeddingBackend
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
)
from src.types import Language

logger = logging.getLogger(__name__)


def embed_texts(
    *,
    texts: list[str],
    language: Language,
    identity: EmbeddingIdentity,
    backend: Optional[EmbeddingBackend] = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Encode `texts` with self-validation against `identity`.

    Args:
        texts: The texts to encode. All must share `language`.
        language: Language literal used by the backend to pick a model.
        identity: The expected identity for the output vectors. Every
            returned vector must have length `identity.embedding_dimensions`.
        backend: Optional backend override, primarily for tests. Defaults
            to a shared `LocalEmbeddingBackend`.
        batch_size: How many texts to encode per backend call. Tuned to
            64 to match the existing embed_episodes default.

    Returns:
        A list of vectors, one per input text, in the same order.

    Raises:
        EmbeddingDimensionContractViolation: A produced vector's length
            differs from `identity.embedding_dimensions`. Systemic halt —
            do not catch at per-show level.
    """
    _backend = backend if backend is not None else LocalEmbeddingBackend()
    expected = identity.embedding_dimensions

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = _backend.embed_batch(batch, language)

        for offset, vec in enumerate(vectors):
            actual = len(vec)
            if actual != expected:
                raise EmbeddingDimensionContractViolation(
                    expected=expected,
                    actual=actual,
                    model_name=identity.model_name,
                    context=(
                        f"language={language}, batch_start={start}, "
                        f"index_in_batch={offset}"
                    ),
                )

        out.extend(vectors)

    return out
