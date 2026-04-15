"""Thin adapters that construct `EmbeddingIdentity` from a DB row or a
cache payload.

Phase 2b-A backfill performs CT2 identity comparison between what the
`episodes` table says a row was embedded with and what the on-disk
cache payload claims. The two sources carry the same logical tuple but
with different field names and structural quirks:

- DB row uses `embedding_model` / `embedding_version` and has no
  `embedding_dimensions` column (dimensions are derived at runtime from
  the in-repo `_DIM_TABLE`).
- Cache payload uses `model_name` / `embedding_version` /
  `embedding_dimensions` and carries dimensions directly.

This adapter isolates those shape differences so backfill code can
treat both sources uniformly. It deliberately does not extend
`EmbeddingIdentity` — Phase 2a froze that primitive.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE


class IdentityAdapterError(ValueError):
    """Raised when a row or payload cannot be mapped to EmbeddingIdentity.

    Attributes:
        source: ``"row"`` or ``"payload"`` — the origin that failed to
                produce an identity. Backfill uses this to classify the
                failure (row-side missing fields → probably neutral;
                payload-side missing fields → anomaly or hard fail).
        reason: short machine-readable label for the concrete cause
                (``"missing_field"``, ``"unknown_model"``).
    """

    def __init__(self, message: str, *, source: str, reason: str) -> None:
        super().__init__(message)
        self.source = source
        self.reason = reason


def _require(value: Any, *, field: str, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdentityAdapterError(
            f"{source} missing required identity field {field!r}",
            source=source,
            reason="missing_field",
        )
    return value


def identity_from_row(row: Mapping[str, Any]) -> EmbeddingIdentity:
    """Build `EmbeddingIdentity` from an `episodes` row.

    Dimensions are looked up in `_DIM_TABLE` using `embedding_model` —
    the DB schema does not carry a dimensions column (Phase 2a decision).
    An unknown model is a dev-time contract violation (`_DIM_TABLE`
    missing an entry); the error is surfaced with
    ``reason="unknown_model"`` so callers can distinguish it from a
    plain missing-field case.
    """
    model_name = _require(row.get("embedding_model"), field="embedding_model", source="row")
    embedding_version = _require(
        row.get("embedding_version"), field="embedding_version", source="row"
    )
    try:
        dims = _DIM_TABLE[model_name]
    except KeyError:
        raise IdentityAdapterError(
            f"row references unknown model {model_name!r} "
            f"(not in _DIM_TABLE); update "
            f"src/pipelines/embedding_identity.py::_DIM_TABLE",
            source="row",
            reason="unknown_model",
        ) from None
    return EmbeddingIdentity(
        model_name=model_name,
        embedding_version=embedding_version,
        embedding_dimensions=dims,
    )


def identity_from_payload(payload: Mapping[str, Any]) -> EmbeddingIdentity:
    """Build `EmbeddingIdentity` from a cache payload dict.

    The payload is expected to carry `model_name`, `embedding_version`,
    and `embedding_dimensions` as written by Phase 2a
    `migrate_embeddings_to_versioned` / `embed_episodes`. Missing or
    mistyped `embedding_dimensions` raises with ``reason="missing_field"``
    so backfill classifies it as a payload-side failure (CT2 / CT3
    triage).
    """
    model_name = _require(
        payload.get("model_name"), field="model_name", source="payload"
    )
    embedding_version = _require(
        payload.get("embedding_version"), field="embedding_version", source="payload"
    )
    dims = payload.get("embedding_dimensions")
    if not isinstance(dims, int) or dims <= 0:
        raise IdentityAdapterError(
            f"payload missing or invalid embedding_dimensions: {dims!r}",
            source="payload",
            reason="missing_field",
        )
    return EmbeddingIdentity(
        model_name=model_name,
        embedding_version=embedding_version,
        embedding_dimensions=dims,
    )
