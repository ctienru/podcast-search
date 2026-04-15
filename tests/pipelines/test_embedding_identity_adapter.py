"""Tests for the EmbeddingIdentity row/payload adapter.

Per 2b-A impl doc Step 1b: the adapter must produce identical
`EmbeddingIdentity` tuples from a DB row and its corresponding cache
payload, and must raise `IdentityAdapterError` with a structured
`source` / `reason` when a required field is missing.
"""

from __future__ import annotations

import pytest

from src.pipelines.embedding_identity import EmbeddingIdentity, _DIM_TABLE
from src.pipelines.embedding_identity_adapter import (
    IdentityAdapterError,
    identity_from_payload,
    identity_from_row,
)


_KNOWN_MODEL = next(iter(_DIM_TABLE))
_KNOWN_DIMS = _DIM_TABLE[_KNOWN_MODEL]


class TestRowAdapter:
    def test_identity_from_row_matches_hand_constructed(self) -> None:
        row = {
            "episode_id": "ep:apple:1001073519:ABC",
            "embedding_model": _KNOWN_MODEL,
            "embedding_version": "text-v1",
        }
        identity = identity_from_row(row)
        assert identity == EmbeddingIdentity(
            model_name=_KNOWN_MODEL,
            embedding_version="text-v1",
            embedding_dimensions=_KNOWN_DIMS,
        )

    def test_row_missing_embedding_model_raises(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_row({"embedding_version": "text-v1"})
        assert excinfo.value.source == "row"
        assert excinfo.value.reason == "missing_field"

    def test_row_missing_embedding_version_raises(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_row({"embedding_model": _KNOWN_MODEL})
        assert excinfo.value.source == "row"
        assert excinfo.value.reason == "missing_field"

    def test_row_unknown_model_raises_with_distinct_reason(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_row(
                {"embedding_model": "nonexistent-model-xyz", "embedding_version": "text-v1"}
            )
        assert excinfo.value.source == "row"
        assert excinfo.value.reason == "unknown_model"


class TestPayloadAdapter:
    def test_identity_from_payload_matches_hand_constructed(self) -> None:
        payload = {
            "model_name": _KNOWN_MODEL,
            "embedding_version": "text-v1",
            "embedding_dimensions": _KNOWN_DIMS,
            "episodes": {},
        }
        identity = identity_from_payload(payload)
        assert identity == EmbeddingIdentity(
            model_name=_KNOWN_MODEL,
            embedding_version="text-v1",
            embedding_dimensions=_KNOWN_DIMS,
        )

    def test_payload_missing_dims_raises(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_payload(
                {"model_name": _KNOWN_MODEL, "embedding_version": "text-v1"}
            )
        assert excinfo.value.source == "payload"
        assert excinfo.value.reason == "missing_field"

    def test_payload_invalid_dims_type_raises(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_payload(
                {
                    "model_name": _KNOWN_MODEL,
                    "embedding_version": "text-v1",
                    "embedding_dimensions": "384",  # str, not int
                }
            )
        assert excinfo.value.source == "payload"

    def test_payload_missing_model_name_raises(self) -> None:
        with pytest.raises(IdentityAdapterError) as excinfo:
            identity_from_payload(
                {"embedding_version": "text-v1", "embedding_dimensions": _KNOWN_DIMS}
            )
        assert excinfo.value.source == "payload"


class TestCrossSourceConsistency:
    def test_row_and_payload_for_same_show_match(self) -> None:
        """CT2 core precondition: given a consistent pair, both adapters
        produce the same `EmbeddingIdentity`. Backfill relies on this
        equivalence to decide Pass vs identity_mismatch."""
        row = {
            "embedding_model": _KNOWN_MODEL,
            "embedding_version": "text-v1",
        }
        payload = {
            "model_name": _KNOWN_MODEL,
            "embedding_version": "text-v1",
            "embedding_dimensions": _KNOWN_DIMS,
        }
        assert identity_from_row(row) == identity_from_payload(payload)

    def test_mismatch_surfaces_via_inequality_not_adapter_error(self) -> None:
        """When the two sources disagree on version, the adapter itself
        must still succeed on each side — the mismatch is a caller-side
        comparison concern, not an adapter concern."""
        row_ident = identity_from_row(
            {"embedding_model": _KNOWN_MODEL, "embedding_version": "text-v1"}
        )
        payload_ident = identity_from_payload(
            {
                "model_name": _KNOWN_MODEL,
                "embedding_version": "text-v2",
                "embedding_dimensions": _KNOWN_DIMS,
            }
        )
        assert row_ident != payload_ident
