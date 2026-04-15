"""Phase 2a: catalog relocation regression tests.

Verify `MODEL_MAP` moved from `src.embedding.backend` to
`src.pipelines.embedding_catalog` without changing its content, and that the
legacy import path still returns the same object (backward compat).

Corresponds to Phase 2a doc §4.6:
- `test_embed_runtime_uses_same_model_catalog_as_before`
"""

from __future__ import annotations


def test_embedding_catalog_has_expected_model_map() -> None:
    """Authoritative MODEL_MAP content is pinned — any change is a correctness event."""
    from src.pipelines.embedding_catalog import MODEL_MAP

    assert MODEL_MAP == {
        "zh": "paraphrase-multilingual-MiniLM-L12-v2",
        "en": "paraphrase-multilingual-MiniLM-L12-v2",
    }


def test_embed_runtime_uses_same_model_catalog_as_before() -> None:
    """Legacy `src.embedding.backend.MODEL_MAP` must be the catalog object.

    Guarantees: (a) catalog relocation did not fork semantics; (b) callers
    still importing from `backend` (e.g. legacy tests, routes comments) pick
    up the same object identity, so future edits to the catalog propagate.
    """
    from src.embedding import backend as backend_module
    from src.pipelines import embedding_catalog

    assert backend_module.MODEL_MAP is embedding_catalog.MODEL_MAP


def test_embed_episodes_uses_catalog_model_map() -> None:
    """`embed_episodes` must read MODEL_MAP from the catalog directly (Phase 2a §3.8)."""
    from src.pipelines import embed_episodes, embedding_catalog

    assert embed_episodes.MODEL_MAP is embedding_catalog.MODEL_MAP
