"""Phase 2a §3.5 Content Identity + §3.8 resolver isolation tests.

Covers:
- `EmbeddingIdentity` shape + immutability
- `_DIM_TABLE` coverage vs MODEL_MAP
- `resolve_expected_identity` per-language return values
- `ArtifactReadyRangeUnavailable` default message
- `EmbeddingDimensionContractViolation` (ET1) carries actionable context
- AST isolation: resolver source does not read env / external files
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError

import pytest

from src.pipelines import embedding_identity as ei
from src.pipelines.embedding_catalog import MODEL_MAP
from src.pipelines.embedding_identity import (
    ArtifactReadyRangeUnavailable,
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
    _DIM_TABLE,
    _LANGUAGE_TO_MODEL_KEY,
    resolve_expected_identity,
)


# ── EmbeddingIdentity shape ──────────────────────────────────────────────────

def test_embedding_identity_has_three_fields() -> None:
    """IC1: identity is (model_name, embedding_version, embedding_dimensions)."""
    ident = EmbeddingIdentity(
        model_name="m", embedding_version="v", embedding_dimensions=1,
    )
    assert ident.model_name == "m"
    assert ident.embedding_version == "v"
    assert ident.embedding_dimensions == 1


def test_embedding_identity_is_frozen() -> None:
    """Prevent accidental mutation — identity is a value type."""
    ident = EmbeddingIdentity(model_name="m", embedding_version="v", embedding_dimensions=1)
    with pytest.raises(FrozenInstanceError):
        ident.model_name = "x"  # type: ignore[misc]


# ── _DIM_TABLE ──────────────────────────────────────────────────────────────

def test_dim_table_has_expected_entry_for_current_model() -> None:
    assert _DIM_TABLE["paraphrase-multilingual-MiniLM-L12-v2"] == 384


def test_dim_table_covers_all_models_in_map() -> None:
    """Any model added to MODEL_MAP must also have a dims entry."""
    missing = {m for m in MODEL_MAP.values() if m not in _DIM_TABLE}
    assert not missing, f"MODEL_MAP has models without _DIM_TABLE entries: {missing}"


# ── resolve_expected_identity ───────────────────────────────────────────────

@pytest.mark.parametrize("language", ["zh-tw", "zh-cn", "en"])
def test_resolve_expected_identity_returns_identity_for_each_language(language) -> None:
    ident = resolve_expected_identity(language=language)
    assert ident.model_name == MODEL_MAP[_LANGUAGE_TO_MODEL_KEY[language]]
    assert ident.embedding_dimensions == _DIM_TABLE[ident.model_name]
    # embedding_version comes from settings.EMBEDDING_TEXT_VERSION — not a
    # literal assertion on value (config may vary per env), just non-empty.
    assert ident.embedding_version
    assert isinstance(ident.embedding_version, str)


def test_resolve_expected_identity_is_keyword_only() -> None:
    """No positional call — avoids silent-arg swap."""
    with pytest.raises(TypeError):
        resolve_expected_identity("zh-tw")  # type: ignore[misc]


def test_resolver_raises_on_missing_dim_table_entry(monkeypatch) -> None:
    """Dev-time contract: MODEL_MAP expanded without _DIM_TABLE update = KeyError."""
    monkeypatch.setitem(MODEL_MAP, "en", "unknown-model-xyz")
    # _DIM_TABLE still has no entry for "unknown-model-xyz"
    with pytest.raises(KeyError, match="_DIM_TABLE has no entry"):
        resolve_expected_identity(language="en")


# ── ArtifactReadyRangeUnavailable ───────────────────────────────────────────

def test_artifact_ready_range_unavailable_default_message_references_phase_2b() -> None:
    exc = ArtifactReadyRangeUnavailable()
    msg = str(exc)
    assert "Phase 2a" in msg
    assert "Phase 2b" in msg
    assert "V1e-A" in msg and "V1e-B" in msg


def test_artifact_ready_range_unavailable_custom_message() -> None:
    exc = ArtifactReadyRangeUnavailable("custom context")
    assert str(exc) == "custom context"


# ── EmbeddingDimensionContractViolation (ET1) ───────────────────────────────

def test_et1_carries_dims_and_model() -> None:
    exc = EmbeddingDimensionContractViolation(
        expected=384, actual=768, model_name="m", context="batch #3",
    )
    assert exc.expected == 384
    assert exc.actual == 768
    assert exc.model_name == "m"
    assert exc.context == "batch #3"
    msg = str(exc)
    assert "384" in msg and "768" in msg and "m" in msg and "batch #3" in msg


def test_et1_omits_context_segment_when_empty() -> None:
    exc = EmbeddingDimensionContractViolation(expected=384, actual=768, model_name="m")
    assert "context" not in str(exc)


# ── AST isolation (§3.8: resolver reads only catalog + settings) ────────────

def test_resolver_dim_source_does_not_read_env_or_external_file() -> None:
    """The resolver module must not touch env vars or open external files.

    EMBEDDING_TEXT_VERSION enters via `settings` (imported at module load);
    dims enters via the in-repo `_DIM_TABLE` constant. Any `os.getenv`,
    `os.environ`, `open(`, or `Path(...).read_*` within this module would
    break the contract that resolver behavior is reproducible from source.
    """
    source = inspect.getsource(ei)
    tree = ast.parse(source)

    forbidden_attrs = {("os", "getenv"), ("os", "environ")}
    forbidden_calls = {"open"}
    forbidden_suffix_methods = {"read_text", "read_bytes", "read"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in forbidden_attrs, (
                f"resolver must not reference {node.value.id}.{node.attr}"
            )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden_calls:
                raise AssertionError(f"resolver must not call {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr in forbidden_suffix_methods:
                raise AssertionError(f"resolver must not call .{func.attr}()")
