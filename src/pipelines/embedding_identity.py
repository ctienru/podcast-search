"""Embedding identity — resolver, in-repo dimension table, systemic exceptions.

Phase 2a §3.5 (Content Identity Contract) + §3.8 (architecture).

This module is the *only* authoritative source of "expected identity" for the
Phase 2a pipeline. Callers (drift detection, cache validation, rebuild
primitive, force_embed) must route through `resolve_expected_identity()`
instead of reading `MODEL_MAP` / `EMBEDDING_TEXT_VERSION` / dims directly, so
that future resolver expansion (per-show overrides, A/B, etc.) happens in one
place.

Key contracts:
- `EmbeddingIdentity` is the triple (model_name, embedding_version, dims) that
  determines cache path slug (IC1) and validates cache freshness (IC2).
- `_DIM_TABLE` is an in-repo constant. It is the single source of truth for
  expected vector dimensions per model. Updating the table is a deliberate
  code change that must accompany any MODEL_MAP change.
- Resolver does NOT read environment variables or external files beyond
  `settings.EMBEDDING_TEXT_VERSION` (imported once at module load; no runtime
  env reads). Enforced by AST test.
- `EmbeddingDimensionContractViolation` (ET1, WV1) is a systemic-halt error
  raised by `embedding_runtime.embed_texts` when actual vector length differs
  from expected. Never caught at per-show level — aborts the entire pipeline.
- `ArtifactReadyRangeUnavailable` is raised by any caller that asks for the
  artifact-ready range before Phase 2b semantic cleanup completes.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.pipelines.embedding_catalog import MODEL_MAP
from src.types import Language


# Map Language literal → MODEL_MAP key. Keep in sync with
# `embed_episodes._LANGUAGE_TO_MODEL_KEY`; a regression test in
# test_embedding_identity.py pins the shape.
_LANGUAGE_TO_MODEL_KEY: dict[Language, str] = {
    "zh-tw": "zh",
    "zh-cn": "zh",
    "en": "en",
}


# In-repo source of truth for expected vector dimensions per model.
# Updating this dict is a deliberate code change; any MODEL_MAP addition MUST
# add the corresponding dim entry here, enforced by
# `test_dim_table_covers_all_models_in_map`.
_DIM_TABLE: dict[str, int] = {
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The triple that identifies a vector's provenance (Phase 2a §3.5 IC1)."""

    model_name: str
    embedding_version: str
    embedding_dimensions: int


def resolve_expected_identity(*, language: Language) -> EmbeddingIdentity:
    """Return the expected `EmbeddingIdentity` for `language`.

    Deliberately keyword-only and without a `context` parameter (see Phase 2a
    v4 decision — no fake extensibility). If the pipeline later needs
    per-show overrides, the signature will be widened in one place.

    Raises:
        KeyError: language maps to a model_name that has no `_DIM_TABLE`
                  entry. This is a dev-time contract violation (MODEL_MAP
                  expanded without updating `_DIM_TABLE`).
    """
    model_key = _LANGUAGE_TO_MODEL_KEY[language]
    model_name = MODEL_MAP[model_key]
    try:
        dims = _DIM_TABLE[model_name]
    except KeyError:
        raise KeyError(
            f"_DIM_TABLE has no entry for model={model_name!r}; "
            f"update src/pipelines/embedding_identity.py::_DIM_TABLE "
            f"before using this model."
        ) from None
    return EmbeddingIdentity(
        model_name=model_name,
        embedding_version=settings.EMBEDDING_TEXT_VERSION,
        embedding_dimensions=dims,
    )


class ArtifactReadyRangeUnavailable(Exception):
    """Raised when a caller requests the artifact-ready set in Phase 2a.

    Phase 2a deliberately does not define a runtime "artifact-ready" set:
    `episodes.embedding_status` semantics are still decoupled from ES state
    (main spec §3.1.3). Phase 2b must pick one of two paths (V1e-A backfill
    or V1e-B retire) before any caller can treat it as a truth source.
    """

    DEFAULT_MESSAGE = (
        "artifact-ready range is unavailable in Phase 2a: requires Phase 2b "
        "semantic cleanup to resolve embedding_status truth (pick V1e-A "
        "backfill or V1e-B retire). See "
        "podcast-daily/2026-04-14-daily-pipeline-orchestration.md §3.1.3 / "
        "§5 Phase 2b."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class EmbeddingDimensionContractViolation(Exception):
    """ET1 systemic halt (WV1): runtime vector dim != expected dim.

    Phase 2a §3.8 Error Taxonomy ET1: when `embedding_runtime.embed_texts`
    produces a vector whose length differs from `identity.embedding_dimensions`,
    the entire pipeline run aborts. Rationale: either `_DIM_TABLE` is wrong
    or the model is wrong — in both cases, every subsequent cache write
    would be poisoned.

    Never caught at per-show level; bubbles past `rebuild_show_cache()` and
    past `embed_and_ingest`'s A1 error funnel.
    """

    def __init__(
        self,
        *,
        expected: int,
        actual: int,
        model_name: str,
        context: str = "",
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.model_name = model_name
        self.context = context
        msg = (
            f"embedding dimension contract violation: "
            f"expected={expected}, actual={actual}, model={model_name!r}"
        )
        if context:
            msg += f" (context: {context})"
        super().__init__(msg)
