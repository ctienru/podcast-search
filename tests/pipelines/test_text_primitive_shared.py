"""Phase 2a TP1 isolation test — embed pipelines must share primitives.

If a pipeline that produces embedding cache files (`embed_episodes`, and
later `show_rebuild`) implemented its own chunking or called the backend's
`embed_batch` directly, the cache identity contract would no longer be
trustworthy: two "same-version" runs could produce different vectors for
the same show because each caller applied its own text logic.

This test enforces that constraint at the source level — callers must
route through `prepare_chunks_for_show` for chunking and `embed_texts`
for encoding.

When `show_rebuild.py` lands, extend the `PIPELINES` list below so the
same constraint applies there.
"""

from __future__ import annotations

import ast
import inspect

from src.pipelines import embed_episodes, show_rebuild

# Modules whose embed-producing behavior must go through shared primitives.
PIPELINES = [embed_episodes, show_rebuild]


def _parse(module) -> ast.Module:
    return ast.parse(inspect.getsource(module))


def _attribute_method_calls(tree: ast.AST) -> list[str]:
    """Return names of every `x.<method>(...)` call in `tree`."""
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    return calls


def test_no_pipeline_calls_embed_batch_directly() -> None:
    """TP1 + WV1: every encoder call must route through `embed_texts`.

    A direct `backend.embed_batch(...)` inside an embed-producing pipeline
    would bypass runtime dimension self-validation and the shared
    text-assembly primitive — both are load-bearing for cache identity.
    """
    for module in PIPELINES:
        calls = _attribute_method_calls(_parse(module))
        assert "embed_batch" not in calls, (
            f"{module.__name__} calls .embed_batch() directly; "
            f"route through src.pipelines.embedding_runtime.embed_texts."
        )


def test_every_pipeline_imports_shared_text_primitive() -> None:
    """TP1: embed-producing pipelines must import `prepare_chunks_for_show`.

    Importing the shared primitive is necessary but not sufficient; paired
    with the `embed_batch` check above, it enforces that chunking goes
    through the primitive rather than an ad-hoc loop.
    """
    for module in PIPELINES:
        source = inspect.getsource(module)
        assert "prepare_chunks_for_show" in source, (
            f"{module.__name__} does not import `prepare_chunks_for_show`; "
            f"chunking must go through src.pipelines.embedding_text."
        )


def test_every_pipeline_imports_shared_runtime_primitive() -> None:
    for module in PIPELINES:
        source = inspect.getsource(module)
        assert "embed_texts" in source, (
            f"{module.__name__} does not import `embed_texts`; "
            f"encoding must go through src.pipelines.embedding_runtime."
        )
