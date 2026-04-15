"""Phase 2a IC3 + V18 audit — single authoritative gate.

§3.5 IC3 names seven modules that form the "primitive / consumer"
side of the Phase 2a architecture. None of them may reach for
`embed_episodes` (the batch entry point) or `embed_and_ingest` (the
pipeline runner), because doing so would pull the whole main-pipeline
import surface into a primitive or into a manual-override tool. The
boundary keeps the primitive reusable, keeps the override tool
auditable, and keeps drift detection from silently depending on the
pipeline it is supposed to observe.

V18 (main spec §10.2a) names the literal `--allow-model-drift`: it
must appear only in `scripts/force_embed.py` and nowhere else under
`src/` or `scripts/`. If it leaks into the orchestrator or the daily
pipeline, the scheduled run could be coaxed into the drift-accepting
branch — a contract violation that would go unnoticed without a
grep-level check.

Some individual module test files already enforce part of this
(e.g. `test_force_embed.py::TestIsolation`, the check_drift tests).
This file is the single consolidated gate that proves the whole
seven-module set is clean at once, so adding a new module to IC3
means touching exactly one place to wire it in.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Iterable

import pytest


# The seven modules named in §3.5 IC3. Identified by import path —
# the test resolves each to its source file via importlib so a
# rename would surface here as a collection error instead of a
# silently-disabled check.
IC3_MODULES: tuple[str, ...] = (
    "src.pipelines.embedding_identity",
    "src.pipelines.embedding_paths",
    "src.pipelines.embedding_text",
    "src.pipelines.embedding_runtime",
    "src.pipelines.show_rebuild",
    "src.tools.check_drift",
    "scripts.force_embed",
)


# Imports that IC3 forbids on the primitive / consumer side. Listed
# as tuples of (forbidden_prefix, human_readable_reason) so the
# failure message tells the next reader *why* it matters, not just
# that it matters.
FORBIDDEN_IMPORTS: tuple[tuple[str, str], ...] = (
    (
        "src.pipelines.embed_episodes",
        "IC3 forbids reaching into the batch entry point from a primitive; "
        "route through the shared chunking + runtime primitives instead.",
    ),
    (
        "src.pipelines.embed_and_ingest",
        "IC3 forbids a primitive depending on the pipeline that consumes it; "
        "that cycle would make the primitive unreusable by force_embed.",
    ),
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _source_path_for(dotted: str) -> Path:
    """Resolve a dotted module path to its source file on disk.

    We parse the file directly (rather than importing the module and
    inspecting `sys.modules`) because the audit cares about the
    *declared* imports in source, not whatever happens to be loaded
    at test time.
    """
    import importlib.util

    spec = importlib.util.find_spec(dotted)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate source for {dotted!r}")
    return Path(spec.origin)


def _imported_module_names(tree: ast.Module) -> set[str]:
    """Every module path that appears in an `import` or `from … import`.

    AST-based on purpose: a docstring or comment that *names* a
    forbidden module (often to explain why it is forbidden) must not
    trip the check. Only actual import nodes count.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


def _project_root() -> Path:
    # tests/pipelines/test_phase2a_ast_audit.py → …/podcast-search
    return Path(__file__).resolve().parents[2]


# ── IC3 seven-module audit ──────────────────────────────────────────────────


@pytest.mark.parametrize("dotted", IC3_MODULES)
@pytest.mark.parametrize("forbidden_prefix,reason", FORBIDDEN_IMPORTS)
def test_ic3_module_does_not_import_forbidden(
    dotted: str, forbidden_prefix: str, reason: str,
) -> None:
    """Every IC3 module must be clean of every forbidden import.

    The cartesian product (7 modules × 2 forbidden imports = 14 test
    cases) is the cheapest way to make a failure message point at
    the exact offender: the parametrize ids name both sides of the
    pairing.
    """
    source = _source_path_for(dotted).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = _imported_module_names(tree)
    offending = [
        name for name in imports
        if name == forbidden_prefix or name.startswith(forbidden_prefix + ".")
    ]
    assert not offending, (
        f"{dotted} imports {offending}. {reason}"
    )


def test_ic3_module_list_matches_design_doc() -> None:
    """Guard against accidentally shrinking the audit set.

    The design doc commits to auditing seven modules (§3.8 architecture
    diagram's "AST gate" list). If someone removes a module from
    `IC3_MODULES` without a corresponding doc update, this test fails
    first so the drift is visible.
    """
    assert len(IC3_MODULES) == 7, (
        f"IC3 names seven modules (§3.8 architecture); current list "
        f"has {len(IC3_MODULES)}: {IC3_MODULES}"
    )


# ── V18: --allow-model-drift literal gate ──────────────────────────────────


def _files_containing(literal: str, roots: Iterable[Path]) -> list[Path]:
    """Every file under `roots` whose text contains `literal`.

    Uses grep rather than a Python walker because a grep command with
    --include is both faster and matches what a reviewer would run by
    hand. The audit is meant to be easy to reproduce outside the
    test harness.
    """
    root = _project_root()
    paths: list[str] = []
    for sub in roots:
        paths.append(str(sub))
    result = subprocess.run(
        ["grep", "-r", "-l", "--include=*.py", literal, *paths],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def test_allow_model_drift_literal_only_in_force_embed() -> None:
    """V18: the `--allow-model-drift` flag must be exclusive to
    `scripts/force_embed.py`. Any other appearance under `src/` or
    `scripts/` means the daily pipeline could be driven into the
    drift-accepting branch, which is the exact thing Phase 2a
    promises cannot happen.
    """
    hits = _files_containing("--allow-model-drift", [Path("src"), Path("scripts")])
    hit_strs = {str(h) for h in hits}
    allowed = {"scripts/force_embed.py"}
    forbidden = hit_strs - allowed
    assert not forbidden, (
        f"--allow-model-drift leaked into non-force_embed code: {forbidden}. "
        f"V18 requires the flag to exist only in scripts/force_embed.py."
    )
