"""Guard against the CB1 integration gap being reintroduced.

Phase 2a's per-show DB metadata commit lives inside
`EmbedAndIngestPipeline.run()`, but that code is dead weight unless
every production entry point hands the pipeline an
`EpisodeStatusRepository`. The earlier implementation batches added
the constructor parameter and the commit branch, but for a while the
CLI never supplied the repo — so the commit silently never fired in
the daily path. These tests exist so that mistake cannot silently
reappear.

Approach:

- For each of the three entry points that construct a pipeline
  (`run_incremental`, `run_backfill`, `upsert_by_show_id`) patch the
  pipeline class and inspect what the entry point passed.
- For the CLI `run()` function, parse its source and check for the
  literal `EpisodeStatusRepository(` — the single place where a
  regression would most likely land.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines import embed_and_ingest
from src.pipelines.embed_and_ingest import (
    run_backfill,
    run_incremental,
    upsert_by_show_id,
)
from src.storage.episode_status import EpisodeStatusRepository


def _pipeline_stub() -> MagicMock:
    """A stand-in pipeline class whose `.run()` returns a benign stats
    dict so cursor-advancement / logging paths do not blow up."""
    stub = MagicMock()
    stub.return_value.run.return_value = {
        "success": 0,
        "errors": 0,
        "total": 0,
    }
    return stub


# ── Plumbing: entry points forward `episode_status_repo` to the pipeline ────


class TestEntrypointForwardsEpisodeStatusRepo:
    def test_run_incremental_passes_repo_to_pipeline(self) -> None:
        repo = MagicMock(spec=EpisodeStatusRepository)
        storage = MagicMock(get_shows_updated_since=lambda _: [])

        pipeline_stub = _pipeline_stub()
        with patch.object(embed_and_ingest, "EmbedAndIngestPipeline", pipeline_stub):
            # force_full=True skips the "no updates" short-circuit so the
            # pipeline actually gets constructed.
            run_incremental(
                storage=storage,
                embedding_backend=None,
                force_full=True,
                episode_status_repo=repo,
            )

        pipeline_stub.assert_called_once()
        kwargs = pipeline_stub.call_args.kwargs
        assert kwargs.get("episode_status_repo") is repo

    def test_run_backfill_forwards_repo_via_run_incremental(self) -> None:
        repo = MagicMock(spec=EpisodeStatusRepository)
        storage = MagicMock(get_shows_updated_since=lambda _: [])

        pipeline_stub = _pipeline_stub()
        with patch.object(embed_and_ingest, "EmbedAndIngestPipeline", pipeline_stub):
            run_backfill(
                storage=storage,
                embedding_backend=None,
                episode_status_repo=repo,
            )

        pipeline_stub.assert_called_once()
        assert pipeline_stub.call_args.kwargs.get("episode_status_repo") is repo

    def test_upsert_by_show_id_passes_repo_to_pipeline(self) -> None:
        repo = MagicMock(spec=EpisodeStatusRepository)

        fake_show = MagicMock(show_id="show:1")
        storage = MagicMock(get_shows=lambda: [fake_show])

        pipeline_stub = _pipeline_stub()
        pipeline_stub.return_value.run.return_value = {"success": 0}

        with patch.object(embed_and_ingest, "EmbedAndIngestPipeline", pipeline_stub):
            upsert_by_show_id(
                "show:1",
                storage=storage,
                embedding_backend=None,
                episode_status_repo=repo,
            )

        pipeline_stub.assert_called_once()
        assert pipeline_stub.call_args.kwargs.get("episode_status_repo") is repo


# ── Default behaviour: None preserves Phase 1 / unit-test call sites ───────


class TestDefaultsStayOptional:
    """Entry points must remain callable without the new parameter so
    every pre-Phase-2a test and caller keeps working. The pipeline
    receives `episode_status_repo=None` and silently skips the commit
    branch — Phase 2a's documented fallback when no repo is wired."""

    def test_run_incremental_defaults_repo_to_none(self) -> None:
        storage = MagicMock(get_shows_updated_since=lambda _: [])
        pipeline_stub = _pipeline_stub()
        with patch.object(embed_and_ingest, "EmbedAndIngestPipeline", pipeline_stub):
            run_incremental(
                storage=storage,
                embedding_backend=None,
                force_full=True,
            )
        assert pipeline_stub.call_args.kwargs.get("episode_status_repo") is None


# ── CLI wiring: run() must construct EpisodeStatusRepository ───────────────


class TestCliWiresRepo:
    """The literal `EpisodeStatusRepository(` must appear inside the
    module-level `run()` function that backs `python -m
    src.pipelines.embed_and_ingest`. Checked by AST rather than
    substring so a comment that merely names the class cannot mask
    its actual absence from the function body."""

    def _run_function_source(self) -> str:
        source = inspect.getsource(embed_and_ingest)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                return ast.unparse(node)
        pytest.fail("module-level `run()` function not found in embed_and_ingest")

    def test_run_constructs_episode_status_repository(self) -> None:
        body = self._run_function_source()
        tree = ast.parse(body)
        names_called: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    names_called.append(func.id)
                elif isinstance(func, ast.Attribute):
                    names_called.append(func.attr)
        assert "EpisodeStatusRepository" in names_called, (
            "CLI run() must construct an EpisodeStatusRepository and "
            "pass it to the pipeline entry points; otherwise the "
            "per-show DB metadata commit is silently disabled."
        )

    def test_run_passes_episode_status_repo_to_every_entrypoint(self) -> None:
        """Every `run_incremental` / `run_backfill` / `upsert_by_show_id`
        call inside `run()` must include `episode_status_repo=` as a
        keyword, otherwise the CLI silently reverts to the None default."""
        body = self._run_function_source()
        tree = ast.parse(body)
        entrypoints = {"run_incremental", "run_backfill", "upsert_by_show_id"}
        missing: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name: Any = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in entrypoints:
                kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
                if "episode_status_repo" not in kwargs:
                    missing.append(name)
        assert not missing, (
            f"CLI run() calls {missing} without `episode_status_repo=`; "
            f"every production entry point must forward the repo."
        )
