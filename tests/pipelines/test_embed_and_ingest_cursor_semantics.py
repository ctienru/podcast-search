"""Phase 1 Sev0 correctness tests — ingest commit semantics.

Covers spec §5.1 / §5.2.1 (from 2026-04-14-phase1-ingest-sev0-implementation.md):
- Partial failure freezes cursor, skips sync_state flush, exits non-zero
- Empty candidate set freezes both cursor fields (R7)
- All-success happy path (regression guard)
- `environment` is a required kwarg on EmbedAndIngestPipeline
- Daily path never writes non-'local' to search_sync_state
- Error funneling (§4.0 / Step 2.5): bulk/transform/flush exceptions counted in stats["errors"]

Design notes:
- In-memory SQLite for sync_state delta assertions (no DB pollution between tests)
- Primary mixed-partial evidence is in-process (mock monkeypatch) — subprocess cannot
  inherit mocks; the subprocess-based test (test_full_failure_real_es_returncode)
  only exercises FULL failure, not mixed partial
- Tests assert post-Phase-1 behavior, so they FAIL against current main and PASS
  after Steps 2-6 land (TDD)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest
from sqlite_utils import Database

from src.pipelines.embed_and_ingest import EmbedAndIngestPipeline
from src.storage.sync_state import SyncStateRepository


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_sync_repo() -> SyncStateRepository:
    """Real repo backed by in-memory SQLite so we can query rows back."""
    return SyncStateRepository(Database(":memory:"))


def _snapshot_sync_state(repo: SyncStateRepository) -> list[tuple]:
    """Return sorted tuples of (entity_id, environment, sync_status) rows."""
    rows = repo._db.execute(
        "SELECT entity_id, environment, sync_status FROM search_sync_state ORDER BY entity_id, environment"
    ).fetchall()
    return sorted(rows)


def _make_pipeline(tmp_path: Path, environment: str = "local", sync_repo=None) -> EmbedAndIngestPipeline:
    """Construct a pipeline wired for in-memory testing.

    NOTE: `environment` is passed as a kwarg; after Step 2 this becomes required
    (kwarg-only). Tests that exercise the required-kwarg behavior must NOT
    use this helper — they must construct EmbedAndIngestPipeline directly.
    """
    embedding_input_dir = tmp_path / "embedding_input" / "episodes"
    cleaned_dir = tmp_path / "cleaned" / "episodes"
    embedding_input_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    pipeline = EmbedAndIngestPipeline(
        environment=environment,
        es_service=MagicMock(),
        embedding_backend=MagicMock(embed_batch=lambda texts, lang: [[0.1] * 384] * len(texts)),
        storage=MagicMock(get_shows=lambda: [], get_shows_updated_since=lambda s: []),
        sync_repo=sync_repo,
    )
    pipeline.EMBEDDING_INPUT_DIR = embedding_input_dir
    pipeline.CLEANED_EPISODES_DIR = cleaned_dir
    return pipeline


def _add_episode(pipeline: EmbedAndIngestPipeline, episode_id: str, show_id: str) -> None:
    """Inject a pre-built episode into the pipeline's caches + on-disk input file."""
    pipeline._cleaned_episode_cache[episode_id] = {
        "episode_id": episode_id,
        "show_id": show_id,
        "target_index": "podcast-episodes-zh-tw",
        "cleaned": {"normalized": {"title": f"Title {episode_id}", "description": "desc"}},
        "original_meta": {"pub_date": None, "duration": None, "audio_url": None, "language": "zh-tw"},
    }
    pipeline._show_cache[show_id] = {
        "show_id": show_id,
        "title": "Test Show",
        "author": "Author",
        "image_url": None,
        "external_urls": {},
        "target_index": "podcast-episodes-zh-tw",
    }
    (pipeline.EMBEDDING_INPUT_DIR / f"{episode_id}.json").write_text(
        json.dumps({"episode_id": episode_id, "show_id": show_id, "embedding_input": {"text": "text"}}),
        encoding="utf-8",
    )


def _fake_bulk(items: Iterable[tuple]):
    """streaming_bulk side_effect that consumes the actions generator then yields items."""
    def _side_effect(client, actions, **kwargs):
        list(actions)  # populate _episode_aliases as production does
        return iter(list(items))
    return _side_effect


# ── §5.1: pipeline.run() level ─────────────────────────────────────────────────


class TestPartialFailure:
    """V1 core: any doc error → no flush, stats.errors > 0."""

    def test_partial_failure_no_sync_state_flush(self, tmp_path):
        """R3: mixed success/error → zero rows marked 'synced' for this batch."""
        repo = _make_sync_repo()
        pipeline = _make_pipeline(tmp_path, sync_repo=repo)
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")

        before = _snapshot_sync_state(repo)

        mixed = [
            (True,  {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}}),
            (False, {"index": {"_id": "ep:002", "error": {"type": "mapper_error"}}}),
        ]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(mixed)):
            pipeline.run()

        after = _snapshot_sync_state(repo)
        assert after == before, f"sync_state should not change on partial failure, got delta: {set(after) - set(before)}"

    def test_partial_failure_returns_nonzero_errors(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        _add_episode(pipeline, "ep:001", "show:1")
        _add_episode(pipeline, "ep:002", "show:1")

        mixed = [
            (True,  {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}}),
            (False, {"index": {"_id": "ep:002", "error": {"type": "x"}}}),
        ]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(mixed)):
            stats = pipeline.run()

        assert stats["errors"] > 0
        assert stats["success"] == 1


class TestEnvironmentKwarg:
    """§3.1.1 architecture rule: environment is caller-injected, no shared-core default."""

    def test_environment_required_at_construction(self):
        """§4.1: constructing EmbedAndIngestPipeline without `environment` must raise."""
        with pytest.raises(TypeError):
            EmbedAndIngestPipeline(
                es_service=MagicMock(),
                embedding_backend=MagicMock(),
                storage=MagicMock(get_shows=lambda: []),
                sync_repo=None,
            )

    def test_daily_incremental_only_writes_local_to_sync_state(self, tmp_path):
        """§3.1.1 規則 F: daily path must write environment='local', never 'production'/'default'."""
        repo = _make_sync_repo()
        pipeline = _make_pipeline(tmp_path, environment="local", sync_repo=repo)
        _add_episode(pipeline, "ep:001", "show:1")

        ok = [(True, {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}})]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(ok)):
            pipeline.run()

        environments = {row[1] for row in _snapshot_sync_state(repo)}
        assert environments == {"local"}, f"daily path must only write 'local', got {environments}"


# ── §4.0 / Step 2.5: error funneling ──────────────────────────────────────────


class TestErrorFunneling:
    """A1 coverage: bulk / transform / flush exceptions must all land in stats['errors']."""

    def test_bulk_exception_counted_in_errors(self, tmp_path):
        """streaming_bulk itself raises (e.g. ES client connection error) → errors > 0."""
        pipeline = _make_pipeline(tmp_path)
        _add_episode(pipeline, "ep:001", "show:1")

        def _raise(*a, **kw):
            list(a[1])  # consume actions as production does
            raise RuntimeError("simulated bulk request exception")

        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_raise):
            stats = pipeline.run()

        assert stats["errors"] > 0, "bulk request exceptions must be counted as errors (A1 funneling)"

    def test_transform_exception_counted_in_errors(self, tmp_path):
        """build_actions / load_embedding_inputs raises → errors > 0 (not uncaught)."""
        pipeline = _make_pipeline(tmp_path)
        _add_episode(pipeline, "ep:001", "show:1")

        def _raise_transform(*a, **kw):
            raise RuntimeError("simulated transform exception")

        with patch.object(pipeline, "build_actions", side_effect=_raise_transform):
            stats = pipeline.run()

        assert stats["errors"] > 0, "transform exceptions must be counted as errors (A1 funneling)"

    def test_flush_exception_counted_in_errors(self, tmp_path):
        """sync_repo.mark_done / commit raises → errors > 0 (don't silently succeed)."""
        repo = MagicMock(spec=SyncStateRepository)
        repo._db = MagicMock()
        repo.mark_done.side_effect = RuntimeError("simulated flush exception")

        pipeline = _make_pipeline(tmp_path, sync_repo=repo)
        _add_episode(pipeline, "ep:001", "show:1")

        ok = [(True, {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}})]
        with patch("src.pipelines.embed_and_ingest.streaming_bulk", side_effect=_fake_bulk(ok)):
            stats = pipeline.run()

        assert stats["errors"] > 0, "flush exceptions must be counted as errors (A1 funneling)"


# ── run_incremental() level: cursor + CLI SystemExit ──────────────────────────


class TestRunIncrementalCursor:
    """§5.1 V1 cursor semantics + R7 empty-candidate freeze."""

    def test_partial_failure_freezes_cursor(self, tmp_path, monkeypatch):
        """R1: partial failure → cursor file byte-identical before and after."""
        from src.pipelines import embed_and_ingest as mod

        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text('{"episodes-zh-tw": {"last_ingest_at": "2020-01-01T00:00:00Z", "last_run_at": "2020-01-01T00:00:00Z"}}')
        before = cursor_path.read_bytes()

        repo = _make_sync_repo()
        # Minimal show stub to satisfy get_shows_updated_since
        show = MagicMock(show_id="show:1", updated_at="2099-01-01T00:00:00Z")
        storage = MagicMock(
            get_shows=lambda: [show],
            get_shows_updated_since=lambda s: [show],
        )

        mixed = [
            (True,  {"index": {"_id": "ep:001", "_index": "episodes-zh-tw-v3"}}),
            (False, {"index": {"_id": "ep:002", "error": {"type": "x"}}}),
        ]

        with patch.object(mod, "streaming_bulk", side_effect=_fake_bulk(mixed)), \
             patch.object(mod, "EmbedAndIngestPipeline") as MockP:
            inst = MagicMock()
            inst.run.return_value = {"success": 1, "errors": 1, "total": 2}
            MockP.return_value = inst
            mod.run_incremental(
                storage=storage,
                embedding_backend=None,
                force_full=True,
                cursor_path=cursor_path,
                sync_repo=repo,
            )

        after = cursor_path.read_bytes()
        assert after == before, "partial failure must not advance cursor"

    def test_all_success_advances_cursor(self, tmp_path, monkeypatch):
        """Happy path regression: errors==0 → cursor updates; sync_state has 'local' rows."""
        from src.pipelines import embed_and_ingest as mod

        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text('{"episodes-zh-tw": {"last_ingest_at": "2020-01-01T00:00:00Z", "last_run_at": "2020-01-01T00:00:00Z"}}')

        repo = _make_sync_repo()
        show = MagicMock(show_id="show:1", updated_at="2099-01-01T00:00:00Z")
        storage = MagicMock(
            get_shows=lambda: [show],
            get_shows_updated_since=lambda s: [show],
        )

        with patch.object(mod, "EmbedAndIngestPipeline") as MockP:
            inst = MagicMock()
            inst.run.return_value = {"success": 2, "errors": 0, "total": 2}
            MockP.return_value = inst
            mod.run_incremental(
                storage=storage,
                embedding_backend=None,
                force_full=True,
                cursor_path=cursor_path,
                sync_repo=repo,
            )

        after = json.loads(cursor_path.read_text())
        # At least one alias's last_ingest_at must have moved past the baseline.
        baseline = "2020-01-01T00:00:00Z"
        assert any(
            v.get("last_ingest_at", baseline) > baseline
            for v in after.values()
        ), f"successful run must advance cursor, got {after}"

    def test_empty_candidate_set_freezes_both_cursor_fields(self, tmp_path, monkeypatch):
        """R7: no updated shows → neither last_ingest_at nor last_run_at moves."""
        from src.pipelines import embed_and_ingest as mod

        cursor_path = tmp_path / "cursor.json"
        initial = '{"episodes-zh-tw": {"last_ingest_at": "2099-01-01T00:00:00Z", "last_run_at": "2099-01-01T00:00:00Z"}}'
        cursor_path.write_text(initial)
        before = cursor_path.read_bytes()

        storage = MagicMock(
            get_shows=lambda: [],
            get_shows_updated_since=lambda s: [],  # empty candidate set
        )

        mod.run_incremental(
            storage=storage,
            embedding_backend=None,
            force_full=False,  # incremental with empty candidate set
            cursor_path=cursor_path,
            sync_repo=None,
        )

        after = cursor_path.read_bytes()
        assert after == before, "empty candidate set must freeze cursor entirely (R7)"


# ── §5.2.1: in-process CLI exit propagation ──────────────────────────────────


class TestCliExitPropagation:
    """§5.2.1: mixed partial in daily/force-full path → SystemExit(1)."""

    def test_partial_failure_cli_systemexit_in_process(self, tmp_path, monkeypatch):
        """Mixed partial in daily/force-full path → SystemExit(1) + cursor frozen + no flush."""
        from src.pipelines import embed_and_ingest as mod

        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text('{"episodes-zh-tw": {"last_ingest_at": "2020-01-01T00:00:00Z", "last_run_at": "2020-01-01T00:00:00Z"}}')
        before = cursor_path.read_bytes()

        # Route CLI through our tmp cursor path
        monkeypatch.setattr(mod.settings, "INGEST_CURSOR_PATH", cursor_path, raising=False)
        monkeypatch.setattr(sys, "argv", ["embed_and_ingest", "--force-full", "--show-ids", "show:1"])

        repo = _make_sync_repo()
        before_rows = _snapshot_sync_state(repo)

        # Patch SyncStateRepository construction in CLI run() so we get our in-memory repo
        monkeypatch.setattr(mod, "SyncStateRepository", lambda db: repo)
        monkeypatch.setattr(mod, "Database", lambda path: MagicMock())
        # CI has no real SQLITE_PATH; short-circuit create_storage() so run_incremental
        # doesn't try to open a non-existent DB file before the mocked Pipeline runs.
        monkeypatch.setattr(mod, "create_storage", lambda: MagicMock())

        with patch.object(mod, "EmbedAndIngestPipeline") as MockP:
            inst = MagicMock()
            inst.run.return_value = {"success": 1, "errors": 1, "total": 2}
            MockP.return_value = inst

            with pytest.raises(SystemExit) as exc:
                mod.run()

        assert exc.value.code == 1
        assert cursor_path.read_bytes() == before, "cursor must not advance on CLI partial failure"
        assert _snapshot_sync_state(repo) == before_rows, "sync_state must not change on CLI partial failure"

    def test_force_full_partial_failure_systemexit(self, tmp_path, monkeypatch):
        """--force-full + mixed partial → SystemExit(1). Same guarantees as daily path."""
        from src.pipelines import embed_and_ingest as mod

        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text('{}')
        monkeypatch.setattr(mod.settings, "INGEST_CURSOR_PATH", cursor_path, raising=False)
        monkeypatch.setattr(sys, "argv", ["embed_and_ingest", "--force-full"])
        monkeypatch.setattr(mod, "SyncStateRepository", lambda db: _make_sync_repo())
        monkeypatch.setattr(mod, "Database", lambda path: MagicMock())
        monkeypatch.setattr(mod, "create_storage", lambda: MagicMock())

        with patch.object(mod, "EmbedAndIngestPipeline") as MockP:
            inst = MagicMock()
            inst.run.return_value = {"success": 3, "errors": 2, "total": 5}
            MockP.return_value = inst

            with pytest.raises(SystemExit) as exc:
                mod.run()

        assert exc.value.code == 1


# ── §5.2.2: process-level FULL failure smoke (integration, requires subprocess) ─


@pytest.mark.integration
class TestProcessLevelFullFailure:
    """§5.2.2 — subprocess cannot inherit mocks; covers FULL failure only, not mixed."""

    def test_full_failure_real_es_returncode(self, tmp_path):
        """Unreachable ES endpoint → non-zero returncode + cursor not advanced.

        NOTE: this subprocess fully boots the pipeline (model load + scans all
        embedding_input files) so it takes several minutes in production layouts.
        Marker `integration` keeps it out of the default CI run.
        """
        import os
        import subprocess

        cursor_path = tmp_path / "cursor.json"
        cursor_path.write_text('{}')
        before = cursor_path.read_bytes()

        # Limit to a real show so the pipeline still has actions to send (and thus
        # actually tries to hit the unreachable ES endpoint), then fails at bulk.
        env = {
            **os.environ,
            "ES_HOST": "http://127.0.0.1:9999",  # unreachable
            "INGEST_CURSOR_PATH": str(cursor_path),
        }
        proc = subprocess.run(
            [sys.executable, "-m", "src.pipelines.embed_and_ingest",
             "--force-full", "--show-ids", "show:apple:1580754352"],
            capture_output=True,
            env=env,
            timeout=600,
        )

        assert proc.returncode != 0, f"full failure must exit non-zero; stderr: {proc.stderr.decode()[:500]}"
        assert cursor_path.read_bytes() == before, "cursor must not advance on full failure"
