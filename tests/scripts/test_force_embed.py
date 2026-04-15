"""Tests for the manual rebuild CLI (`scripts/force_embed.py`).

Covers:
- Selection resolution: `--show-ids`, `--episode-ids`, unresolvable
  inputs, target_index → language mapping.
- Per-show orchestration: happy path, dry-run, per-show recoverable
  failure (ET2), systemic halt (ET1) with preserve semantics for
  already-committed shows.
- Commit policy: the tool invokes `mark_embedded_daily` (shared
  with `embed_and_ingest` CB1), which writes
  `embedding_status='done'` at the artifact-ready boundary.
  `mark_embedded_batch` stays on the legacy standalone path;
  `mark_embedding_metadata_only` stays for any future path that
  keeps status untouched.
- Exit codes: 0 / 1 / 3 / 4 / 5. Exit code 2 (missing
  `--allow-model-drift`) is enforced by argparse, covered by a
  SystemExit test.
- Import isolation: the script never reaches for
  `embed_episodes`, the ES service, or `SyncStateRepository`.
- V18: orchestrator code paths never mention `--allow-model-drift`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlite_utils import Database

from scripts import force_embed
from scripts.force_embed import (
    EXIT_ALL_UNRESOLVABLE,
    EXIT_EMPTY_INPUT,
    EXIT_OK,
    EXIT_PER_SHOW_FAILURE,
    EXIT_SYSTEMIC_HALT,
    ForceEmbedSummary,
    ResolvedShow,
    _split_csv,
    main,
    resolve_selection,
    run_force_embed,
)
from src.pipelines.embedding_catalog import MODEL_MAP
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
)
from src.pipelines.show_rebuild import ShowRebuildResult


# ── Fixtures ────────────────────────────────────────────────────────────────

MODEL = MODEL_MAP["zh"]


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name=MODEL,
        embedding_version="text-v1",
        embedding_dimensions=384,
    )


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "crawler.db")
    db.executescript("""
        CREATE TABLE shows (
            show_id TEXT PRIMARY KEY,
            target_index TEXT
        );
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            show_id TEXT,
            embedding_status TEXT,
            embedding_model TEXT,
            embedding_version TEXT,
            last_embedded_at TEXT,
            updated_at TEXT
        );
    """)
    return db


def _add_show(db: Database, show_id: str, target_index: str = "podcast-episodes-zh-tw") -> None:
    db["shows"].insert({"show_id": show_id, "target_index": target_index}, pk="show_id")


def _add_episode(db: Database, episode_id: str, show_id: str) -> None:
    db["episodes"].insert(
        {"episode_id": episode_id, "show_id": show_id},
        pk="episode_id",
    )


def _ok_result(show_id: str) -> ShowRebuildResult:
    return ShowRebuildResult(
        show_id=show_id,
        status="ok",
        cache_written=True,
        episode_count=2,
        identity_used=_identity(),
        new_last_embedded_at=datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc),
        error_code=None,
        error_message=None,
    )


def _failed_result(show_id: str, code: str = "embedding_runtime_error") -> ShowRebuildResult:
    return ShowRebuildResult(
        show_id=show_id,
        status="failed",
        cache_written=False,
        episode_count=0,
        identity_used=_identity(),
        new_last_embedded_at=None,
        error_code=code,
        error_message=f"boom on {show_id}",
    )


# ── _split_csv ──────────────────────────────────────────────────────────────


class TestSplitCsv:
    def test_empty_returns_empty_list(self) -> None:
        assert _split_csv("") == []
        assert _split_csv(None) == []

    def test_trims_and_dedups(self) -> None:
        assert _split_csv("a, b,  a , c ") == ["a", "b", "c"]


# ── resolve_selection ───────────────────────────────────────────────────────


class TestResolveSelection:
    def test_show_ids_are_resolved_via_target_index(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:zh", "podcast-episodes-zh-tw")
        _add_show(db, "show:en", "podcast-episodes-en")

        resolved, unresolvable = resolve_selection(
            db, show_ids=["show:zh", "show:en"], episode_ids=[],
        )

        assert resolved == [
            ResolvedShow(show_id="show:en", language="en"),
            ResolvedShow(show_id="show:zh", language="zh-tw"),
        ]
        assert unresolvable == []

    def test_show_with_unknown_target_index_goes_unresolvable(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:wild", "podcast-episodes-klingon")

        resolved, unresolvable = resolve_selection(
            db, show_ids=["show:wild"], episode_ids=[],
        )

        assert resolved == []
        assert unresolvable == ["show:wild"]

    def test_episode_ids_resolve_to_owning_show(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, "ep:a", "show:1")
        _add_episode(db, "ep:b", "show:1")

        resolved, unresolvable = resolve_selection(
            db, show_ids=[], episode_ids=["ep:a", "ep:b"],
        )

        # Both episodes collapse into their single owning show.
        assert resolved == [ResolvedShow(show_id="show:1", language="zh-tw")]
        assert unresolvable == []

    def test_unknown_episode_id_marked_unresolvable(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, "ep:real", "show:1")

        resolved, unresolvable = resolve_selection(
            db, show_ids=[], episode_ids=["ep:real", "ep:ghost"],
        )

        assert [r.show_id for r in resolved] == ["show:1"]
        assert unresolvable == ["ep:ghost"]

    def test_show_and_episode_inputs_merged_and_sorted(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:1", "podcast-episodes-zh-tw")
        _add_show(db, "show:2", "podcast-episodes-en")
        _add_episode(db, "ep:x", "show:2")

        resolved, unresolvable = resolve_selection(
            db, show_ids=["show:1"], episode_ids=["ep:x"],
        )

        # Sorted ascending by show_id so repeat runs hit shows in the
        # same order — preserve semantics depend on this determinism.
        assert resolved == [
            ResolvedShow(show_id="show:1", language="zh-tw"),
            ResolvedShow(show_id="show:2", language="en"),
        ]
        assert unresolvable == []


# ── run_force_embed ─────────────────────────────────────────────────────────


class TestRunForceEmbed:
    def _prep_db(self, tmp_path: Path) -> Database:
        db = _make_db(tmp_path)
        _add_show(db, "show:1")
        _add_episode(db, "ep:1", "show:1")
        _add_episode(db, "ep:2", "show:1")
        return db

    def test_happy_path_commits_artifact_ready_metadata(self, tmp_path: Path) -> None:
        db = self._prep_db(tmp_path)
        resolved = [ResolvedShow(show_id="show:1", language="zh-tw")]

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            return_value=_ok_result("show:1"),
        ):
            summary = run_force_embed(
                db=db,
                resolved_shows=resolved,
                cache_dir=tmp_path / "cache",
                embedding_input_dir=tmp_path / "inputs",
                dry_run=False,
            )

        assert summary.exit_code == EXIT_OK
        assert summary.rebuild_attempted == 1
        assert summary.rebuild_succeeded == 1
        assert summary.rebuild_failed == []
        assert summary.db_metadata_updated == 2

        row = db.execute(
            "SELECT embedding_model, embedding_version, embedding_status, last_embedded_at "
            "FROM episodes WHERE episode_id = 'ep:1'"
        ).fetchone()
        assert row[0] == MODEL
        assert row[1] == "text-v1"
        # Phase 2b-A V1e-A: force_embed writes artifact-ready status.
        assert row[2] == "done"
        assert row[3] == "2026-04-15T12:00:00+00:00"

    def test_dry_run_skips_primitive_and_db(self, tmp_path: Path) -> None:
        db = self._prep_db(tmp_path)
        resolved = [ResolvedShow(show_id="show:1", language="zh-tw")]

        with patch(
            "scripts.force_embed.rebuild_show_cache",
        ) as rebuild_mock:
            summary = run_force_embed(
                db=db,
                resolved_shows=resolved,
                cache_dir=tmp_path / "cache",
                embedding_input_dir=None,
                dry_run=True,
            )

        rebuild_mock.assert_not_called()
        assert summary.exit_code == EXIT_OK
        assert summary.rebuild_attempted == 0
        assert summary.rebuild_succeeded == 0
        assert summary.db_metadata_updated == 0

        # DB metadata untouched.
        row = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'ep:1'"
        ).fetchone()
        assert row[0] is None

    def test_per_show_failure_continues_and_exits_4(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _add_show(db, "show:a")
        _add_show(db, "show:b")
        _add_episode(db, "ep:a1", "show:a")
        _add_episode(db, "ep:b1", "show:b")

        resolved = [
            ResolvedShow(show_id="show:a", language="zh-tw"),
            ResolvedShow(show_id="show:b", language="zh-tw"),
        ]

        def fake_rebuild(*, show_id: str, **_: object) -> ShowRebuildResult:
            if show_id == "show:a":
                return _failed_result("show:a")
            return _ok_result("show:b")

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            side_effect=fake_rebuild,
        ):
            summary = run_force_embed(
                db=db,
                resolved_shows=resolved,
                cache_dir=tmp_path / "cache",
                embedding_input_dir=None,
                dry_run=False,
            )

        assert summary.exit_code == EXIT_PER_SHOW_FAILURE
        assert summary.rebuild_attempted == 2
        assert summary.rebuild_succeeded == 1
        assert summary.rebuild_failed == [("show:a", "embedding_runtime_error")]
        # show:b was committed; show:a was not.
        assert summary.db_metadata_updated == 1

        a_model = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'ep:a1'"
        ).fetchone()[0]
        b_model = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'ep:b1'"
        ).fetchone()[0]
        assert a_model is None
        assert b_model == MODEL

    def test_systemic_halt_preserves_earlier_shows(self, tmp_path: Path) -> None:
        """ET1 on the second show: first show's commit remains; second
        and third are not attempted."""
        db = _make_db(tmp_path)
        for sid in ("show:1", "show:2", "show:3"):
            _add_show(db, sid)
            _add_episode(db, f"{sid}:ep", sid)

        resolved = [
            ResolvedShow(show_id="show:1", language="zh-tw"),
            ResolvedShow(show_id="show:2", language="zh-tw"),
            ResolvedShow(show_id="show:3", language="zh-tw"),
        ]

        call_count = {"n": 0}

        def fake_rebuild(*, show_id: str, **_: object) -> ShowRebuildResult:
            call_count["n"] += 1
            if show_id == "show:1":
                return _ok_result("show:1")
            if show_id == "show:2":
                raise EmbeddingDimensionContractViolation(
                    expected=384, actual=512, model_name=MODEL,
                )
            pytest.fail("show:3 must not be attempted after ET1")

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            side_effect=fake_rebuild,
        ):
            summary = run_force_embed(
                db=db,
                resolved_shows=resolved,
                cache_dir=tmp_path / "cache",
                embedding_input_dir=None,
                dry_run=False,
            )

        assert summary.exit_code == EXIT_SYSTEMIC_HALT
        assert summary.rebuild_systemic_halted is True
        assert summary.rebuild_attempted == 2  # show:1 + in-flight show:2
        assert summary.rebuild_succeeded == 1  # only show:1
        assert call_count["n"] == 2  # show:3 was not touched

        # show:1 preserved.
        row_1 = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'show:1:ep'"
        ).fetchone()
        assert row_1[0] == MODEL

        # show:2 in-flight — no DB commit.
        row_2 = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'show:2:ep'"
        ).fetchone()
        assert row_2[0] is None

        # show:3 not attempted.
        row_3 = db.execute(
            "SELECT embedding_model FROM episodes WHERE episode_id = 'show:3:ep'"
        ).fetchone()
        assert row_3[0] is None

    def test_commit_uses_mark_embedded_daily_not_legacy_writers(self, tmp_path: Path) -> None:
        """V1e-A: force_embed's artifact-ready commit must route through
        the shared `mark_embedded_daily` writer, not the legacy
        `mark_embedded_batch` (embed_episodes only) nor
        `mark_embedding_metadata_only` (status-untouched path)."""
        db = self._prep_db(tmp_path)
        resolved = [ResolvedShow(show_id="show:1", language="zh-tw")]

        fake_repo = MagicMock()
        fake_repo.mark_embedded_daily.return_value = 2

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            return_value=_ok_result("show:1"),
        ), patch(
            "scripts.force_embed.EpisodeStatusRepository",
            return_value=fake_repo,
        ):
            run_force_embed(
                db=db,
                resolved_shows=resolved,
                cache_dir=tmp_path / "cache",
                embedding_input_dir=None,
                dry_run=False,
            )

        assert fake_repo.mark_embedded_daily.call_count == 1
        assert fake_repo.mark_embedded_batch.call_count == 0
        assert fake_repo.mark_embedding_metadata_only.call_count == 0
        kwargs = fake_repo.mark_embedded_daily.call_args.kwargs
        assert sorted(kwargs["episode_ids"]) == ["ep:1", "ep:2"]
        assert kwargs["model"] == MODEL
        assert kwargs["version"] == "text-v1"  # cleaned form only


# ── main() CLI ──────────────────────────────────────────────────────────────


class TestMainCli:
    def test_missing_allow_flag_exits_2(self, tmp_path: Path, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--show-ids", "show:1"])
        # argparse exits 2 on missing required options — our chosen
        # slot for this error class.
        assert excinfo.value.code == 2

    def test_empty_input_exits_1(self, tmp_path: Path, capsys) -> None:
        rc = main(["--allow-model-drift", "--db", str(tmp_path / "x.db")])
        assert rc == EXIT_EMPTY_INPUT

    def test_all_unresolvable_exits_3(self, tmp_path: Path, capsys) -> None:
        db_path = tmp_path / "crawler.db"
        _make_db(db_path.parent)

        rc = main([
            "--allow-model-drift",
            "--db", str(db_path),
            "--episode-ids", "ep:ghost",
        ])
        assert rc == EXIT_ALL_UNRESOLVABLE

        output = capsys.readouterr().out
        assert "ep:ghost" in output
        assert "resolved_unique_shows:    0" in output

    def test_happy_path_through_main(self, tmp_path: Path, capsys) -> None:
        db_path = tmp_path / "crawler.db"
        db = _make_db(db_path.parent)
        _add_show(db, "show:1")
        _add_episode(db, "ep:1", "show:1")

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            return_value=_ok_result("show:1"),
        ):
            rc = main([
                "--allow-model-drift",
                "--db", str(db_path),
                "--show-ids", "show:1",
                "--cache-dir", str(tmp_path / "cache"),
            ])

        assert rc == EXIT_OK
        output = capsys.readouterr().out
        assert "rebuild_succeeded:        1" in output
        assert "exit_code:                0" in output


# ── Import / V18 isolation ──────────────────────────────────────────────────


def _module_symbols(module) -> set[str]:
    """Collect every imported module path mentioned in `module`.

    AST-based so a docstring naming a forbidden module doesn't trip
    the check.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


class TestIsolation:
    def test_force_embed_does_not_import_embed_episodes(self) -> None:
        names = _module_symbols(force_embed)
        assert not any(
            n == "src.pipelines.embed_episodes" or n.endswith(".embed_episodes")
            for n in names
        ), f"force_embed must not import embed_episodes; imports: {names}"

    def test_force_embed_does_not_import_embed_and_ingest(self) -> None:
        names = _module_symbols(force_embed)
        assert not any(
            "embed_and_ingest" in n for n in names
        ), f"force_embed must not import embed_and_ingest; imports: {names}"

    def test_force_embed_does_not_import_es_service_or_sync_state(self) -> None:
        names = _module_symbols(force_embed)
        for forbidden in (
            "ElasticsearchService",
            "SyncStateRepository",
            "search_sync_state",
            "es_service",
        ):
            for n in names:
                assert forbidden not in n, (
                    f"force_embed must not import {forbidden}; saw: {n}"
                )


class TestV18:
    """The `--allow-model-drift` flag must live only inside force_embed
    and the Phase 2a design doc. Any other appearance (orchestrator,
    daily pipeline) would mean the daily run could be coaxed into
    drifting — a contract violation."""

    def _project_root(self) -> Path:
        # tests/scripts/test_force_embed.py → …/podcast-search
        return Path(__file__).resolve().parents[2]

    def test_allow_model_drift_not_present_in_src_or_other_scripts(self) -> None:
        root = self._project_root()
        result = subprocess.run(
            ["grep", "-r", "-l", "--allow-model-drift", "src/", "scripts/"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        hits = [line for line in result.stdout.splitlines() if line.strip()]
        # Only scripts/force_embed.py and the test file are allowed to
        # reference the flag literally.
        allowed = {"scripts/force_embed.py"}
        disallowed = [h for h in hits if h not in allowed]
        assert disallowed == [], (
            f"--allow-model-drift appeared in unexpected paths: {disallowed}"
        )


# ── Step 7: advisory output + embedding_status='done' ───────────────────────


def _parse_advisory_block(stderr_text: str) -> dict | None:
    """Extract the JSON between the PHASE2B_ADVISORY marker lines, if any."""
    begin = force_embed._ADVISORY_BEGIN
    end = force_embed._ADVISORY_END
    if begin not in stderr_text or end not in stderr_text:
        return None
    start = stderr_text.index(begin) + len(begin)
    stop = stderr_text.index(end)
    import json as _json
    return _json.loads(stderr_text[start:stop].strip())


def _capture_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[str, str, int]:
    """Run `main(argv)` with sys.stdout / sys.stderr monkey-patched to
    StringIO buffers so advisory tests don't fight with pytest's own
    capture mechanism (which interacts badly with `setup_logging`'s
    handler reseating). Returns (stdout_text, stderr_text, exit_code)."""
    import io as _io
    out_buf = _io.StringIO()
    err_buf = _io.StringIO()
    monkeypatch.setattr("sys.stdout", out_buf)
    monkeypatch.setattr("sys.stderr", err_buf)
    rc = main(argv)
    return out_buf.getvalue(), err_buf.getvalue(), rc


class TestAdvisoryEmission:
    def _setup_two_shows(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "crawler.db"
        db = _make_db(db_path.parent)
        _add_show(db, "show:1")
        _add_show(db, "show:2")
        _add_episode(db, "ep:1", "show:1")
        _add_episode(db, "ep:2", "show:2")
        return db_path

    def test_advisory_emitted_on_successful_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = self._setup_two_shows(tmp_path)
        with patch(
            "scripts.force_embed.rebuild_show_cache",
            side_effect=lambda show_id, **_: _ok_result(show_id),
        ):
            _, err, rc = _capture_main(monkeypatch, [
                "--allow-model-drift", "--db", str(db_path),
                "--show-ids", "show:1,show:2",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert rc == EXIT_OK
        advisory = _parse_advisory_block(err)
        assert advisory is not None, (
            "advisory marker block must appear on stderr"
        )
        assert sorted(advisory["affected_show_ids"]) == ["show:1", "show:2"]

    def test_advisory_suppressed_on_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = self._setup_two_shows(tmp_path)
        _, err, rc = _capture_main(monkeypatch, [
            "--allow-model-drift", "--db", str(db_path),
            "--show-ids", "show:1",
            "--dry-run",
            "--cache-dir", str(tmp_path / "cache"),
        ])
        assert rc == EXIT_OK
        assert force_embed._ADVISORY_BEGIN not in err

    def test_advisory_suppressed_on_per_show_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exit 4 (partial success) must NOT emit advisory — operator has
        to diagnose failures before being handed off."""
        db_path = self._setup_two_shows(tmp_path)
        with patch(
            "scripts.force_embed.rebuild_show_cache",
            side_effect=lambda show_id, **_: _failed_result(show_id),
        ):
            _, err, rc = _capture_main(monkeypatch, [
                "--allow-model-drift", "--db", str(db_path),
                "--show-ids", "show:1",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert rc == EXIT_PER_SHOW_FAILURE
        assert force_embed._ADVISORY_BEGIN not in err

    def test_advisory_suppressed_on_systemic_halt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = self._setup_two_shows(tmp_path)

        def _raise(show_id, **_):
            raise EmbeddingDimensionContractViolation(
                model_name=MODEL, expected=384, actual=100,
            )

        with patch("scripts.force_embed.rebuild_show_cache", side_effect=_raise):
            _, err, rc = _capture_main(monkeypatch, [
                "--allow-model-drift", "--db", str(db_path),
                "--show-ids", "show:1",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert rc == EXIT_SYSTEMIC_HALT
        assert force_embed._ADVISORY_BEGIN not in err

    def test_advisory_not_emitted_on_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """stdout is reserved for the human-readable summary; advisory
        must live ONLY on stderr so shell pipelines and operator
        tooling can parse without stderr->stdout redirection."""
        db_path = self._setup_two_shows(tmp_path)
        with patch(
            "scripts.force_embed.rebuild_show_cache",
            side_effect=lambda show_id, **_: _ok_result(show_id),
        ):
            out, _, _ = _capture_main(monkeypatch, [
                "--allow-model-drift", "--db", str(db_path),
                "--show-ids", "show:1",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert force_embed._ADVISORY_BEGIN not in out
        assert force_embed._ADVISORY_END not in out


class TestAdvisorySchema:
    def test_advisory_payload_schema_fields(self) -> None:
        """Direct unit-level check of `_build_advisory` — schema shape."""
        payload = force_embed._build_advisory(["show:a", "show:b"])
        assert payload["schema_version"] == 1
        assert payload["state"] == "artifact_only"
        assert payload["affected_show_ids"] == ["show:a", "show:b"]
        assert isinstance(payload["canonical_handoff_command_template"], str)
        assert isinstance(payload["next_step_commands"], list)

    def test_canonical_handoff_command_template_exact_string(self) -> None:
        """Template is part of the CLI contract. Any rename of
        embed_and_ingest invocation here breaks Phase 2b-A §3.5 and
        must go through a schema_version bump."""
        payload = force_embed._build_advisory(["show:x"])
        assert (
            payload["canonical_handoff_command_template"]
            == "python -m src.pipelines.embed_and_ingest --show-id <show_id>"
        )

    def test_next_step_commands_expand_from_template(self) -> None:
        ids = ["show:a", "show:b:weird/slash"]
        payload = force_embed._build_advisory(ids)
        template = payload["canonical_handoff_command_template"]
        for i, sid in enumerate(ids):
            assert payload["next_step_commands"][i] == template.replace(
                "<show_id>", sid,
            )

    def test_next_step_commands_length_matches_affected(self) -> None:
        payload = force_embed._build_advisory(["s:1", "s:2", "s:3"])
        assert len(payload["next_step_commands"]) == len(
            payload["affected_show_ids"],
        )


class TestStatusDoneOnSuccess:
    def test_force_embed_sets_embedding_status_done(
        self, tmp_path: Path
    ) -> None:
        """V1e-A CL4: after a successful force_embed, the row's
        `embedding_status` must read 'done'."""
        db_path = tmp_path / "crawler.db"
        db = _make_db(db_path.parent)
        _add_show(db, "show:1")
        _add_episode(db, "ep:1", "show:1")

        with patch(
            "scripts.force_embed.rebuild_show_cache",
            return_value=_ok_result("show:1"),
        ):
            rc = main([
                "--allow-model-drift", "--db", str(db_path),
                "--show-ids", "show:1",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert rc == EXIT_OK
        row = db.execute(
            "SELECT embedding_status FROM episodes WHERE episode_id='ep:1'"
        ).fetchone()
        assert row[0] == "done"


class TestAdvisorySurfaceUniqueness:
    """Only one template string may exist in force_embed.py. If
    someone accidentally paste a second variant of the handoff command,
    advisory consumers wouldn't know which is canonical."""

    def test_force_embed_contains_exactly_one_handoff_template(self) -> None:
        source = Path("scripts/force_embed.py").read_text()
        template = "python -m src.pipelines.embed_and_ingest --show-id <show_id>"
        assert source.count(template) == 1, (
            "canonical handoff template string must appear exactly once"
        )
