"""Fixture-only contract tests for the create-only observation diagnostic CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.seeding import derive_seed

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "observation_value_diagnostic", ROOT / "src/scripts/observation_value_diagnostic.py"
)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_tape_is_joint_normal_action_only_and_deterministic() -> None:
    first = cli._tape(91, 32, 6)
    second = cli._tape(91, 32, 6)
    assert first.shape == (32, 1, 6)
    assert first.dtype.name == "int64"
    assert first.tolist() == second.tolist()
    assert set(first.flat).issubset({0, 1, 2})


def test_resolved_protocol_has_exact_bounded_collection_and_disjoint_seed_streams() -> None:
    protocol = cli.resolved_protocol("4e2dd475a3de7f81321b42b373e36afaa4a6f10c", 20260912, ROOT)
    assert protocol["collection"]["world_count"] == 24
    assert protocol["collection"]["development_world_count"] == 8
    assert protocol["collection"]["max_steps_per_world"] == 256
    assert protocol["collection"]["max_pairs"] == 72
    assert protocol["collection"]["frames"] == [16, 80, 160]
    assert protocol["collection"]["horizons"] == [1, 8, 16, 32]
    assert protocol["collection"]["primary_horizon"] == 16
    assert protocol["limits"]["agent_slots"] == 202752
    assert 24 * 256 * 6 == 36864
    assert 72 * 2 * 6 * 32 * 6 == 165888
    streams = [
        protocol["seeds"]["root"],
        protocol["seeds"]["bootstrap"],
        *protocol["seeds"]["world"],
        *(seed for slot_seeds in protocol["seeds"]["policy"] for seed in slot_seeds),
        *protocol["seeds"]["tape"].values(),
    ]
    assert len(streams) == len(set(streams))
    assert cli.derive_seed is derive_seed


def test_resource_counter_uses_agent_slots_not_world_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = {
        "natural_agent_slots": 36864,
        "counterfactual_agent_slots": 165888,
    }
    monkeypatch.setattr(
        cli,
        "_resource_snapshot",
        lambda *_: {"elapsed_seconds": 0.0, "rss_bytes": 0, "available_bytes": 2**63 - 1},
    )
    cli._check_resources(0.0, counters)
    counters["counterfactual_agent_slots"] += 1
    with pytest.raises(cli.ResourceStop, match="agent_slots"):
        cli._check_resources(0.0, counters)


def test_seed_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "derive_seed", lambda *_: 7)
    with pytest.raises(RuntimeError, match="C0"):
        cli.resolved_protocol("revision", 9, ROOT)


def test_direct_script_help_has_repo_import_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src/scripts/observation_value_diagnostic.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "freeze" in completed.stdout and "run" in completed.stdout


def test_manifest_semantic_projection_rejects_config_or_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = cli.resolved_protocol("revision", 9, ROOT)
    manifest["manifest_digest"] = cli.digest_without(manifest, "manifest_digest")
    path = (tmp_path / "manifest.json").resolve()
    path.write_bytes(cli.canonical(manifest))
    expected_hash = cli.sha256_file(path)
    monkeypatch.setattr(cli, "_clean_revision", lambda *_: True)
    monkeypatch.setattr(cli, "source_closure", lambda *_: manifest["source_closure"])
    monkeypatch.setattr(cli, "_REPO_ROOT", ROOT)
    assert cli.verify_manifest(path, expected_hash)["schema"] == cli.SCHEMA
    changed = json.loads(path.read_text())
    changed["collection"]["frames"] = [15, 80, 160]
    changed["manifest_digest"] = cli.digest_without(changed, "manifest_digest")
    path.write_bytes(cli.canonical(changed))
    with pytest.raises(RuntimeError, match="semantic projection"):
        cli.verify_manifest(path, cli.sha256_file(path))


def test_freeze_requires_a_new_absolute_output_directory(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        cli.freeze(Path("relative"), "not-a-revision", 1)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        cli.freeze(existing, "not-a-revision", 1)


def test_verify_manifest_rejects_command_hash_drift_before_source_access(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": cli.SCHEMA}))
    with pytest.raises(RuntimeError, match="SHA"):
        cli.verify_manifest(path.resolve(), "0" * 64)


def test_run_refuses_existing_output_before_constructing_a_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(cli, "verify_manifest", lambda *_: {"probe_config": {}, "batch_config": {}})
    with pytest.raises(FileExistsError, match="run output"):
        cli.run(manifest.resolve(), "a" * 64)


def test_atomic_json_never_overwrites_an_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "terminal.json"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        cli._atomic_json(target, {"replacement": True})
    assert target.read_text() == "original"


def test_run_preflight_config_error_writes_failed_terminal_without_world_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = (tmp_path / "manifest.json").resolve()
    manifest.write_text("{}")
    monkeypatch.setattr(
        cli,
        "verify_manifest",
        lambda *_: {"probe_config": {"unknown": 1}, "batch_config": {}},
    )
    terminal = cli.run(manifest, "a" * 64)
    payload = json.loads(terminal.read_text())
    assert payload["status"] == "failed"
    assert payload["completed_world_count"] == 0
    assert (terminal.parent / "raw.jsonl").exists()


def test_partial_terminal_explicitly_marks_every_unrun_world_and_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = (tmp_path / "manifest.json").resolve()
    manifest.write_text("{}")
    monkeypatch.setattr(
        cli,
        "verify_manifest",
        lambda *_: {"probe_config": {"unknown": 1}, "batch_config": {}},
    )
    # A setup exception is a failed terminal and must still expose all unrun work.
    terminal = cli.run(manifest, "a" * 64)
    rows = [json.loads(line) for line in (terminal.parent / "raw.jsonl").read_text().splitlines()]
    assert sum(row.get("kind") == "world" for row in rows) == cli.WORLD_COUNT
    assert sum(row.get("reason") == "unrun" for row in rows) == cli.WORLD_COUNT * len(cli.FRAMES)
