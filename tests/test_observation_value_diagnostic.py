"""Fixture-only contract tests for the create-only observation diagnostic CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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
    assert protocol["collection"]["horizons"] == [1, 8, 16, 32]
    streams = [
        protocol["seeds"]["root"],
        protocol["seeds"]["bootstrap"],
        *protocol["seeds"]["world"],
        *protocol["seeds"]["policy"],
        *protocol["seeds"]["tape"],
    ]
    assert len(streams) == len(set(streams))


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
