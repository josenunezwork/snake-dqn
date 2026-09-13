"""Fixture-only contract tests for the create-only observation diagnostic CLI."""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.seeding import derive_seed
from src.evaluation import observation_probe

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "observation_value_diagnostic", ROOT / "src/scripts/observation_value_diagnostic.py"
)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _tiny_manifest(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Freeze a deterministic, real-simulator-sized-down run protocol."""
    monkeypatch.setattr(cli, "WORLD_COUNT", 2)
    monkeypatch.setattr(cli, "DEVELOPMENT_WORLDS", 1)
    monkeypatch.setattr(cli, "MAX_STEPS", 3)
    monkeypatch.setattr(cli, "MAX_PAIRS", 4)
    monkeypatch.setattr(cli, "FRAMES", (0, 1))
    monkeypatch.setattr(cli, "HORIZONS", (1,))
    monkeypatch.setattr(cli, "PRIMARY_HORIZON", 1)
    monkeypatch.setattr(cli, "SNAKES", 6)
    monkeypatch.setattr(cli, "TAPE_STEPS", 1)
    manifest = cli.resolved_protocol("tiny-real-run", 2026091201, ROOT)
    manifest["batch_config"].update({"initial_food": 0, "max_food": 0})
    return manifest


def _tiny_run_paths(tmp_path: Path) -> tuple[Path, Path]:
    manifest = (tmp_path / "manifest.json").resolve()
    manifest.write_text("{}")
    return manifest, manifest.parent / "run"


def _raw_rows(terminal: Path) -> list[dict]:
    return [json.loads(line) for line in (terminal.parent / "raw.jsonl").read_text().splitlines()]


def test_tiny_real_run_completes_all_worlds_with_full_h1_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(monkeypatch)
    manifest_path, run_dir = _tiny_run_paths(tmp_path)
    monkeypatch.setattr(cli, "verify_manifest", lambda *_: manifest)

    terminal = cli.run(manifest_path, "a" * 64)
    payload, rows = json.loads(terminal.read_text()), _raw_rows(terminal)
    candidates = [row for row in rows if row.get("kind") == "candidate"]
    branch_inputs = [row for row in rows if row.get("kind") == "branch_inputs"]
    branch_results = [row for row in rows if row.get("kind") == "branch_result"]
    worlds = [row for row in rows if row.get("kind") == "world"]

    assert payload["status"] == "completed"
    assert payload["completed_world_count"] == 2
    assert {row["world_id"] for row in candidates} == {"world-00", "world-01"}
    assert {(row["world_id"], row["frame"]) for row in candidates} == {
        ("world-00", 0),
        ("world-00", 1),
        ("world-01", 0),
        ("world-01", 1),
    }
    assert all(row["status"] == "accepted" and row["reason"] is None for row in candidates)
    assert len(branch_inputs) == len(candidates) == 4
    assert len(branch_results) == 8
    assert worlds == [
        {
            "kind": "world",
            "world_id": "world-00",
            "status": "completed",
            "world_end_reason": "step_cap",
            "natural_world_ticks": 3,
        },
        {
            "kind": "world",
            "world_id": "world-01",
            "status": "completed",
            "world_end_reason": "step_cap",
            "natural_world_ticks": 3,
        },
    ]
    assert payload["counters"] == {
        "worlds": 2,
        "natural_world_ticks": 6,
        "natural_agent_slots": 36,
        "natural_valid_transition_agent_slots": 36,
        "counterfactual_world_ticks": 24,
        "counterfactual_agent_slots": 144,
        "counterfactual_valid_transition_agent_slots": 144,
        "pairs": 4,
    }
    assert run_dir.is_dir()
    for row in branch_inputs:
        for key in ("left_snapshot_path", "right_snapshot_path", "tape_path"):
            path = Path(row[key])
            assert path.is_file()
            with path.open("rb") as handle:
                pickle.load(handle)
    for row in candidates:
        for result_key in ("left_returns", "right_returns"):
            result = row[result_key]
            assert result["horizons_are_total_executed_steps"] is True
            assert result["source_fingerprint"]
            for action in (0, 1, 2):
                record = result["actions"][str(action)]
                assert record["actual_steps"] == 1
                assert record["valid_agent_transitions"] == 6
                assert record["discounted_return_by_horizon"] == {
                    "1": record["first_step"]["reward"]
                }


def test_tiny_real_run_final_manifest_drift_suppresses_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(monkeypatch)
    manifest_path, _ = _tiny_run_paths(tmp_path)
    calls = 0

    def drift_on_final_verify(*_args):
        nonlocal calls
        calls += 1
        if calls == 4:  # initial verification + one per world + final verification
            raise RuntimeError("approved semantic projection drift: final")
        return manifest

    monkeypatch.setattr(cli, "verify_manifest", drift_on_final_verify)
    payload = json.loads(cli.run(manifest_path, "a" * 64).read_text())
    assert calls == 4
    assert payload["status"] == "failed"
    assert "projection drift" in payload["cause"]
    assert payload["completed_world_count"] == 2
    assert payload["development"] is None
    assert payload["holdout"] is None


def test_tiny_real_run_persistent_psutil_failure_is_terminal_resource_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(monkeypatch)
    manifest_path, _ = _tiny_run_paths(tmp_path)
    monkeypatch.setattr(cli, "verify_manifest", lambda *_: manifest)
    monkeypatch.setattr(
        cli.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(cli.psutil.Error("unavailable")),
    )

    payload = json.loads(cli.run(manifest_path, "a" * 64).read_text())
    assert payload["status"] == "partial"
    assert payload["cause"].startswith("available_memory_unavailable")
    assert payload["completed_world_count"] == 0
    assert payload["resource"]["resource_error"].startswith("available_memory_unavailable")


def test_tiny_real_run_branch_exception_retains_inputs_and_marks_active_candidate_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(monkeypatch)
    manifest_path, _ = _tiny_run_paths(tmp_path)
    monkeypatch.setattr(cli, "verify_manifest", lambda *_: manifest)

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected finite branch failure")

    monkeypatch.setattr(observation_probe, "finite_action_returns", boom)
    terminal = cli.run(manifest_path, "a" * 64)
    payload, rows = json.loads(terminal.read_text()), _raw_rows(terminal)
    inputs = [row for row in rows if row.get("kind") == "branch_inputs"]
    current = [
        row
        for row in rows
        if row.get("kind") == "candidate" and row["world_id"] == "world-00" and row["frame"] == 0
    ]

    assert payload["status"] == "failed"
    assert len(inputs) == 1
    assert all(
        Path(inputs[0][key]).is_file()
        for key in ("left_snapshot_path", "right_snapshot_path", "tape_path")
    )
    assert len(current) == 1
    assert current[0]["status"] == "failed"
    assert current[0]["reason"] == "RuntimeError: injected finite branch failure"
    assert any(row.get("world_id") == "world-01" and row.get("status") == "unrun" for row in rows)


def test_tiny_real_run_last_resource_snapshot_failure_downgrades_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _tiny_manifest(monkeypatch)
    manifest_path, _ = _tiny_run_paths(tmp_path)
    monkeypatch.setattr(cli, "verify_manifest", lambda *_: manifest)
    monkeypatch.setattr(cli, "_check_resources", lambda *_: {})
    snapshots = 0

    def snapshot(*_args):
        nonlocal snapshots
        snapshots += 1
        if snapshots == 7:  # Four candidate and two world-complete heartbeats, then terminal.
            raise cli.ResourceStop("available_memory_unavailable:final")
        return {"elapsed_seconds": 0.0, "rss_bytes": 0, "available_bytes": 2**63 - 1}

    monkeypatch.setattr(cli, "_resource_snapshot", snapshot)
    payload = json.loads(cli.run(manifest_path, "a" * 64).read_text())
    assert snapshots == 7
    assert payload["status"] == "partial"
    assert payload["cause"] == "available_memory_unavailable:final"
    assert payload["completed_world_count"] == 2
    assert payload["development"] is None and payload["holdout"] is None
    assert payload["decision"] == "INCONCLUSIVE_NOT_ADVANCED"
    assert payload["incomplete_variant_work_possible"] is True


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
    for key in cli._THREAD_KEYS:
        monkeypatch.setenv(key, "1")
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
