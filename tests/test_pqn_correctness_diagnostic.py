"""Runtime-path tests for the immutable H0 diagnostic pipeline."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "src/scripts/pqn_correctness_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("pqn_correctness_diagnostic", MODULE_PATH)
assert SPEC and SPEC.loader
h0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h0
SPEC.loader.exec_module(h0)


class FakeProcess:
    pid = 8675309

    def __init__(self, returncode: int = 0, alive_polls: int = 0) -> None:
        self.returncode = None if alive_polls else returncode
        self._final = returncode
        self._alive_polls = alive_polls
        self.waited = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._alive_polls:
            self._alive_polls -= 1
            return None
        if self.returncode is None:
            self.returncode = self._final
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        self.returncode = self._final
        return self.returncode


class FakeChild:
    class Memory:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    def __init__(self, rss: int = 0) -> None:
        self.rss = rss

    def memory_info(self) -> "FakeChild.Memory":
        return self.Memory(self.rss)


def plan(tmp_path: Path, mode: str = "smoke", steps: int = 2) -> Any:
    projection = {"validated": True} if mode == "screen" else None
    return h0.DiagnosticPlan(
        mode=mode,
        output_dir=tmp_path / "run",
        requested_seed=42,
        device="cpu",
        num_envs=1,
        rollout_len=1,
        arms=h0.expected_arms(mode, 42, steps),
        limits=h0.DiagnosticLimits(
            wall_seconds=2,
            evaluation_wall_seconds=2,
            no_heartbeat_seconds=2,
            poll_seconds=0.001,
            term_grace_seconds=0.001,
        ),
        projection=projection,
    )


def write_valid_evaluation_receipt(arm_dir: Path) -> dict[str, Any]:
    receipt = {"schema": "fake-evaluation/v1", "status": "completed"}
    receipt["receipt_digest"] = h0._digest_without(receipt, "receipt_digest")
    path = arm_dir / "fake_evaluation_terminal.json"
    h0._write_json(path, receipt)
    return {"path": str(path), "sha256": h0.sha256_file(path)}


def lineage_for_bytes(arm_dir: Path, manifest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    requested = next(
        item["requested_steps"] for item in manifest["arms"] if item["name"] == arm_dir.name
    )
    return True, {
        "initial_sha256": h0.sha256_file(arm_dir / "initial.pth"),
        "final_sha256": h0.sha256_file(arm_dir / "final.pth"),
        "actual_agent_steps": requested + 1,
        "requested_agent_steps": requested,
        "overshoot_steps": 1,
        "update_count": 1,
        "elapsed_seconds": 0.1,
    }


def successful_popen(processes: list[FakeProcess], *, fail_index: int | None = None):
    def start(command: list[str], **kwargs: Any) -> FakeProcess:
        assert kwargs["start_new_session"] is True
        assert "-u" in command
        spec = json.loads(Path(command[-1]).read_text())
        arm_dir = Path(spec["arm_dir"])
        assert arm_dir.is_dir()
        index = len(processes)
        if index == fail_index:
            process = FakeProcess(h0.EXIT_FAILURE)
            h0._worker_terminal(
                arm_dir,
                status="failed",
                cause="worker_exception",
                requested_steps=spec["requested_steps"],
                actual_steps=0,
                update_count=0,
                started=0.0,
                resources={"thresholds": spec["limits"]},
                exception={"type": "FakeFailure", "message": "planned"},
            )
        else:
            (arm_dir / "initial.pth").write_bytes(f"initial-{index}".encode())
            (arm_dir / "final.pth").write_bytes(f"final-{index}".encode())
            process = FakeProcess()
            h0._worker_terminal(
                arm_dir,
                status="completed",
                cause="budget_reached",
                requested_steps=spec["requested_steps"],
                actual_steps=spec["requested_steps"] + 1,
                update_count=1,
                started=0.0,
                resources={"thresholds": spec["limits"]},
            )
        processes.append(process)
        return process

    return start


def fake_evaluator(
    arm_dir: Path, manifest: dict[str, Any], timeout: float
) -> tuple[bool, dict[str, Any]]:
    assert timeout == manifest["limits"]["evaluation_wall_seconds"]
    return True, write_valid_evaluation_receipt(arm_dir)


def complete_checkpoint_state(
    manifest: dict[str, Any], arm_name: str, steps: int, updates: int
) -> dict[str, Any]:
    config = h0.PQNConfig(**manifest["configs"][arm_name])
    contracts = {
        "action_mask_contract": h0.pqn_action_mask_contract(config),
        "reward_contract": h0.pqn_reward_contract(config),
        "target_contract": h0.pqn_target_contract(config),
        "sampler_contract": h0.pqn_sampler_contract(config),
        "optimizer_contract": h0.pqn_optimizer_contract(config),
        "runtime_contract": h0.asdict(
            h0.RuntimeModeContract(
                mode="pqn_train",
                training=True,
                respawn=False,
                hero_terminal=True,
                population_floor=True,
                reset_strategy="batch_episode",
            )
        ),
    }
    state = {
        "effective_world": manifest["world"],
        "effective_world_digest": manifest["world_digest"],
        "effective_seed": manifest["configs"][arm_name]["seed"],
        "source_revision": manifest["git"]["commit"],
        "agent_steps": steps,
        "update_counter": updates,
        "h0_manifest_digest": manifest["manifest_digest"],
        "obs_spec": config.obs_spec,
        "recipe": config.recipe,
        "mechanics_version": config.mechanics_version,
        "reward_version": config.reward_version,
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "lr": config.lr,
        "adam_eps": config.adam_eps,
        "eps_start": config.eps_start,
        "eps_end": config.eps_end,
        "eps_decay_steps": config.eps_decay_steps,
        "sgd_epochs": config.sgd_epochs,
        "pad_sgd_batches": config.pad_sgd_batches,
        "sgd_seed": config.sgd_seed,
        "field_sources": config.field_sources,
    }
    for name, contract in contracts.items():
        state[name] = contract
        state[f"{name}_digest"] = h0.canonical_digest(contract)
    return state


def test_modes_validate_exact_cardinality_positive_budgets_and_fixed_recipe(
    tmp_path: Path,
) -> None:
    assert [len(h0.expected_arms(mode, 1, 3)) for mode in ("smoke", "shakedown", "screen")] == [
        1,
        1,
        5,
    ]
    candidate = plan(tmp_path)
    config = h0.build_config(candidate, candidate.arms[0])
    assert (config.recipe, config.obs_spec, config.reward_version, config.mechanics_version) == (
        "corrected-v3",
        "raster31v3",
        2,
        2,
    )
    assert config.eps_decay_steps == 500_000
    assert config.flip_augment is False and config.sgd_epochs is None
    assert (config.max_length, config.max_capacity, config.starvation_max, config.kill_scale) == (
        150,
        400,
        500,
        0.3,
    )
    assert set(config.field_sources) == set(config.__dataclass_fields__) - {"field_sources"}
    assert config.field_sources["eps_decay_steps"] == "h0_fixed_protocol"
    with pytest.raises(ValueError, match="positive"):
        h0.expected_arms("smoke", 1, 0)
    invalid = h0.DiagnosticPlan(**{**candidate.__dict__, "arms": ()})
    with pytest.raises(ValueError, match="exactly 1"):
        h0.freeze_manifest(invalid)


def test_manifest_is_new_immutable_complete_and_covers_serving_and_evaluation(
    tmp_path: Path,
) -> None:
    candidate = plan(tmp_path)
    candidate.output_dir.mkdir()
    manifest = h0.freeze_manifest(candidate)
    assert len(manifest["world"]) == len(h0.fields(h0.EffectiveWorldConfig))
    assert manifest["world"] == manifest["e1_promotion_profile"]["descriptor"]["world"]
    assert manifest["evaluation"]["actual_seeds"] != manifest["evaluation"]["raw_seeds"]
    assert all(
        Path(path).is_absolute()
        for path in (manifest["output_dir"], manifest["evaluation"]["config_path"])
    )
    assert any(path.startswith("src/evaluation/") for path in manifest["source_hashes"])
    assert any(path.startswith("web/backend/") for path in manifest["source_hashes"])
    assert h0._verify_boundary(manifest) is None
    with pytest.raises(FileExistsError):
        h0.run_plan(candidate)


def test_manifest_original_bytes_and_sidecar_are_not_adopted_after_mutation(
    tmp_path: Path,
) -> None:
    manifest = h0.freeze_manifest(plan(tmp_path))
    path = Path(manifest["_manifest_path"])
    path.write_bytes(path.read_bytes() + b" ")
    assert h0._verify_boundary(manifest) == "protocol_drift"


def test_worker_descriptor_seeds_before_constructor_validates_initial_world_and_overshoot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path, steps=1)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    arm_dir = candidate.output_dir / arm.name
    arm_dir.mkdir()
    events: list[str] = []

    @dataclass
    class Telemetry:
        update: int = 0
        agent_steps: int = 2

    class Trainer:
        def __init__(self, config: Any, *, device: torch.device) -> None:
            assert events == ["seed"]
            assert config.seed == arm.training_seed and device == torch.device("cpu")
            self.cfg = config
            self.agent_steps = 0
            self.update_idx = 0

        def checkpoint_state(self) -> dict[str, Any]:
            return complete_checkpoint_state(manifest, arm.name, self.agent_steps, self.update_idx)

        def update(self) -> Telemetry:
            self.agent_steps = 2
            self.update_idx = 1
            return Telemetry()

    real_seed = h0.initialize_run_seed

    def seed(value: int) -> Any:
        events.append("seed")
        return real_seed(value)

    monkeypatch.setattr(h0, "initialize_run_seed", seed)
    monkeypatch.setattr(h0, "PQNTrainer", Trainer)
    monkeypatch.setattr(h0.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(h0.torch, "set_num_interop_threads", lambda _: None)
    for name in h0.THREAD_ENV:
        monkeypatch.setenv(name, "2")
    spec = h0._expected_worker_spec(manifest, arm.name)
    spec["worker_spec_digest"] = h0._digest_without(spec, "worker_spec_digest")
    spec_path = candidate.output_dir / "worker.json"
    h0._write_json(spec_path, spec)
    assert h0.worker_main(spec_path) == 0
    terminal = json.loads((arm_dir / "terminal.json").read_text())
    assert terminal["actual_clocks"] == {
        "requested_agent_steps": 1,
        "actual_agent_steps": 2,
        "last_accepted_step": 2,
        "overshoot_steps": 1,
        "update_count": 1,
        "elapsed_seconds": terminal["actual_clocks"]["elapsed_seconds"],
    }
    initial = torch.load(arm_dir / "initial.pth", map_location="cpu", weights_only=False)
    assert initial["agent_steps"] == initial["update_counter"] == 0
    telemetry = json.loads((arm_dir / "telemetry.jsonl").read_text())
    assert telemetry["monotonic"] > 0
    assert telemetry["requested_agent_steps"] == 1
    assert telemetry["overshoot_steps"] == 1


def test_worker_no_progress_failure_preserves_actual_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path, steps=3)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    (candidate.output_dir / arm.name).mkdir()

    @dataclass
    class Telemetry:
        update: int = 0

    class Trainer:
        def __init__(self, config: Any, *, device: torch.device) -> None:
            self.agent_steps = 1
            self.update_idx = 1

        def checkpoint_state(self) -> dict[str, Any]:
            state = complete_checkpoint_state(manifest, arm.name, self.agent_steps, self.update_idx)
            if not (candidate.output_dir / arm.name / "initial.pth").exists():
                state["agent_steps"] = state["update_counter"] = 0
                self.agent_steps = self.update_idx = 0
            return state

        def update(self) -> Telemetry:
            self.agent_steps = 1
            self.update_idx = 1
            return Telemetry()

    monkeypatch.setattr(h0, "PQNTrainer", Trainer)
    monkeypatch.setattr(h0.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(h0.torch, "set_num_interop_threads", lambda _: None)
    for name in h0.THREAD_ENV:
        monkeypatch.setenv(name, "2")
    spec = h0._expected_worker_spec(manifest, arm.name)
    spec["worker_spec_digest"] = h0._digest_without(spec, "worker_spec_digest")
    spec_path = candidate.output_dir / "worker.json"
    h0._write_json(spec_path, spec)
    assert h0.worker_main(spec_path) == h0.EXIT_FAILURE
    clocks = json.loads((candidate.output_dir / arm.name / "terminal.json").read_text())[
        "actual_clocks"
    ]
    assert clocks["actual_agent_steps"] == clocks["last_accepted_step"] == 1
    assert clocks["update_count"] == 1


def test_worker_rejects_descriptor_that_differs_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    (candidate.output_dir / arm.name).mkdir()
    spec = h0._expected_worker_spec(manifest, arm.name)
    spec["device"] = "mps"
    spec["worker_spec_digest"] = h0._digest_without(spec, "worker_spec_digest")
    path = candidate.output_dir / "worker.json"
    h0._write_json(path, spec)
    assert h0.worker_main(path) == h0.EXIT_PROTOCOL
    terminal = json.loads((candidate.output_dir / arm.name / "terminal.json").read_text())
    assert terminal["cause"] == "protocol_drift"


def test_unreadable_worker_descriptor_emits_detailed_exit_72_receipt(
    tmp_path: Path,
) -> None:
    arm_dir = tmp_path / "smoke-1"
    arm_dir.mkdir()
    spec_path = tmp_path / "smoke-1.worker.json"
    spec_path.write_text("{")
    assert h0.worker_main(spec_path) == h0.EXIT_PROTOCOL
    terminal = json.loads((arm_dir / "terminal.json").read_text())
    assert terminal["cause"] == "protocol_drift"
    assert terminal["exception_or_tripwire"]["type"] == "JSONDecodeError"


def test_screen_exactly_once_retains_failed_arm_and_evaluates_after_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path, "screen")
    processes: list[FakeProcess] = []
    evaluated: list[str] = []

    def evaluate(arm_dir: Path, manifest: dict[str, Any], timeout: float):
        assert processes[-1].waited
        evaluated.append(arm_dir.name)
        return fake_evaluator(arm_dir, manifest, timeout)

    monkeypatch.setattr(
        h0.psutil, "virtual_memory", lambda: type("V", (), {"available": 99 * h0.GIB})()
    )
    results = h0.run_plan(
        candidate,
        popen=successful_popen(processes, fail_index=2),
        evaluator=evaluate,
        process_factory=lambda _: FakeChild(),
        checkpoint_inspector=lineage_for_bytes,
    )
    assert len(processes) == len(results) == 5
    assert [item["outcome"] for item in results] == [
        "completed",
        "completed",
        "learner_exit_failure",
        "completed",
        "completed",
    ]
    assert evaluated == ["screen-1", "screen-2", "screen-4", "screen-5"]
    assert all("--gate" not in json.dumps(item) for item in results)


def test_zero_exit_without_terminal_receipt_cannot_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path)
    monkeypatch.setattr(
        h0.psutil, "virtual_memory", lambda: type("V", (), {"available": 99 * h0.GIB})()
    )
    result = h0.run_plan(
        candidate,
        popen=lambda *args, **kwargs: FakeProcess(),
        evaluator=lambda *args: pytest.fail("must not evaluate"),
        process_factory=lambda _: FakeChild(),
    )[0]
    assert result["outcome"] == "learner_exit_failure"
    assert result["cause"] == "missing_or_invalid_terminal_receipt"


def test_launch_errors_retain_all_five_arms_without_retry(tmp_path: Path) -> None:
    candidate = plan(tmp_path, "screen")
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> FakeProcess:
        nonlocal calls
        calls += 1
        raise OSError("no process")

    results = h0.run_plan(candidate, popen=fail)
    assert calls == len(results) == 5
    assert {item["cause"] for item in results} == {"launch_failure"}


def test_unconfirmed_exit_aborts_plan_and_records_remaining_arms_unrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path, "screen")
    starts: list[bool] = []
    monkeypatch.setattr(
        h0,
        "_monitor_process",
        lambda *args, **kwargs: {
            "cause": "watchdog_timeout",
            "termination": "unconfirmed_exit",
            "confirmed_exit": False,
        },
    )
    results = h0.run_plan(
        candidate,
        popen=lambda *args, **kwargs: (starts.append(True) or FakeProcess(alive_polls=99)),
    )
    assert len(starts) == 1
    assert [item["outcome"] for item in results] == ["learner_exit_failure"] + ["unrun"] * 4
    assert all(
        (candidate.output_dir / arm.name / "result.json").is_file() for arm in candidate.arms
    )


def test_watchdog_uses_latest_heartbeat_after_initial_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heartbeat = tmp_path / "heartbeat.jsonl"
    h0._append_jsonl(heartbeat, {"monotonic": 1.0, "agent_steps": 0})
    h0._append_jsonl(heartbeat, {"monotonic": 5.0, "agent_steps": 7})
    process = FakeProcess(alive_polls=99)
    monkeypatch.setattr(h0.os, "killpg", lambda pid, sig: setattr(process, "returncode", -sig))
    monkeypatch.setattr(
        h0.psutil, "virtual_memory", lambda: type("V", (), {"available": 99 * h0.GIB})()
    )
    clock = itertools.chain((0.0, 8.0, 8.0, 8.0, 8.0), itertools.repeat(8.0))
    limits = h0.asdict(plan(tmp_path).limits)
    limits["no_heartbeat_seconds"] = 2
    result = h0._monitor_process(
        process,
        limits,
        wall_seconds=100,
        heartbeat_path=heartbeat,
        now=lambda: next(clock),
        sleep=lambda _: None,
        process_factory=lambda _: FakeChild(),
    )
    assert result["cause"] == "watchdog_timeout"
    assert result["latest_heartbeat"]["agent_steps"] == 7


def test_psutil_access_denied_stops_and_reaps_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess(alive_polls=99)
    monkeypatch.setattr(h0.os, "killpg", lambda pid, sig: setattr(process, "returncode", -sig))

    def denied(_: int) -> Any:
        raise h0.psutil.AccessDenied(pid=process.pid)

    result = h0._monitor_process(
        process,
        h0.asdict(plan(tmp_path).limits),
        wall_seconds=2,
        heartbeat_path=None,
        now=itertools.count().__next__,
        sleep=lambda _: None,
        process_factory=denied,
    )
    assert result["cause"] == "process_monitor_error"
    assert result["confirmed_exit"] and process.waited


def test_checkpoint_lineage_requires_complete_world_zero_initial_and_terminal(
    tmp_path: Path,
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    arm_dir = candidate.output_dir / arm.name
    arm_dir.mkdir()
    torch.save(complete_checkpoint_state(manifest, arm.name, 0, 0), arm_dir / "initial.pth")
    torch.save(complete_checkpoint_state(manifest, arm.name, 3, 1), arm_dir / "final.pth")
    h0._worker_terminal(
        arm_dir,
        status="completed",
        cause="budget_reached",
        requested_steps=2,
        actual_steps=3,
        update_count=1,
        started=h0.time.monotonic(),
        resources={"thresholds": manifest["limits"]},
    )
    ok, receipt = h0.checkpoint_lineage(arm_dir, manifest)
    assert ok and receipt["overshoot_steps"] == 1
    state = torch.load(arm_dir / "initial.pth", map_location="cpu", weights_only=False)
    state["effective_world"].pop("max_length")
    torch.save(state, arm_dir / "initial.pth")
    assert h0.checkpoint_lineage(arm_dir, manifest)[1]["reason"] == "incomplete_effective_world"


def test_checkpoint_lineage_rejects_forged_recipe_contract(
    tmp_path: Path,
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    arm_dir = candidate.output_dir / arm.name
    arm_dir.mkdir()
    initial = complete_checkpoint_state(manifest, arm.name, 0, 0)
    final = complete_checkpoint_state(manifest, arm.name, 2, 1)
    initial["target_contract_digest"] = "0" * 64
    torch.save(initial, arm_dir / "initial.pth")
    torch.save(final, arm_dir / "final.pth")
    assert (
        h0.checkpoint_lineage(arm_dir, manifest)[1]["reason"] == "target_contract_digest_mismatch"
    )


@pytest.mark.parametrize("phase", ["after_learner", "pre_eval", "post_eval"])
def test_source_drift_is_checked_at_every_late_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    candidate = plan(tmp_path)
    processes: list[FakeProcess] = []
    positions = {"after_learner": 1, "pre_eval": 2, "post_eval": 3}
    calls = iter(["source_drift" if index == positions[phase] else None for index in range(4)])
    evaluated: list[bool] = []

    def evaluate(arm_dir: Path, manifest: dict[str, Any], timeout: float):
        evaluated.append(True)
        return fake_evaluator(arm_dir, manifest, timeout)

    monkeypatch.setattr(h0, "_verify_boundary", lambda manifest: next(calls))
    monkeypatch.setattr(
        h0.psutil,
        "virtual_memory",
        lambda: type("V", (), {"available": 99 * h0.GIB})(),
    )
    result = h0.run_plan(
        candidate,
        popen=successful_popen(processes),
        evaluator=evaluate,
        process_factory=lambda _: FakeChild(),
        checkpoint_inspector=lineage_for_bytes,
    )[0]
    assert result["outcome"] == "source_drift"
    assert evaluated == ([] if phase in {"after_learner", "pre_eval"} else [True])


def test_evaluator_launch_is_cpu_bounded_and_uses_absolute_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm_dir = candidate.output_dir / candidate.arms[0].name
    arm_dir.mkdir()
    (arm_dir / "initial.pth").write_bytes(b"initial")
    (arm_dir / "final.pth").write_bytes(b"final")
    accepted = h0._freeze_checkpoint_pair(
        arm_dir,
        {
            "initial_sha256": h0.sha256_file(arm_dir / "initial.pth"),
            "final_sha256": h0.sha256_file(arm_dir / "final.pth"),
        },
    )
    captured: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        captured.update({"command": command, "env": kwargs["env"]})
        return FakeProcess()

    monkeypatch.setattr(h0.subprocess, "Popen", popen)
    monkeypatch.setattr(
        h0,
        "_monitor_process",
        lambda *args, **kwargs: {
            "cause": None,
            "confirmed_exit": True,
            "termination": "natural_exit",
        },
    )
    ok, receipt = h0._run_evaluation(
        arm_dir, manifest, manifest["limits"]["evaluation_wall_seconds"]
    )
    assert not ok and receipt["reason"] == "evaluation_artifact_validation"
    assert captured["env"]["SNAKE_DQN_DEVICE"] == "cpu"
    assert "--gate" not in captured["command"]
    assert str(Path(accepted["checkpoints"]["initial"]["path"]).resolve()) in captured["command"]
    assert Path(manifest["evaluation"]["config_path"]).is_absolute()


def test_projection_uses_actual_steps_rate_and_rejects_incompatible_execution(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema": h0.SHAKE_RECEIPT_SCHEMA,
        "status": "completed",
        "mode": "shakedown",
        "device": "mps",
        "requested_steps": 100_000,
        "actual_steps": 125_000,
        "elapsed_seconds": 100.0,
        "source_digest": h0.canonical_digest(h0._source_closure()),
        "protocol_digest": h0.canonical_digest(h0._protocol_descriptor()),
        "manifest_sha256": "a" * 64,
        "num_envs": 16,
        "rollout_len": 16,
    }
    receipt["receipt_digest"] = h0._digest_without(receipt, "receipt_digest")
    path = tmp_path / "shake.json"
    h0._write_json(path, receipt)
    projection = h0.load_shakedown_projection(
        path,
        device="mps",
        requested_steps=500_000,
        num_envs=16,
        rollout_len=16,
    )
    assert projection["uncapped_seconds"] == 500.0
    assert projection["projected_wall_seconds"] == 500.0
    with pytest.raises(ValueError, match="num_envs"):
        h0.load_shakedown_projection(
            path,
            device="mps",
            requested_steps=500_000,
            num_envs=32,
            rollout_len=16,
        )


def test_private_worker_cli_and_public_mode_budgets() -> None:
    parsed = h0.parse_args(["--worker-spec", "/tmp/spec.json"])
    assert parsed.worker_spec == Path("/tmp/spec.json")
    with pytest.raises(SystemExit):
        h0.parse_args(
            ["--mode", "screen", "--out-dir", "/tmp/x", "--seed", "1", "--total-steps", "100000"]
        )


@pytest.mark.parametrize("breach", ["rss", "available", "mps", "wall"])
def test_actual_monitor_stops_resources_or_wall_and_suppresses_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breach: str
) -> None:
    candidate = plan(tmp_path)
    if breach == "wall":
        candidate = replace(candidate, limits=replace(candidate.limits, wall_seconds=0.1))
    process = FakeProcess(alive_polls=99)
    evaluated: list[bool] = []

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        spec = json.loads(Path(command[-1]).read_text())
        if breach == "mps":
            h0._append_jsonl(
                Path(spec["arm_dir"]) / "heartbeat.jsonl",
                {
                    "monotonic": 0.0,
                    "agent_steps": 0,
                    "mps_driver_bytes": candidate.limits.mps_driver_bytes + 1,
                },
            )
        return process

    monkeypatch.setattr(h0.os, "killpg", lambda pid, sig: setattr(process, "returncode", -sig))
    available = candidate.limits.available_bytes - 1 if breach == "available" else 99 * h0.GIB
    monkeypatch.setattr(
        h0.psutil,
        "virtual_memory",
        lambda: type("V", (), {"available": available})(),
    )
    rss = candidate.limits.rss_bytes + 1 if breach == "rss" else 0
    result = h0.run_plan(
        candidate,
        popen=popen,
        evaluator=lambda *args: (evaluated.append(True), {})[1],
        now=itertools.count().__next__,
        sleep=lambda _: None,
        process_factory=lambda _: FakeChild(rss),
    )[0]
    expected = "wall_stop" if breach == "wall" else "resource_stop"
    assert result["outcome"] == expected
    assert result["monitor"]["confirmed_exit"]
    assert result["monitor"]["termination"] in {"terminated", "killed"}
    assert evaluated == []
    if breach == "rss":
        assert result["monitor"]["max_rss_bytes"] > candidate.limits.rss_bytes
    if breach == "available":
        assert result["monitor"]["min_available_bytes"] < candidate.limits.available_bytes
    if breach == "mps":
        assert (
            result["monitor"]["latest_heartbeat"]["mps_driver_bytes"]
            > candidate.limits.mps_driver_bytes
        )


def test_term_grace_escalates_to_sigkill_and_confirms_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(alive_polls=99)
    sent: list[int] = []

    def killpg(pid: int, sig: int) -> None:
        sent.append(sig)
        if sig == h0.signal.SIGKILL:
            process.returncode = -sig

    monkeypatch.setattr(h0.os, "killpg", killpg)
    assert (
        h0.stop_process(
            process,
            1,
            now=itertools.count().__next__,
            sleeper=lambda _: None,
        )
        == "killed"
    )
    assert sent == [h0.signal.SIGTERM, h0.signal.SIGKILL]
    assert process.waited


@pytest.mark.parametrize(
    ("incident_class", "checkpoint_expected"),
    [("max_abs_q", True), ("non_finite", False)],
)
def test_worker_tripwire_only_archives_finite_incidents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incident_class: str,
    checkpoint_expected: bool,
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    arm_dir = candidate.output_dir / arm.name
    arm_dir.mkdir()

    class Trainer:
        def __init__(self, config: Any, *, device: torch.device) -> None:
            self.agent_steps = 0
            self.update_idx = 0

        def checkpoint_state(self) -> dict[str, Any]:
            return complete_checkpoint_state(manifest, arm.name, self.agent_steps, self.update_idx)

        def update(self) -> None:
            raise h0.TripwireError(
                "alarm",
                incident_class=incident_class,
                incident={"observed": 99},
            )

        def save_incident_checkpoint(self, path: str, exc: Exception) -> None:
            Path(path).write_bytes(b"finite incident")

    monkeypatch.setattr(h0, "PQNTrainer", Trainer)
    monkeypatch.setattr(h0.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(h0.torch, "set_num_interop_threads", lambda _: None)
    for name in h0.THREAD_ENV:
        monkeypatch.setenv(name, "2")
    spec = h0._expected_worker_spec(manifest, arm.name)
    spec["worker_spec_digest"] = h0._digest_without(spec, "worker_spec_digest")
    spec_path = candidate.output_dir / "tripwire.worker.json"
    h0._write_json(spec_path, spec)
    assert h0.worker_main(spec_path) == h0.EXIT_TRIPWIRE
    assert (arm_dir / "incident.pth").exists() is checkpoint_expected
    terminal = json.loads((arm_dir / "terminal.json").read_text())
    assert terminal["status"] == "tripwire"
    assert terminal["exception_or_tripwire"]["class"] == incident_class


def test_corrupt_checkpoint_bytes_become_unreadable_lineage(tmp_path: Path) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm_dir = candidate.output_dir / candidate.arms[0].name
    arm_dir.mkdir()
    (arm_dir / "initial.pth").write_bytes(b"truncated pickle")
    (arm_dir / "final.pth").write_bytes(b"truncated pickle")
    ok, detail = h0.checkpoint_lineage(arm_dir, manifest)
    assert not ok and detail["reason"] == "unreadable_checkpoint"


def test_source_drift_between_screen_arms_persists_every_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = plan(tmp_path, "screen")
    processes: list[FakeProcess] = []
    boundary_calls = 0

    def boundary(manifest: dict[str, Any]) -> str | None:
        nonlocal boundary_calls
        boundary_calls += 1
        return None if boundary_calls <= 4 else "source_drift"

    monkeypatch.setattr(h0, "_verify_boundary", boundary)
    monkeypatch.setattr(
        h0.psutil,
        "virtual_memory",
        lambda: type("V", (), {"available": 99 * h0.GIB})(),
    )
    results = h0.run_plan(
        candidate,
        popen=successful_popen(processes),
        evaluator=fake_evaluator,
        process_factory=lambda _: FakeChild(),
        checkpoint_inspector=lineage_for_bytes,
    )
    assert [item["outcome"] for item in results] == ["completed"] + ["source_drift"] * 4
    assert len(processes) == 1
    assert all((candidate.output_dir / arm.name / "result.json").exists() for arm in candidate.arms)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "profile",
        "config_hash",
        "config_bytes",
        "seeds",
        "candidate_null",
        "snapshots_null",
    ],
)
def test_evaluator_accepts_only_manifest_bound_e1_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
) -> None:
    candidate = plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm_dir = candidate.output_dir / candidate.arms[0].name
    arm_dir.mkdir()
    (arm_dir / "initial.pth").write_bytes(b"initial")
    (arm_dir / "final.pth").write_bytes(b"final")
    accepted = h0._freeze_checkpoint_pair(
        arm_dir,
        {
            "initial_sha256": h0.sha256_file(arm_dir / "initial.pth"),
            "final_sha256": h0.sha256_file(arm_dir / "final.pth"),
        },
    )

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        config_snapshot = arm_dir / "frozen-config.yaml"
        config_snapshot.write_bytes(Path(manifest["evaluation"]["config_path"]).read_bytes())
        if mutation == "config_bytes":
            config_snapshot.write_bytes(b"changed")
        snapshots = [
            {"sha256": item["sha256"], "snapshot_path": item["path"]}
            for item in accepted["checkpoints"].values()
        ]
        evaluator_receipt = {
            "config": {
                "sha256": (
                    "0" * 64
                    if mutation == "config_hash"
                    else manifest["evaluation"]["config_sha256"]
                ),
                "snapshot_path": str(config_snapshot),
            },
            "evaluation_profile": (
                {"forged": True} if mutation == "profile" else manifest["e1_promotion_profile"]
            ),
            "checkpoint_snapshots": ([None] if mutation == "snapshots_null" else snapshots),
        }
        receipt_path = arm_dir / "fake-e1-receipt.json"
        receipt_path.write_text(json.dumps(evaluator_receipt))
        output = Path(command[command.index("--json-output") + 1])
        payload = {
            "mode": "eval",
            "config": manifest["evaluation"]["config_path"],
            "engine": "simd",
            "frames": manifest["evaluation"]["frames"],
            "seeds": ([1] if mutation == "seeds" else manifest["evaluation"]["actual_seeds"]),
            "baseline": accepted["checkpoints"]["initial"]["path"],
            "mixes": manifest["evaluation"]["mixes"],
            "opponent_pool": manifest["evaluation"]["opponents"],
            "candidates": (
                [None]
                if mutation == "candidate_null"
                else [{"candidate": accepted["checkpoints"]["final"]["path"]}]
            ),
            "evaluation_inputs": {"receipt": str(receipt_path)},
        }
        output.write_text(json.dumps(payload))
        return FakeProcess()

    monkeypatch.setattr(h0.subprocess, "Popen", popen)
    monkeypatch.setattr(
        h0,
        "_monitor_process",
        lambda *args, **kwargs: {
            "cause": None,
            "confirmed_exit": True,
            "termination": "natural_exit",
        },
    )
    ok, receipt = h0._run_evaluation(
        arm_dir, manifest, manifest["limits"]["evaluation_wall_seconds"]
    )
    assert ok is (mutation is None)
    if mutation is None:
        assert h0._valid_evaluation_receipt(receipt)
    else:
        assert receipt["reason"] == "evaluation_artifact_validation"
