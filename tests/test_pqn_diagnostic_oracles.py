"""Independent orchestration oracles for the bounded H0 diagnostic runner.

These tests exercise the public entry points and injectable process seams.  They
do not construct a real learner subprocess or run the evaluator; the fake child
only supplies lifecycle and checkpoint bytes needed to test coordinator policy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "src/scripts/pqn_correctness_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("h0_diagnostic_oracles", MODULE_PATH)
assert SPEC and SPEC.loader
h0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h0
SPEC.loader.exec_module(h0)


class Process:
    """A subprocess-shaped object with a confirmed wait boundary."""

    pid = 2187

    def __init__(self, returncode: int = 0, alive_polls: int = 1) -> None:
        self._final = returncode
        self._alive_polls = alive_polls
        self.returncode: int | None = None if alive_polls else returncode
        self.waited = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._alive_polls:
            self._alive_polls -= 1
            return None
        self.returncode = self._final
        return self.returncode

    def wait(self, **_: Any) -> int:
        self.waited = True
        self.returncode = self._final
        return self.returncode


class Child:
    class Memory:
        rss = 0

    def memory_info(self) -> Memory:
        return self.Memory()


def make_plan(tmp_path: Path, mode: str = "smoke") -> Any:
    return h0.DiagnosticPlan(
        mode=mode,
        output_dir=tmp_path / "run",
        requested_seed=90210,
        device="cpu",
        num_envs=1,
        rollout_len=1,
        arms=h0.expected_arms(mode, 90210, 2),
        limits=h0.DiagnosticLimits(
            wall_seconds=5,
            evaluation_wall_seconds=5,
            no_heartbeat_seconds=5,
            poll_seconds=0.001,
            term_grace_seconds=0.001,
        ),
        projection={"validated": True} if mode == "screen" else None,
    )


def complete_payload(manifest: dict[str, Any], arm_name: str, steps: int) -> dict[str, Any]:
    config = h0.PQNConfig(**manifest["configs"][arm_name])
    contracts = {
        "action_mask_contract": h0.pqn_action_mask_contract(config),
        "reward_contract": h0.pqn_reward_contract(config),
        "target_contract": h0.pqn_target_contract(config),
        "sampler_contract": h0.pqn_sampler_contract(config),
        "optimizer_contract": h0.pqn_optimizer_contract(config),
        "runtime_contract": asdict(
            h0.RuntimeModeContract("pqn_train", True, False, True, True, "batch_episode")
        ),
    }
    value = {
        "h0_manifest_digest": manifest["manifest_digest"],
        "effective_world": manifest["world"],
        "effective_world_digest": manifest["world_digest"],
        "agent_steps": steps,
        "update_counter": int(steps > 0),
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
        value[name] = contract
        value[f"{name}_digest"] = h0.canonical_digest(contract)
    provenance = h0.RunProvenance(
        effective_seed=config.seed,
        observation_digest=h0.RASTER31V3_CONTRACT.digest,
        world_digest=manifest["world_digest"],
        runtime_digest=value["runtime_contract_digest"],
        reward_digest=value["reward_contract_digest"],
        target_digest=value["target_contract_digest"],
        sampler_digest=value["sampler_contract_digest"],
        optimizer_digest=value["optimizer_contract_digest"],
        model_head_digest=h0.ModelHeadContract("pqn", "dueling_q", 6).digest,
        source_revision=manifest["git"]["commit"],
    )
    value.update(provenance.to_metadata())
    value.update(h0.ModelHeadContract("pqn", "dueling_q", 6).to_metadata())
    return value


def write_eval_receipt(arm_dir: Path) -> dict[str, str]:
    receipt = {"status": "completed"}
    receipt["receipt_digest"] = h0._digest_without(receipt, "receipt_digest")
    path = arm_dir / "fake-eval.json"
    h0._write_json(path, receipt)
    return {"path": str(path), "sha256": h0.sha256_file(path)}


def test_private_worker_dispatch_uses_actual_main_without_public_required_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "worker.json"
    spec_path.write_text("{}")
    seen: list[Path] = []

    def dispatch(path: Path) -> int:
        seen.append(path)
        return 23

    monkeypatch.setattr(h0, "worker_main", dispatch)
    assert h0.main(["--worker-spec", str(spec_path)]) == 23
    assert seen == [spec_path]


def test_worker_seeds_before_keyword_device_trainer_and_emits_initial_and_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = make_plan(tmp_path)
    manifest = h0.freeze_manifest(candidate)
    arm = candidate.arms[0]
    events: list[str] = []

    @dataclass
    class Telemetry:
        update: int = 0

    class KeywordOnlyTrainer:
        def __init__(self, config: Any, *, device: torch.device) -> None:
            assert events == ["seed"]
            assert config.seed == arm.training_seed
            assert device == torch.device("cpu")
            self.agent_steps = 0
            self.update_idx = 0

        def checkpoint_state(self) -> dict[str, Any]:
            return complete_payload(manifest, arm.name, self.agent_steps)

        def update(self) -> Telemetry:
            self.agent_steps = arm.requested_steps
            self.update_idx = 1
            return Telemetry()

    real_seed = h0.initialize_run_seed

    def seed(value: int) -> Any:
        events.append("seed")
        return real_seed(value)

    monkeypatch.setattr(h0, "initialize_run_seed", seed)
    monkeypatch.setattr(h0, "PQNTrainer", KeywordOnlyTrainer)
    monkeypatch.setattr(h0.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(h0.torch, "set_num_interop_threads", lambda _: None)
    for name in h0.THREAD_ENV:
        monkeypatch.setenv(name, "2")
    arm_dir = candidate.output_dir / arm.name
    arm_dir.mkdir()
    spec = h0._expected_worker_spec(manifest, arm.name)
    spec["worker_spec_digest"] = h0._digest_without(spec, "worker_spec_digest")
    spec_path = tmp_path / "worker-spec.json"
    spec_path.write_text(json.dumps(spec))

    assert h0.worker_main(spec_path) == 0
    assert events == ["seed"]
    assert (arm_dir / "initial.pth").is_file()
    assert (arm_dir / "final.pth").is_file()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.pop("final.pth"), "missing_initial_or_final_checkpoint"),
        (lambda value: value.update({"effective_world_digest": "wrong"}), "world_lineage_mismatch"),
        (lambda value: value.update({"h0_manifest_digest": "wrong"}), "manifest_lineage_mismatch"),
    ],
)
def test_exit_zero_cannot_evaluate_without_exact_checkpoint_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, reason: str
) -> None:
    candidate = make_plan(tmp_path)
    evaluated: list[bool] = []
    monkeypatch.setattr(
        h0.psutil, "virtual_memory", lambda: type("V", (), {"available": 99 * h0.GIB})()
    )

    def zero_exit_with_bad_lineage(command: list[str], **_: Any) -> Process:
        spec = json.loads(Path(command[-1]).read_text())
        manifest = json.loads(Path(spec["manifest_path"]).read_text())
        arm_dir = Path(spec["arm_dir"])
        initial = complete_payload(manifest, arm_dir.name, 0)
        final = complete_payload(manifest, arm_dir.name, spec["requested_steps"])
        if reason == "missing_initial_or_final_checkpoint":
            torch.save(initial, arm_dir / "initial.pth")
        else:
            mutate(final)
            torch.save(initial, arm_dir / "initial.pth")
            torch.save(final, arm_dir / "final.pth")
        h0._worker_terminal(
            arm_dir,
            status="completed",
            cause="budget_reached",
            requested_steps=spec["requested_steps"],
            actual_steps=spec["requested_steps"],
            update_count=1,
            started=h0.time.monotonic(),
            resources={"thresholds": spec["limits"]},
        )
        return Process(returncode=0, alive_polls=0)

    results = h0.run_plan(
        candidate,
        popen=zero_exit_with_bad_lineage,
        evaluator=lambda *args: (evaluated.append(True), {})[1],
        process_factory=lambda _: Child(),
    )
    assert results[0]["outcome"] == "learner_exit_failure"
    assert results[0]["checkpoint_lineage"]["reason"] == reason
    assert evaluated == []


def test_screen_has_exactly_five_single_attempt_arms_and_evaluates_only_after_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = make_plan(tmp_path, "screen")
    started: list[tuple[str, Process]] = []
    evaluated: list[str] = []

    def popen(command: list[str], **_: Any) -> Process:
        spec = json.loads(Path(command[-1]).read_text())
        arm_dir = Path(spec["arm_dir"])
        process = Process(returncode=12 if len(started) == 2 else 0)
        started.append((arm_dir.name, process))
        if process._final == 0:
            (arm_dir / "initial.pth").write_bytes(b"initial")
            (arm_dir / "final.pth").write_bytes(b"final")
            status, cause = "completed", "budget_reached"
        else:
            status, cause = "failed", "worker_exception"
        h0._worker_terminal(
            arm_dir,
            status=status,
            cause=cause,
            requested_steps=spec["requested_steps"],
            actual_steps=spec["requested_steps"] if status == "completed" else 0,
            update_count=1 if status == "completed" else 0,
            started=h0.time.monotonic(),
            resources={"thresholds": spec["limits"]},
        )
        return process

    def evaluate(
        arm_dir: Path, _manifest: dict[str, Any], _timeout: float
    ) -> tuple[bool, dict[str, Any]]:
        process = started[-1][1]
        assert process.waited, "evaluation must follow confirmed learner exit"
        evaluated.append(arm_dir.name)
        return True, write_eval_receipt(arm_dir)

    def lineage(arm_dir: Path, manifest: dict[str, Any]):
        requested = manifest["arms"][int(arm_dir.name.rsplit("-", 1)[1]) - 1]["requested_steps"]
        return True, {
            "initial_sha256": h0.sha256_file(arm_dir / "initial.pth"),
            "final_sha256": h0.sha256_file(arm_dir / "final.pth"),
            "actual_agent_steps": requested,
            "requested_agent_steps": requested,
            "overshoot_steps": 0,
            "update_count": 1,
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(
        h0.psutil, "virtual_memory", lambda: type("V", (), {"available": 99 * h0.GIB})()
    )
    results = h0.run_plan(
        candidate,
        popen=popen,
        evaluator=evaluate,
        sleep=lambda _: None,
        process_factory=lambda _: Child(),
        checkpoint_inspector=lineage,
    )
    assert [name for name, _ in started] == [arm.name for arm in candidate.arms]
    assert len(started) == len(results) == 5
    assert [result["outcome"] for result in results] == [
        "completed",
        "completed",
        "learner_exit_failure",
        "completed",
        "completed",
    ]
    assert evaluated == [candidate.arms[index].name for index in (0, 1, 3, 4)]
    assert all(process.waited for _, process in started)
