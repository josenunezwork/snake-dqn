"""Durable simulator-D TRAIN-01 PPO entrypoint (never a gate or release tool)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from research.sandbox_a_resident.training_acceptance import run_distribution_screen
from research.sandbox_a_resident.training_compact import (
    materialize_compact_resume,
    publish_compact_snapshot,
    verify_decoded_state,
)
from research.sandbox_a_resident.training_compile import TrainingCompile
from research.sandbox_a_resident.training_cpu_reference import cpu_distribution_factory
from research.sandbox_a_resident.training_learner import (
    GracefulSignalStop,
    PPOConfig,
    PPORollout,
    PPOTrainer,
)
from research.sandbox_a_resident.training_model import (
    B8Actor,
    ClockedPopArtCritic,
    canonical_legal_mask,
    categorical,
)
from research.sandbox_a_resident.training_prep import PreparationConfig, PreparationFailure, prepare
from research.sandbox_a_resident.training_proxy import (
    MixMetrics,
    ProxyConfig,
    evaluate,
    paired,
    serializable,
)
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    DeviceRosterRuntime,
)
from research.sandbox_a_resident.training_task import (
    ResidentTrainingTask,
    TrainingTaskConfig,
    seed_allocations,
)
from research.sandbox_a_resident.training_units import (
    COMPACT_RESUME_FIELD,
    allocation,
    load_ledger,
    migration_source_control,
    publish_snapshot,
    source_migration,
    unit_id,
)
from research.sandbox_a_resident.units import digest, publish
from src.model.inference_agent import InferenceAgent

DIAGNOSTIC_PROXY_UPDATE = 16
SIGUSR1_GRACE_SECONDS = 15.0
DEFAULT_SIGUSR1_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
SIGUSR1_MIN_REMAINING_SECONDS_FOR_SNAPSHOT = 3.0
SIGUSR1_MARKER_NAME = "sigusr1-checkpoint.json"
SIGUSR1_MARKER_SCHEMA = "sandbox-a-train-sigusr1-checkpoint/v1"


class Sigusr1Controller:
    """Bounded SIGUSR1 grace tracker for clean minibatch/tick termination."""

    def __init__(
        self,
        *,
        grace_seconds: float = SIGUSR1_GRACE_SECONDS,
        max_snapshot_bytes: int = DEFAULT_SIGUSR1_MAX_SNAPSHOT_BYTES,
    ) -> None:
        if not np.isfinite(grace_seconds) or grace_seconds <= 0:
            raise ValueError("grace_seconds must be finite and positive")
        if type(max_snapshot_bytes) is not int or max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes must be a nonnegative integer")
        self.grace_seconds = float(grace_seconds)
        self.max_snapshot_bytes = max_snapshot_bytes
        self.received_at: float | None = None
        self.signal_count = 0

    def request_stop(self, *_args: Any) -> None:
        if self.received_at is None:
            self.received_at = time.monotonic()
        self.signal_count += 1

    def should_stop(self) -> bool:
        return self.received_at is not None

    def elapsed_seconds(self) -> float:
        if self.received_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self.received_at)

    def remaining_seconds(self) -> float:
        return self.grace_seconds - self.elapsed_seconds()


def _estimate_snapshot_bytes(*components: Any) -> int:
    """Estimate serialized state bytes in O(tensor_count) without copying buffers."""
    seen_storages: set[int] = set()
    total = 65_536

    def walk(value: Any) -> None:
        nonlocal total
        if isinstance(value, torch.Tensor):
            try:
                storage = value.untyped_storage()
                pointer = int(storage.data_ptr())
                size = int(storage.nbytes())
            except RuntimeError:
                pointer = 0
                size = int(value.numel() * value.element_size())
            if pointer == 0 or pointer not in seen_storages:
                if pointer != 0:
                    seen_storages.add(pointer)
                total += size
            return
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for component in components:
        if component is None:
            continue
        if hasattr(component, "state_dict") and callable(component.state_dict):
            walk(component.state_dict())
        else:
            walk(component)
    return total


@dataclass(frozen=True)
class TrainControl:
    """Frozen finite endpoint and non-overlapping sandbox allocation descriptor."""

    study: str
    source_commit: str
    allocation_index: int
    outer_iterations: int = 384
    worlds: int = 1024
    rollout_horizon: int = 256
    task_horizon: int = 5000
    snapshot_interval: int = 16
    proxy_interval: int = 64
    proxy_worlds_per_mix: int = 96
    proxy_horizon: int = 5000
    train_seed_offset: int = 0
    test_only: bool = False
    compile_mode: str = "default"

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TrainControl":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        foreign = set(value) - allowed
        if foreign:
            raise ValueError(f"unknown immutable control fields: {sorted(foreign)}")
        return cls(**value)

    def __post_init__(self) -> None:
        if self.study != "sandbox-a-train01" or len(self.source_commit) != 40:
            raise ValueError("foreign TRAIN-01 control identity")
        if type(self.allocation_index) is not int or self.allocation_index < 0:
            raise ValueError("allocation_index must be a nonnegative integer")
        if type(self.train_seed_offset) is not int or self.train_seed_offset < 0:
            raise ValueError("train_seed_offset must be a nonnegative integer")
        if self.outer_iterations < 1 or self.worlds < 3 or self.rollout_horizon < 1:
            raise ValueError("positive finite training endpoint required")
        train_capacity = (self.outer_iterations * self.rollout_horizon + 1) * self.worlds
        if self.allocation_index >= train_capacity:
            raise ValueError("allocation_index must lie inside the audited train allocation")

        if self.task_horizon != 5000 or self.snapshot_interval < 1 or self.proxy_interval < 1:
            raise ValueError("TRAIN-01 requires H5000 and positive cadences")
        if type(self.test_only) is not bool:
            raise ValueError("test_only must be boolean")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune", "none"}:
            raise ValueError("unsupported compile mode")
        if self.proxy_horizon < 1:
            raise ValueError("proxy_horizon must be positive")
        if not self.test_only and (
            self.outer_iterations != 384
            or self.worlds != 1024
            or self.rollout_horizon != 256
            or self.proxy_interval != 64
            or self.proxy_worlds_per_mix != 96
            or self.proxy_horizon != 5000
        ):
            raise ValueError("production TRAIN-01 recipe is frozen")
        seed_allocations(asdict(self))

    @property
    def actor_rows(self) -> int:
        return self.outer_iterations * self.worlds * self.rollout_horizon


def proxy_kind(boundary: int, control: TrainControl) -> str | None:
    """Return the fixed detached-monitoring schedule without a selection rule."""
    if boundary == DIAGNOSTIC_PROXY_UPDATE:
        return "diagnostic-update-16"
    if boundary % control.proxy_interval == 0:
        return "regular"
    return None


def rollout_monitoring(rollout: PPORollout) -> dict[str, Any]:
    """Return detached transition support, retaining the explicit denominators."""
    rows: dict[str, dict[str, int | float]] = {}
    for mix_id, name in enumerate(("scripted", "frozen", "mixed")):
        selected = rollout.mix_ids == mix_id
        denominator = int(selected.sum().item())
        rewards = rollout.rewards[selected]
        rows[name] = {
            "rows": denominator,
            # TRAIN-01's task reward is the post-step live hero mass / 1000.
            "mass_sum": float((rewards.sum() * 1000.0).item()),
            "terminal_count": int(rollout.terminal[selected].sum().item()),
            "horizon_count": int(rollout.horizon[selected].sum().item()),
        }
    return {"by_mix": rows, "rows": int(rollout.actions.numel())}


def _seed(control: TrainControl, index: int) -> int:
    """Domain-separated deterministic seed without using a protected bank."""
    raw = hashlib.sha256(
        f"a100-dev/{control.study}/{control.train_seed_offset + index}".encode()
    ).digest()
    return int.from_bytes(raw[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "collection": generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    torch.set_rng_state(state["torch"].to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(
            [value.to(device="cpu", dtype=torch.uint8) for value in state["cuda"]]
        )
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["collection"].to(device="cpu", dtype=torch.uint8))


def _task_state(task: Any) -> dict[str, Any]:
    if not hasattr(task, "state_dict"):
        raise TypeError("resident task must expose exact state_dict for durable resume")
    return task.state_dict()


def _preparation_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """Expose preparation diagnostics without duplicating exact task tensor state."""
    return {key: value for key, value in state.items() if key != "task_state"}


def _durable_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop wall-clock telemetry from the exact-resume state.

    Timings are useful in the mutable progress/terminal summary, but they are
    intentionally not trajectory state: serializing them would make a resumed
    snapshot differ from an uninterrupted one despite identical simulation,
    sampling, and optimizer state.
    """
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"collector_seconds", "learner_seconds"}
        }
        for row in metrics
    ]


_METADATA_ONLY_ROW_KEYS = frozenset(
    {
        "distribution_screen",
        "preparation",
        "zero_update_b8_greedy",
        "proxy_baseline",
        "proxy_cache_migration",
        "monitoring_note",
    }
)


def _find_learner_metrics_row(
    metrics: list[dict[str, Any]], iteration: int
) -> dict[str, Any] | None:
    """Locate the PPO learner row for ``iteration``, ignoring appended cache/note rows."""
    if iteration <= 0:
        return None
    return next(
        (
            row
            for row in reversed(metrics)
            if row.get("iteration") == iteration
            and not any(key in row for key in _METADATA_ONLY_ROW_KEYS)
        ),
        None,
    )


def _has_completed_proxy_at_iteration(
    metrics: list[dict[str, Any]], iteration: int, *, kind: str | None = None
) -> bool:
    """Return whether any row for ``iteration`` already carries a completed proxy dict."""
    for row in metrics:
        if row.get("iteration") != iteration:
            continue
        proxy_payload = row.get("proxy")
        if isinstance(proxy_payload, Mapping) and (
            kind is None or proxy_payload.get("kind") == kind
        ):
            return True
    return False


def _load_task_state(task: Any, state: dict[str, Any]) -> None:
    if not hasattr(task, "load_state_dict"):
        raise TypeError("resident task must expose exact load_state_dict for durable resume")
    task.load_state_dict(state)


@torch.no_grad()
def collect(
    task: Any,
    actor: B8Actor,
    critic: ClockedPopArtCritic,
    ticks: int,
    generator: torch.Generator,
    compiled: TrainingCompile | None = None,
    on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> PPORollout:
    """Collect exactly ``ticks * E`` fresh hero rows and retain natural cut facts."""
    observations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logps: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    terminals: list[torch.Tensor] = []
    horizons: list[torch.Tensor] = []
    clocks: list[torch.Tensor] = []
    logical_lengths: list[torch.Tensor] = []
    last_progress = time.monotonic()
    for tick in range(ticks):
        local, glob, scalars, legal = compiled.observe() if compiled is not None else task.observe()
        # Scalar zero is a log-normalized B8 input.  Detached diagnostics need
        # the simulator's logical mass at the pre-action instant instead.
        length = getattr(task, "hero_length", None)
        length = length() if callable(length) else length
        if not isinstance(length, torch.Tensor) or length.shape != (local.shape[0],):
            raise TypeError("resident task must expose pre-action logical hero lengths")
        remaining = (task.config.horizon - task.age).clamp_min(0).float() / task.config.horizon
        logits = (
            compiled.actor_forward(local, glob, scalars)
            if compiled is not None
            else actor(local, glob, scalars)
        )
        legal = canonical_legal_mask(legal)
        distribution = categorical(logits, legal)
        action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        observations.append((local.clone(), glob.clone(), scalars.clone()))
        logical_lengths.append(length.clone())
        masks.append(legal.clone())
        actions.append(action)
        logps.append(distribution.log_prob(action))
        values.append(critic(local, glob, scalars, remaining))
        clocks.append(remaining)
        step = task.step(action)
        rewards.append(step.reward)
        terminals.append(step.terminal)
        horizons.append(step.cut)
        if on_progress is not None and time.monotonic() - last_progress >= 60:
            on_progress("collect", {"ticks": tick + 1, "total": ticks})
            last_progress = time.monotonic()
        if should_stop is not None and should_stop():
            raise GracefulSignalStop("collect-tick", minibatch_completed=False)
    local, glob, scalars, _ = compiled.observe() if compiled is not None else task.observe()
    remaining = (task.config.horizon - task.age).clamp_min(0).float() / task.config.horizon
    return PPORollout(
        local=torch.stack([row[0] for row in observations]),
        glob=torch.stack([row[1] for row in observations]),
        scalars=torch.stack([row[2] for row in observations]),
        legal_masks=torch.stack(masks),
        actions=torch.stack(actions),
        logp=torch.stack(logps),
        rewards=torch.stack(rewards),
        values=torch.stack(values),
        terminal=torch.stack(terminals),
        horizon=torch.stack(horizons),
        last_value=critic(local, glob, scalars, remaining),
        last_bootstrap=~(terminals[-1] | horizons[-1]),
        remaining_horizon=torch.stack(clocks),
        mix_ids=task.roster_id.expand(ticks, -1),
        logical_lengths=torch.stack(logical_lengths),
    )


def _snapshot_state(
    iteration: int,
    control: TrainControl,
    actor: B8Actor,
    critic: ClockedPopArtCritic,
    trainer: PPOTrainer,
    task: Any,
    generator: torch.Generator,
    metrics: list[dict[str, Any]],
    preparation: dict[str, Any] | None = None,
    action_window: torch.Tensor | None = None,
) -> dict[str, Any]:
    return {
        "schema": "sandbox-a-train-state/v1",
        "iteration": iteration,
        "control": asdict(control),
        "control_allocation": allocation(asdict(control)),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "trainer": trainer.state_dict(),
        "task": _task_state(task),
        "rng": _rng_state(generator),
        "metrics": _durable_metrics(metrics),
        "preparation": preparation,
        "action_window": (
            torch.empty(0, dtype=torch.long) if action_window is None else action_window.cpu()
        ),
    }


def _restore(
    state: dict[str, Any],
    control: TrainControl,
    actor: B8Actor,
    critic: ClockedPopArtCritic,
    trainer: PPOTrainer,
    task: Any,
    generator: torch.Generator,
    source_control: dict[str, Any] | None = None,
) -> int:
    current_control = asdict(control)
    expected_control = current_control
    if source_control is not None:
        # Keep direct callers just as constrained as ledger-backed resume.
        source_migration(source_control, current_control)
        expected_control = source_control
    expected_allocations = {allocation(expected_control)}
    if (
        state.get("schema") != "sandbox-a-train-state/v1"
        or state.get("control_allocation") not in expected_allocations
        or state.get("control") != expected_control
    ):
        raise ValueError("foreign snapshot allocation")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    trainer.load_state_dict(state["trainer"])
    _load_task_state(task, state["task"])
    _restore_rng(state["rng"], generator)
    return int(state["iteration"])


def _export_actor(
    output: Path, iteration: int, actor: B8Actor, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Completion-last served-format actor export alongside the full snapshot."""
    # Ego2sNet construction inside export consumes global Torch initialization
    # draws before weights are overwritten.  Preserve them so a boundary export
    # cannot alter the exact post-snapshot trajectory on an uninterrupted run.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        return publish(
            output / f"iteration-{iteration:04d}-actor.pth",
            lambda path: actor.export_ego2s(path, metadata),
        )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def _load_vector61(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a frozen vector61 opponent through the project's ordinary loader."""
    agent = InferenceAgent.from_checkpoint(str(path), device=device)
    if getattr(agent, "obs_spec", None) != "vector61":
        raise ValueError(f"{path} is not the required vector61 opponent checkpoint")
    return agent.network.eval()


def _progress(output: Path, value: dict[str, Any]) -> None:
    """Atomically refresh operational telemetry; it is never resume evidence."""
    temporary = output / "progress.json.partial"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, output / "progress.json")


SNAPSHOT_FORMATS = ("raw", "compact")


def run(
    control: TrainControl,
    *,
    checkpoint: Path,
    frp3: Path,
    champion: Path,
    output: Path,
    device: str,
    ledger: Path | None = None,
    input_root: Path | None = None,
    monitoring: bool = True,
    on_progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    stop_controller: Sigusr1Controller | None = None,
    snapshot_format: str = "raw",
    source_packet_id: str | None = None,
    early_pre_actor_snapshot: bool = False,
) -> dict[str, Any]:
    """Run/resume the bounded trajectory; it never retries failed collection."""
    if type(early_pre_actor_snapshot) is not bool:
        raise TypeError("early_pre_actor_snapshot must be boolean")
    if type(monitoring) is not bool:
        raise TypeError("monitoring must be boolean")
    if snapshot_format not in SNAPSHOT_FORMATS:
        raise ValueError(f"unsupported snapshot_format: {snapshot_format!r}")
    if digest(checkpoint) != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise ValueError("TRAIN-01 requires the pinned B8 s3 checkpoint")
    if digest(frp3) != FRP3_S12_SHA256 or digest(champion) != CHAMPION_SHA256:
        raise ValueError("frozen roster checkpoint hash differs from gate identity")
    output.mkdir(parents=True, exist_ok=True)
    interruptible_stage: str | None = None

    def emit(stage: str, values: Mapping[str, Any] | None = None) -> None:
        value = {"stage": stage, **dict(values or {})}
        _progress(output, value)
        print(f"TRAIN01_PROGRESS {json.dumps(value, sort_keys=True)}", flush=True)
        if on_progress is not None:
            on_progress(stage, value)
        if (
            interruptible_stage is not None
            and stop_controller is not None
            and stop_controller.should_stop()
        ):
            raise GracefulSignalStop(interruptible_stage, minibatch_completed=False)

    def _publish_state_snapshot(
        boundary: int,
        writer: Callable[[Path], None],
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if snapshot_format == "compact":
            emit("snapshot-compact-start", {"iteration": boundary})
            published = publish_compact_snapshot(
                output,
                boundary,
                writer,
                metadata=metadata,
                source_packet_id=source_packet_id,
            )
            emit(
                "snapshot-compact-complete",
                {
                    "iteration": boundary,
                    "derivative_bytes": published["compact_snapshot"]["derivative"]["size"],
                },
            )
            return published
        return publish_snapshot(output, boundary, writer, metadata=metadata)

    emit("initializing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device)
    seed = _seed(control, control.allocation_index)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    generator = torch.Generator(device=dev).manual_seed(seed)
    actor = B8Actor(checkpoint).to(dev)
    critic = ClockedPopArtCritic(actor.reference).to(dev)
    trainer = PPOTrainer(actor, critic, PPOConfig())
    audited_allocations = {row["phase"]: row for row in seed_allocations(asdict(control))}
    train_allocation = audited_allocations["train"]
    task = ResidentTrainingTask(
        TrainingTaskConfig(
            worlds=control.worlds,
            allocation_offset=control.train_seed_offset,
            allocation_phase_offset=train_allocation["offset"],
            allocation_count=train_allocation["count"],
            max_episodes=control.outer_iterations * control.rollout_horizon + 1,
            tiny=control.test_only,
        ),
        device=str(dev),
    )
    provenance = {
        "b8_s3_sha256": digest(checkpoint),
        "frp3_s12_sha256": digest(frp3),
        "champion_sha256": digest(champion),
        "roster_definition": "research.t34_round0.bank._rows",
        "engine": "simulator-D",
    }
    snapshot_header = {
        "allocation": allocation(asdict(control)),
        "source_commit": control.source_commit,
        "provenance": provenance,
    }
    frozen = {
        FRP3_S12_SHA256: _load_vector61(frp3, dev),
        CHAMPION_SHA256: _load_vector61(champion, dev),
    }
    opponents = DeviceRosterRuntime(
        task.sim, task.rosters, b8_network=actor.reference, checkpoint_networks=frozen
    )
    if not hasattr(task, "attach_opponents"):
        raise TypeError("resident task must expose attach_opponents for roster runtime")
    task.attach_opponents(opponents)
    compiled = TrainingCompile(actor, task, mode=control.compile_mode)
    emit("compile-self-check")
    compile_check = compiled.runtime_self_check(control.worlds)
    emit("compile-self-check-complete")
    # The screen is a mandatory simulator-D gross-error diagnostic before the
    # real trajectory, deliberately separate from any efficacy/gate evidence.
    proxy_factory: Any = None
    start, metrics, prep_state, resumed = 0, [], None, False
    last_heartbeat = time.monotonic()
    action_window = torch.empty(0, dtype=torch.long)
    resume_source_control: dict[str, Any] | None = None
    permitted_boundaries = {
        0,
        control.outer_iterations,
        *range(control.snapshot_interval, control.outer_iterations, control.snapshot_interval),
    }
    last_published_iteration: int | None = None
    completed_consistent_iteration: int | None = None
    completed_consistent_row: dict[str, Any] | None = None
    stop_kwargs: dict[str, Any] = (
        {"should_stop": stop_controller.should_stop} if stop_controller is not None else {}
    )
    if ledger is not None and ledger.exists():
        ledger_payload = json.loads(ledger.read_text())
        predecessor = migration_source_control(ledger_payload, asdict(control))
        if predecessor != asdict(control):
            resume_source_control = predecessor
        compact_manifest: dict[str, Any] | None = None
        if COMPACT_RESUME_FIELD in ledger_payload:
            # The allowlisted compact derivative is decoded into its
            # byte-identical original before the ordinary ledger verification
            # re-hashes it; the decoded state is checked tensor-by-tensor below.
            if input_root is None:
                raise ValueError("compact resume requires the uploaded input root")
            emit("compact-resume-decode")
            compact_manifest = materialize_compact_resume(ledger_payload, input_root)
            emit("compact-resume-decoded", {"unit": compact_manifest["unit"]})
        verified = load_ledger(ledger, asdict(control), input_root=input_root)
        if verified:
            if not set(verified).issubset(permitted_boundaries):
                raise ValueError("ledger contains a snapshot outside the frozen cadence/endpoint")
            latest = verified[max(verified)]
            # Decode durable metadata/RNG on CPU. Module, optimizer and task
            # loaders explicitly place their numerical state on the target device.
            state = torch.load(latest["exports"][0]["path"], map_location="cpu", weights_only=False)
            if compact_manifest is not None:
                if latest.get("id") != compact_manifest["unit"]:
                    raise ValueError("compact resume unit is not the selected snapshot")
                # Every decoded tensor and the non-tensor structure must equal
                # the original snapshot's manifest before anything is restored.
                verify_decoded_state(state, compact_manifest)
                emit("compact-resume-verified", {"tensors": compact_manifest["tensor_count"]})
            if int(state.get("iteration", -1)) != int(latest["iteration"]):
                raise ValueError("ledger descriptor and snapshot iteration differ")
            start = _restore(
                state,
                control,
                actor,
                critic,
                trainer,
                task,
                generator,
                source_control=resume_source_control,
            )
            metrics = list(state.get("metrics", []))
            action_window = state.get("action_window", action_window).cpu()
            prep_state = state.get("preparation")
            resumed = True
            last_published_iteration = start
            completed_consistent_iteration = start
            if prep_state is not None and prep_state.get("temperature") is not None:
                with torch.no_grad():
                    actor.temperature.fill_(float(prep_state["temperature"]))
    try:
        if stop_controller is not None and stop_controller.should_stop():
            raise GracefulSignalStop("compile-self-check", minibatch_completed=False)
        if not control.test_only and start == 0 and not resumed:
            proxy_factory = task.proxy_factory(opponents, list(audited_allocations.values()))
            os.environ.setdefault("SNAKE_CKPT_DIR", str(frp3.parent))
            cpu_actor = copy.deepcopy(actor.reference).to(dev).eval()
            engineering_offset = int(audited_allocations["engineering"]["offset"])
            interruptible_stage = "distribution-screen"
            try:
                emit("distribution-screen-start")
                screen = run_distribution_screen(
                    cpu_factory=cpu_distribution_factory(
                        {FRP3_S12_SHA256: frp3, CHAMPION_SHA256: champion},
                        inference_device=dev,
                        root_offset=engineering_offset,
                    ),
                    device_factory=lambda mix, worlds, horizon: proxy_factory(
                        mix,
                        worlds,
                        horizon,
                        engineering_offset + ("scripted", "frozen", "mixed").index(mix) * 32,
                    ),
                    cpu_actor=cpu_actor,
                    device_actor=actor.reference,
                    on_progress=emit,
                )
            finally:
                interruptible_stage = None
            metrics.append({"iteration": 0, "distribution_screen": screen})
            publish(
                output / "distribution-screen.json",
                lambda path: path.write_text(json.dumps(screen, sort_keys=True) + "\n"),
            )
            emit(
                "distribution-screen-complete",
                {"material_gap_count": len(screen["material_gaps"])},
            )
            if screen["material_gaps"]:
                _progress(
                    output, {"stage": "distribution-screen-failed", "distribution_screen": screen}
                )
                raise RuntimeError(
                    "material distribution gap: diagnose sandbox adapter before training"
                )

        prep_config = PreparationConfig(diagnostics=monitoring)

        def prep_task_factory(phase: str) -> ResidentTrainingTask:
            if phase not in {"calibration", "refit"}:
                raise ValueError("unknown frozen preparation phase")
            allocation = audited_allocations["calibration" if phase == "calibration" else "critic"]
            worlds = (
                prep_config.calibration_worlds
                if phase == "calibration"
                else prep_config.prep_worlds
            )
            prepared_task = ResidentTrainingTask(
                TrainingTaskConfig(
                    worlds=worlds,
                    allocation_offset=allocation["offset"],
                    allocation_phase_offset=allocation["offset"],
                    allocation_count=allocation["count"],
                    max_episodes=allocation["count"] // worlds,
                ),
                device=str(dev),
            )
            prepared_task.attach_opponents(
                DeviceRosterRuntime(
                    prepared_task.sim,
                    prepared_task.rosters,
                    b8_network=actor.reference,
                    checkpoint_networks=frozen,
                )
            )
            return prepared_task

        prep_stage = None if prep_state is None else prep_state.get("stage")
        if prep_stage == "failed":
            raise PreparationFailure("refusing PPO resume from a failed preparation endpoint")
        if prep_stage not in ("ready", "skipped") and not control.test_only:

            def calibrated_boundary(
                state: Any, _actor: B8Actor, _critic: ClockedPopArtCritic, optimizer: Any
            ) -> None:
                emit("calibrated-boundary-write")
                trainer.critic_optimizer = optimizer
                payload = _snapshot_state(
                    0,
                    control,
                    actor,
                    critic,
                    trainer,
                    task,
                    generator,
                    metrics,
                    state.state_dict(),
                )
                publish(
                    output / "calibrated-boundary.pt",
                    lambda path, value=payload: torch.save(value, path),
                )
                publish(
                    output / "calibrated-boundary.json",
                    lambda path: path.write_text(
                        json.dumps(
                            {
                                "schema": "sandbox-a-train-calibrated-boundary/v1",
                                "metadata": {
                                    **snapshot_header,
                                    "phase": "calibrated",
                                    "prepare_phase": state.stage,
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    ),
                )

            interruptible_stage = "preparation"
            try:
                emit("preparation-start", {"prepare_phase": prep_stage})
                try:
                    prepared = prepare(
                        prep_task_factory,
                        actor,
                        critic,
                        prep_config,
                        generator,
                        state=prep_state,
                        on_calibrated=calibrated_boundary,
                        on_progress=emit,
                    )
                except PreparationFailure as error:
                    interruptible_stage = None
                    # The preparation helper carries its exact independent task state.
                    failure = None if error.state is None else error.state.state_dict()
                    failed = _snapshot_state(
                        0, control, actor, critic, trainer, task, generator, metrics, failure
                    )
                    _publish_state_snapshot(
                        0,
                        lambda path, value=failed: torch.save(value, path),
                        metadata={
                            **snapshot_header,
                            "phase": "preparation-failed",
                            "prepare_phase": "failed",
                        },
                    )
                    raise
            finally:
                interruptible_stage = None
            emit("preparation-complete")
            prep_state = prepared.state.state_dict()
            trainer.critic_optimizer = prepared.critic_optimizer
            metrics.append({"iteration": 0, "preparation": _preparation_metrics(prep_state)})
        elif prep_state is None:
            # Deliberately conspicuous CPU fixture path.  It is unavailable unless
            # the immutable control says test_only and cannot qualify a real run.
            prep_state = {"schema": "sandbox-a-train01-prep/test-only", "stage": "skipped"}
            metrics.append({"iteration": 0, "preparation": _preparation_metrics(prep_state)})
        # Proxy adapters may allocate helper tensors while constructing a factory.
        # Monitoring is wholly detached from the training trajectory: when it is
        # disabled, do not construct/evaluate the cache or mutate telemetry.
        factory = None
        proxy = None
        baseline_cache = None
        if monitoring:
            if proxy_factory is None:
                proxy_factory = task.proxy_factory(opponents, list(audited_allocations.values()))
            factory = proxy_factory
            proxy = ProxyConfig(
                horizon=control.proxy_horizon,
                worlds_per_mix=control.proxy_worlds_per_mix,
                seed_base=int(audited_allocations["monitor"]["offset"]),
            )

            def cache_identity(source_commit: str) -> str:
                return hashlib.sha256(
                    json.dumps(
                        {
                            "source_commit": source_commit,
                            "provenance": provenance,
                            "world_config": asdict(task.sim.cfg),
                            "proxy": asdict(proxy),
                            "rosters": [roster.identity for roster in task.rosters],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()

            proxy_identity = cache_identity(control.source_commit)
            baseline_cache = next(
                (row.get("proxy_baseline") for row in reversed(metrics) if "proxy_baseline" in row),
                None,
            )
            if baseline_cache is not None:
                saved_identity = next(
                    row.get("proxy_identity")
                    for row in reversed(metrics)
                    if "proxy_baseline" in row
                )
                if resume_source_control is not None:
                    predecessor_identity = cache_identity(resume_source_control["source_commit"])
                    if saved_identity != predecessor_identity:
                        raise ValueError("cached B8 proxy identity differs from bound predecessor")
                    # Preserve the verified immutable B8 cache bytes.  Rebind the
                    # latest cache metadata to the successor identity instead of
                    # silently recomputing its common-random-number bank.
                    metrics.append(
                        {
                            "iteration": start,
                            "proxy_baseline": baseline_cache,
                            "proxy_config": asdict(proxy),
                            "proxy_identity": proxy_identity,
                            "proxy_cache_migration": {
                                "from_source_commit": resume_source_control["source_commit"],
                                "to_source_commit": control.source_commit,
                                "from_proxy_identity": predecessor_identity,
                                "to_proxy_identity": proxy_identity,
                            },
                        }
                    )
                elif saved_identity != proxy_identity:
                    raise ValueError(
                        "cached B8 proxy identity differs from frozen engine/roster/config"
                    )
        if start == 0 and not resumed:
            from research.sandbox_a_resident.training_proxy import masked_argmax

            with torch.no_grad():
                local, glob, scalars, legal = task.observe()
                logits = actor(local, glob, scalars)
                reference_q = actor.reference(local, glob, scalars)
                actor_action = masked_argmax(logits, legal)
                reference_action = masked_argmax(reference_q, legal)
                difference = actor_action != reference_action
                fallback = torch.tensor((True, True, True, False, False, False), device=dev)
                allowed = torch.where(legal.any(-1, keepdim=True), legal, fallback)
                top = reference_q.masked_fill(~allowed, -torch.inf).topk(2).values
                margin = top[:, 0] - top[:, 1]
                if difference.any():
                    raise RuntimeError(
                        "zero-update B8 greedy agreement failed on fresh task observations"
                    )
                metrics.append(
                    {
                        "iteration": 0,
                        "zero_update_b8_greedy": {
                            "rows": int(reference_action.numel()),
                            "mismatches": int(difference.sum()),
                            "min_top_two_margin": float(margin.min()),
                            "same_device": str(dev),
                        },
                    }
                )
            completed_consistent_iteration = 0
        if start == 0 and not resumed and early_pre_actor_snapshot:
            initial = _snapshot_state(
                0, control, actor, critic, trainer, task, generator, metrics, prep_state
            )
            _publish_state_snapshot(
                0,
                lambda path, value=initial: torch.save(value, path),
                metadata={
                    **snapshot_header,
                    "actor_rows": 0,
                    "phase": "pre-actor",
                    "prepare_phase": prep_state.get("stage"),
                },
            )
            _export_actor(
                output,
                0,
                actor,
                {"iteration": 0, "source_commit": control.source_commit, **provenance},
            )
            last_published_iteration = 0
        if monitoring and baseline_cache is None:
            assert factory is not None and proxy is not None
            emit("proxy-baseline-start")
            baseline = evaluate(actor.reference, factory, proxy, emit, **stop_kwargs)
            emit("proxy-baseline-complete")
            baseline_cache = serializable(baseline)
            metrics.append(
                {
                    "iteration": 0,
                    "proxy_baseline": baseline_cache,
                    "proxy_config": asdict(proxy),
                    "proxy_identity": proxy_identity,
                    "zero_actor_proxy": (
                        "identity reuse after fresh-observation B8 greedy agreement"
                    ),
                }
            )

        def detached_proxy(boundary: int, kind: str) -> dict[str, Any]:
            """Run one fixed-bank monitor with no nomination or stop consequence."""
            assert factory is not None and proxy is not None and baseline_cache is not None
            emit("proxy-candidate-start", {"iteration": boundary, "kind": kind})
            candidate = evaluate(actor, factory, proxy, emit, **stop_kwargs)
            emit("proxy-candidate-complete", {"iteration": boundary, "kind": kind})
            baseline = {mix: MixMetrics(**value) for mix, value in baseline_cache.items()}
            return {
                "kind": kind,
                "diagnostic_only": kind == "diagnostic-update-16",
                "nominates": False,
                "config": asdict(proxy),
                "baseline": baseline_cache,
                "candidate": serializable(candidate),
                "paired": paired(candidate, baseline),
            }

        # A train01-07 snapshot at the exact new boundary contains the correct
        # post-update-16 weights but predates this detached check.  Measure it once
        # before collecting update 17, then persist it in the successor snapshot.
        # Once successor telemetry is present, a further resume never repeats it.
        has_update16_proxy = _has_completed_proxy_at_iteration(
            metrics, DIAGNOSTIC_PROXY_UPDATE, kind="diagnostic-update-16"
        )
        if monitoring and resumed and start == DIAGNOSTIC_PROXY_UPDATE and not has_update16_proxy:
            diagnostic_row = _find_learner_metrics_row(metrics, start)
            if diagnostic_row is None:
                raise ValueError("resume snapshot lacks its update-16 metrics row")
            diagnostic_row["proxy"] = detached_proxy(start, "diagnostic-update-16")
            diagnostic_row.pop("proxy_deferred_on_sigusr1", None)
        elif monitoring and resumed and start > DIAGNOSTIC_PROXY_UPDATE and not has_update16_proxy:
            # Do not backdate a diagnostic against later weights.  It was absent
            # from the predecessor bytes and is explicitly reported as missed.
            missing_note = "update-16 diagnostic unavailable in predecessor snapshot"
            if not any(row.get("monitoring_note") == missing_note for row in metrics):
                metrics.append({"iteration": start, "monitoring_note": missing_note})
        resumed_proxy_kind = proxy_kind(start, control) if start > 0 else None
        if (
            monitoring
            and resumed
            and resumed_proxy_kind is not None
            and start != DIAGNOSTIC_PROXY_UPDATE
            and not _has_completed_proxy_at_iteration(metrics, start)
        ):
            resumed_row = _find_learner_metrics_row(metrics, start)
            if resumed_row is None:
                raise ValueError(f"resume snapshot lacks its update-{start} metrics row")
            resumed_row["proxy"] = detached_proxy(start, resumed_proxy_kind)
            resumed_row.pop("proxy_deferred_on_sigusr1", None)
        if start == 0 and not resumed and not early_pre_actor_snapshot:
            initial = _snapshot_state(
                0, control, actor, critic, trainer, task, generator, metrics, prep_state
            )
            _publish_state_snapshot(
                0,
                lambda path, value=initial: torch.save(value, path),
                metadata={
                    **snapshot_header,
                    "actor_rows": 0,
                    "phase": "pre-actor",
                    "prepare_phase": prep_state.get("stage"),
                },
            )
            _export_actor(
                output,
                0,
                actor,
                {"iteration": 0, "source_commit": control.source_commit, **provenance},
            )
            last_published_iteration = 0
        for iteration in range(start, control.outer_iterations):
            if stop_controller is not None and stop_controller.should_stop():
                raise GracefulSignalStop("before-collect", minibatch_completed=False)
            collect_started = time.monotonic()
            emit("collect-start", {"iteration": iteration + 1})
            rollout = collect(
                task,
                actor,
                critic,
                control.rollout_horizon,
                generator,
                compiled,
                emit,
                **stop_kwargs,
            )
            emit("learner-start", {"iteration": iteration + 1})
            if not rollout.legal_masks.gather(-1, rollout.actions[..., None]).all():
                raise RuntimeError("collector selected an illegal action")
            action_window = torch.cat((action_window, rollout.actions.detach().cpu().reshape(-1)))
            while action_window.numel() >= 100_000:
                completed, action_window = action_window[:100_000], action_window[100_000:]
                if torch.bincount(completed, minlength=6).max().item() > 90_000:
                    raise RuntimeError(
                        "action collapse exceeds 90 percent over a complete 100k-row window"
                    )
            learner_started = time.monotonic()
            row = {
                "iteration": iteration + 1,
                **trainer.update(rollout, diagnostics=monitoring, **stop_kwargs),
                "collector_seconds": learner_started - collect_started,
                "learner_seconds": time.monotonic() - learner_started,
                "compile": {"self_check": compile_check, "report": compiled.report},
            }
            if monitoring:
                row["transition_support"] = rollout_monitoring(rollout)
            metrics.append(row)
            emit("learner-complete", {"iteration": iteration + 1})
            boundary = iteration + 1
            completed_consistent_iteration = boundary
            completed_consistent_row = row
            kind = proxy_kind(boundary, control)
            if monitoring and kind is not None:
                # The task factory is supplied by the adapter so each candidate and
                # baseline starts an independent, identical proxy RNG stream.  The
                # update-16 result is diagnostic only: it never expands nomination
                # or changes retry/stop behaviour.
                try:
                    row["proxy"] = detached_proxy(boundary, kind)
                except GracefulSignalStop:
                    row["proxy_deferred_on_sigusr1"] = kind
                    raise
            if boundary % control.snapshot_interval == 0 or boundary == control.outer_iterations:
                state = _snapshot_state(
                    boundary,
                    control,
                    actor,
                    critic,
                    trainer,
                    task,
                    generator,
                    metrics,
                    prep_state,
                    action_window,
                )
                _publish_state_snapshot(
                    boundary,
                    lambda path, value=state: torch.save(value, path),
                    metadata={
                        **snapshot_header,
                        "actor_rows": trainer.actor_rows,
                        "metrics": row,
                        "prepare_phase": prep_state.get("stage"),
                    },
                )
                _export_actor(
                    output,
                    boundary,
                    actor,
                    {"iteration": boundary, "source_commit": control.source_commit, **provenance},
                )
                last_published_iteration = boundary
            emit("iteration-complete", {"iteration": boundary})
            now = time.monotonic()
            if now - last_heartbeat >= 120 or boundary == control.outer_iterations:
                _progress(
                    output,
                    {
                        "stage": "ppo",
                        "iteration": boundary,
                        "actor_rows": trainer.actor_rows,
                        "preparation": _preparation_metrics(prep_state),
                        "elapsed_seconds": now - last_heartbeat,
                    },
                )
                last_heartbeat = now
            if (
                stop_controller is not None
                and stop_controller.should_stop()
                and boundary < control.outer_iterations
            ):
                raise GracefulSignalStop("iteration-complete", minibatch_completed=False)
    except GracefulSignalStop as stopped:
        interruptible_stage = None
        estimated_bytes = _estimate_snapshot_bytes(actor, critic, trainer, task)
        remaining_grace = (
            SIGUSR1_GRACE_SECONDS
            if stop_controller is None
            else stop_controller.remaining_seconds()
        )
        max_bytes = (
            DEFAULT_SIGUSR1_MAX_SNAPSHOT_BYTES
            if stop_controller is None
            else stop_controller.max_snapshot_bytes
        )
        grace_budget = (
            SIGUSR1_GRACE_SECONDS if stop_controller is None else stop_controller.grace_seconds
        )
        can_attempt_boundary = (
            completed_consistent_iteration is not None
            and completed_consistent_iteration != last_published_iteration
            and completed_consistent_iteration in permitted_boundaries
            and prep_state is not None
            and prep_state.get("stage") in ("ready", "skipped")
        )
        sigusr1_action = "last_known_good_marker"
        if not can_attempt_boundary:
            if (
                completed_consistent_iteration is not None
                and completed_consistent_iteration == last_published_iteration
            ):
                sigusr1_reason = "latest_boundary_already_published"
            elif (
                completed_consistent_iteration is not None
                and completed_consistent_iteration not in permitted_boundaries
            ):
                sigusr1_reason = "off_cadence_boundary_rollback_to_last_snapshot"
            else:
                sigusr1_reason = "mid_iteration_rollback_to_last_snapshot"
        elif estimated_bytes > max_bytes:
            sigusr1_reason = "snapshot_exceeds_15s_byte_budget"
        elif remaining_grace < SIGUSR1_MIN_REMAINING_SECONDS_FOR_SNAPSHOT:
            sigusr1_reason = "insufficient_remaining_grace_budget"
        else:
            boundary_to_write = completed_consistent_iteration
            assert boundary_to_write is not None and prep_state is not None
            state_payload = _snapshot_state(
                boundary_to_write,
                control,
                actor,
                critic,
                trainer,
                task,
                generator,
                metrics,
                prep_state,
                None if boundary_to_write == 0 else action_window,
            )
            snap_meta: dict[str, Any] = {
                **snapshot_header,
                "actor_rows": 0 if boundary_to_write == 0 else trainer.actor_rows,
                "prepare_phase": prep_state.get("stage"),
                "sigusr1_boundary_save": True,
            }
            if boundary_to_write == 0:
                snap_meta["phase"] = "pre-actor"
            elif completed_consistent_row is not None:
                snap_meta["metrics"] = completed_consistent_row
                if "proxy_deferred_on_sigusr1" in completed_consistent_row:
                    snap_meta["deferred_proxy"] = completed_consistent_row[
                        "proxy_deferred_on_sigusr1"
                    ]
            _publish_state_snapshot(
                boundary_to_write,
                lambda path, value=state_payload: torch.save(value, path),
                metadata=snap_meta,
            )
            _export_actor(
                output,
                boundary_to_write,
                actor,
                {
                    "iteration": boundary_to_write,
                    "source_commit": control.source_commit,
                    **provenance,
                },
            )
            last_published_iteration = boundary_to_write
            sigusr1_action = "full_snapshot_published"
            sigusr1_reason = "permitted_boundary_within_15s_budget"

        elapsed_grace = 0.0 if stop_controller is None else stop_controller.elapsed_seconds()
        marker = {
            "schema": SIGUSR1_MARKER_SCHEMA,
            "signal": "SIGUSR1",
            "action": sigusr1_action,
            "reason": sigusr1_reason,
            "stopped_stage": stopped.stage,
            "minibatch_completed": bool(stopped.minibatch_completed),
            "stopped_epoch": stopped.epoch,
            "stopped_minibatch": stopped.minibatch,
            "completed_consistent_iteration": completed_consistent_iteration,
            "last_known_good_iteration": last_published_iteration,
            "last_known_good_unit": (
                None if last_published_iteration is None else unit_id(last_published_iteration)
            ),
            "estimated_snapshot_bytes": estimated_bytes,
            "max_snapshot_bytes": max_bytes,
            "grace_budget_seconds": grace_budget,
            "grace_elapsed_seconds": round(elapsed_grace, 6),
        }
        marker_path = output / SIGUSR1_MARKER_NAME
        if not marker_path.exists():
            publish(
                marker_path,
                lambda path, value=marker: path.write_text(
                    json.dumps(value, sort_keys=True) + "\n"
                ),
            )
        emit("sigusr1-stopped", marker)
        return {
            "schema": "sandbox-a-train-sigusr1-stop/v1",
            "status": "SIGUSR1_STOPPED",
            "allocation": allocation(asdict(control)),
            "actor_rows": trainer.actor_rows,
            "iterations": last_published_iteration,
            "sigusr1": marker,
        }
    summary = {
        "schema": "sandbox-a-train-summary/v1",
        "allocation": allocation(asdict(control)),
        "actor_rows": trainer.actor_rows,
        "iterations": control.outer_iterations,
        "metrics": metrics,
    }
    publish(
        output / "summary.json",
        lambda path: path.write_text(json.dumps(summary, sort_keys=True) + "\n"),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frp3", required=True, type=Path)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--stall-timeout-seconds", type=float, default=900)
    parser.add_argument("--stall-diagnostics", type=Path)
    parser.add_argument(
        "--sigusr1-grace-seconds",
        type=float,
        default=SIGUSR1_GRACE_SECONDS,
    )
    parser.add_argument(
        "--sigusr1-max-snapshot-bytes",
        type=int,
        default=int(
            os.environ.get(
                "SANDBOX_A_SIGUSR1_MAX_SNAPSHOT_BYTES",
                str(DEFAULT_SIGUSR1_MAX_SNAPSHOT_BYTES),
            )
        ),
    )
    parser.add_argument(
        "--snapshot-format",
        choices=SNAPSHOT_FORMATS,
        default=None,
        help="snapshot persistence format (defaults to SANDBOX_A_SNAPSHOT_FORMAT or compact)",
    )
    parser.add_argument(
        "--source-packet-id",
        default=os.environ.get("SANDBOX_A_PACKET_ID"),
        help="optional packet id bound into compact snapshot manifests",
    )
    args = parser.parse_args()
    control = TrainControl.from_json(json.loads(args.control.read_text()))
    resolved_snapshot_format = (
        args.snapshot_format
        if args.snapshot_format is not None
        else os.environ.get("SANDBOX_A_SNAPSHOT_FORMAT", "raw" if control.test_only else "compact")
    )
    stop_controller = Sigusr1Controller(
        grace_seconds=args.sigusr1_grace_seconds,
        max_snapshot_bytes=args.sigusr1_max_snapshot_bytes,
    )
    previous_sigusr1 = None
    if hasattr(signal, "SIGUSR1"):
        previous_sigusr1 = signal.signal(signal.SIGUSR1, stop_controller.request_stop)
    from research.sandbox_a_resident.stall_watchdog import StallWatchdog

    try:
        with StallWatchdog(
            args.stall_diagnostics or args.output / "stall-watchdog", args.stall_timeout_seconds
        ) as watchdog:
            summary = run(
                control,
                checkpoint=args.checkpoint,
                frp3=args.frp3,
                champion=args.champion,
                output=args.output,
                device=args.device,
                ledger=args.ledger,
                input_root=args.input_root,
                on_progress=lambda stage, values: watchdog.progress(stage, values),
                stop_controller=stop_controller,
                snapshot_format=resolved_snapshot_format,
                source_packet_id=args.source_packet_id,
                early_pre_actor_snapshot=os.environ.get("SANDBOX_A_EARLY_PRE_ACTOR_SNAPSHOT")
                == "1",
            )
    finally:
        if hasattr(signal, "SIGUSR1") and previous_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, previous_sigusr1)
    if summary.get("status") == "SIGUSR1_STOPPED":
        sigusr1 = summary["sigusr1"]
        print(
            f"STOP_SANDBOX_A_TRAIN_SIGUSR1 action={sigusr1['action']} "
            f"last_known_good_iteration={sigusr1['last_known_good_iteration']} "
            f"actor_rows={summary['actor_rows']}"
        )
        return 0
    print(
        f"PASS_SANDBOX_A_TRAIN iterations={summary['iterations']} "
        f"actor_rows={summary['actor_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
