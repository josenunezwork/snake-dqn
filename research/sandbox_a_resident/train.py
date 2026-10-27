"""Durable simulator-D TRAIN-01 PPO entrypoint (never a gate or release tool)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.sandbox_a_resident.training_acceptance import run_distribution_screen
from research.sandbox_a_resident.training_compile import TrainingCompile
from research.sandbox_a_resident.training_cpu_reference import cpu_distribution_factory
from research.sandbox_a_resident.training_learner import PPOConfig, PPORollout, PPOTrainer
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic, categorical
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
from research.sandbox_a_resident.training_task import ResidentTrainingTask, TrainingTaskConfig
from research.sandbox_a_resident.training_units import allocation, load_ledger, publish_snapshot
from research.sandbox_a_resident.units import digest, publish
from src.model.inference_agent import InferenceAgent


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
        if self.outer_iterations < 1 or self.worlds < 3 or self.rollout_horizon < 1:
            raise ValueError("positive finite training endpoint required")

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
            or self.proxy_worlds_per_mix != 96
            or self.proxy_horizon != 5000
        ):
            raise ValueError("production TRAIN-01 recipe is frozen")

    @property
    def actor_rows(self) -> int:
        return self.outer_iterations * self.worlds * self.rollout_horizon


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
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["collection"])


def _task_state(task: Any) -> dict[str, Any]:
    if not hasattr(task, "state_dict"):
        raise TypeError("resident task must expose exact state_dict for durable resume")
    return task.state_dict()


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
    for _ in range(ticks):
        local, glob, scalars, legal = compiled.observe() if compiled is not None else task.observe()
        remaining = (task.config.horizon - task.age).clamp_min(0).float() / task.config.horizon
        logits = (
            compiled.actor_forward(local, glob, scalars)
            if compiled is not None
            else actor(local, glob, scalars)
        )
        distribution = categorical(logits, legal)
        action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        observations.append((local.clone(), glob.clone(), scalars.clone()))
        masks.append(legal.clone())
        actions.append(action)
        logps.append(distribution.log_prob(action))
        values.append(critic(local, glob, scalars, remaining))
        clocks.append(remaining)
        step = task.step(action)
        rewards.append(step.reward)
        terminals.append(step.terminal)
        horizons.append(step.cut)
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
) -> int:
    if state.get("schema") != "sandbox-a-train-state/v1" or state.get(
        "control_allocation"
    ) != allocation(asdict(control)):
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
) -> dict[str, Any]:
    """Run/resume the bounded trajectory; it never retries failed collection."""
    if digest(checkpoint) != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise ValueError("TRAIN-01 requires the pinned B8 s3 checkpoint")
    if digest(frp3) != FRP3_S12_SHA256 or digest(champion) != CHAMPION_SHA256:
        raise ValueError("frozen roster checkpoint hash differs from gate identity")
    output.mkdir(parents=True, exist_ok=True)
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
    task = ResidentTrainingTask(
        TrainingTaskConfig(
            worlds=control.worlds,
            allocation_offset=control.train_seed_offset,
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
    compile_check = compiled.runtime_self_check(32)
    # The screen is a mandatory simulator-D gross-error diagnostic before the
    # real trajectory, deliberately separate from any efficacy/gate evidence.
    proxy_factory = task.proxy_factory(opponents)
    start, metrics, prep_state, resumed = 0, [], None, False
    last_heartbeat = time.monotonic()
    action_window = torch.empty(0, dtype=torch.long)
    if ledger is not None and ledger.exists():
        verified = load_ledger(ledger, asdict(control), input_root=input_root)
        if verified:
            permitted = {
                0,
                control.outer_iterations,
                *range(
                    control.snapshot_interval, control.outer_iterations, control.snapshot_interval
                ),
            }
            if not set(verified).issubset(permitted):
                raise ValueError("ledger contains a snapshot outside the frozen cadence/endpoint")
            latest = verified[max(verified)]
            state = torch.load(latest["exports"][0]["path"], map_location=dev, weights_only=False)
            if int(state.get("iteration", -1)) != int(latest["iteration"]):
                raise ValueError("ledger descriptor and snapshot iteration differ")
            start = _restore(state, control, actor, critic, trainer, task, generator)
            metrics = list(state.get("metrics", []))
            action_window = state.get("action_window", action_window).cpu()
            prep_state = state.get("preparation")
            resumed = True
            if prep_state is not None and prep_state.get("temperature") is not None:
                with torch.no_grad():
                    actor.temperature.fill_(float(prep_state["temperature"]))
    if not control.test_only and start == 0:
        os.environ.setdefault("SNAKE_CKPT_DIR", str(frp3.parent))
        cpu_actor = copy.deepcopy(actor.reference).cpu().eval()
        screen = run_distribution_screen(
            cpu_factory=cpu_distribution_factory(
                {FRP3_S12_SHA256: frp3, CHAMPION_SHA256: champion}
            ),
            device_factory=lambda mix, worlds, horizon: proxy_factory(
                mix, worlds, horizon, 600_000_000 + ("scripted", "frozen", "mixed").index(mix) * 32
            ),
            cpu_actor=cpu_actor,
            device_actor=actor.reference,
        )
        metrics.append({"iteration": 0, "distribution_screen": screen})

    def prep_task_factory(phase: str) -> ResidentTrainingTask:
        if phase not in {"calibration", "refit"}:
            raise ValueError("unknown frozen preparation phase")
        cap = 5_242_880 if phase == "calibration" else 8_388_608
        prepared_task = ResidentTrainingTask(
            TrainingTaskConfig(
                worlds=24,
                allocation_offset=200_000_000 if phase == "calibration" else 300_000_000,
                max_episodes=cap // 24 + 1,
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

        try:
            prepared = prepare(
                prep_task_factory,
                actor,
                critic,
                PreparationConfig(),
                generator,
                state=prep_state,
                on_calibrated=calibrated_boundary,
            )
        except PreparationFailure as error:
            # The preparation helper carries its exact independent task state.
            failure = None if error.state is None else error.state.state_dict()
            failed = _snapshot_state(
                0, control, actor, critic, trainer, task, generator, metrics, failure
            )
            publish_snapshot(
                output,
                0,
                lambda path, value=failed: torch.save(value, path),
                metadata={
                    **snapshot_header,
                    "phase": "preparation-failed",
                    "prepare_phase": "failed",
                },
            )
            raise
        prep_state = prepared.state.state_dict()
        trainer.critic_optimizer = prepared.critic_optimizer
        metrics.append({"iteration": 0, "preparation": prep_state})
    elif prep_state is None:
        # Deliberately conspicuous CPU fixture path.  It is unavailable unless
        # the immutable control says test_only and cannot qualify a real run.
        prep_state = {"schema": "sandbox-a-train01-prep/test-only", "stage": "skipped"}
        metrics.append({"iteration": 0, "preparation": prep_state})
    # Proxy adapters may allocate helper tensors while constructing a factory.
    # Keep that setup outside the resumed trajectory's RNG stream.
    factory = proxy_factory
    proxy = ProxyConfig(
        horizon=control.proxy_horizon,
        worlds_per_mix=control.proxy_worlds_per_mix,
        seed_base=400_000_000,
    )
    proxy_identity = hashlib.sha256(
        json.dumps(
            {
                "source_commit": control.source_commit,
                "provenance": provenance,
                "world_config": asdict(task.sim.cfg),
                "proxy": asdict(proxy),
                "rosters": [roster.identity for roster in task.rosters],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    baseline_cache = next(
        (row.get("proxy_baseline") for row in metrics if "proxy_baseline" in row), None
    )
    if baseline_cache is not None:
        saved_identity = next(
            row.get("proxy_identity") for row in metrics if "proxy_baseline" in row
        )
        if saved_identity != proxy_identity:
            raise ValueError("cached B8 proxy identity differs from frozen engine/roster/config")
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
    if baseline_cache is None:
        baseline = evaluate(actor.reference, factory, proxy)
        baseline_cache = serializable(baseline)
        metrics.append(
            {
                "iteration": 0,
                "proxy_baseline": baseline_cache,
                "proxy_config": asdict(proxy),
                "proxy_identity": proxy_identity,
                "zero_actor_proxy": "identity reuse after fresh-observation B8 greedy agreement",
            }
        )
    if start == 0 and not resumed:
        initial = _snapshot_state(
            0, control, actor, critic, trainer, task, generator, metrics, prep_state
        )
        publish_snapshot(
            output,
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
    for iteration in range(start, control.outer_iterations):
        collect_started = time.monotonic()
        rollout = collect(task, actor, critic, control.rollout_horizon, generator, compiled)
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
            **trainer.update(rollout),
            "collector_seconds": learner_started - collect_started,
            "learner_seconds": time.monotonic() - learner_started,
            "compile": {"self_check": compile_check, "report": compiled.report},
        }
        metrics.append(row)
        boundary = iteration + 1
        if boundary % control.proxy_interval == 0:
            # The task factory is supplied by the adapter so each candidate and
            # baseline starts an independent, identical proxy RNG stream.
            candidate = evaluate(actor, factory, proxy)
            baseline = {mix: MixMetrics(**value) for mix, value in baseline_cache.items()}
            row["proxy"] = {
                "config": asdict(proxy),
                "baseline": baseline_cache,
                "candidate": serializable(candidate),
                "paired": paired(candidate, baseline),
            }
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
            publish_snapshot(
                output,
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
        now = time.monotonic()
        if now - last_heartbeat >= 120 or boundary == control.outer_iterations:
            _progress(
                output,
                {
                    "stage": "ppo",
                    "iteration": boundary,
                    "actor_rows": trainer.actor_rows,
                    "preparation": prep_state,
                    "elapsed_seconds": now - last_heartbeat,
                },
            )
            last_heartbeat = now
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
    args = parser.parse_args()
    control = TrainControl.from_json(json.loads(args.control.read_text()))
    summary = run(
        control,
        checkpoint=args.checkpoint,
        frp3=args.frp3,
        champion=args.champion,
        output=args.output,
        device=args.device,
        ledger=args.ledger,
        input_root=args.input_root,
    )
    print(
        f"PASS_SANDBOX_A_TRAIN iterations={summary['iterations']} "
        f"actor_rows={summary['actor_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
