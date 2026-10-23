"""A3 P1 actor-enabled learner ceiling benchmark.

This is an operational performance ruler, never a P1 qualification or an
alternative P1 recipe.  Production dimensions are deliberately fixed at the
registered 128 worlds by 128 steps.  ``--tiny`` is only a serial Mac/CUDA
entrypoint smoke: it labels its result non-scientific and cannot be aggregated
with an A3 result.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from research.bl_ppo.exports import export_json

KILL_SECONDS = 1.6384
PRODUCTION_WORLDS = 128
PRODUCTION_STEPS = 128


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _clock(device: str) -> float:
    _sync(device)
    return time.perf_counter()


def _gpu_busy() -> float | None:
    """Sample utilisation once; it is context, never the timing authority."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        return float(completed.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _timed_update(
    learner: Any,
    rollout: Any,
    device: str,
    flattened: bool,
    *,
    sample_busy: bool,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    samples: list[float] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            value = _gpu_busy()
            if value is not None:
                samples.append(value)
            stop.wait(0.2)

    thread = threading.Thread(target=sampler, daemon=True) if sample_busy else None
    if thread is not None:
        thread.start()
    try:
        result = _one_update(
            learner, rollout, device, flattened, capture_gradients=capture_gradients
        )
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=4)
    result["gpu_busy_samples"] = samples
    return result


class FlattenedFLearner:
    """Confined F-only forward replacement used for the labelled comparison.

    The production learner is untouched.  Flattening creates a logical
    ``[T*N]`` batch, but processes it in bounded chunks: the full 16,384-row
    B8 raster batch can exceed smaller L4 allocations. This bounds workspace,
    while autograd still retains all trainable activations until backward.
    Concatenating in time order preserves loss semantics; kernel reassociation is
    measured by the comparison receipt.
    """

    def __init__(self, learner: Any, *, chunk_rows: int = 1024) -> None:
        self.learner = learner
        self.chunk_rows = chunk_rows

    def forward(self, rollout: Any) -> tuple[Any, Any]:
        import torch

        if self.learner.policy.is_recurrent:
            raise ValueError("A3 flattening is F-arm only")
        if int(rollout.mask().any(dim=1).nonzero()[0]) != 0:
            raise ValueError("A3 F benchmark requires no recurrent burn-in prefix")
        time_steps, worlds = rollout.shape
        local = rollout.local.reshape(time_steps * worlds, *rollout.local.shape[2:])
        global_ = rollout.global_.reshape(time_steps * worlds, *rollout.global_.shape[2:])
        scalars = rollout.scalars.reshape(time_steps * worlds, rollout.scalars.shape[-1])
        horizon = (
            rollout.remaining_horizon.reshape(-1) if rollout.remaining_horizon is not None else None
        )
        logits, values = [], []
        for start in range(0, local.shape[0], self.chunk_rows):
            stop = min(start + self.chunk_rows, local.shape[0])
            current_horizon = None if horizon is None else horizon[start:stop]
            row_logits, row_values, _ = self.learner.policy(
                local[start:stop],
                global_[start:stop],
                scalars[start:stop],
                remaining_horizon=current_horizon,
            )
            logits.append(row_logits)
            values.append(row_values)
        return torch.cat(logits).reshape(time_steps, worlds, 6), torch.cat(values).reshape(
            time_steps, worlds
        )


class _NoDuplicateTeacherAdapter:
    """P1 adapter variant that omits only an unused diagnostic teacher replay."""

    def __init__(self, policy: Any, device: str) -> None:
        from research.bl_ppo.run import PersistentPolicyAdapter

        # Preserve the real adapter's complete rollout construction/state.
        self._base = PersistentPolicyAdapter(policy, device)
        self._teacher_logits: Any | None = None

        def cache(_module: Any, _inputs: Any, output: Any) -> None:
            self._teacher_logits = output.detach()

        self._teacher_hook = policy.teacher.register_forward_hook(cache)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, obs: dict[str, Any], reset: Any) -> Any:
        import numpy as np
        import torch
        from torch.distributions import Categorical

        base = self._base
        local, global_, scalars = (
            torch.as_tensor(obs["local"], device=base.device),
            torch.as_tensor(obs["global"], device=base.device),
            torch.as_tensor(obs["scalars"], device=base.device),
        )
        mask = torch.as_tensor(obs["mask"], device=base.device, dtype=torch.bool)
        reset_t = torch.as_tensor(reset, device=base.device, dtype=torch.bool)
        horizon = torch.as_tensor(
            (5000 - np.asarray(obs["frame"], dtype=np.float32)) / 5000,
            device=base.device,
        )
        if base.hidden is None and base.policy.is_recurrent:
            base.hidden = base.policy.initial_hidden(len(local), base.device)
        hidden_before = None if base.hidden is None else base.hidden.detach().clone()
        if not base.records and base.hidden is not None:
            base.rollout_initial_hidden = base.hidden.detach().clone()
        with torch.no_grad():
            logits, values, base.hidden = base.policy(
                local, global_, scalars, base.hidden, reset_t, remaining_horizon=horizon
            )
            logits = logits.masked_fill(~mask, float("-inf"))
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            logp = distribution.log_prob(actions)
            if self._teacher_logits is None:
                raise RuntimeError("optimized adapter did not observe teacher logits")
            if base.device.type == "cuda" and self._teacher_logits.device.type != "cuda":
                raise RuntimeError("A3 requires direct CUDA teacher argmax")
            b8_actions = self._teacher_logits.masked_fill(~mask, float("-inf")).argmax(dim=-1)
        base.records.append(
            {
                "actions": actions.cpu(),
                "logp": logp.cpu(),
                "values": values.cpu(),
                "mask": mask.cpu(),
                "reset": reset_t.cpu(),
                "hidden_before": hidden_before,
                "remaining_horizon": horizon.cpu(),
                "critic_features": base.policy.last_critic_features.cpu(),
                # Direct argmax uses the teacher output cached by policy.forward;
                # no second teacher call is made for this diagnostic field.
                "b8_actions": b8_actions.cpu(),
            }
        )
        return actions.cpu().numpy()

    def take_rollout(self, raw: Any, next_obs: dict[str, Any]) -> Any:
        return self._base.take_rollout(raw, next_obs)


def _make_adapter(policy: Any, device: str, *, optimized: bool) -> Any:
    # Use the real P1 adapter and collector path; no synthetic rollout is used.
    from research.bl_ppo.run import PersistentPolicyAdapter

    return (
        _NoDuplicateTeacherAdapter(policy, device)
        if optimized
        else PersistentPolicyAdapter(policy, device)
    )


class _SerialCollector:
    """The real collector shard without spawned workers, for ``--tiny`` only."""

    def __init__(self, config: Any) -> None:
        import numpy as np

        from research.bl_ppo.collector import _GridShard

        self.config = config
        self.shard = _GridShard(config, 0, np.arange(config.worlds, dtype=np.int64) + 1)

    def __enter__(self) -> "_SerialCollector":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def peek(self) -> dict[str, Any]:
        return self.shard.observe()

    def collect(self, policy: Any, *, steps: int | None = None) -> Any:
        import numpy as np

        from research.bl_ppo.collector import RolloutBatch

        count = self.config.rollout_steps if steps is None else int(steps)
        started = time.perf_counter()
        rows = []
        for _ in range(count):
            observation = self.shard.observe()
            action = policy(observation, observation["reset"])
            rows.append(self.shard.step(action))
        # Match PersistentCollector's public world-major, then time order.
        flat = [
            (world, tick, row, index)
            for tick, row in enumerate(rows)
            for index, world in enumerate(row["world_id"])
        ]
        flat.sort(key=lambda item: (int(item[0]), item[1]))

        def stacked(name: str) -> Any:
            return np.stack([item[2][name][item[3]] for item in flat])

        return RolloutBatch(
            local=stacked("local"),
            global_=stacked("global"),
            scalars=stacked("scalars"),
            mask=stacked("mask"),
            action=np.asarray([x[2]["action"][x[3]] for x in flat], dtype=np.int64),
            reward=np.asarray([x[2]["reward"][x[3]] for x in flat], dtype=np.float32),
            done=np.asarray([x[2]["done"][x[3]] for x in flat], dtype=bool),
            survived_horizon=np.asarray([x[2]["survived_horizon"][x[3]] for x in flat], dtype=bool),
            reset=np.asarray([x[2]["reset"][x[3]] for x in flat], dtype=bool),
            world_id=np.asarray([x[0] for x in flat], dtype=np.int64),
            frame=np.asarray([x[2]["frame"][x[3]] for x in flat], dtype=np.int32),
            length=np.asarray([x[2]["length"][x[3]] for x in flat], dtype=np.int32),
            mix=np.asarray([x[2]["mix"][x[3]] for x in flat], dtype="U8"),
            episode=np.asarray([x[2]["episode"][x[3]] for x in flat], dtype=np.int64),
            active_transitions=len(flat),
            allocated_transitions=self.config.worlds * count,
            wall_seconds=time.perf_counter() - started,
        )


def _make_collector(*, worlds: int, steps: int, tiny: bool, processes: int, seed: int) -> Any:
    from research.bl_ppo.bank import all_seeds, materialize
    from research.bl_ppo.collector import CollectorConfig, PersistentCollector

    config = CollectorConfig(
        worlds=worlds,
        processes=processes,
        rollout_steps=steps,
        horizon=5000,
        roster="frozen-equal-mixes",
        seed=seed,
        excluded_seeds=frozenset(all_seeds(materialize())),
        native_v8=True,
        preflight=tiny,
    )
    return _SerialCollector(config) if tiny else PersistentCollector(config)


def _build_rollout(
    collector: Any, adapter: Any, *, sample_busy: bool = False
) -> tuple[Any, float, float]:
    samples = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            value = _gpu_busy()
            if value is not None:
                samples.append(value)
            stop.wait(0.2)

    thread = threading.Thread(target=sampler, daemon=True) if sample_busy else None
    if thread:
        thread.start()
    start = _clock(str(adapter.device))
    try:
        raw = collector.collect(adapter)
        collection = _clock(str(adapter.device)) - start
        start = _clock(str(adapter.device))
        rollout = adapter.take_rollout(raw, collector.peek())
        staging = _clock(str(adapter.device)) - start
        adapter.a3_raw = raw
    finally:
        stop.set()
        if thread:
            thread.join(timeout=4)
    adapter.a3_collection_gpu_busy = samples
    return rollout, collection, staging


def _one_update(
    learner: Any, rollout: Any, device: str, flattened: bool, *, capture_gradients: bool = False
) -> dict[str, Any]:
    """Run a real actor-enabled PPO update, optionally replacing only its forward."""
    captured: list[dict[str, Any]] = []
    steps = [0]
    original_step = learner.optimizer.step

    def capture_step(*args: Any, **kwargs: Any) -> Any:
        if capture_gradients:
            captured.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in learner.policy.named_parameters()
                    if parameter.grad is not None
                }
            )
        steps[0] += 1
        return original_step(*args, **kwargs)

    learner.optimizer.step = capture_step
    start = _clock(device)
    if flattened:
        original = learner._forward_rollout
        learner._forward_rollout = FlattenedFLearner(learner).forward
        try:
            metrics = learner.update(rollout, actor_enabled=True)
        finally:
            learner._forward_rollout = original
    else:
        metrics = learner.update(rollout, actor_enabled=True)
    elapsed = _clock(device) - start
    learner.optimizer.step = original_step
    return {"seconds": elapsed, "metrics": metrics, "gradients": captured, "adam_steps": steps[0]}


def _gradient_delta(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    import torch

    if len(reference) != len(candidate):
        return {"allclose": False, "reason": "different Adam-step counts"}
    if not reference or not any(reference):
        return {"allclose": False, "reason": "no optimizer gradients"}
    differences, scale, allclose = [], [], True
    for baseline_epoch, optimized_epoch in zip(reference, candidate):
        if set(baseline_epoch) != set(optimized_epoch):
            return {"allclose": False, "reason": "gradient parameter sets differ"}
        for name, grad in baseline_epoch.items():
            other = optimized_epoch[name]
            allclose &= bool(torch.allclose(grad, other, atol=2e-5, rtol=2e-4))
            differences.append((grad - other).abs().reshape(-1))
            scale.append(torch.maximum(grad.abs(), other.abs()).reshape(-1))
    joined, denominator = torch.cat(differences), torch.cat(scale).clamp_min(1e-12)
    return {
        "max_abs": float(joined.max().cpu()),
        "mean_abs": float(joined.mean().cpu()),
        "normalized_elementwise_max": float((joined / denominator).max().cpu()),
        "allclose": allclose,
        "epochs": len(reference),
    }


def _comparison(policy: Any, rollout: Any, device: str) -> dict[str, Any]:
    """Compare baseline and flattened forward from an identical actor-enabled state."""
    from research.bl_ppo.learner import PPOLearner

    baseline_policy, optimized_policy = copy.deepcopy(policy), copy.deepcopy(policy)
    baseline = PPOLearner(baseline_policy, critic_refit_updates=0)
    optimized = PPOLearner(optimized_policy, critic_refit_updates=0)
    baseline_result = _one_update(
        baseline, copy.deepcopy(rollout), device, False, capture_gradients=True
    )
    optimized_result = _one_update(
        optimized, copy.deepcopy(rollout), device, True, capture_gradients=True
    )
    difference = _gradient_delta(baseline_result["gradients"], optimized_result["gradients"])
    tolerance = {"absolute": 2e-5, "relative": 2e-4}
    passed = bool(difference.get("allclose")) and difference.get("epochs") == 2
    return {
        "variant": "optimized_flattened_f_forward_and_duplicate_teacher_removal_candidate",
        "comparison": "identical fixed actor-enabled batch and initial parameters",
        "gradient_equality": "per-parameter gradients captured immediately before every Adam step",
        "difference": difference,
        "tolerance": tolerance,
        "passed": passed,
        "reassociation": (
            "flattened row ordering is time-major; bounded chunk kernels may "
            "reassociate fp32 reductions"
        ),
        "duplicate_teacher": (
            "candidate is separately labelled; production run.py and learner.py "
            "are not modified by A3"
        ),
    }


def _summary(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "median_seconds": statistics.median(rows),
        "p95_seconds": max(rows),
        "samples": float(len(rows)),
    }


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one independent immutable benchmark unit for a wrapper-owned ledger."""
    import torch

    if args.repeats < 3:
        raise ValueError("A3 requires at least three warmed repeats")
    if args.tiny:
        worlds, steps, processes = 1, 3, 1
    else:
        worlds, steps, processes = PRODUCTION_WORLDS, PRODUCTION_STEPS, 12
    if not args.tiny and (
        args.worlds != PRODUCTION_WORLDS or args.rollout_steps != PRODUCTION_STEPS
    ):
        raise ValueError("A3 production shape is frozen at 128 worlds x 128 steps")
    if args.mode not in {"cached", "collection", "cycle", "comparison"}:
        raise ValueError("unknown A3 mode")
    if not torch.cuda.is_available() and not args.tiny:
        raise RuntimeError("production A3 requires CUDA; use --tiny only for an operational smoke")
    device = str(args.device)
    if args.temperature is None and not args.tiny:
        raise ValueError("production A3 requires the frozen P1 calibrated --temperature")
    temperature = 0.7 if args.temperature is None else args.temperature
    torch.manual_seed(args.seed)
    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PPOPolicy

    policy = PPOPolicy("F", args.checkpoint, temperature).to(device).train()
    # Explicitly benchmark-only: this bypasses no P1 qualification and never
    # touches the registered runtime/control plane.
    learner = PPOLearner(policy, critic_refit_updates=0)
    adapter = _make_adapter(policy, device, optimized=args.variant == "optimized")
    from research.bl_ppo.diagnostics import E8Diagnostics

    # Timing the real P1 bookkeeping does not create a fixed-bank quality claim.
    adapter.a3_diagnostics = E8Diagnostics(require_fixed_bank=True)
    samples: list[dict[str, float | None]] = []
    comparison: dict[str, Any] | None = None
    with _make_collector(
        worlds=worlds, steps=steps, tiny=args.tiny, processes=processes, seed=args.seed
    ) as collector:
        warmup, _, _ = _build_rollout(collector, adapter)
        _one_update(learner, warmup, device, False)
        cached_rollout, _, _ = _build_rollout(collector, adapter)
        initial_policy = copy.deepcopy(policy)
        initial_learner_state = copy.deepcopy(learner.learner_state_dict())
        for _ in range(args.repeats):
            timers: dict[str, float | None] = {
                "collection": 0.0,
                "staging_h2d": 0.0,
                "fwd_bwd_adam": 0.0,
                "scoring": 0.0,
                "export": 0.0,
                "gpu_busy_percent": None,
            }
            if args.mode == "cached":
                # A separate clone warms CUDA; each measured repeat starts from
                # identical parameters and the identical prestaged on-policy batch.
                warm_policy = copy.deepcopy(initial_policy)
                _one_update(
                    PPOLearner(warm_policy, critic_refit_updates=0),
                    copy.deepcopy(cached_rollout),
                    device,
                    False,
                )
                measured = PPOLearner(copy.deepcopy(initial_policy), critic_refit_updates=0)
                measured.load_learner_state_dict(copy.deepcopy(initial_learner_state))
                result = _timed_update(
                    measured,
                    copy.deepcopy(cached_rollout),
                    device,
                    args.variant == "optimized",
                    sample_busy=not args.tiny,
                )
                timers["fwd_bwd_adam"] = result["seconds"]
                timers["gpu_busy_percent"] = (
                    statistics.mean(result["gpu_busy_samples"])
                    if result["gpu_busy_samples"]
                    else None
                )
                timers["adam_steps"] = float(result["adam_steps"])
                timers["status"] = (
                    "COMPLETE" if result["adam_steps"] == 2 else "INCOMPLETE(KL skip)"
                )
            elif args.mode == "collection":
                _, collection, staging = _build_rollout(
                    collector, adapter, sample_busy=not args.tiny
                )
                timers["collection"], timers["staging_h2d"] = collection, staging
                busy = adapter.a3_collection_gpu_busy
                timers["gpu_busy_percent"] = statistics.mean(busy) if busy else None
            else:
                rollout, collection, staging = _build_rollout(
                    collector, adapter, sample_busy=not args.tiny
                )
                timers["collection"], timers["staging_h2d"] = collection, staging
                if args.mode == "comparison" and comparison is None:
                    # The collected log-probabilities match this pre-update policy.
                    comparison = _comparison(policy, rollout, device)
                result = _timed_update(
                    learner,
                    rollout,
                    device,
                    args.variant == "optimized" or args.mode == "comparison",
                    sample_busy=not args.tiny,
                )
                timers["fwd_bwd_adam"] = result["seconds"]
                timers["complete_cycle"] = collection + staging + result["seconds"]
                timers["gpu_busy_percent"] = (
                    statistics.mean(adapter.a3_collection_gpu_busy + result["gpu_busy_samples"])
                    if adapter.a3_collection_gpu_busy + result["gpu_busy_samples"]
                    else None
                )
                timers["adam_steps"] = float(result["adam_steps"])
                timers["status"] = (
                    "COMPLETE" if result["adam_steps"] == 2 else "INCOMPLETE(KL skip)"
                )
            started = _clock(device)
            if args.mode != "cached":
                adapter.action_windows.observe(adapter.last_actions, adapter.last_b8_actions)
                adapter.a3_diagnostics.observe(
                    adapter.a3_raw,
                    actions=adapter.last_actions.transpose(0, 1).reshape(-1),
                    b8_actions=adapter.last_b8_actions.transpose(0, 1).reshape(-1),
                )
            timers["scoring"] = _clock(device) - started
            if "complete_cycle" in timers:
                timers["complete_cycle"] += timers["scoring"]
            samples.append(timers)
    learning = _summary([float(row["fwd_bwd_adam"] or 0.0) for row in samples])
    incomplete = any(row.get("status") == "INCOMPLETE(KL skip)" for row in samples)
    receipt: dict[str, Any] = {
        "schema": "a3-p1-learner-ceiling/v1",
        "unit_id": args.unit_id,
        "scientific": False,
        "purpose": "WP5 operational ceiling only; not P1 qualification or a recipe change",
        "shape": {"worlds": worlds, "rollout_steps": steps, "transitions": worlds * steps},
        "arm": "F",
        "actor_enabled": True,
        "critic_refit_updates": 0,
        "device": device,
        "mode": args.mode,
        "variant": args.variant,
        "warmup": "one real collector+actor PPO cycle before measured repeats",
        "phase_timers_seconds": samples,
        "cached_learning": learning if args.mode == "cached" else None,
        "status": "INCOMPLETE(KL skip)" if incomplete else "COMPLETE",
        "kill": {
            "cached_actor_enabled_seconds_gt": KILL_SECONDS,
            "status": (
                "PAUSE_WP5"
                if args.mode == "cached" and learning["median_seconds"] > KILL_SECONDS
                else "NOT_TRIGGERED"
            ),
        },
        "comparison": comparison,
        "memory": (
            "F flattening bounds workspace at 1024 rows, while retaining the full autograd graph; "
            "OOM is INCOMPLETE, never a reason to reduce the registered 128x128 shape."
        ),
    }
    receipt["export_timing_path"] = "export-timing.json"
    started = _clock(device)
    export = export_json(args.output / f"unit-{args.unit_id}.json", receipt)
    elapsed = _clock(device) - started
    export_json(args.output / "export-timing.json", {"unit": export, "export_seconds": elapsed})
    receipt["export"] = export
    receipt["export_seconds"] = elapsed
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument(
        "--mode", choices=("cached", "collection", "cycle", "comparison"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="baseline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_unit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
