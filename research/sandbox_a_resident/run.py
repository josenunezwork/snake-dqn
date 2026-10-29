"""Finite same-allocation timing assay; exports timing/parity only, no efficacy."""

from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import os
import platform
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.core import Recipe, ResidentAssay, distribution, gae
from research.sandbox_a_resident.proof import prove
from research.sandbox_a_resident.units import (
    allocation,
    cell_specs,
    digest,
    load_ledger,
    publish,
    unit_id,
    unit_stem,
)
from src.model.ego2s_network import load_ego2s_checkpoint


def sync(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def numerical_settings() -> dict[str, bool]:
    """Return the Torch switches that materially affect the timing/proof claim."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


@contextmanager
def numerical_mode(*, deterministic: bool, tf32: bool):
    """Apply proof or timing numerics and restore process-global Torch state exactly."""
    before = numerical_settings()
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        yield numerical_settings()
    finally:
        torch.use_deterministic_algorithms(
            before["deterministic_algorithms"],
            warn_only=before["deterministic_algorithms_warn_only"],
        )
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]


def cycle_stability(cycles: list[dict[str, float]], rows_per_cycle: int) -> dict[str, float]:
    """Report excluded-warmup cycle-rate stability without concealing slow tail cycles."""
    rates = [rows_per_cycle / row["seconds"] for row in cycles]
    mean = sum(rates) / len(rates)
    variance = sum((rate - mean) ** 2 for rate in rates) / len(rates)
    return {
        "rows_per_second_mean": mean,
        "rows_per_second_min": min(rates),
        "rows_per_second_max": max(rates),
        "rows_per_second_cv": variance**0.5 / mean if mean else 0.0,
        "first_to_last_rows_per_second_ratio": rates[-1] / rates[0] if rates[0] else 0.0,
    }


def profile_snapshot(runner: ResidentAssay) -> dict[str, Any]:
    """Copy the runtime's device-side phase profile without changing its timing path."""
    profile = getattr(runner, "phase_profile", {})
    if callable(profile):
        profile = profile()
    if not isinstance(profile, dict):
        return {}
    return dict(profile)


def aggregate_phase_profiles(
    cycles: list[dict[str, Any]], worlds: int, horizon: int
) -> dict[str, Any]:
    """Summarize measured collection phase timing with an equivalent hero-row rate."""
    profiles = [cycle["phase_profile"] for cycle in cycles if cycle.get("phase_profile")]
    if not profiles:
        return {}
    grouped: dict[str, list[float]] = {}
    tick_profiles: dict[str, list[float]] = {}
    for profile in profiles:
        for name, milliseconds in profile.get("milliseconds", {}).items():
            grouped.setdefault(name, []).append(float(milliseconds))
        for name, milliseconds in profile.get("ms_per_tick", {}).items():
            tick_profiles.setdefault(name, []).append(float(milliseconds))
    milliseconds = {name: sum(values) / len(values) for name, values in grouped.items()}
    return {
        "mode": profiles[-1].get("mode", "unknown"),
        "cycles": len(profiles),
        "ticks_per_cycle": horizon,
        "milliseconds": milliseconds,
        "ms_per_tick": {name: sum(values) / len(values) for name, values in tick_profiles.items()},
        "hero_rows_per_second": {
            name: worlds * horizon * 1000 / value
            for name, value in milliseconds.items()
            if value > 0
        },
    }


def compile_snapshot(
    runner: ResidentAssay, compile_mode: str, phase: str | None = None
) -> dict[str, Any]:
    """Record requested compilation and every subgraph's eager fallback outcome."""
    report = getattr(runner, "compile_report", None)
    if isinstance(report, dict) and report:
        snapshot = copy.deepcopy(report)
        unused = (
            ("ActorCritic_collect", "v8_decide", "build_observation", "anchors")
            if phase == "cached"
            else ("ActorCritic_train",) if phase == "collection" else ()
        )
        for name, entry in snapshot.items():
            if not isinstance(entry, dict) or "status" not in entry:
                continue
            entry["counter_scope"] = "run_cache_cumulative"
            if name in unused:
                entry["status"] = "not_applicable"
                entry["counter_scope"] = "not_applicable"
                for field in (
                    "compile_factory_seconds",
                    "compile_specialization_seconds",
                    "compile_specialization_calls",
                    "validation_seconds",
                    "validation_calls",
                    "output_check",
                ):
                    entry.pop(field, None)
        return snapshot
    return {
        "requested_mode": compile_mode,
        "status": "eager" if compile_mode == "none" else "unreported",
        "subgraphs": {},
    }


class CellCompileCache:
    """Reuse tensor graphs and their parameter owner across independent timing cells.

    Only compiled code/model storage is reused. Each cell keeps its fresh simulator,
    RNG, rollout, model initialization and optimizer state. No assay is retained.
    """

    attributes = ("_forward_collect", "_forward_train", "_v8", "_observation", "_anchors")

    def __init__(self) -> None:
        self.entries: dict[tuple, tuple[Any, dict, tuple]] = {}

    def attach(self, runner: ResidentAssay, mode: str, *, backend: str | None = None) -> bool:
        sim = runner.sim
        key = (mode, backend or "inductor", str(runner.device), sim.cap, sim.grid_h, sim.grid_w)
        cached = self.entries.get(key)
        if cached is None:
            if backend is None:
                runner.enable_compile(mode)
            else:
                runner.enable_compile(mode, backend=backend)
            self.entries[key] = (
                runner.actor,
                runner.compile_report,
                tuple(getattr(runner, name) for name in self.attributes),
            )
            return False
        actor, report, blocks = cached
        actor.load_state_dict(runner.actor.state_dict())
        runner.actor = actor
        runner.optimizer = torch.optim.Adam(actor.parameters(), lr=runner.recipe.learning_rate)
        runner.compile_report, runner.compile_mode = report, mode
        for name, block in zip(self.attributes, blocks, strict=True):
            setattr(runner, name, block)
        return True


def compiler_metrics() -> dict[str, tuple[float, int]]:
    """Torch's compiler timers include lazy backward and TF32 specializations.

    Timer names are emitted verbatim; nested timers must never be added together.
    They are diagnostics alongside wall time, not a subtraction from throughput.
    """
    from torch._dynamo.utils import compilation_time_metrics

    return {name: (sum(values), len(values)) for name, values in compilation_time_metrics.items()}


def compiler_delta(before: dict, after: dict) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "seconds": seconds - before.get(name, (0, 0))[0],
            "calls": calls - before.get(name, (0, 0))[1],
        }
        for name, (seconds, calls) in after.items()
        if calls > before.get(name, (0, 0))[1]
    }


def synthetic_observations(recipe: Recipe, device: torch.device) -> tuple:
    """Reconstruct frozen cached observations without saving gigabytes of random bytes."""
    n = recipe.worlds * recipe.horizon
    obs_gen = torch.Generator(device=device).manual_seed(74031)
    obs = (
        torch.randint(0, 256, (n, 7, 31, 31), dtype=torch.uint8, device=device, generator=obs_gen),
        torch.randint(0, 256, (n, 4, 37, 37), dtype=torch.uint8, device=device, generator=obs_gen),
        torch.rand((n, 60), device=device, generator=obs_gen),
    )
    shape = (recipe.horizon, recipe.worlds)
    return tuple(x.reshape(*shape, *x.shape[1:]) for x in obs)


@torch.no_grad()
def synthetic(runner: ResidentAssay) -> dict[str, Any]:
    """Frozen synthetic on-device learner tensors; no collection throughput claim."""
    recipe, device, gen = runner.recipe, runner.device, runner.generator
    n = recipe.worlds * recipe.horizon
    obs = tuple(x.flatten(0, 1) for x in synthetic_observations(recipe, device))
    mask = torch.ones((n, 6), dtype=torch.bool, device=device)
    values, logps, actions = [], [], []
    for start in range(0, n, recipe.minibatch):
        sl = slice(start, start + recipe.minibatch)
        logits, value = runner.actor(*(x[sl] for x in obs))
        dist = distribution(logits, mask[sl])
        action = torch.multinomial(dist.probs, 1, generator=gen).squeeze(1)
        values.append(value)
        logps.append(dist.log_prob(action))
        actions.append(action)
    shape = (recipe.horizon, recipe.worlds)
    value = torch.cat(values).reshape(shape)
    reward = torch.randn(shape, device=device, generator=gen)
    terminal = torch.zeros(shape, dtype=torch.bool, device=device)
    next_value = torch.cat((value[1:], torch.zeros_like(value[:1])))
    advantage, returns = gae(
        reward, value, next_value, terminal, terminal, recipe.gamma, recipe.gae_lambda
    )
    return {
        "obs": tuple(x.reshape(*shape, *x.shape[1:]) for x in obs),
        "mask": mask.reshape(*shape, 6),
        "value": value,
        "logp": torch.cat(logps).reshape(shape),
        "action": torch.cat(actions).reshape(shape),
        "advantage": advantage,
        "returns": returns,
    }


def package_unit(
    path: Path, runner: ResidentAssay, report: dict[str, Any], cached: dict | None
) -> None:
    """Snapshot includes cached tensors when they are the phase's future input."""
    with tempfile.TemporaryDirectory(prefix="sandbox-a-export-") as directory:
        checkpoint = Path(directory) / "state.pt"
        torch.save(
            {
                "state": runner.snapshot(),
                "phase": report["phase"],
                "completed_cycles": len(report["cycles"]),
                "cached": (
                    None if cached is None else {k: v for k, v in cached.items() if k != "obs"}
                ),
                "cached_observation_seed": 74031 if cached is not None else None,
            },
            checkpoint,
        )
        report_bytes = json.dumps(report, sort_keys=True).encode()
        with tarfile.open(path, "w") as archive:
            archive.add(checkpoint, arcname="state.pt")
            info = tarfile.TarInfo("timing.json")
            info.size = len(report_bytes)
            archive.addfile(info, io.BytesIO(report_bytes))


def restore_unit_checkpoint(path: Path, runner: ResidentAssay) -> dict | None:
    """Restore the immutable phase boundary and reconstruct exact frozen learner inputs."""
    with tarfile.open(path, "r") as archive:
        stream = archive.extractfile("state.pt")
        if stream is None:
            raise ValueError("checkpoint payload absent")
        payload = torch.load(io.BytesIO(stream.read()), weights_only=False)
    runner.restore(payload["state"])
    cached = payload["cached"]
    if cached is not None:
        if payload.get("cached_observation_seed") != 74031:
            raise ValueError("unknown cached observation generator")
        cached["obs"] = synthetic_observations(runner.recipe, runner.device)
    return cached


def run_cell(
    control: dict,
    worlds: int,
    phase: str,
    offset: int,
    checkpoint: Path,
    output: Path,
    device: str,
    deadline: float,
    episode_limit: int | None = None,
    compile_cache: CellCompileCache | None = None,
) -> dict:
    begin = sync(device)
    compiler_before = compiler_metrics()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    recipe = Recipe(
        worlds=worlds,
        horizon=control["horizon"],
        minibatch=control["minibatch"],
        epochs=control["epochs"],
        episode_limit=control["episode_limit"] if episode_limit is None else episode_limit,
        seed_offset=offset,
        max_episodes=1 + (control["cycles"] + control["warmups"]) * control["horizon"],
        tiny=control.get("tiny_arena", control["tiny"]),
        **control.get("recipe_parameters", {}),
    )
    runner = ResidentAssay(recipe, load_ego2s_checkpoint(str(checkpoint), device), device)
    compile_mode = control.get("compile_mode", "none")
    if compile_mode not in ("none", "default"):
        raise ValueError("compile_mode must be none or default")
    compile_cache_reused = False
    compile_bucket_warmup: list[int] = []
    if compile_mode != "none":
        if runner.device.type != "cuda":
            raise ValueError("compiled timing cell requires CUDA")
        compile_cache_reused = (compile_cache or CellCompileCache()).attach(runner, compile_mode)
        if phase != "cached":
            compile_bucket_warmup = runner.warm_compile_buckets()
    cached = synthetic(runner) if phase == "cached" else None
    setup_seconds = sync(device) - begin
    # Excluded warmup cycles are charged to total cell time, not steady-state rate.
    for _ in range(control["warmups"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("runaway deadline during warmup")
        data = cached if phase == "cached" else runner.collect()
        if phase != "collection":
            runner.update(data)
    warmup_end = sync(device)
    compiler_warm = compiler_metrics()
    cycles = []
    boundary_counts = torch.zeros((control["cycles"], 2), dtype=torch.long, device=runner.device)
    for index in range(control["cycles"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("runaway deadline during fixed cycles")
        started = sync(device)
        compiler_cycle = compiler_metrics()
        data = cached if phase == "cached" else runner.collect()
        phase_profile = {} if phase == "cached" else profile_snapshot(runner)
        if phase != "cached":
            boundary_counts[index].copy_(
                torch.stack((data["terminal"].sum(), data["truncated"].sum()))
            )
        collected = sync(device)
        if phase != "collection":
            runner.update(data)
        ended = sync(device)
        cycles.append(
            {
                "cycle": index,
                "hero_rows": worlds * recipe.horizon,
                "collection_gae_resets_seconds": collected - started,
                "learner_seconds": ended - collected,
                "seconds": ended - started,
                "hero_rows_per_second": worlds * recipe.horizon / (ended - started),
                "phase_profile": phase_profile,
                "compiler_activity": compiler_delta(compiler_cycle, compiler_metrics()),
            }
        )
        print(
            json.dumps(
                {
                    "progress": unit_id(worlds, phase, episode_limit),
                    "done": index + 1,
                    "total": control["cycles"],
                    "seconds": ended - started,
                }
            ),
            flush=True,
        )
    rows_per_cycle = worlds * recipe.horizon
    rows = rows_per_cycle * control["cycles"]
    elapsed = sum(row["seconds"] for row in cycles)
    # Transfer counts only after every timed cycle. No extra per-cycle host sync.
    counts = boundary_counts.cpu().tolist()
    for row, (terminal, truncated) in zip(cycles, counts):
        row["terminal_rows"] = terminal
        row["truncated_rows"] = truncated
    terminal_rows = sum(row[0] for row in counts)
    truncated_rows = sum(row[1] for row in counts)
    invalid_cycles = (
        [row["cycle"] for row in cycles if not row["truncated_rows"]]
        if not control["tiny"] and phase != "cached"
        else []
    )
    invalid_reasons = [
        f"measured cycle {index} did not exercise time-limit bootstrap" for index in invalid_cycles
    ]
    invalid_reasons.extend(
        f"measured cycle {row['cycle']} included compiler work; steady-state timing invalid"
        for row in cycles
        if row["compiler_activity"]
    )
    report = {
        "schema": "sandbox-a-timing/v3",
        "status": "INVALID" if invalid_reasons else "PASS",
        "invalid_reasons": invalid_reasons,
        "label": "simulator-D",
        "id": unit_id(worlds, phase, episode_limit),
        "allocation": allocation(control),
        "recipe": asdict(recipe),
        "phase": phase,
        "cycles": cycles,
        "hero_rows": rows,
        "cached_rows": rows if phase == "cached" else 0,
        "useful_hero_rows": 0 if phase == "cached" else rows,
        "steady_rows_per_second": rows / elapsed,
        "steady_useful_hero_rows_per_second": None if phase == "cached" else rows / elapsed,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_end - begin - setup_seconds,
        "scope": "one hero plus five greedy_food anchors; not P1 recipe equivalence",
        "host_tensor_staging_in_cycle": False,
        "reset_count": int(runner.reset_count),
        "measured_terminal_rows": terminal_rows,
        "measured_truncated_rows": truncated_rows,
        "cycle_stability": cycle_stability(cycles, rows_per_cycle),
        "numerical_settings": numerical_settings(),
        "compile_mode": compile_mode,
        "compile_report": compile_snapshot(runner, compile_mode, phase),
        "compile_cache_reused": compile_cache_reused,
        "compile_bucket_warmup": compile_bucket_warmup,
        "compiler_setup_warmup": compiler_delta(compiler_before, compiler_warm),
        "compiler_timing_note": (
            "Torch named timers overlap; setup/warmup excluded from steady-state; "
            "measured compilation invalidates timing"
        ),
        "phase_profile": aggregate_phase_profiles(cycles, worlds, recipe.horizon),
        "efficacy_selection": False,
        "peak_device_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        ),
    }
    stem = unit_stem(worlds, phase, episode_limit)
    path = output / f"{stem}.tar"
    export_start = sync(device)
    descriptor = publish(
        path,
        lambda target, active=runner, frozen=cached: package_unit(target, active, report, frozen),
    )
    export_seconds = sync(device) - export_start
    # Transfer/cleanup are unknowable to the remote process. Mac aggregator adds them.
    total_seconds = sync(device) - begin
    result = {
        **report,
        "export_seconds": export_seconds,
        "cell_seconds_including_setup_warmup_export": total_seconds,
        "rows_per_second_with_export": rows / (elapsed + export_seconds),
        "useful_hero_rows_per_second_with_export": (
            None if phase == "cached" else rows / (elapsed + export_seconds)
        ),
        "fully_charged_rows_per_second": rows / total_seconds,
        "payload": descriptor,
    }
    publish(
        output / f"{stem}.json",
        lambda target: target.write_text(json.dumps(result, indent=2) + "\n"),
    )
    del runner, cached, data
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    if "checkpoint_sha256" in control and digest(args.checkpoint) != control["checkpoint_sha256"]:
        raise ValueError("frozen input checkpoint differs")
    if (
        "source_commit" in control
        and os.environ.get("SANDBOX_A_SOURCE_COMMIT") != control["source_commit"]
    ):
        raise ValueError("frozen allocation source commit differs")
    if args.device == "cpu" and not control["tiny"]:
        raise ValueError("Mac permits tiny preparation only")
    if args.device == "cuda" and (
        not torch.cuda.is_available()
        or not any(gpu in torch.cuda.get_device_name() for gpu in ("H100", "A100"))
    ):
        raise ValueError("H100 or A100 required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3403)
    verified = load_ledger(args.ledger, control)
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + control["timeout_seconds"]
    results, skipped = [], []
    start = time.perf_counter()
    # prove owns its strict deterministic/TF32-off mode and restores this process afterwards.
    proof = prove(
        load_ego2s_checkpoint(str(args.checkpoint), args.device),
        args.device,
        control.get("parity_seed_offset", 90_000_000),
    )
    proof_settings = proof["numerical_settings"]
    publish(args.output / "parity.json", lambda target: target.write_text(json.dumps(proof) + "\n"))
    # Four world shapes and the finite prewarmed R2 buckets share pure graph
    # owners. Timing forward, backward and validation keep the same TF32 mode.
    # This scoped allowance covers the fixed matrix, never measured recompiles.
    with (
        torch._dynamo.config.patch(recompile_limit=64, accumulated_recompile_limit=256),
        numerical_mode(deterministic=False, tf32=True) as timing_settings,
    ):
        compile_cache = CellCompileCache()
        for worlds, phase, episode_limit, offset in cell_specs(control):
            identity = unit_id(worlds, phase, episode_limit)
            if identity in verified:
                skipped.append(identity)
                continue
            results.append(
                run_cell(
                    control,
                    worlds,
                    phase,
                    offset,
                    args.checkpoint,
                    args.output,
                    args.device,
                    deadline,
                    episode_limit,
                    compile_cache,
                )
            )
    summary = {
        "schema": "sandbox-a-run/v3",
        "status": "INVALID" if any(r["status"] == "INVALID" for r in results) else "PASS",
        "label": "simulator-D",
        "allocation": allocation(control),
        "results": results,
        "skipped_verified": skipped,
        "seconds": time.perf_counter() - start,
        "device": args.device,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "checkpoint_sha256": digest(args.checkpoint),
        "cuda": torch.version.cuda,
        "hardware": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        "source_commit": os.environ.get("SANDBOX_A_SOURCE_COMMIT", "local-unpinned-smoke"),
        "parity": proof,
        "proof_numerical_settings": proof_settings,
        "timing_numerical_settings": timing_settings,
        "compile_mode": control.get("compile_mode", "none"),
        "compiler_recompile_limit": 64,
        "numerical_outputs": "timing/parity only; no efficacy or candidate selection",
    }
    publish(
        args.output / "summary.json",
        lambda target: target.write_text(json.dumps(summary, indent=2) + "\n"),
    )
    print(
        f"{summary['status']}_SANDBOX_A units={len(results)} skipped={len(skipped)} "
        "allocations=0_by_entrypoint",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
