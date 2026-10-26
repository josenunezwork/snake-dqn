"""Finite same-allocation timing assay; exports timing/parity only, no efficacy."""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import platform
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.core import Recipe, ResidentAssay, distribution, gae
from research.sandbox_a_resident.proof import prove
from research.sandbox_a_resident.units import (
    PHASES,
    allocation,
    digest,
    load_ledger,
    publish,
    unit_id,
)
from src.model.ego2s_network import load_ego2s_checkpoint


def sync(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


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
) -> dict:
    begin = sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    recipe = Recipe(
        worlds=worlds,
        horizon=control["horizon"],
        minibatch=control["minibatch"],
        epochs=control["epochs"],
        episode_limit=control["episode_limit"],
        seed_offset=offset,
        max_episodes=1 + (control["cycles"] + control["warmups"]) * control["horizon"],
        tiny=control.get("tiny_arena", control["tiny"]),
        **control.get("recipe_parameters", {}),
    )
    runner = ResidentAssay(recipe, load_ego2s_checkpoint(str(checkpoint), device), device)
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
    cycles = []
    for index in range(control["cycles"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("runaway deadline during fixed cycles")
        started = sync(device)
        data = cached if phase == "cached" else runner.collect()
        collected = sync(device)
        if phase != "collection":
            runner.update(data)
        ended = sync(device)
        cycles.append(
            {
                "cycle": index,
                "collection_gae_resets_seconds": collected - started,
                "learner_seconds": ended - collected,
                "seconds": ended - started,
            }
        )
        print(
            json.dumps(
                {
                    "progress": unit_id(worlds, phase),
                    "done": index + 1,
                    "total": control["cycles"],
                    "seconds": ended - started,
                }
            ),
            flush=True,
        )
    rows = worlds * recipe.horizon * control["cycles"]
    elapsed = sum(row["seconds"] for row in cycles)
    report = {
        "schema": "sandbox-a-timing/v1",
        "label": "simulator-D",
        "id": unit_id(worlds, phase),
        "allocation": allocation(control),
        "recipe": asdict(recipe),
        "phase": phase,
        "cycles": cycles,
        "hero_rows": rows,
        "steady_rows_per_second": rows / elapsed,
        "steady_useful_hero_rows_per_second": None if phase == "cached" else rows / elapsed,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_end - begin - setup_seconds,
        "scope": "one hero plus five greedy_food anchors; not P1 recipe equivalence",
        "host_tensor_staging_in_cycle": False,
        "reset_count": int(runner.reset_count),
        "efficacy_selection": False,
        "peak_device_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        ),
    }
    path = output / f"e{worlds}-{phase}.tar"
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
        output / f"e{worlds}-{phase}.json",
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
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("A100 required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3403)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verified = load_ledger(args.ledger, control)
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + control["timeout_seconds"]
    results, skipped = [], []
    start = time.perf_counter()
    proof = prove(load_ego2s_checkpoint(str(args.checkpoint), args.device), args.device)
    publish(args.output / "parity.json", lambda target: target.write_text(json.dumps(proof) + "\n"))
    for shape_index, worlds in enumerate(control["worlds"]):
        for phase_index, phase in enumerate(PHASES):
            identity = unit_id(worlds, phase)
            if identity in verified:
                skipped.append(identity)
                continue
            offset = (shape_index * len(PHASES) + phase_index) * 1_000_000
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
                )
            )
    summary = {
        "schema": "sandbox-a-run/v1",
        "status": "PASS",
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
        "numerical_outputs": "timing/parity only; no efficacy or candidate selection",
    }
    publish(
        args.output / "summary.json",
        lambda target: target.write_text(json.dumps(summary, indent=2) + "\n"),
    )
    print(
        f"PASS_SANDBOX_A units={len(results)} skipped={len(skipped)} allocations=0_by_entrypoint",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
