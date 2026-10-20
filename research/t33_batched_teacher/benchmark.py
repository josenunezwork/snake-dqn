"""Bounded resident-world throughput probe; exports no checkpoints or labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numba
import numpy as np
import psutil
import torch

from research.phase0_search_20261019 import bank
from research.phase0_search_20261019 import teacher as phase0
from research.phase0_search_20261019.preflight import fixture
from research.t33_batched_teacher.ablation import B8, B8_SHA, calibration_rows, sha
from research.t33_batched_teacher.evidence import identity
from research.t33_batched_teacher.inference import FixedInference
from research.t33_batched_teacher.qualify_inference import collect, qualify
from research.t33_batched_teacher.teacher import BatchedTeacher, SearchConfig


def complete(path: Path, value: dict) -> None:
    """Create the supervisor's immutable payload/hash/completion export triple."""
    json.dumps(value, allow_nan=False)
    bank.write_create(path, value)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(sha(path) + "\n")
    with path.with_name(path.name + ".complete").open("x") as handle:
        handle.write("complete\n")


def memory_admission(
    source: Any,
    worlds: int,
    rollouts: int,
    device: str,
    host_available: int,
    gpu_available: int | None,
    *,
    horizon: int = 128,
    ring_margin: int = 1,
) -> dict:
    """Conservative peak estimate before creating any expanded batch.

    Counting views twice is intentional headroom. Python food/RNG objects,
    clone peaks, staging, and fixed CNN activations are additional allowances.
    Admission is a planning bound, never proof that an allocator will succeed.
    """
    if worlds < 1 or rollouts < 1:
        raise ValueError("world and rollout counts must be positive")

    def unique_backing_bytes(owner: Any) -> int:
        """Count each ndarray allocation once; padded interior views alias it."""
        allocations: dict[int, int] = {}
        for value in vars(owner).values():
            if not isinstance(value, np.ndarray) or not value.ndim or value.shape[0] != 1:
                continue
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            allocations[id(root)] = max(allocations.get(id(root), 0), int(root.nbytes))
        return sum(allocations.values())

    source_sim_arrays = unique_backing_bytes(source.sim)
    sim_arrays = source_sim_arrays
    # merge_worlds re-caps the int64 body rings before expansion.  Model the
    # resulting resident row rather than the source's strict-H5000 capacity.
    old_bodies = getattr(source.sim, "bodies", None)
    seg_count = getattr(source.sim, "seg_count", None)
    recap_capacity = None
    if isinstance(old_bodies, np.ndarray) and isinstance(seg_count, np.ndarray):
        recap_capacity = int(np.max(seg_count)) + horizon + ring_margin
        recapped_bodies = (
            int(old_bodies.shape[1]) * recap_capacity * 2 * np.dtype(np.int64).itemsize
        )
        sim_arrays += recapped_bodies - old_bodies.nbytes
    runtime_arrays = unique_backing_bytes(source.runtime)
    branches = worlds * 6 * rollouts
    resident_bytes_per_row = sim_arrays + runtime_arrays + 65536 + 12444
    # Assembly owns a resident output plus a same-sized explicit clone headroom;
    # source-world clones peak separately before the gathered batch is complete.
    clone_headroom = branches * sim_arrays
    source_clone_headroom = worlds * source_sim_arrays
    host = branches * resident_bytes_per_row + clone_headroom + source_clone_headroom
    # fp16 CNN workspaces/activations plus persistent inputs. Large resident
    # shapes can exceed a 24GB L4 even when host RAM admits their simulator.
    gpu = branches * (512 * 1024 + 12444) + 256 * 1024**2 if device == "cuda" else 0
    allowed = host <= host_available and (gpu_available is None or gpu <= gpu_available)
    return {
        "requested_source_worlds": worlds,
        "resident_source_worlds": worlds,
        "branch_worlds": branches,
        "model_batch_rows": branches,
        "chunk_count": 1,
        "estimated_host_peak_bytes": host,
        "resident_bytes_per_row": resident_bytes_per_row,
        "clone_headroom_bytes": clone_headroom,
        "source_clone_headroom_bytes": source_clone_headroom,
        "row_count": branches,
        "recap_capacity": recap_capacity,
        "host_admission_cap_bytes": host_available,
        "estimated_gpu_peak_bytes": gpu,
        "gpu_activation_bytes_per_row": 512 * 1024 if device == "cuda" else 0,
        "gpu_admission_cap_bytes": gpu_available,
        "admitted": allowed,
        "status": "admitted" if allowed else "skipped_memory_admission",
    }


def utilization() -> float | None:
    """Take an optional bounded GPU snapshot; event timing decides the binder."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return float(result.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def prepared(world: Any) -> Any:
    """Capture an exact prepared seam without advancing a benchmark template."""

    class Ready(Exception):
        pass

    def select(sim: Any) -> np.ndarray:
        world.runtime.prepare(sim)
        raise Ready

    try:
        world.sim.step_with_policy(select)
    except Ready:
        # The selector has observed the fully prepared lifecycle seam. The
        # simulator sets this flag only after a successful selector return;
        # this intentional capture abort must mark the same prepared boundary.
        world.sim._policy_phase_prepared = True
    return world


class TimedTeacher(BatchedTeacher):
    """Cooperative per-forward deadline; the external supervisor also caps time."""

    deadline: float = float("inf")
    heartbeat: float = 0.0

    def _q(self, sim: Any, rows: np.ndarray, net: Any, **kwargs: Any) -> np.ndarray:
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("benchmark cell reached its wall cap")
        if time.perf_counter() - self.heartbeat >= 110:
            print(
                json.dumps(
                    {
                        "event": "benchmark_progress",
                        "resident_rows": len(rows),
                        "seconds_remaining": max(0.0, self.deadline - time.perf_counter()),
                    }
                ),
                flush=True,
            )
            self.heartbeat = time.perf_counter()
        return super()._q(sim, rows, net, **kwargs)


def run_cell(
    templates: list[Any],
    config: SearchConfig,
    worlds: int,
    device: str,
    seconds: float,
    iterations: int,
) -> dict:
    """Measure one actual resident shape; no microbatch substitution."""
    sources = [templates[i % len(templates)].clone() for i in range(worlds)]
    adapter = FixedInference(sources[0].hero.net, device)
    search = TimedTeacher(
        config, device=device, common_random=True, inference=adapter, opponent_inference=adapter
    )
    # JIT/compile and fixed-shape warmup excluded from the timed interval.
    search.deadline = time.perf_counter() + max(120.0, seconds)
    search.choose(sources)
    search.profile(reset=True)
    adapter.reset_profile()
    started = time.perf_counter()
    search.deadline = started + seconds
    done = 0
    rows = []
    termination = "iteration_or_wall_limit"
    phases, inference = search.profile(), adapter.profile()
    while time.perf_counter() < search.deadline and (iterations == 0 or done < iterations):
        before = time.perf_counter()
        try:
            actions, diagnostics = search.choose(sources)
        except TimeoutError:
            termination = "wall_cap_partial_iteration_excluded"
            break
        elapsed = time.perf_counter() - before
        done += 1
        phases, inference = search.profile(), adapter.profile()
        rows.append(
            {
                "iteration": done,
                "seconds": elapsed,
                "searched_decisions": sum(d["searched"] for d in diagnostics),
                "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "benchmark_iteration",
                    "worlds": worlds,
                    "done": done,
                    "elapsed_s": time.perf_counter() - started,
                    "remaining_s": max(0.0, search.deadline - time.perf_counter()),
                    "decisions_per_s": worlds / elapsed,
                }
            ),
            flush=True,
        )
    wall = time.perf_counter() - started
    # All CUDA events are synchronized by inference.q/vector_q before returning.
    gpu = inference["gpu_s"] + inference["vector_gpu_s"]
    h2d = inference["h2d_s"] + inference["vector_h2d_s"]
    steps = max(1, phases["steps"])
    measured = sum(r["seconds"] for r in rows)
    host = max(0.0, measured - gpu - h2d)
    decisions = sum(r["searched_decisions"] for r in rows)
    result = {
        "status": "complete" if done else "no_complete_iterations",
        "termination": termination,
        "iterations": done,
        "wall_s": wall,
        "completed_iteration_wall_s": measured,
        "excluded_wall_s": max(0.0, wall - measured),
        "searched_decisions": decisions,
        "decisions_per_s": decisions / measured if measured else 0.0,
        "host_ms_per_step": 1000 * host / steps,
        "h2d_ms_per_step": 1000 * h2d / steps,
        "gpu_ms_per_step": 1000 * gpu / steps,
        "host_fraction": host / measured if measured else 0.0,
        "gpu_fraction": gpu / measured if measured else 0.0,
        "h2d_fraction": h2d / measured if measured else 0.0,
        "gpu_utilization_snapshot_percent": utilization() if device == "cuda" else None,
        "phases": phases,
        "stage_us_per_resident_row_step": {
            k: 1e6
            * v
            / max(1, phases.get("resident_row_steps", steps * worlds * 6 * config.rollouts))
            for k, v in phases.items()
            if k.endswith("_s")
        },
        "stage_us_per_active_row_step": {
            k: 1e6
            * v
            / max(1, phases.get("active_row_steps", steps * worlds * 6 * config.rollouts))
            for k, v in phases.items()
            if k.endswith("_s")
        },
        "inference": inference,
        "rows": rows,
        "fixture": "repeated three disjoint L800 timing snapshots, not independent labels",
    }
    del search, adapter, sources
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def recommendation(cells: list[dict], device: str) -> str:
    """Record the spec's follow-up rules only for complete measured CUDA cells."""
    valid = [r for r in cells if r.get("status") == "complete" and r.get("iterations", 0) > 0]
    if device != "cuda" or not valid:
        return "UNVERIFIED: no complete CUDA throughput cell"
    # Use the fastest fully measured resident cell, fixed rule before results.
    cell = max(valid, key=lambda r: r["decisions_per_s"])
    if cell["host_fraction"] >= 0.6:
        return (
            "HOST_BOUND: consider three L4 hosts; GPU cropping dropped by Fable; "
            "no dispatch authorized"
        )
    if cell["gpu_fraction"] >= 0.6:
        return "GPU_BOUND: consider one separately operated 20-minute A100 probe"
    return "MIXED: neither host nor GPU reaches 60%; do not select an accelerator by inference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=B8)
    parser.add_argument("--worlds", default="256,1024")
    parser.add_argument(
        "--resident-rows",
        help="comma-separated resident row shapes, divisible by 6*N; overrides worlds",
    )
    parser.add_argument(
        "--fallback-worlds",
        type=int,
        default=64,
        help="separately labeled resident shape if no requested shape fits; 0 disables",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--seconds", type=float, default=1200.0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--host-memory-gib", type=float)
    parser.add_argument("--qualify-states", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    worlds = [int(x) for x in args.worlds.split(",")]
    if args.resident_rows:
        requested_rows = [int(x) for x in args.resident_rows.split(",")]
        if args.rollouts < 1 or any(r < 1 or r % (6 * args.rollouts) for r in requested_rows):
            parser.error("resident rows must be positive multiples of 6*N")
        worlds = [r // (6 * args.rollouts) for r in requested_rows]
    if min(worlds) < 1 or args.seconds <= 0 or args.iterations < 0 or args.threads < 1:
        parser.error("positive world/time/thread limits required")
    if platform.system() == "Darwin" and args.threads > 4:
        parser.error("Mac runs are capped at four threads")
    if args.device == "cuda" and args.qualify_states < 1000:
        parser.error("CUDA benchmark requires >=1000-state fp16 qualification")
    if sha(args.checkpoint) != B8_SHA:
        raise ValueError("B8 checkpoint hash mismatch")
    phase0.configure_cpu()
    # Phase0 imports pin all native pools to 1. Larger Numba pools must be
    # explicitly configured before Python starts (NUMBA_NUM_THREADS).
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    numba.config.reload_config()
    numba.set_num_threads(args.threads)
    if platform.system() == "Darwin":
        from research.phase0_search_20261019.runner import operational_guard

        operational_guard()
    value = {
        "schema": "t33-throughput/v1",
        "device": args.device,
        "platform": platform.platform(),
        "threads": args.threads,
        "source": identity(),
        "checkpoint_sha256": B8_SHA,
        "bank_sha256": sha(Path(phase0.__file__).with_name("WORLD_BANK.json")),
        "requested_source_worlds": worlds,
        "requested_resident_rows": [w * 6 * args.rollouts for w in worlds],
        "cells": [],
        "status": "incomplete",
    }
    qualification_data = None
    if args.qualify_states:
        net, obs, masks = collect(args.checkpoint, args.qualify_states)
        value["qualification"] = qualify(net, obs, masks, device=args.device)
        if args.device == "cuda":
            qualification_data = (net, obs, masks)
        if args.device == "cuda" and not value["qualification"]["cuda_qualification"]:
            complete(args.out, value)
            return 2
    else:
        value["qualification"] = {"status": "LOCAL_SMOKE_ONLY_CUDA_UNVERIFIED"}
    templates = [
        prepared(fixture(row, args.checkpoint, 800)) for row in calibration_rows(smoke=True)
    ]
    config = SearchConfig(horizon=args.horizon, rollouts=args.rollouts, length_gate=0)
    value["search"] = asdict(config)
    host_cap = (
        int(args.host_memory_gib * 1024**3)
        if args.host_memory_gib
        else int(psutil.virtual_memory().available * 0.6)
    )
    gpu_cap = int(torch.cuda.mem_get_info()[0] * 0.7) if args.device == "cuda" else None
    admissions = [
        memory_admission(
            templates[0], size, args.rollouts, args.device, host_cap, gpu_cap, horizon=args.horizon
        )
        for size in worlds
    ]
    for admission in admissions:
        admission["benchmark_role"] = "requested"
    execution_worlds = list(worlds)
    if not any(c["admitted"] for c in admissions) and args.fallback_worlds > 0:
        fallback = memory_admission(
            templates[0],
            args.fallback_worlds,
            args.rollouts,
            args.device,
            host_cap,
            gpu_cap,
            horizon=args.horizon,
        )
        fallback["benchmark_role"] = "fallback_requested_resident_shapes_unverified"
        admissions.append(fallback)
        execution_worlds.append(args.fallback_worlds)
    cell_seconds = args.seconds / max(1, sum(c["admitted"] for c in admissions))
    for size, admitted in zip(execution_worlds, admissions):
        if admitted["admitted"]:
            try:
                if qualification_data is not None:
                    net, obs, masks = qualification_data
                    admitted["resident_qualification"] = qualify(
                        net, obs, masks, device=args.device, batch_size=admitted["model_batch_rows"]
                    )
                    if not admitted["resident_qualification"]["cuda_qualification"]:
                        admitted["status"] = "inference_qualification_failed"
                        value["cells"].append(admitted)
                        continue
                admitted.update(
                    run_cell(
                        templates,
                        config,
                        size,
                        args.device,
                        cell_seconds,
                        args.iterations,
                    )
                )
            except (RuntimeError, MemoryError, TimeoutError) as exc:
                admitted.update(status="failed", error=str(exc))
        value["cells"].append(admitted)
        print(json.dumps({"event": "benchmark_cell", **admitted}), flush=True)
    measured = [c for c in value["cells"] if c.get("iterations", 0) > 0]
    failed = any(
        c["status"] in ("failed", "no_complete_iterations", "inference_qualification_failed")
        for c in value["cells"]
    )
    value["status"] = "incomplete" if failed else "measured" if measured else "no_complete_cells"
    value["recommendation"] = recommendation(value["cells"], args.device)
    complete(args.out, value)
    return 0 if measured and not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
