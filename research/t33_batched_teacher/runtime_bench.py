"""Measure T33's Vector61 carry runtime on the registered Phase-0 timing fixture.

This harness intentionally drives only ``prepare -> BatchSim.step -> observe``.
It isolates the carry builder from frozen-network inference, but keeps the real
Phase-0 frozen roster geometry and the 48 independent rollout-world shape.
Warm-up is excluded.  It also checks canonical runtime versus one- and
two-thread T33 runtime bytes before emitting a create-only JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np

from research.phase0_search_20261019 import preflight, teacher
from research.t33_batched_teacher.runtime import BatchedVector61Runtime

ROOT = Path(__file__).resolve().parents[2]
# Phase-0's import-time local safety cap is one thread.  This isolated benchmark
# needs a two-thread ceiling to test the wrapper, while each measured arm still
# sets its actual count explicitly with ``numba.set_num_threads`` below.
os.environ["NUMBA_NUM_THREADS"] = "2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timing_row() -> dict[str, Any]:
    bank = json.loads((ROOT / "research/phase0_search_20261019/WORLD_BANK.json").read_text())
    return next(row for row in bank["fixtures"]["timing"] if row["mix"] == "frozen")


def _replica(worlds: int, *, accelerated: bool) -> tuple[Any, Any]:
    from research.phase0_search_20261019.profile_teacher import _fixture

    source = _fixture(_timing_row(), preflight.B8, 800)
    replica = teacher._replicate(source, worlds)
    if not accelerated:
        return replica.sim, replica.runtime
    runtime = BatchedVector61Runtime(params=source.runtime._params)
    runtime.bind(replica.sim, replica.runtime.controlled)
    return replica.sim, runtime


def _runtime_digest(runtime: Any) -> str:
    digest = hashlib.sha256()
    for owner in (runtime, runtime.memory):
        for name in sorted(vars(owner)):
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                digest.update(name.encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def _sim_digest(sim: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(vars(sim)):
        value = getattr(sim, name)
        if isinstance(value, np.ndarray):
            digest.update(name.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _drive(sim: Any, runtime: Any, frames: int) -> dict[str, Any]:
    actions = np.ones((sim.E, sim.S), dtype=np.int64)
    prepare_s = 0.0
    observe_s = 0.0
    started = time.perf_counter()
    for frame in range(frames):
        phase_started = time.perf_counter()
        runtime.prepare(sim)
        prepare_s += time.perf_counter() - phase_started
        actions[:, :] = frame % 3
        sim.step(actions)
        phase_started = time.perf_counter()
        runtime.observe_step(sim)
        observe_s += time.perf_counter() - phase_started
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "exclusive_runtime_s": {
            "prepare_s": prepare_s,
            "observe_s": observe_s,
            "total_s": prepare_s + observe_s,
        },
        "sim_digest": _sim_digest(sim),
        "runtime_digest": _runtime_digest(runtime),
        "profile": runtime.profile() if isinstance(runtime, BatchedVector61Runtime) else None,
    }


def _parity(worlds: int, frames: int) -> dict[str, Any]:
    numba.set_num_threads(1)
    canonical_sim, canonical_runtime = _replica(worlds, accelerated=False)
    canonical = _drive(canonical_sim, canonical_runtime, frames)
    checks = {"canonical": canonical}
    for threads in (1, 2):
        numba.set_num_threads(threads)
        sim, runtime = _replica(worlds, accelerated=True)
        observed = _drive(sim, runtime, frames)
        checks[f"parallel_{threads}"] = observed
        if observed["sim_digest"] != canonical["sim_digest"]:
            raise RuntimeError(f"sim digest mismatch at {threads} threads")
        if observed["runtime_digest"] != canonical["runtime_digest"]:
            raise RuntimeError(f"runtime digest mismatch at {threads} threads")
    return checks


def _measure(
    label: str, worlds: int, frames: int, rounds: int, threads: int
) -> list[dict[str, Any]]:
    measurements = []
    for round_index in range(1, rounds + 1):
        numba.set_num_threads(threads)
        sim, runtime = _replica(worlds, accelerated=label == "parallel")
        observed = _drive(sim, runtime, frames)
        observed["round"] = round_index
        measurements.append(observed)
    return measurements


def run(worlds: int, frames: int, rounds: int) -> dict[str, Any]:
    # Compile both specializations outside the measured interval.
    _parity(min(2, worlds), 1)
    parity = _parity(worlds, frames)
    canonical = _measure("canonical", worlds, frames, rounds, threads=1)
    parallel_one = _measure("parallel", worlds, frames, rounds, threads=1)
    parallel_two = _measure("parallel", worlds, frames, rounds, threads=2)

    def median(rows: list[dict[str, Any]]) -> float:
        return float(np.median([row["elapsed_s"] for row in rows]))

    baseline = median(canonical)
    return {
        "schema": "t33-batched-teacher-runtime-bench/v1",
        "command": [sys.executable, *sys.argv],
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "warmup_excluded": True,
            "thread_counts": [1, 2],
        },
        "inputs": {
            "worlds": worlds,
            "frames": frames,
            "rounds": rounds,
            "fixture": "phase0 timing/frozen, synthetic valid L800 serpentines",
            "checkpoint": str(preflight.B8),
            "checkpoint_sha256": _sha256(preflight.B8),
        },
        "source_sha256": {
            "research/t33_batched_teacher/runtime.py": _sha256(
                ROOT / "research/t33_batched_teacher/runtime.py"
            ),
            "research/t33_batched_teacher/runtime_bench.py": _sha256(Path(__file__)),
            "src/simd_env/vector61_nb.py": _sha256(ROOT / "src/simd_env/vector61_nb.py"),
            "src/simd_env/vector61_policy.py": _sha256(ROOT / "src/simd_env/vector61_policy.py"),
        },
        "parity": parity,
        "measurements": {
            "canonical_numba_1_thread": canonical,
            "t33_parallel_1_thread": parallel_one,
            "t33_parallel_2_threads": parallel_two,
        },
        "median_speedup_vs_canonical": {
            "t33_parallel_1_thread": baseline / median(parallel_one),
            "t33_parallel_2_threads": baseline / median(parallel_two),
        },
        "timing_interpretation": {
            "elapsed_s": "whole prepare -> BatchSim.step -> observe wall time",
            "exclusive_runtime_s": "caller-timed prepare/observe intervals; "
            "excluded from step time",
            "prepare_s_and_observe_s": "parent wall timers; each includes food_s and kernel_s",
            "food_s_and_kernel_s": "nested featurizer phases; do not add to parent timers",
        },
        "limitations": [
            "This measures only runtime carry construction and BatchSim stepping, "
            "not policy forwards.",
            "The result is a local Mac fixture measurement, not a Colab throughput result.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=48)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.frames, args.rounds) < 1:
        parser.error("--worlds, --frames, and --rounds must be positive")
    if args.out.exists():
        parser.error("refusing to overwrite an existing receipt")
    result = run(args.worlds, args.frames, args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "speedup": result["median_speedup_vs_canonical"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
