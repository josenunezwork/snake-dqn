"""Matched dense/static vector61 full-tick benchmark with a CPU byte oracle.

This is a development measurement only.  A feature mismatch is terminal for
the route and a roster whose E1024 full-tick median speedup is below two is a
terminal STOP_ROUTE.  It deliberately does not claim the A1 2,000/s bar.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import sparse_units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes


class SparseTerminal(RuntimeError):
    """A prospective sparse-route stop condition."""


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _features_cpu_oracle(sim: Any, runtime: Any) -> np.ndarray:
    """Use the canonical CPU vector61 featurizer on the recorded DeviceSim state."""
    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory

    # A repeated resident batch retains its one-world template cfg. Build a
    # read-only feature snapshot with the actual E; the canonical featurizer
    # consumes these arrays directly and never steps mechanics or RNG here.
    reference = SimpleNamespace(
        cfg=sim.cfg, E=int(sim.E), S=int(sim.S), s=sim.cfg.segment_size, cap=int(sim.cap)
    )
    for name in ("bodies", "head_ptr", "seg_count", "length", "alive", "direction"):
        setattr(reference, name, getattr(sim, name).detach().cpu().numpy())
    food = sim.food_cells.detach().cpu().numpy()
    counts = sim.food_count.detach().cpu().numpy()
    reference.food_cells = [
        [tuple(map(int, cell)) for cell in food[env, : int(counts[env])]]
        for env in range(int(sim.E))
    ]
    params = runtime.params
    if params is None:
        raise SparseTerminal("vector61 runtime was not bound")
    rows = np.stack(
        np.unravel_index(np.arange(int(sim.E) * int(sim.S)), (int(sim.E), int(sim.S))), axis=1
    )
    memory = Vector61Memory(int(sim.E), int(sim.S))
    # The corrected candidate stores CPU-exact pixel distances; the original
    # dense comparator stores cells. Match the recorded carry in either mode.
    scale = 1.0 if os.environ.get("GPU_SIM_VECTOR_FLOOD") == "static" else params.segment_size
    memory.prev_dist[:] = runtime.prev_dist.detach().cpu().numpy() * scale
    memory.prev_id[:] = runtime.prev_id.detach().cpu().numpy()
    return Vector61Featurizer(params).featurize(reference, rows, memory, update_memory=False)


def _byte_oracle(sim: Any, policy: Any) -> dict[str, Any]:
    """Record every feature byte mismatch against the CPU contract before timing."""
    runtime = policy.vector_runtime
    if runtime is None:
        raise SparseTerminal("full-tick policy omitted vector61 runtime")
    actual = (
        runtime.features(sim, update_memory=False)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    expected = _features_cpu_oracle(sim, runtime).astype(np.float32, copy=False)
    if actual.shape != expected.shape:
        return {
            "match": False,
            "reason": "shape",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    equal = actual.tobytes() == expected.tobytes()
    mismatch = np.argwhere(actual.view(np.uint32) != expected.view(np.uint32))
    return {
        "match": bool(equal),
        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "rows": int(actual.shape[0]),
        "columns": int(actual.shape[1]),
        "first_mismatch": (
            None
            if not len(mismatch)
            else {
                "row": int(mismatch[0, 0]),
                "column": int(mismatch[0, 1]),
                "actual_bits": int(actual.view(np.uint32)[tuple(mismatch[0])]),
                "expected_bits": int(expected.view(np.uint32)[tuple(mismatch[0])]),
            }
        ),
    }


def _recorded_fixture(
    output_dir: Path, mix: str, worlds: int, sim: Any, policy: Any, oracle: dict[str, Any]
) -> dict[str, Any]:
    """Export one replayable world, never the E1024 resident tensor population."""
    runtime = policy.vector_runtime
    assert runtime is not None
    target = output_dir / "recorded-states" / f"{mix}-static-e{worlds}.npz"
    if target.exists():
        raise FileExistsError(f"immutable recorded state already exists: {target}")
    arrays = {
        "config_json": np.array(json.dumps(asdict(sim.cfg), sort_keys=True)),
        "params_json": np.array(json.dumps(asdict(runtime.params), sort_keys=True)),
        "bodies": sim.bodies[0].detach().cpu().numpy(),
        "head_ptr": sim.head_ptr[0].detach().cpu().numpy(),
        "seg_count": sim.seg_count[0].detach().cpu().numpy(),
        "length": sim.length[0].detach().cpu().numpy(),
        "direction": sim.direction[0].detach().cpu().numpy(),
        "alive": sim.alive[0].detach().cpu().numpy(),
        "food_cells": sim.food_cells[0].detach().cpu().numpy(),
        "food_present": sim.food_present[0].detach().cpu().numpy(),
        "food_count": sim.food_count[0].detach().cpu().numpy(),
        "prev_dist": runtime.prev_dist[0].detach().cpu().numpy(),
        "prev_id": runtime.prev_id[0].detach().cpu().numpy(),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > 16 * 1024**2:
        raise SparseTerminal("recorded sparse state exceeds the 16 MiB fixture bound")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    from research.gpu_sim_spike import a1_prof_units

    a1_prof_units._atomic_create(target, payload)
    a1_prof_units._atomic_create(
        target.with_name(target.name + ".sha256"), f"{digest}  {target.name}\n".encode()
    )
    a1_prof_units._atomic_create(target.with_name(target.name + ".complete"), b"complete\n")
    return {"path": str(target.relative_to(output_dir)), "sha256": digest, "size": len(payload)}


def _measure_backend(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    backend: str,
    repeats: int,
    warmups: int,
    deadline: float,
    output_dir: Path | None = None,
    repeat_offset: int = 0,
) -> dict[str, Any]:
    """Use fresh, identically reset resident states for one backend's full ticks."""
    if backend not in {"dense", "static"}:
        raise ValueError("unknown vector61 flood backend")
    os.environ["GPU_SIM_VECTOR_FLOOD"] = backend
    samples: list[dict[str, Any]] = []
    oracle: dict[str, Any] | None = None
    for repeat in range(repeats):
        if time.perf_counter() >= deadline:
            raise TimeoutError("sparse benchmark wall cap reached before a repeat")
        ruler._release_cell(device)
        sim = policy = template = policy_template = collection = None
        try:
            sim, policy, template, policy_template = ruler._build_case(
                checkpoint, "long-" + mix, worlds, device
            )
            collection = ruler._collection(1, worlds, sim.device)
            with sim.stage("reset"):
                ruler._reset(sim, policy, template, policy_template)
            # The oracle records a reset state and is intentionally outside timing.
            current = _byte_oracle(sim, policy)
            if oracle is None:
                oracle = current
                if backend == "static" and output_dir is not None:
                    current["recorded_fixture"] = _recorded_fixture(
                        output_dir, mix, worlds, sim, policy, current
                    )
            if backend == "static" and not current["match"]:
                return {"backend": backend, "feature_oracle": current, "samples": samples}
            for _ in range(warmups):
                with sim.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
            _sync(device)
            _sync(device)
            started = time.perf_counter()
            # Reset is resident work and stays inside each timed complete tick.
            with sim.stage("reset"):
                ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
            _sync(device)
            tick = time.perf_counter() - started
            samples.append(
                {
                    "repeat": repeat_offset + repeat,
                    "tick_seconds": tick,
                    "complete_world_steps_per_second": worlds / tick,
                    "collection_checksum": int(collection["length"].sum().detach().cpu()),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "sparse_bench_repeat_complete",
                        "backend": backend,
                        "mix": mix,
                        "worlds": worlds,
                        "repeat": repeat + 1,
                        "repeats": repeats,
                        "complete_world_steps_per_second": samples[-1][
                            "complete_world_steps_per_second"
                        ],
                    }
                ),
                flush=True,
            )
        finally:
            del collection, policy_template, template, policy, sim
            gc.collect()
            ruler._release_cell(device)
    ticks = [sample["tick_seconds"] for sample in samples]
    return {
        "backend": backend,
        "feature_oracle": oracle,
        "samples": samples,
        "median_tick_seconds": statistics.median(ticks),
        "median_complete_world_steps_per_second": statistics.median(
            sample["complete_world_steps_per_second"] for sample in samples
        ),
    }


def _merge_backend(backend: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve every paired repeat while reporting the registered median."""
    oracle = next(
        (row["feature_oracle"] for row in rows if not row["feature_oracle"]["match"]),
        rows[0]["feature_oracle"],
    )
    oracle = dict(oracle)
    if "recorded_fixture" not in oracle:
        for row in rows:
            fixture = row["feature_oracle"].get("recorded_fixture")
            if fixture is not None:
                oracle["recorded_fixture"] = fixture
                break
    samples = [sample for row in rows for sample in row["samples"]]
    if not samples:
        return {"backend": backend, "feature_oracle": oracle, "samples": []}
    return {
        "backend": backend,
        "feature_oracle": oracle,
        "samples": samples,
        "median_tick_seconds": statistics.median(sample["tick_seconds"] for sample in samples),
        "median_complete_world_steps_per_second": statistics.median(
            sample["complete_world_steps_per_second"] for sample in samples
        ),
    }


def _cell(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    warmups: int,
    deadline: float,
    *,
    enforce_speed: bool,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    # Alternate order within each matched repeat; every individual run constructs
    # a fresh identical resident state and includes reset in its tick clock.
    by_backend: dict[str, list[dict[str, Any]]] = {"dense": [], "static": []}
    for repeat in range(repeats):
        order = ("dense", "static") if repeat % 2 == 0 else ("static", "dense")
        for backend in order:
            row = _measure_backend(
                checkpoint,
                mix,
                worlds,
                device,
                backend,
                1,
                warmups,
                deadline,
                output_dir if backend == "static" and repeat == 0 else None,
                repeat,
            )
            by_backend[backend].append(row)
            # The dense port's known non-free-space float drift is recorded as
            # baseline diagnosis. Static is a deliberately corrected candidate
            # and must satisfy the full canonical 61-column CPU bytes.
            if backend == "static" and not row["feature_oracle"]["match"]:
                dense, static = _merge_backend("dense", by_backend["dense"]), (
                    _merge_backend("static", by_backend["static"]) if by_backend["static"] else None
                )
                return {
                    "mix": mix,
                    "worlds": worlds,
                    "dense": dense,
                    "static": static,
                    "verdict": "INVALID_FEATURE_MISMATCH",
                }
    dense, static = _merge_backend("dense", by_backend["dense"]), _merge_backend(
        "static", by_backend["static"]
    )
    speedup = dense["median_tick_seconds"] / static["median_tick_seconds"]
    return {
        "mix": mix,
        "worlds": worlds,
        "dense": dense,
        "static": static,
        "baseline_dense_cpu_diagnostic_only": not dense["feature_oracle"]["match"],
        "candidate_static_full61_cpu_exact": True,
        "speedup": speedup,
        "verdict": (
            "PASS_CELL" if (not enforce_speed or speedup >= 2.0) else "STOP_ROUTE_SPEEDUP_LT_2X"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise SparseTerminal("sparse main/canary requires an A100 CUDA runtime")
    if args.device == "cpu" and not args.smoke:
        raise SparseTerminal("CPU is only allowed for the bounded Mac preflight")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    commit = source_commit()
    ns = sparse_units.namespace(commit, smoke=args.smoke, canary=args.canary)
    allocated = sparse_units.allocation(smoke=args.smoke, canary=args.canary)
    prior = sparse_units.load_ledger(args.ledger, ns, allocated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, deadline = time.perf_counter(), time.perf_counter() + args.seconds
    completed: list[str] = []
    cells: list[dict[str, Any]] = []
    worlds, repeats, warmups = (
        (1, 1, 0) if args.smoke else (1024, 1 if args.canary else 3, 0 if args.canary else 3)
    )
    for unit_id in allocated:
        if unit_id in prior:
            value = prior[unit_id]
        else:
            mix = unit_id.split("-")[1]
            value = _cell(
                args.checkpoint,
                mix,
                worlds,
                args.device,
                repeats,
                warmups,
                deadline,
                enforce_speed=not (args.smoke or args.canary),
                output_dir=args.output_dir,
            )
            sparse_units.publish_unit(args.output_dir, ns, allocated, unit_id, value)
        cells.append(value)
        completed.append(unit_id)
        if value["verdict"] != "PASS_CELL":
            return {
                "schema": "gpu-sim-spike-sparse-bench/v1",
                "status": (
                    "INVALID" if value["verdict"] == "INVALID_FEATURE_MISMATCH" else "STOP_ROUTE"
                ),
                "terminal_unit": unit_id,
                "cells": cells,
                "completed_units": completed,
                "namespace": ns,
                "commit": commit,
                "source_sha256": source_hashes(),
                "device": args.device,
                "entrypoint_run_wall_seconds": time.perf_counter() - started,
                "wall_cap_seconds": args.seconds,
                "kill_rule": "feature mismatch or below-2x E1024 speedup terminates route",
            }
    return {
        "schema": "gpu-sim-spike-sparse-bench/v1",
        "status": "PREFLIGHT" if args.smoke else "MEASURED_PENDING_MAC_REVIEW",
        "cells": cells,
        "completed_units": completed,
        "skipped_verified_units": sorted(prior),
        "namespace": ns,
        "commit": commit,
        "source_sha256": source_hashes(),
        "device": args.device,
        "platform": platform.platform(),
        "wall_cap_seconds": args.seconds,
        "entrypoint_run_wall_seconds": time.perf_counter() - started,
        "kill_rule": "feature mismatch or below-2x E1024 speedup terminates route",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    try:
        receipt = run(args)
    except Exception as exc:
        receipt = {
            "schema": "gpu-sim-spike-sparse-bench/v1",
            "status": "INCOMPLETE",
            "reason": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "commit": source_commit(),
            "source_sha256": source_hashes(),
            "device": args.device,
        }
    sparse_units.publish_json(args.output_dir / "sparse-bench.json", receipt)
    print(
        json.dumps(
            {"status": receipt["status"], "output": str(args.output_dir / "sparse-bench.json")},
            sort_keys=True,
        ),
        flush=True,
    )
    if receipt["status"] not in {"PREFLIGHT", "MEASURED_PENDING_MAC_REVIEW"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
