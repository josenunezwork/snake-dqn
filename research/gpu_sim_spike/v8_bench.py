"""Matched-state V8 lane A/B packet; native timing never proves CPU closed-loop identity."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import torch

from research.gpu_sim_spike import a1_prof
from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import sparse_bench
from research.gpu_sim_spike import v8_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes


class ParityMismatch(ValueError):
    """A scientific mismatch, never a mechanical completion trigger."""


def _bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def _digest(sim: Any, policy: Any, collection: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for group in (vars(sim), vars(policy.vector_runtime), collection):
        for key, value in sorted(group.items()):
            if isinstance(value, torch.Tensor):
                digest.update(key.encode())
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(_bytes(value))
    return digest.hexdigest()


@contextlib.contextmanager
def _arm(candidate: bool) -> Iterator[None]:
    with patch.dict(
        os.environ,
        {
            "GPU_SIM_V8_INDEXED": "1" if candidate else "0",
            "GPU_SIM_VECTOR_GEOMETRY": "native" if candidate else "torch",
        },
    ):
        yield


@contextlib.contextmanager
def _instrument(sim: Any, timer: a1_prof.ExclusiveTiming) -> Iterator[None]:
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    original = DeviceVectorRuntime._free_space_all

    def flood(*args: Any, **kwargs: Any) -> Any:
        with timer.stage("vector61_flood_fill"):
            return original(*args, **kwargs)

    with (
        a1_prof.instrument(sim, timer),
        patch.object(DeviceVectorRuntime, "_free_space_all", flood),
    ):
        yield


def _case(checkpoint: Path, mix: str, worlds: int, device: str, candidate: bool) -> tuple:
    with _arm(candidate):
        sim, policy, template, policy_template = ruler._build_case(
            checkpoint, "long-" + mix, worlds, device
        )
        if candidate:
            from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

            policy.vector_runtime = IndexedVectorRuntime.from_runtime(policy.vector_runtime)
            policy_template.vector_runtime = IndexedVectorRuntime.from_runtime(
                policy_template.vector_runtime
            )
    return (
        sim,
        policy,
        template,
        policy_template,
        ruler._collection(1, worlds, sim.device),
        candidate,
    )


def _tick(case: tuple, device: str, timer: a1_prof.ExclusiveTiming | None = None) -> tuple:
    sim, policy, template, policy_template, collection, candidate = case
    with _arm(candidate), torch.inference_mode():
        a1_prof.sync(device)
        began = time.perf_counter()
        if timer is None:
            ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
        else:
            with _instrument(sim, timer), timer.stage("unscoped"):
                with timer.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
        a1_prof.sync(device)
        elapsed = time.perf_counter() - began
        # Transfers and digesting are outside the authoritative plain interval.
        return elapsed, _digest(sim, policy, collection)


def _first_verified_world(sim: Any, runtime: Any) -> tuple[Any, Any]:
    """Prove replication before using a single independent CPU world as oracle."""
    one, carry = copy.copy(sim), copy.copy(runtime)
    for name in (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "food_cells",
        "food_present",
        "food_count",
    ):
        value = getattr(sim, name)
        if not torch.equal(value, value[:1].expand_as(value)):
            raise ValueError("oracle fixture is not replicated: " + name)
        setattr(one, name, value[:1])
    for name in ("prev_dist", "prev_id"):
        value = getattr(runtime, name)
        if _bytes(value) != _bytes(value[:1].expand_as(value)):
            raise ValueError("oracle carry fixture is not replicated: " + name)
        setattr(carry, name, value[:1])
    one.E = 1
    return one, carry


def _oracles(case: tuple, output: Path, mix: str, worlds: int) -> dict[str, Any]:
    """Compare actual full-population outputs to CPU on a proven repeated world."""
    from research.gpu_sim_spike.v8_device import v8_decide
    from src.simd_env.v8_nb import v8_decide as cpu_v8

    sim, policy, _, _, _, candidate = case
    with _arm(candidate), torch.inference_mode():
        one, carry = _first_verified_world(sim, policy.vector_runtime)
        expected = sparse_bench._features_cpu_oracle(one, carry)
        actual = policy.vector_runtime.features(sim, update_memory=False)
        expected_all = np.tile(expected, (worlds, 1))
        vector_match = _bytes(actual) == expected_all.tobytes()
        vector = {
            "match": vector_match,
            "rows": int(actual.shape[0]),
            "columns": 61,
            "independent_cpu_worlds": 1,
            "replication_verified_worlds": worlds,
            "actual_sha256": hashlib.sha256(_bytes(actual)).hexdigest(),
            "expected_sha256": hashlib.sha256(expected_all.tobytes()).hexdigest(),
        }
        # Independent CPU v8 reads a snapshot, never candidate construction helpers.
        reference = copy.copy(one)
        for name, value in vars(one).items():
            if isinstance(value, torch.Tensor):
                setattr(reference, name, value.detach().cpu().numpy())
        reference.grid_w, reference.grid_h = int(sim.grid_w), int(sim.grid_h)
        diagnostics = []
        fields = (
            "action",
            "base",
            "v5_action",
            "v7_action",
            "outcome",
            "v5_reason",
            "counts",
            "cap",
            "need",
            "area_cap",
            "area",
            "landing",
            "head_risky",
            "head_waived",
            "v5_eligible",
        )
        for slot in range(sim.S):
            # Nonuniform Q and exact ties cover ordering without changing network math.
            q0 = np.array([[0.25, 0.5, 0.5, 0.75, 1.0, 0.125]], dtype=np.float32)
            m0 = np.ones((1, 6), dtype=bool)
            ref = cpu_v8(reference, np.array([[0, slot]], np.int64), q0, m0, features=True)
            rows = torch.stack(
                (
                    torch.arange(worlds, device=sim.device),
                    torch.full((worlds,), slot, device=sim.device),
                ),
                1,
            )
            got = v8_decide(
                sim,
                rows,
                torch.as_tensor(q0, device=sim.device).expand(worlds, -1),
                torch.as_tensor(m0, device=sim.device).expand(worlds, -1),
                features=True,
            )
            for field in fields:
                expected_field = np.repeat(getattr(ref, field), worlds, axis=0)
                actual_field = getattr(got, field).detach().cpu().numpy()
                if (
                    actual_field.dtype != expected_field.dtype
                    or actual_field.tobytes() != expected_field.tobytes()
                ):
                    raise ParityMismatch(f"CPU v8 byte mismatch: {mix}/slot{slot}/{field}")
            diagnostics.append({"slot": slot, "fields": len(fields), "match": True})
        if not vector_match:
            raise ParityMismatch(f"CPU vector61 byte mismatch: {mix}/E{worlds}")
        fixture = _recorded_fixture(output, mix, worlds, sim, policy)
        return {
            "vector61": vector,
            "v8": diagnostics,
            "recorded_fixture": fixture,
            "cpu_closed_loop_identity": False,
        }


def _recorded_fixture(output: Path, mix: str, worlds: int, sim: Any, policy: Any) -> dict[str, Any]:
    from research.gpu_sim_spike.a1_prof_units import _atomic_create

    arrays = {
        "config_json": np.array(json.dumps(asdict(sim.cfg), sort_keys=True)),
        "params_json": np.array(json.dumps(asdict(policy.vector_runtime.params), sort_keys=True)),
        "v8_q": np.array([[0.25, 0.5, 0.5, 0.75, 1.0, 0.125]], np.float32),
        "v8_mask": np.ones((1, 6), bool),
        "v8_lambda": np.array(8.0),
    }
    for prefix, obj in (("", sim), ("runtime_", policy.vector_runtime)):
        for key, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                selected = value[:1] if value.ndim and value.shape[0] == sim.E else value
                arrays[prefix + key] = selected.detach().cpu().numpy()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > 16 * 1024**2:
        raise ValueError("V8 recorded fixture exceeds16MiB")
    target = output / "recorded-states" / f"v8-{mix}-e{worlds}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_create(target, payload)
    _atomic_create(Path(str(target) + ".sha256"), f"{digest}  {target.name}\n".encode())
    _atomic_create(Path(str(target) + ".complete"), b"complete\n")
    return {"path": target.relative_to(output).as_posix(), "sha256": digest, "size": len(payload)}


def _cpu_feature_expected(sim: Any, runtime: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from types import SimpleNamespace

    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory

    one, carry = _first_verified_world(sim, runtime)
    reference = SimpleNamespace(cfg=one.cfg, E=1, S=sim.S, s=sim.cfg.segment_size, cap=sim.cap)
    for key in ("bodies", "head_ptr", "seg_count", "length", "alive", "direction"):
        setattr(reference, key, getattr(one, key).detach().cpu().numpy())
    count = int(one.food_count[0].detach().cpu())
    reference.food_cells = [
        [tuple(map(int, x)) for x in one.food_cells[0, :count].detach().cpu().numpy()]
    ]
    memory = Vector61Memory(1, sim.S)
    memory.prev_dist[:] = carry.prev_dist.detach().cpu().numpy()
    memory.prev_id[:] = carry.prev_id.detach().cpu().numpy()
    rows = np.stack((np.zeros(sim.S, np.int64), np.arange(sim.S)), 1)
    expected = Vector61Featurizer(runtime.params).featurize(
        reference, rows, memory, update_memory=True
    )
    return expected, memory.prev_dist, memory.prev_id


def decide(row: dict[str, Any], *, enforce_speed: bool = True) -> str:
    if row.get("verdict") == "INVALID":
        return "INVALID"
    if (
        not row.get("same_action_mechanics_byte_identical")
        or not row.get("feature_call_cpu_parity")
        or not row.get("cpu_oracle", {}).get("vector61", {}).get("match")
    ):
        return "INVALID"
    if enforce_speed and row["worlds"] == 1024:
        for stage in ("vector61_non_flood", "v8_combined"):
            speed = row["stage_seconds"][stage]["speedup"]
            if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed < 2:
                return "STOP_ROUTE"
    return "PASS_CELL"


# Only these normalizations are declared CPU repairs. Every other feature byte
# remains subject to the pre-change optimization oracle.
REPAIR_COLUMNS = (4, *range(5, 24), *range(40, 47), 50, 51, 52)


def _mechanics_digest(sim: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(vars(sim).items()):
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(_bytes(value))
    return digest.hexdigest()


def _proof(cases: dict[str, tuple], device: str) -> dict[str, Any]:
    """Separate CPU feature proof from common-action mechanics regression."""
    tape: list[torch.Tensor] = []
    baseline_features: list[np.ndarray] = []
    repair_columns: set[int] = set()
    calls = 0
    digests = []
    for name in ("baseline", "candidate"):
        case = cases[name]
        sim, policy = case[:2]
        ruler._reset(sim, policy, case[2], case[3])
        runtime = policy.vector_runtime
        original_features, original_step = runtime.features, sim.step
        index = 0

        def features(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal index, calls
            expected, next_dist, next_id = _cpu_feature_expected(sim, runtime)
            previous_dist = runtime.prev_dist.detach().cpu().numpy().copy()
            previous_id = runtime.prev_id.detach().cpu().numpy().copy()
            result = original_features(*args, **kwargs)
            array = result.detach().cpu().numpy()
            expected_all = np.tile(expected, (int(sim.E), 1))
            if array.shape != expected_all.shape:
                raise ValueError("proof requires the frozen complete-row lifecycle")
            if name == "baseline":
                baseline_features.append(array.copy())
            else:
                if array.tobytes() != expected_all.tobytes():
                    raise ParityMismatch("candidate lifecycle feature bytes differ from CPU")
                expected_dist = previous_dist.copy()
                expected_id = previous_id.copy()
                if kwargs.get("update_memory", True):
                    mask = kwargs.get("memory_mask")
                    store = sim.alive.detach().cpu().numpy().copy()
                    if mask is not None:
                        store &= mask.detach().cpu().numpy().reshape(store.shape)
                    expected_dist = np.where(store, np.tile(next_dist, (sim.E, 1)), expected_dist)
                    expected_id = np.where(store, np.tile(next_id, (sim.E, 1)), expected_id)
                if (
                    _bytes(runtime.prev_dist) != expected_dist.tobytes()
                    or _bytes(runtime.prev_id) != expected_id.tobytes()
                ):
                    raise ParityMismatch("candidate lifecycle carry bytes differ from CPU")
                old = baseline_features[index]
                differences = np.argwhere(old.view(np.uint32) != array.view(np.uint32))
                changed = set(map(int, differences[:, 1])) if len(differences) else set()
                if not changed <= set(REPAIR_COLUMNS):
                    raise ParityMismatch("unexplained geometry optimization byte difference")
                repair_columns.update(changed)
                calls += 1
            index += 1
            return result

        def step(actions: torch.Tensor) -> Any:
            if name == "baseline":
                tape.append(actions.clone())
                return original_step(actions)
            if len(tape) != 1:
                raise ValueError("proof action tape differs from one-tick contract")
            return original_step(tape[0])

        with (
            _arm(name == "candidate"),
            patch.object(runtime, "features", features),
            patch.object(sim, "step", step),
        ):
            ruler._full_tick(sim, policy, case[4], 0)
            a1_prof.sync(device)
        if name == "candidate" and index != len(baseline_features):
            raise ValueError("feature lifecycle call count differs")
        digests.append(_mechanics_digest(sim))
    return {
        "same_action_mechanics_byte_identical": digests[0] == digests[1],
        "feature_call_cpu_parity": calls > 0,
        "verified_lifecycle_feature_calls": calls,
        "observed_declared_cpu_repair_columns": sorted(repair_columns),
    }


def measure(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    warmups: int,
    deadline: float,
    output: Path,
) -> dict[str, Any]:
    cases: dict[str, tuple] = {}
    try:
        cases = {
            name: _case(checkpoint, mix, worlds, device, name == "candidate")
            for name in ("baseline", "candidate")
        }
        oracle = _oracles(cases["candidate"], output, mix, worlds)
        proof = _proof(cases, device)
        arms = {name: {"plain_ticks": [], "digests": [], "stages": []} for name in cases}
        for case in cases.values():
            for _ in range(warmups):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("V8 wall cap during warmup")
                _tick(case, device)
        for repeat in range(repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for name in order:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("V8 wall cap during repeat")
                elapsed, digest = _tick(cases[name], device)
                arms[name]["plain_ticks"].append(elapsed)
                arms[name]["digests"].append(digest)
                timer = a1_prof.ExclusiveTiming(device)
                diagnostic_wall, event_digest = _tick(cases[name], device, timer)
                if event_digest != digest:
                    raise ParityMismatch("instrumented/plain state-carry byte mismatch")
                arms[name]["stages"].append(
                    {"repeat": repeat, "wall_seconds": diagnostic_wall, **timer.result()}
                )
        baseline, candidate = arms["baseline"], arms["candidate"]
        clock = "exclusive_cuda_elapsed_seconds" if device == "cuda" else "exclusive_host_seconds"
        names = {
            "vector61_non_flood": ("vector61_other",),
            "b8_v8_bundle": ("b8_v8_features",),
            "opponent_v8_bundle": ("opponent_v8_bundle",),
            "v8_combined": ("b8_v8_features", "opponent_v8_bundle"),
        }
        stages = {}
        for label, components in names.items():
            values = {
                arm: statistics.median(
                    sum(r[clock][k] for k in components) for r in arms[arm]["stages"]
                )
                for arm in arms
            }
            stages[label] = {
                **values,
                "speedup": (
                    values["baseline"] / values["candidate"] if values["candidate"] > 0 else None
                ),
            }
        return {
            "mix": mix,
            "worlds": worlds,
            "repeats": repeats,
            "warmups": warmups,
            "baseline": baseline,
            "candidate": candidate,
            "cpu_oracle": oracle,
            "recorded_fixture": oracle["recorded_fixture"],
            **proof,
            "state_carry_byte_identical": baseline["digests"] == candidate["digests"],
            "cpu_closed_loop_identity": False,
            "stage_seconds": stages,
            "full_tick_speedup": statistics.median(baseline["plain_ticks"])
            / statistics.median(candidate["plain_ticks"]),
        }
    finally:
        cases.clear()
        ruler._release_cell(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" and not args.smoke:
        raise ValueError("CPU is smoke/preflight only")
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("V8 requires A100")
    if args.seconds <= 0:
        raise ValueError("wall cap must be positive")
    os.environ["GPU_SIM_VECTOR_FLOOD"] = "static"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    commit = source_commit()
    ns = units.namespace(commit, smoke=args.smoke, canary=args.canary)
    allocation = units.allocation(smoke=args.smoke, canary=args.canary)
    prior = units.load_ledger(args.ledger, ns, allocation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    complete, cells = [], []
    status = "PREFLIGHT" if args.smoke else "MEASURED_PENDING_MAC_REVIEW"
    for unit in allocation:
        if unit in prior:
            value = prior[unit]
        else:
            _, mix, world = unit.split("-")
            print(
                json.dumps(
                    {
                        "event": "unit_start",
                        "unit": unit,
                        "done": len(complete),
                        "total": len(allocation),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            try:
                value = measure(
                    args.checkpoint,
                    mix,
                    int(world[1:]),
                    args.device,
                    1 if args.smoke else 3,
                    0 if args.smoke else 3,
                    started + args.seconds,
                    args.output_dir,
                )
            except ParityMismatch as exc:
                value = {
                    "mix": mix,
                    "worlds": int(world[1:]),
                    "verdict": "INVALID",
                    "reason": str(exc),
                    "cpu_closed_loop_identity": False,
                }
            value["verdict"] = decide(value, enforce_speed=not args.smoke)
            units.publish_unit(args.output_dir, ns, allocation, unit, value)
        complete.append(unit)
        cells.append(value)
        verdict = decide(value, enforce_speed=not args.smoke)
        if verdict != "PASS_CELL":
            status = verdict
            break
    receipt = {
        "schema": "gpu-sim-spike-v8-bench/v1",
        "status": status,
        "commit": commit,
        "source_sha256": source_hashes(),
        "device": args.device,
        "namespace": ns,
        "allocation": list(allocation),
        "completed_units": complete,
        "cells": cells,
        "skipped_verified_units": sorted(prior),
        "elapsed_seconds": time.perf_counter() - started,
        "wall_cap_seconds": args.seconds,
        "platform": platform.platform(),
        "cpu_closed_loop_identity": False,
    }
    units.publish_json(args.output_dir / "v8-bench.json", receipt)
    return receipt


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
    try:
        receipt = run(args)
    except Exception as exc:
        units.publish_json(
            args.output_dir / "v8-bench-failure.json",
            {"status": "INCOMPLETE", "error": str(exc), "type": type(exc).__name__},
        )
        raise
    if receipt["status"] not in {"PREFLIGHT", "MEASURED_PENDING_MAC_REVIEW"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
