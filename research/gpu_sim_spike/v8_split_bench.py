"""Prospective R/G/GV attribution packet; frozen kernels and no gameplay claims."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import torch

from research.gpu_sim_spike import a1_prof
from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import v8_bench as legacy
from research.gpu_sim_spike import v8_split_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes

ARMS = ("R", "G", "GV")
ORDERS = (ARMS, ("G", "GV", "R"), ("GV", "R", "G"))
KERNEL_PIN = "85180521e87ffc3ddc79296bfb09dcd2c89fdc62"
REPAIR_COLUMNS = legacy.REPAIR_COLUMNS
ParityMismatch = legacy.ParityMismatch
# Timing accumulators cannot influence policies or mechanics; all other fields
# must be serializable and are included. Indexed transient views must be cleared.
INSTRUMENTATION_FIELDS = {"_timing", "host_stages"}
TRANSIENT_FIELDS = {"_geometry_sim", "_geometry_body_view"}


@contextlib.contextmanager
def _arm(name: str) -> Iterator[None]:
    if name not in ARMS:
        raise ValueError("unknown split arm")
    with patch.dict(
        os.environ,
        {
            "GPU_SIM_V8_INDEXED": "1" if name == "GV" else "0",
            "GPU_SIM_VECTOR_GEOMETRY": "native" if name != "R" else "torch",
            "GPU_SIM_VECTOR_FLOOD": "static",
        },
    ):
        yield


def _case(checkpoint: Path, mix: str, worlds: int, device: str, name: str) -> tuple:
    from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

    with _arm(name):
        sim, policy, template, policy_template = ruler._build_case(
            checkpoint, "long-" + mix, worlds, device
        )
        if name != "R":
            for target in (policy, policy_template):
                target.vector_runtime = IndexedVectorRuntime.from_runtime(target.vector_runtime)
        if isinstance(policy.vector_runtime, IndexedVectorRuntime) != (name != "R"):
            raise ParityMismatch("actual geometry adapter differs from arm")
    return sim, policy, template, policy_template, ruler._collection(1, worlds, sim.device), name


def _value(value: Any) -> Any:
    """Typed, lossless inventory; refuse unknown persistent objects."""
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(legacy._bytes(value)).hexdigest(),
        }
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__qualname__, "fields": _value(dataclasses.asdict(value))}
    if isinstance(value, torch.device):
        return {"type": "device", "value": str(value)}
    if isinstance(value, torch.nn.Module):
        return {
            "type": type(value).__qualname__,
            "configuration_and_state": _value(vars(value)),
        }
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, set):
        return {"type": "set", "items": [_value(v) for v in sorted(value)]}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [_value(v) for v in value]}
    if isinstance(value, float):
        return {"type": "float", "hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": type(value).__name__, "value": value}
    raise ParityMismatch(f"uninventoried persistent type {type(value).__qualname__}")


def inventory(obj: Any) -> dict[str, Any]:
    result = {}
    for key, value in sorted(vars(obj).items()):
        if key in INSTRUMENTATION_FIELDS:
            continue
        if key in TRANSIENT_FIELDS:
            if value is not None:
                raise ParityMismatch("stale indexed carry: " + key)
            continue
        result[key] = _value(value)
    return result


def _snapshot(case: tuple) -> dict[str, Any]:
    sim, policy = case[:2]
    p = {k: v for k, v in vars(policy).items() if k != "vector_runtime"}
    return {
        "sim": inventory(sim),
        "runtime": inventory(policy.vector_runtime),
        "policy": _value({k: v for k, v in p.items() if k not in INSTRUMENTATION_FIELDS}),
        "collection": _value(case[4]),
    }


def _digest(case: tuple) -> str:
    return hashlib.sha256(json.dumps(_snapshot(case), sort_keys=True).encode()).hexdigest()


def _tick(case: tuple, device: str, timer: Any = None) -> tuple[float, str]:
    sim, policy, template, policy_template, collection, name = case
    with _arm(name), torch.inference_mode():
        a1_prof.sync(device)
        began = time.perf_counter()
        if timer is None:
            ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
        else:
            with legacy._instrument(sim, timer), timer.stage("unscoped"):
                with timer.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
        a1_prof.sync(device)
        elapsed = time.perf_counter() - began
    return elapsed, _digest(case)


def _oracles(
    case: tuple, output: Path, mix: str, worlds: int, fixture: Any = None
) -> dict[str, Any]:
    name = case[-1]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(legacy, "_arm", _arm))
        if fixture is not None:
            stack.enter_context(patch.object(legacy, "_recorded_fixture", return_value=fixture))
        result = legacy._oracles(case, output / name, mix, worlds)
    if fixture is None:
        result["recorded_fixture"]["path"] = name + "/" + result["recorded_fixture"]["path"]
    return result


def _proof(cases: dict[str, tuple], device: str) -> dict[str, Any]:
    """CPU lifecycle bytes plus frozen-device full-state same-action identity."""
    tape, baseline_features, final = [], [], {}
    repair_columns: set[int] = set()
    call_counts = {}
    for name in ARMS:
        case = cases[name]
        sim, policy = case[:2]
        ruler._reset(sim, policy, case[2], case[3])
        runtime = policy.vector_runtime
        original_features, original_step = runtime.features, sim.step
        index = 0

        def features(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal index
            expected, next_dist, next_id = legacy._cpu_feature_expected(sim, runtime)
            previous_dist = runtime.prev_dist.detach().cpu().numpy().copy()
            previous_id = runtime.prev_id.detach().cpu().numpy().copy()
            result = original_features(*args, **kwargs)
            array = result.detach().cpu().numpy()
            expected_all = np.tile(expected, (sim.E, 1))
            if array.shape != expected_all.shape:
                raise ParityMismatch("unexpected feature lifecycle row selection")
            if name == "R":
                baseline_features.append(array.copy())
            else:
                if array.tobytes() != expected_all.tobytes():
                    raise ParityMismatch(name + " lifecycle feature bytes differ from CPU")
                store = sim.alive.detach().cpu().numpy().copy()
                mask = kwargs.get("memory_mask")
                if mask is not None:
                    store &= mask.detach().cpu().numpy().reshape(store.shape)
                if not kwargs.get("update_memory", True):
                    store[:] = False
                want_dist = np.where(store, np.tile(next_dist, (sim.E, 1)), previous_dist)
                want_id = np.where(store, np.tile(next_id, (sim.E, 1)), previous_id)
                if (
                    legacy._bytes(runtime.prev_dist) != want_dist.tobytes()
                    or legacy._bytes(runtime.prev_id) != want_id.tobytes()
                ):
                    raise ParityMismatch(name + " carry differs from CPU")
                old = baseline_features[index]
                changed = set(
                    map(int, np.argwhere(old.view(np.uint32) != array.view(np.uint32))[:, 1])
                )
                if not changed <= set(REPAIR_COLUMNS):
                    raise ParityMismatch("undeclared geometry repair column")
                repair_columns.update(changed)
            index += 1
            return result

        def step(actions: torch.Tensor) -> Any:
            if name == "R":
                tape.append(actions.clone())
            if len(tape) != 1:
                raise ParityMismatch("unexpected tick/action tape")
            return original_step(tape[0])

        with (
            _arm(name),
            patch.object(runtime, "features", features),
            patch.object(sim, "step", step),
        ):
            ruler._full_tick(sim, policy, case[4], 0)
            a1_prof.sync(device)
        if index != len(baseline_features):
            raise ParityMismatch("feature call count differs")
        call_counts[name] = index
        final[name] = _snapshot(case)
    if not (final["R"]["sim"] == final["G"]["sim"] == final["GV"]["sim"]):
        raise ParityMismatch("frozen-device mechanics/config/lifecycle/RNG inventory differs")
    # G/GV isolate only v8 construction; every resulting field must agree.
    if final["G"] != final["GV"]:
        raise ParityMismatch("G/GV complete output/carry/config inventory differs")
    # R differs only in enumerated CPU repair bytes and their direct caches.
    # Validate the remaining policy/runtime inventory rather than hiding it.
    for key in ("policy", "collection"):
        if final["R"][key] != final["G"][key]:
            raise ParityMismatch("unrelated policy state differs")
    allowed = {"prev_dist", "prev_id", "_carry_state", "sel_state"}
    r, g = final["R"]["runtime"], final["G"]["runtime"]
    if set(r) != set(g) or any(r[k] != g[k] for k in r if k not in allowed):
        raise ParityMismatch("unrelated runtime state differs")
    return {
        "feature_call_cpu_parity": True,
        "same_action_mechanics_byte_identical": True,
        "full_inventory_compared": True,
        "g_gv_byte_identical": True,
        "verified_lifecycle_feature_calls": call_counts,
        "observed_declared_cpu_repair_columns": sorted(repair_columns),
        "inventory": {k: list(v) for k, v in final["G"].items()},
        "inventory_instrumentation_exclusions": sorted(INSTRUMENTATION_FIELDS),
    }


STAGES = {
    "vector61_non_flood": ("vector61_other",),
    "flood": ("vector61_flood_fill",),
    "b8_v8_bundle": ("b8_v8_features",),
    "opponent_v8_bundle": ("opponent_v8_bundle",),
    "v8_combined": ("b8_v8_features", "opponent_v8_bundle"),
    "movement_events": ("movement_collision_events",),
}


def statistics_for(arms: dict[str, Any], device: str) -> dict[str, Any]:
    clock = "exclusive_cuda_elapsed_seconds" if device == "cuda" else "exclusive_host_seconds"
    result = {}
    for label, components in STAGES.items():
        result[label] = {
            arm: statistics.median(
                sum(r[clock][k] for k in components) for r in arms[arm]["stages"]
            )
            for arm in ARMS
        }
    result["remaining"] = {
        arm: statistics.median(
            sum(
                v for k, v in row[clock].items() if k not in {x for c in STAGES.values() for x in c}
            )
            for row in arms[arm]["stages"]
        )
        for arm in ARMS
    }
    return result


def decide(row: dict[str, Any], *, enforce_speed: bool = True) -> str:
    if row.get("verdict") == "INVALID" or not all(
        row.get(k) is True
        for k in (
            "feature_call_cpu_parity",
            "same_action_mechanics_byte_identical",
            "full_inventory_compared",
            "g_gv_byte_identical",
            "state_carry_byte_identical",
        )
    ):
        return "INVALID"
    if enforce_speed and row["worlds"] == 1024:
        stage = row["stage_seconds"]["vector61_non_flood"]
        plain = {a: statistics.median(row[a]["plain_ticks"]) for a in ARMS}
        if stage["R"] / stage["G"] < 2 or plain["R"] / plain["G"] < 2:
            return "STOP_SPLIT"
    return "PASS_CELL"


def select(rows: list[dict[str, Any]], *, canary: bool = False) -> str:
    outcomes = [decide(r, enforce_speed=not canary) for r in rows]
    if "INVALID" in outcomes:
        return "INVALID"
    # Missing coverage preceding a claimed planned stop is never a scientific stop.
    if "STOP_SPLIT" in outcomes:
        return "STOP_SPLIT"
    expected = [
        (m, w)
        for w in ((64,) if canary else (1024, 256, 64))
        for m in (("scripted", "frozen", "mixed") if canary else ("frozen", "mixed", "scripted"))
    ]
    if [(r["mix"], r["worlds"]) for r in rows] != expected:
        return "INCOMPLETE"
    if canary:
        return "PASS_CANARY"
    keep = all(
        (
            all(v < g for v, g in zip(r["GV"]["plain_ticks"], r["G"]["plain_ticks"]))
            if r["worlds"] == 1024
            else statistics.median(r["GV"]["plain_ticks"])
            <= statistics.median(r["G"]["plain_ticks"])
        )
        for r in rows
    )
    return "SELECT_GV" if keep else "SELECT_G"


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
    cases = {}
    try:
        cases = {a: _case(checkpoint, mix, worlds, device, a) for a in ARMS}
        oracles = {"G": _oracles(cases["G"], output, mix, worlds)}
        oracles["GV"] = _oracles(cases["GV"], output, mix, worlds, oracles["G"]["recorded_fixture"])
        proof = _proof(cases, device)
        arms = {a: {"plain_ticks": [], "digests": [], "stages": []} for a in ARMS}

        def check() -> None:
            if time.perf_counter() >= deadline:
                raise TimeoutError("V8 split packet wall expired")

        for case in cases.values():
            for _ in range(warmups):
                check()
                _tick(case, device)
        for repeat in range(repeats):
            for name in ORDERS[repeat]:
                check()
                elapsed, digest = _tick(cases[name], device)
                timer = a1_prof.ExclusiveTiming(device)
                check()
                wall, event_digest = _tick(cases[name], device, timer)
                if digest != event_digest:
                    raise ParityMismatch("plain/event full inventory differs")
                arms[name]["plain_ticks"].append(elapsed)
                arms[name]["digests"].append(digest)
                arms[name]["stages"].append(
                    {"repeat": repeat, "wall_seconds": wall, **timer.result()}
                )
                print(
                    json.dumps(
                        {
                            "event": "repeat_done",
                            "mix": mix,
                            "worlds": worlds,
                            "arm": name,
                            "repeat": repeat,
                        }
                    ),
                    flush=True,
                )
        check()
        return {
            "mix": mix,
            "worlds": worlds,
            "repeats": repeats,
            "warmups": warmups,
            "arm_orders": [list(ORDERS[i]) for i in range(repeats)],
            **arms,
            **proof,
            "cpu_oracle": oracles,
            "recorded_fixture": oracles["G"]["recorded_fixture"],
            "state_carry_byte_identical": arms["G"]["digests"] == arms["GV"]["digests"],
            "stage_seconds": statistics_for(arms, device),
            "cpu_closed_loop_identity": False,
        }
    finally:
        cases.clear()
        ruler._release_cell(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke and args.canary or (args.device != "cuda" and not args.smoke):
        raise ValueError("CPU is smoke only; modes are exclusive")
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("V8 split requires A100")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("positive finite wall required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    commit = source_commit()
    ns = units.namespace(commit, smoke=args.smoke, canary=args.canary)
    allocation = units.allocation(smoke=args.smoke, canary=args.canary)
    prior = units.load_ledger(args.ledger, ns, allocation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from research.gpu_sim_spike.v8_split_discrepancies import probe

    discrepancy = probe(args.checkpoint, args.device)
    expected_path = os.environ.get("V8_SPLIT_EXPECTED_DISCREPANCIES")
    if args.device == "cuda" and not expected_path:
        raise ParityMismatch("native admission requires frozen discrepancy input")
    if expected_path:
        expected = json.loads(Path(os.environ["COLAB_INPUT_DIR"], expected_path).read_text())
        if discrepancy != expected:
            raise ParityMismatch(
                "native inherited discrepancy bytes differ from frozen Mac register"
            )
    units.publish_json(args.output_dir / "discrepancy-register.json", discrepancy)
    rows, complete = [], []
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
                    1 if args.smoke or args.canary else 3,
                    0 if args.smoke else 1 if args.canary else 3,
                    started + args.seconds,
                    args.output_dir,
                )
            except ParityMismatch as exc:
                value = {
                    "mix": mix,
                    "worlds": int(world[1:]),
                    "verdict": "INVALID",
                    "reason": str(exc),
                }
            value["verdict"] = decide(value, enforce_speed=not (args.smoke or args.canary))
            units.publish_unit(args.output_dir, ns, allocation, unit, value)
        complete.append(unit)
        rows.append(value)
        if decide(value, enforce_speed=not (args.smoke or args.canary)) != "PASS_CELL":
            break
    status = "PREFLIGHT" if args.smoke else "MEASURED_PENDING_MAC_REVIEW"
    if any(r["verdict"] != "PASS_CELL" for r in rows):
        status = next(r["verdict"] for r in rows if r["verdict"] != "PASS_CELL")
    result = {
        "schema": "gpu-sim-spike-v8-split-bench/v1",
        "status": status,
        "commit": commit,
        "source_sha256": source_hashes(),
        "device": args.device,
        "namespace": ns,
        "allocation": list(allocation),
        "completed_units": complete,
        "cells": rows,
        "skipped_verified_units": sorted(prior),
        "elapsed_seconds": time.perf_counter() - started,
        "wall_cap_seconds": args.seconds,
        "platform": platform.platform(),
        "cpu_closed_loop_identity": False,
    }
    units.publish_json(args.output_dir / "v8-split-bench.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "output-dir", "ledger"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--canary", action="store_true")
    args = p.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        units.publish_json(
            args.output_dir / "v8-split-failure.json",
            {
                "status": "INVALID" if isinstance(exc, ParityMismatch) else "INCOMPLETE",
                "error": str(exc),
                "type": type(exc).__name__,
            },
        )
        raise
    if result["status"] not in {"PREFLIGHT", "MEASURED_PENDING_MAC_REVIEW"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
