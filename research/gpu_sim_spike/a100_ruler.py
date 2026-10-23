"""A1's repaired, full-tick A100 ruler.

This is intentionally separate from the earlier E1 dense-label benchmark.  Its
unit is one resident six-snake world tick: reset, policy forwards, vector61/v8,
movement, food, corpse handling, observation paint, and collection writes.  The
small E=6 control is run with the unrepaired corpse implementation in the same
allocation; it is a causal control, not a throughput result for the engine.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import realistic_benchmark as realistic
from research.gpu_sim_spike.benchmark import source_commit, source_hashes, write_receipt

SHAPES = (6, 256, 1024)
MIXES = realistic.MIXES
STAGES = (
    "movement",
    "deaths_corpse",
    "food",
    "vector61",
    "v8_floods",
    "opponent_forwards",
    "b8_forward",
    "observation_paint",
    "collection",
    "reset",
)


class RulerError(RuntimeError):
    """The measured package cannot support A1's fixed conclusion rules."""


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _memory_now(device: str) -> dict[str, int | None]:
    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None, "free_bytes": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _release_cell(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _p95(values: list[float]) -> float:
    return realistic._p95(values)


def _set_event_path(sim: Any, repaired: bool) -> None:
    """Bind the E=6 control to the immutable pre-repair implementation."""
    if repaired:
        return
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.device_sim_frozen import DeviceSim as FrozenDeviceSim

    # FrozenDeviceSim predates instrumentation, while DevicePolicy and this
    # harness use the harmless no-op/off timing context even for the control.
    class RulerControl(FrozenDeviceSim):
        stage = DeviceSim.stage
        set_instrumentation = DeviceSim.set_instrumentation
        consume_timing = DeviceSim.consume_timing

    sim.__class__ = RulerControl


def _set_instrumented(sim: Any, enabled: bool) -> None:
    if hasattr(sim, "set_instrumentation"):
        sim.set_instrumentation(enabled)
        return
    raise RulerError("DeviceSim lacks A1's instrumented stage/launch interface")


def _consume_instrumentation(sim: Any) -> dict[str, Any]:
    if not hasattr(sim, "consume_timing"):
        raise RulerError("DeviceSim lacks A1's instrumented stage/launch receipt interface")
    value = sim.consume_timing()
    if not isinstance(value, dict):
        raise RulerError("DeviceSim instrumentation receipt is not an object")
    stages = value.get("stage_timers")
    launches = value.get("launch_counts")
    host_stages = value.get("host_stages")
    if (
        not isinstance(stages, dict)
        or not isinstance(launches, dict)
        or not isinstance(host_stages, dict)
    ):
        raise RulerError(
            "DeviceSim instrumentation lacks stage_timers, launch_counts, or host_stages"
        )
    missing = set(STAGES) - set(stages)
    if missing:
        raise RulerError("DeviceSim instrumentation omitted stages: " + ", ".join(sorted(missing)))
    if any(not isinstance(stages[name], (int, float)) or stages[name] < 0 for name in STAGES):
        raise RulerError("DeviceSim instrumentation has invalid stage seconds")
    if any(not isinstance(launches.get(name), int) or launches[name] < 0 for name in STAGES):
        raise RulerError("DeviceSim instrumentation has invalid launch counts")
    return {
        "stage_seconds": {name: float(stages[name]) for name in STAGES},
        "launch_counts": {name: int(launches[name]) for name in STAGES},
        "host_stages": {name: float(host_stages.get(name, 0.0)) for name in STAGES},
    }


def _build_case(checkpoint: Path, mix: str, worlds: int, device: str) -> tuple[Any, Any, Any, Any]:
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim

    world = realistic._long_world(checkpoint, mix, (800, 900, 1000, 1100, 1300, 1500))
    one_sim = DeviceSim.from_reference(world.sim, device=device)
    one_policy = DevicePolicy.from_world(world, device=device, sim=one_sim)
    sim = one_sim.repeat_interleave(worlds)
    policy = one_policy.repeat_interleave(worlds)
    expected = torch.tensor((800, 900, 1000, 1100, 1300, 1500), device=sim.device)[None].expand(
        worlds, -1
    )
    if sim.E != worlds or not bool(
        (sim.alive.all() & sim.length.eq(expected).all()).detach().cpu()
    ):
        raise RulerError("ruler did not allocate the requested six-long-body resident world shape")
    sim._prepared = False
    policy.vector_prepared = False
    return sim, policy, sim.clone(), policy.repeat_interleave(1)


def _reset(sim: Any, policy: Any, sim_template: Any, policy_template: Any) -> None:
    realistic._resident_reset(sim, sim_template)
    realistic._resident_reset(policy.vector_runtime, policy_template.vector_runtime)
    policy.vector_prepared = policy_template.vector_prepared


def _collection(ticks: int, worlds: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "length": torch.empty((ticks, worlds), dtype=torch.long, device=device),
        "alive": torch.empty((ticks, worlds), dtype=torch.bool, device=device),
        "action": torch.empty((ticks, worlds), dtype=torch.long, device=device),
        "qmax": torch.empty((ticks, worlds), dtype=torch.float32, device=device),
    }


@torch.inference_mode()
def _full_tick(sim: Any, policy: Any, collection: dict[str, torch.Tensor], index: int) -> None:
    """Execute the normal resident tick and account collector writes as a stage."""
    sim.prepare()
    hero_q, _legal, hero = policy.evaluate(sim)
    actions = policy.actions(sim, hero)
    sim.step(actions)
    policy.after_step(sim)
    with sim.stage("collection"):
        collection["length"][index].copy_(sim.length[:, 0])
        collection["alive"][index].copy_(sim.alive[:, 0])
        collection["action"][index].copy_(actions[:, 0])
        collection["qmax"][index].copy_(hero_q.max(dim=1).values)


def _measure_cell(
    *,
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    ticks: int,
    warmups: int,
    repaired: bool,
    instrumented: bool,
    deadline: float,
) -> dict[str, Any]:
    """Measure one shape/mix with all worlds resident concurrently."""
    samples: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    setup_seconds = 0.0
    for repeat in range(repeats):
        if time.perf_counter() >= deadline:
            raise TimeoutError("A1 ruler exhausted its bound before a repeat")
        _release_cell(device)
        memory_before = _memory_now(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sim = policy = sim_template = policy_template = collection = None
        try:
            built = time.perf_counter()
            sim, policy, sim_template, policy_template = _build_case(
                checkpoint, mix, worlds, device
            )
            setup_seconds += time.perf_counter() - built
            _set_event_path(sim, repaired)
            _set_instrumented(sim, instrumented)
            collection = _collection(ticks, worlds, sim.device)
            for warmup in range(warmups):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("A1 ruler exhausted its bound during warmup")
                with sim.stage("reset"):
                    _reset(sim, policy, sim_template, policy_template)
                _full_tick(sim, policy, {key: value[:1] for key, value in collection.items()}, 0)
                if instrumented:
                    _consume_instrumentation(sim)
                if warmup + 1 == warmups:
                    _sync(device)
            tick_seconds: list[float] = []
            for tick in range(ticks):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("A1 ruler exhausted its bound during a measured tick")
                # A full-tick ruler includes the resident reset.  Start before it
                # and synchronize after collection, so neither reset nor queued
                # CUDA work disappears from the endpoint measurement.
                started = time.perf_counter()
                with sim.stage("reset"):
                    _reset(sim, policy, sim_template, policy_template)
                _full_tick(sim, policy, collection, tick)
                _sync(device)
                tick_seconds.append(time.perf_counter() - started)
                if instrumented:
                    stage_rows.append(_consume_instrumentation(sim))
            samples.append(
                {
                    "repeat": repeat,
                    "tick_seconds": tick_seconds,
                    "median_tick_seconds": statistics.median(tick_seconds),
                    "p95_tick_seconds": _p95(tick_seconds),
                    "complete_steps_per_second": worlds * ticks / sum(tick_seconds),
                    "peak_device_memory": realistic._peak_memory(device),
                    "memory_before_repeat": memory_before,
                    "collection_checksum": int(collection["length"].sum().detach().cpu()),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "a1_ruler_repeat_complete",
                        "mix": mix,
                        "worlds": worlds,
                        "repeat": repeat + 1,
                        "repeats": repeats,
                        "complete_steps_per_second": samples[-1]["complete_steps_per_second"],
                    }
                ),
                flush=True,
            )
        finally:
            # Drop live tensors before empty_cache; the allocator cannot release
            # referenced storage. This also runs for OOM/warmup/trace failures.
            del collection, policy_template, sim_template, policy, sim
            _release_cell(device)
        samples[-1]["memory_after_release"] = _memory_now(device)
    ticks_all = [value for sample in samples for value in sample["tick_seconds"]]
    result = {
        "mix": mix,
        "requested_worlds": worlds,
        "actual_concurrent_worlds": worlds,
        "bodies": [800, 900, 1000, 1100, 1300, 1500],
        "five_real_opponents": True,
        "repaired_event_path": repaired,
        "warmed_repeats": repeats,
        "ticks_per_repeat": ticks,
        "samples": samples,
        "median_tick_seconds": statistics.median(ticks_all),
        "p95_tick_seconds": _p95(ticks_all),
        "median_complete_steps_per_second": statistics.median(
            sample["complete_steps_per_second"] for sample in samples
        ),
        "pooled_complete_steps_per_second_diagnostic": worlds * len(ticks_all) / sum(ticks_all),
        "peak_device_memory": {
            name: (
                max(sample["peak_device_memory"][name] for sample in samples)
                if samples[0]["peak_device_memory"][name] is not None
                else None
            )
            for name in samples[0]["peak_device_memory"]
        },
        "setup_seconds": setup_seconds,
    }
    if instrumented:
        stage_seconds = {
            name: sum(row["stage_seconds"][name] for row in stage_rows) for name in STAGES
        }
        launches = {name: sum(row["launch_counts"][name] for row in stage_rows) for name in STAGES}
        total = sum(stage_seconds.values())
        result["instrumented"] = {
            "samples": len(stage_rows),
            "stage_seconds": stage_seconds,
            "stage_shares": {
                name: stage_seconds[name] / total if total else 0.0 for name in STAGES
            },
            "launch_counts": launches,
            "host_stages": {
                name: sum(row["host_stages"][name] for row in stage_rows) for name in STAGES
            },
            "unaccounted_e2e_seconds": max(
                0.0,
                statistics.median(sample["median_tick_seconds"] for sample in samples)
                - total / max(1, len(stage_rows)),
            ),
        }
    return result


def classify(
    rows: list[dict[str, Any]], control: dict[str, Any], parity: dict[str, Any]
) -> dict[str, Any]:
    """Apply A1's frozen tiers without turning a parity self-check into a pass bar."""
    required = {(mix, worlds) for mix in MIXES for worlds in SHAPES}
    observed = {(row.get("mix"), row.get("requested_worlds")) for row in rows}
    if observed != required or len(rows) != len(required):
        return {"tier": "INCOMPLETE", "reason": "A1 requires all nine shape/mix cells"}
    if parity.get("classification") == "bug":
        return {"tier": "INVALID", "reason": "unexplained CPU action-tape mismatch"}
    speedup = control["median_tick_seconds"] / next(
        row["median_tick_seconds"]
        for row in rows
        if row["mix"] == control["mix"] and row["requested_worlds"] == 6
    )
    by_mix = {
        row["mix"]: row["median_complete_steps_per_second"]
        for row in rows
        if row["requested_worlds"] == 1024
    }
    event_path = "PASS" if speedup >= 5 else "REJECTED_TRACE_STAGE_GE_50_REQUIRED"
    if any(rate < 2000 for rate in by_mix.values()):
        return {
            "tier": "STOP_ENGINE",
            "e6_speedup": speedup,
            "event_path_hypothesis": event_path,
            "e1024_by_mix": by_mix,
        }
    if speedup < 5:
        return {
            "tier": "EVENT_PATH_REJECTED",
            "e6_speedup": speedup,
            "event_path_hypothesis": event_path,
            "e1024_by_mix": by_mix,
        }
    if all(rate >= 20000 for rate in by_mix.values()):
        return {
            "tier": "FUND_WP1C_WP2",
            "e6_speedup": speedup,
            "event_path_hypothesis": event_path,
            "e1024_by_mix": by_mix,
        }
    return {
        "tier": "FUND_WP1B",
        "e6_speedup": speedup,
        "event_path_hypothesis": event_path,
        "e1024_by_mix": by_mix,
    }


def _parity_check(args: argparse.Namespace, deadline: float) -> dict[str, Any]:
    # Keep this tiny action-tape classification single-process on the Mac.
    # The reference and candidate have independent simulator/policy state.
    from research.gpu_sim_spike.benchmark import _candidate_json
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.parity import compare_decision
    from research.gpu_sim_spike.reference import cpu_decision, fixture_worlds, rescore_label
    from research.gpu_sim_spike.rollout import dense_decision

    _, parity_world = fixture_worlds(args.checkpoint, ["tiny-scripted"])[0]
    reference = cpu_decision(parity_world, depth=1)
    parity_sim = DeviceSim.from_reference(parity_world.sim, device=args.device)
    parity_policy = DevicePolicy.from_world(parity_world, device=args.device, sim=parity_sim)
    values = dense_decision(parity_sim, parity_policy, depth=1, deadline=deadline)
    candidate = _candidate_json(values, depth=1, snakes=parity_sim.S)
    comparison = compare_decision(reference, candidate)
    rescored = {
        str(action): rescore_label(parity_world, tape, depth=1)
        for action, tape in enumerate(candidate["action_trace"])
    }
    mechanics_mismatches = [
        action
        for action, outcome in candidate["outcomes"].items()
        if outcome["alive_steps"] != rescored[str(action)]["alive_steps"]
        or outcome["mass_integral"] != rescored[str(action)]["mass_integral"]
    ]
    if mechanics_mismatches:
        comparison["classification"] = "bug"
    comparison["same_tape_mechanics_mismatches"] = mechanics_mismatches
    parity = {
        "classification": comparison["classification"],
        "recorded_action_tape": "a1-units/parity-tape.json",
        "comparison": comparison,
        "scope": "one tiny H1 tape; classification only, not engine qualification",
    }
    return {
        "parity": parity,
        "tape": {
            "reference": reference,
            "candidate": candidate,
            "comparison": comparison,
            "cpu_rescored_action_tapes": rescored,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RulerError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from research.gpu_sim_spike import a1_units

    shapes = (6, 1024) if args.smoke else SHAPES
    mixes = MIXES
    repeats = 1 if args.smoke else 3
    ticks = 1 if args.smoke else 8
    warmups = 0 if args.smoke else args.warmup_ticks
    deadline = time.perf_counter() + args.seconds
    setup_started = time.perf_counter()
    ns = a1_units.namespace(source_commit(), args.hardware, args.smoke)
    allocated = a1_units.allocation(args.smoke)
    prior = a1_units.load_ledger(args.ledger, ns, allocated)
    rows, trace = [], []

    def unit(unit_id: str, produce: Any) -> dict[str, Any]:
        args.active_unit = unit_id
        print(
            json.dumps({"event": "a1_unit_start", "unit_id": unit_id, "resumed": unit_id in prior}),
            flush=True,
        )
        if unit_id in prior:
            value = prior[unit_id]
        else:
            value = produce()
            if time.perf_counter() >= deadline:
                raise TimeoutError("A1 unit exceeded its fixed wall bound")
        a1_units.publish_unit(args.output.parent, ns, allocated, unit_id, value)
        _release_cell(args.device)
        return value

    for mix in mixes:
        for worlds in shapes:

            def measure() -> dict[str, Any]:
                common = dict(
                    checkpoint=args.checkpoint,
                    mix=mix,
                    worlds=worlds,
                    device=args.device,
                    repaired=True,
                    deadline=deadline,
                )
                measured = _measure_cell(
                    **common, repeats=repeats, ticks=ticks, warmups=warmups, instrumented=False
                )
                attributed = _measure_cell(
                    **common, repeats=1, ticks=1, warmups=0, instrumented=True
                )
                return {"measured": measured, "trace": attributed}

            value = unit(f"cell-{mix.removeprefix('long-')}-e{worlds}", measure)
            rows.append(value["measured"])
            trace.append(value["trace"])
    measured_wall_seconds = time.perf_counter() - setup_started
    control = unit(
        "control-e6",
        lambda: _measure_cell(
            checkpoint=args.checkpoint,
            mix=mixes[0],
            worlds=6,
            device=args.device,
            repeats=1,
            ticks=1,
            warmups=0,
            repaired=False,
            instrumented=False,
            deadline=deadline,
        ),
    )
    parity = unit("parity-tape", lambda: _parity_check(args, deadline))["parity"]
    receipt: dict[str, Any] = {
        "schema": "gpu-sim-spike-a1-ruler/v1",
        "status": "PREFLIGHT" if args.device == "cpu" else "MEASURED_PENDING_INDEPENDENT_REVIEW",
        "device": args.device,
        "shapes": list(shapes),
        "mixes": list(mixes),
        "repaired_cells": rows,
        "unrepaired_e6_control": control,
        "instrumented_pass": trace,
        "host_stages": {
            name: sum(cell["instrumented"]["host_stages"][name] for cell in trace)
            for name in STAGES
        },
        "parity_self_check": parity,
        "construction_setup_seconds": sum(row["setup_seconds"] for row in rows),
        "measured_ruler_wall_seconds": measured_wall_seconds,
        "export_seconds": {"status": "SUPERVISOR_MEASURED_TERMINAL_RECEIPT", "value": None},
        "source_commit": source_commit(),
        "source_sha256": source_hashes(),
    }
    receipt["kill_tier"] = {
        "tier": "SMOKE_ONLY" if args.smoke else "SEALED_PENDING_MAC_AGGREGATION"
    }
    receipt["namespace"] = ns
    receipt["hardware"] = args.hardware
    receipt["verified_units_reused"] = sorted(prior)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-ticks", type=int, default=3)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--hardware", choices=("A100-40GB", "A100-80GB", "H100-80GB"), default="A100-40GB"
    )
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_ticks < 0 or args.seconds <= 0:
        parser.error("warmup ticks must be nonnegative and seconds must be positive")
    if args.output.exists():
        parser.error("output already exists; ruler receipts are immutable")
    started = time.perf_counter()
    try:
        receipt = run(args)
    except (RulerError, RuntimeError, ValueError, TimeoutError) as exc:
        receipt = {
            "schema": "gpu-sim-spike-a1-ruler/v1",
            "status": "INCOMPLETE",
            "verdict": "NO_CONTINUE",
            "reason": str(exc),
            "failure_traceback": traceback.format_exc(),
            "active_unit": getattr(args, "active_unit", None),
            "memory_at_failure": _memory_now(args.device),
            "peak_memory_at_failure": (
                realistic._peak_memory(args.device)
                if args.device == "cpu" or torch.cuda.is_available()
                else None
            ),
            "device": args.device,
            "source_commit": source_commit(),
            "source_sha256": source_hashes(),
        }
    receipt["entrypoint_run_wall_seconds"] = time.perf_counter() - started
    export_started = time.perf_counter()
    write_receipt(args.output, receipt)
    write_receipt(
        args.output.with_name("host-export.json"),
        {
            "schema": "gpu-sim-spike-a1-local-export/v1",
            "local_export_seconds": time.perf_counter() - export_started,
            "scope": "local JSON/sidecars only; provider export in supervisor terminal receipt",
        },
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
    if receipt["status"] == "INCOMPLETE" or receipt.get("kill_tier", {}).get("tier") == "INVALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
