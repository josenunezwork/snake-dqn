"""Production-shape resident-world throughput measurement for E1.

This deliberately measures the normal policy/simulator tick rather than the
serial six-way label branch.  A case owns every requested environment on one
device at once; no Python loop may turn a requested world count into a sequence
of smaller batches.
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch

MIXES = ("long-scripted", "long-frozen", "long-mixed")
L4_FLOOR = 20_000


def comma_ints(value: str, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    """Parse a strict positive comma-separated integer argument."""
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < minimum for item in parsed):
        raise ValueError(f"{name} values must be >= {minimum}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


def validate_shape(
    worlds: Iterable[int], lengths: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Require all requested production scales and six non-padding long bodies."""
    world_tuple, length_tuple = tuple(worlds), tuple(lengths)
    if not world_tuple or any(world < 1 for world in world_tuple):
        raise ValueError("realistic worlds must be positive")
    if len(length_tuple) != 6 or min(length_tuple) < 800 or max(length_tuple) != 1500:
        raise ValueError("realistic lengths require six bodies, all >=800, maximum exactly1500")
    return world_tuple, length_tuple


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_memory(device: str) -> dict[str, int | None]:
    if device != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _long_world(checkpoint: Path, case_name: str, lengths: tuple[int, ...]) -> Any:
    """Build one real mix then inject a fixed six-slot long-body world."""
    from research.gpu_sim_spike.reference import fixture_worlds
    from research.t34_exit_b8.trajectory import prepare_world
    from src.simd_env.grid_scenarios import inject_serpentines

    _, world = fixture_worlds(checkpoint, [case_name])[0]
    if int(world.sim.S) != len(lengths):
        raise ValueError("realistic benchmark requires five real opponents plus the hero")
    # Six bands have two empty rows between them.  The production 145x83 map
    # needs eleven body rows for L1500 (141 * 11), which still fits six bands.
    width = int(world.sim.grid_w) - 4
    box_height = max(8, (max(lengths) + width - 1) // width)
    if 2 + (len(lengths) - 1) * (box_height + 2) + box_height > int(world.sim.grid_h) - 2:
        raise ValueError("realistic long-body layout does not fit this arena")
    inject_serpentines([world.sim], list(lengths), box_height=box_height)
    world.runtime.bind(world.sim, world.runtime.controlled.copy())
    world.sim._policy_phase_prepared = False
    prepare_world(world)
    return world


def _step(sim: Any, policy: Any, collection: dict[str, torch.Tensor], index: int) -> None:
    """One fully resident policy/simulator tick plus representative collection writes."""
    sim.prepare()
    hero_q, _legal, hero = policy.evaluate(sim)
    actions = policy.actions(sim, hero)
    sim.step(actions)
    policy.after_step(sim)
    # These are the collector's per-world fields, retained on the device.  The
    # write is intentional workload, not a host-side accounting shortcut.
    collection["length"][index].copy_(sim.length[:, 0])
    collection["alive"][index].copy_(sim.alive[:, 0])
    collection["action"][index].copy_(actions[:, 0])
    collection["qmax"][index].copy_(hero_q.max(dim=1).values)


def _resident_reset(destination: Any, template: Any) -> None:
    """Copy a same-shaped device template without allocating or touching host state."""
    for key, value in destination.__dict__.items():
        source = getattr(template, key, None)
        if isinstance(value, torch.Tensor) and isinstance(source, torch.Tensor):
            if value.shape == source.shape and value.device == source.device:
                value.copy_(source)
    # DeviceSim's selector seam is intentionally a Python boolean.  It must
    # return to the template state too, otherwise a reset row can skip its
    # food/respawn preparation on the next timed tick.
    if hasattr(destination, "_prepared") and hasattr(template, "_prepared"):
        destination._prepared = template._prepared


def _p95(values: list[float]) -> float:
    """Nearest-rank p95; three repeats/ticks deliberately select the slowest."""
    if not values:
        raise ValueError("p95 needs values")
    return float(sorted(values)[max(0, (95 * len(values) + 99) // 100 - 1)])


def floor_status(cases: list[dict[str, Any]], *, l4: bool) -> str:
    """Require every production shape; serial/tiny/missing rows cannot pass."""
    if not l4:
        return "PRECHECK_ONLY_NOT_L4_DATA"
    required = {(mix, worlds) for mix in MIXES for worlds in (256, 1024, 4096)}
    by_key = {(case["case"], case["requested_worlds"]): case for case in cases}
    if any(case["requested_worlds"] != case["actual_concurrent_worlds"] for case in cases):
        return "STOP_ENGINE_THESIS_THIS_WEEK"
    measured = [
        case["median_full_world_steps_per_second"]
        for key, case in by_key.items()
        if key in required and key[1] in (1024, 4096)
    ]
    if any(rate < L4_FLOOR for rate in measured):
        return "STOP_ENGINE_THESIS_THIS_WEEK"
    if not required <= by_key.keys() or len(by_key) != len(cases):
        return "INCOMPLETE_LONG_BODY_COVERAGE"
    return "FLOOR_MET_PENDING_COST_AND_PARITY_AUDIT"


def run_realistic(args: Any) -> dict[str, Any]:
    """Run all mixes/scales with reset, resident collection writes, and p95 ticks."""
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim

    requested_worlds, lengths = validate_shape(args.realistic_worlds, args.realistic_lengths)
    if args.realistic_ticks < 1 or args.realistic_warmup_ticks < 0 or args.realistic_repeats < 3:
        raise ValueError("realistic ticks must be positive, warmup nonnegative, repeats >= 3")
    if args.policy_backend != "device":
        raise ValueError("realistic mode requires the resident device policy backend")
    # Mac CPU probes prove shape/setup only.  They never claim a 4096-world CPU
    # result and are explicitly excluded from the L4 floor.
    max_worlds = args.realistic_cpu_max_worlds if args.device == "cpu" else None
    cases: list[dict[str, Any]] = []
    deadline = getattr(args, "_absolute_deadline", time.perf_counter() + args.seconds)
    for case_name in MIXES:
        for requested in requested_worlds:
            actual = min(requested, max_worlds) if max_worlds is not None else requested
            if actual < 1:
                raise ValueError("realistic CPU max worlds must be positive")
            repeats: list[dict[str, Any]] = []
            for repeat in range(args.realistic_repeats):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("realistic benchmark exhausted wall budget; no retry")
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                world = _long_world(args.checkpoint, case_name, lengths)
                reset_started = time.perf_counter()
                one_sim = DeviceSim.from_reference(world.sim, device=args.device)
                one_policy = DevicePolicy.from_world(world, device=args.device, sim=one_sim)
                sim = one_sim.repeat_interleave(actual)
                policy = one_policy.repeat_interleave(actual)
                # A full tick includes frame/food/respawn preparation. Source
                # S0 fixtures arrive at an already prepared selector seam;
                # resetting that seam would omit lifecycle work every tick.
                sim._prepared = False
                policy.vector_prepared = False
                # Allocation is complete before timing: every environment is one
                # live long-body world, not dead-row padding or serial chunks.
                expected_lengths = torch.tensor(lengths, device=sim.device)[None].expand(actual, -1)
                valid = (
                    sim.alive.all()
                    & sim.length.eq(expected_lengths).all()
                    & sim.seg_count.eq(expected_lengths).all()
                )
                if sim.E != actual or not bool(valid.detach().cpu()):
                    raise RuntimeError(
                        "realistic world allocation failed its long-body concurrency contract"
                    )
                collection = {
                    "length": torch.empty(
                        (args.realistic_ticks, actual), dtype=torch.long, device=sim.device
                    ),
                    "alive": torch.empty(
                        (args.realistic_ticks, actual), dtype=torch.bool, device=sim.device
                    ),
                    "action": torch.empty(
                        (args.realistic_ticks, actual), dtype=torch.long, device=sim.device
                    ),
                    "qmax": torch.empty(
                        (args.realistic_ticks, actual), dtype=torch.float32, device=sim.device
                    ),
                }
                sim_template = sim.clone()
                policy_template = policy.repeat_interleave(1)
                _sync(args.device)
                reset_seconds = time.perf_counter() - reset_started
                for warm in range(args.realistic_warmup_ticks):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("realistic benchmark exhausted wall budget; no retry")
                    _resident_reset(sim, sim_template)
                    _resident_reset(policy.vector_runtime, policy_template.vector_runtime)
                    policy.vector_prepared = policy_template.vector_prepared
                    _step(sim, policy, {key: value[:1] for key, value in collection.items()}, 0)
                    if warm + 1 == args.realistic_warmup_ticks:
                        _sync(args.device)
                tick_seconds: list[float] = []
                measured_started = time.perf_counter()
                for tick in range(args.realistic_ticks):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("realistic benchmark exhausted wall budget; no retry")
                    tick_started = time.perf_counter()
                    # The reset is a device-resident, full-shape copy and is
                    # timed as part of every production-shaped tick. It keeps
                    # a post-death row from becoming cheap dead-padding.
                    _resident_reset(sim, sim_template)
                    _resident_reset(policy.vector_runtime, policy_template.vector_runtime)
                    policy.vector_prepared = policy_template.vector_prepared
                    _step(sim, policy, collection, tick)
                    _sync(args.device)
                    tick_seconds.append(time.perf_counter() - tick_started)
                measured_seconds = time.perf_counter() - measured_started
                # Materialize only a tiny checksum scalar after timing.  This
                # verifies collection writes execute without making download a
                # fake component of every tick.
                collection_checksum = int(collection["length"].sum().detach().cpu())
                repeat_row = {
                    "repeat": repeat,
                    "requested_worlds": requested,
                    "actual_concurrent_worlds": actual,
                    "reset_seconds": reset_seconds,
                    "warmup_ticks_excluded": args.realistic_warmup_ticks,
                    "measured_ticks": args.realistic_ticks,
                    "tick_seconds": tick_seconds,
                    "p95_tick_seconds": _p95(tick_seconds),
                    "full_world_steps": actual * args.realistic_ticks,
                    "full_world_steps_per_second": actual * args.realistic_ticks / measured_seconds,
                    "collection_write": {"resident": True, "checksum": collection_checksum},
                    "peak_device_memory": _peak_memory(args.device),
                    "resident_reset": {
                        "per_measured_tick": True,
                        "reset_worlds": actual * args.realistic_ticks,
                        "template_device": str(sim.device),
                        "trajectory_note": "fixed-root reset per tick; conservative stress",
                    },
                }
                repeats.append(repeat_row)
                # A completed repeat remains independently recoverable when a
                # later scale hits the bounded deadline.
                from research.gpu_sim_spike.benchmark import write_receipt

                write_receipt(
                    args.output.parent / f"realistic-{case_name}-{requested}-{repeat:02d}.json",
                    repeat_row,
                )
                del (
                    collection,
                    policy_template,
                    sim_template,
                    policy,
                    sim,
                    one_policy,
                    one_sim,
                    world,
                )
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            rates = [entry["full_world_steps_per_second"] for entry in repeats]
            cases.append(
                {
                    "case": case_name,
                    "requested_worlds": requested,
                    "actual_concurrent_worlds": actual,
                    "long_body_lengths": list(lengths),
                    "five_real_opponents": True,
                    "resets_included": True,
                    "collection_writes_included": True,
                    "repeats": repeats,
                    "median_full_world_steps_per_second": statistics.median(rates),
                    "p95_tick_seconds": _p95(
                        [tick for entry in repeats for tick in entry["tick_seconds"]]
                    ),
                }
            )
    fully_concurrent = args.device == "cuda" and all(
        case["requested_worlds"] == case["actual_concurrent_worlds"] for case in cases
    )
    l4 = args.device == "cuda" and "L4" in torch.cuda.get_device_name(0)
    return {
        "schema": "gpu-sim-spike-realistic-throughput/v1",
        "unit": "full six-snake world-step",
        "device": args.device,
        "required_worlds": list(requested_worlds),
        "required_mixes": list(MIXES),
        "long_body_contract": {"min_length": min(lengths), "max_length": max(lengths)},
        "floor": {"l4_world_steps_per_second": L4_FLOOR, "status": floor_status(cases, l4=l4)},
        "cases": cases,
        "preflight_clamped": args.device == "cpu",
        "concurrency_verified": fully_concurrent,
    }
