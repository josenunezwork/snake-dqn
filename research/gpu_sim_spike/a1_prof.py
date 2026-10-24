"""A1-PROF: outcome-free full-tick attribution; never labels or gate evidence."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import torch

from research.gpu_sim_spike import a1_prof_units as units
from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike.benchmark import source_commit, source_hashes

PREFIX = "a1prof::"
# Bound stage attribution uncertainty to two percentage points of kernel time.
# This does not gate or adjust uninstrumented throughput or CUDA-event timing.
MAX_UNATTRIBUTED_KERNEL_TIME_SHARE = 0.02
STAGE_MAP = {
    "movement": "movement_collision_events",
    "deaths_corpse": "movement_collision_events",
    "food": "food_corpse",
    "vector61": "vector61_other",
    "v8_floods": "b8_v8_features",
    "observation_paint": "b8_v8_features",
    "b8_forward": "policy_forward",
    "opponent_forwards": "opponent_policies",
    "reset": "resident_reset",
    "collection": "collection",
}
STAGES = tuple(dict.fromkeys(STAGE_MAP.values())) + (
    "vector61_flood_fill",
    "opponent_v8_bundle",
    "rng",
    "unscoped",
)
OBSERVATION = {"vector61_other", "vector61_flood_fill", "b8_v8_features", "opponent_v8_bundle"}


def sync(device: str) -> None:
    """Finish device work at tick boundaries, outside individual stage scopes."""
    if device == "cuda":
        torch.cuda.synchronize()


class ExclusiveTiming:
    """Segment the default stream at every nested scope boundary without overlap.

    Events measure elapsed stream intervals (including launch starvation), not
    kernel-busy time. A separate Kineto pass measures kernels and copies.
    """

    def __init__(self, device: str, *, events: bool = True, annotations: bool = False) -> None:
        self.device, self.events, self.annotations = device, events, annotations
        self.stack: list[str] = []
        self.segments: list[tuple[str, float, Any, Any]] = []
        self.started = 0.0
        self.first: Any = None

    def _start(self) -> None:
        self.first = None
        if self.events and self.device == "cuda":
            self.first = torch.cuda.Event(enable_timing=True)
            self.first.record()
        self.started = time.perf_counter()

    def _finish(self) -> None:
        end = None
        elapsed = time.perf_counter() - self.started
        if self.first is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        self.segments.append((self.stack[-1], elapsed, self.first, end))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        name = STAGE_MAP.get(name, name)
        if name not in STAGES:
            raise ValueError(f"unregistered profiling stage: {name}")
        if self.stack:
            self._finish()
        self.stack.append(name)
        annotation = torch.profiler.record_function(PREFIX + name) if self.annotations else None
        if annotation is not None:
            annotation.__enter__()
        self._start()
        try:
            yield
        finally:
            self._finish()
            if annotation is not None:
                annotation.__exit__(None, None, None)
            self.stack.pop()
            if self.stack:
                self._start()

    def result(self) -> dict[str, Any]:
        host = dict.fromkeys(STAGES, 0.0)
        gpu = dict.fromkeys(STAGES, 0.0) if self.device == "cuda" and self.events else None
        for name, elapsed, start, end in self.segments:
            host[name] += elapsed
            if gpu is not None:
                gpu[name] += start.elapsed_time(end) / 1000
        return {"exclusive_host_seconds": host, "exclusive_cuda_elapsed_seconds": gpu}


@contextmanager
def instrument(sim: Any, timer: ExclusiveTiming) -> Iterator[None]:
    """Read-only call wrappers; preserve arguments, results, RNG and engine source."""
    from research.gpu_sim_spike import device_policy
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    def wrapper(method: Any, stage: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with timer.stage(stage):
                return method(*args, **kwargs)

        return wrapped

    original_v8 = device_policy.v8_decide

    def v8_bundle(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("features", False):
            return original_v8(*args, **kwargs)
        with timer.stage("opponent_v8_bundle"):
            return original_v8(*args, **kwargs)

    # Class-level wrappers include compacted food sub-batches' RNG calls.
    with (
        patch.object(sim, "stage", timer.stage),
        patch.object(device_policy, "v8_decide", v8_bundle),
        patch.object(DeviceSim, "_random_cells", wrapper(DeviceSim._random_cells, "rng")),
        patch.object(DeviceSim, "_append_many", wrapper(DeviceSim._append_many, "food_corpse")),
        patch.object(DeviceSim, "_append_food", wrapper(DeviceSim._append_food, "food_corpse")),
        patch.object(DeviceSim, "_compact_food", wrapper(DeviceSim._compact_food, "food_corpse")),
        patch.object(
            DeviceVectorRuntime,
            "_free_space",
            wrapper(DeviceVectorRuntime._free_space, "vector61_flood_fill"),
        ),
    ):
        yield


def profiler_summary(events: Any, device: str) -> dict[str, Any]:
    """Attribute each correlated kernel once to its nearest enclosing stage.

    PyTorch FunctionEvent.kernels contains directly correlated device activities;
    ancestors' inclusive device totals must not be summed with these activities.
    CPU runtime events carry the linked op's thread and therefore scope ancestry.
    """
    events = list(events)
    stages: dict[str, Any] = {
        stage: {
            "kernel_count": 0,
            "kernel_seconds": 0.0,
            "launch_api_count": 0,
            "launch_api_seconds": 0.0,
            "python_gap_seconds": 0.0,
            "self_device_memory_net_bytes": 0,
            "self_device_allocation_bytes": 0,
        }
        for stage in (*STAGES, "unattributed")
    }
    copies: dict[str, Any] = {}
    kernel_names: dict[str, Any] = {}
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        parent = event
        while parent is not None and not parent.name.startswith(PREFIX):
            parent = parent.cpu_parent
        name = parent.name[len(PREFIX) :] if parent is not None else "unscoped"
        target = stages[name]
        memory = getattr(event, "self_device_memory_usage", 0)
        target["self_device_memory_net_bytes"] += memory
        target["self_device_allocation_bytes"] += max(memory, 0)
        if event.name.startswith(PREFIX):
            target["python_gap_seconds"] += event.self_cpu_time_total / 1e6
        if event.name.startswith(("cudaLaunch", "cuLaunch")):
            target["launch_api_count"] += 1
            target["launch_api_seconds"] += event.cpu_time_total / 1e6
        for kernel in event.kernels:
            if kernel.name.startswith(("Memcpy", "Memset")):
                transfer = copies.setdefault(kernel.name, {"count": 0, "seconds": 0.0})
                transfer["count"] += 1
                transfer["seconds"] += kernel.duration / 1e6
            else:
                target["kernel_count"] += 1
                target["kernel_seconds"] += kernel.duration / 1e6
                key = name + ":" + kernel.name
                row = kernel_names.setdefault(key, {"count": 0, "seconds": 0.0})
                row["count"] += 1
                row["seconds"] += kernel.duration / 1e6
    observed = sum(row["kernel_count"] for row in stages.values())
    raw_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.name != "[memory]"
        and not event.name.startswith(("Memcpy", "Memset"))
    ]
    raw = len(raw_events)
    attributed_seconds = sum(row["kernel_seconds"] for row in stages.values())
    raw_seconds = sum(event.time_range.elapsed_us() for event in raw_events) / 1e6
    if device == "cuda" and (
        raw == 0
        or observed > raw
        or not math.isfinite(raw_seconds)
        or not math.isfinite(attributed_seconds)
        or raw_seconds <= 0
        or attributed_seconds < 0
        or attributed_seconds > raw_seconds + max(1e-12, raw_seconds * 1e-9)
    ):
        raise RuntimeError(
            "profiler kernel correlation invalid: "
            f"attributed={observed}, raw={raw}, "
            f"attributed_seconds={attributed_seconds}, raw_seconds={raw_seconds}"
        )
    missing = raw - observed if device == "cuda" else 0
    missing_seconds = max(0.0, raw_seconds - attributed_seconds) if device == "cuda" else 0.0
    missing_share = missing_seconds / raw_seconds if raw_seconds > 0 else 0.0
    stages["unattributed"].update(kernel_count=missing, kernel_seconds=missing_seconds)
    if missing or missing_seconds:
        kernel_names["unattributed:correlation_remainder"] = {
            "count": missing,
            "seconds": missing_seconds,
        }
    # Subtraction of second-valued floats can put an exact 2% gap a few ulps
    # above 0.02. Ignore only rounding at the boundary (absolute share 1e-12).
    if (
        device == "cuda"
        and missing_share > MAX_UNATTRIBUTED_KERNEL_TIME_SHARE
        and not math.isclose(
            missing_share, MAX_UNATTRIBUTED_KERNEL_TIME_SHARE, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise RuntimeError(
            "profiler kernel correlation gap exceeds time threshold: "
            f"attributed={observed}, raw={raw}, unattributed={missing}, "
            f"unattributed_seconds={missing_seconds}, raw_seconds={raw_seconds}, "
            f"unattributed_time_share={missing_share:.9f}, "
            f"maximum={MAX_UNATTRIBUTED_KERNEL_TIME_SHARE}"
        )
    return {
        "stages": stages,
        "kernel_count": observed + missing,
        "attributed_cuda_kernel_count": observed,
        "raw_cuda_kernel_count": raw,
        "unattributed_cuda_kernel_count": missing,
        "unattributed_cuda_kernel_count_share": missing / raw if raw else 0.0,
        "attributed_cuda_kernel_seconds": attributed_seconds,
        "raw_cuda_kernel_seconds": raw_seconds,
        "unattributed_cuda_kernel_seconds": missing_seconds,
        "unattributed_cuda_kernel_time_share": missing_share,
        "max_unattributed_cuda_kernel_time_share": MAX_UNATTRIBUTED_KERNEL_TIME_SHARE,
        "device_activities": copies,
        "host_device_transfers": {
            key: value
            for key, value in copies.items()
            if any(token in key.lower() for token in ("htod", "dtoh", "h2d", "d2h"))
        },
        "kernel_inventory": kernel_names,
        "cuda_available": device == "cuda",
        "notes": (
            "Python gap is scope self CPU time; launch API is separate, not added to GPU time. "
            "Unattributed is the raw minus correlated kernel remainder, never assigned to a stage. "
            "At most 2% unattributed kernel time bounds stage-share uncertainty to 2 percentage "
            "points; uninstrumented throughput and CUDA-event timings are independent."
        ),
    }


def metrics(baseline: list[float], timings: list[dict[str, Any]], worlds: int) -> dict[str, Any]:
    """Keep uninstrumented throughput distinct from instrumented attribution."""
    wall = statistics.mean(row["tick_wall_seconds"] for row in timings)
    host = {
        key: statistics.mean(row["exclusive_host_seconds"][key] for row in timings)
        for key in STAGES
    }
    cuda = None
    if timings[0]["exclusive_cuda_elapsed_seconds"] is not None:
        cuda = {
            key: statistics.mean(row["exclusive_cuda_elapsed_seconds"][key] for row in timings)
            for key in STAGES
        }
    attribution = cuda if cuda is not None else host
    observation = sum(attribution[key] for key in OBSERVATION)
    return {
        "complete_world_steps_per_second": worlds * len(baseline) / sum(baseline),
        "uninstrumented_mean_tick_seconds": statistics.mean(baseline),
        "instrumented_mean_tick_seconds": wall,
        "instrumentation_wall_ratio": wall / statistics.mean(baseline),
        "exclusive_host_seconds_per_tick": host,
        "exclusive_cuda_elapsed_seconds_per_tick": cuda,
        "boundary_and_sync_seconds_per_tick": wall - sum(attribution.values()),
        "host_boundary_seconds_per_tick": wall - sum(host.values()),
        "flood_fill_share_of_instrumented_tick": attribution["vector61_flood_fill"] / wall,
        "attribution_clock": "cuda_event_elapsed" if cuda is not None else "cpu_wall",
        "flood_fill_host_enqueue_share": host["vector61_flood_fill"] / wall,
        "flood_fill_cuda_share_of_segment_elapsed": (
            cuda["vector61_flood_fill"] / sum(cuda.values()) if cuda else None
        ),
        "non_observation_remainder_seconds_per_tick": wall - observation,
        "conservative_remainder_retaining_opponent_v8_seconds_per_tick": (
            wall - observation + attribution["opponent_v8_bundle"]
        ),
        "remainder_definition": (
            "Broad observation removes whole B8/v8 feature-and-decision bundles; "
            "conservative remainder retains whole opponent v8 bundle. "
            "Opponent policies stage is network forward/scripted anchors only."
        ),
        "non_observation_cuda_elapsed_seconds_per_tick": (
            sum(value for key, value in cuda.items() if key not in OBSERVATION) if cuda else None
        ),
        "e1024_round24_reference_seconds": 0.512 if worlds == 1024 else None,
        "reference_comparison": (
            "descriptive only; event intervals include launch gaps; no decision label"
        ),
    }


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("A1-PROF workload wall cap reached")


def progress(done: int, total: int, started: float, **details: Any) -> None:
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "event": "a1_prof_progress",
                "done": done,
                "total": total,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed * (total - done) / done if done else None,
                **details,
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def measure_cell(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    repeats: int,
    ticks: int,
    warmups: int,
    deadline: float,
) -> dict[str, Any]:
    """Same fixed long-body ruler seam, fresh reset for every complete tick."""
    _check_deadline(deadline)
    ruler._release_cell(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sim, policy, template, policy_template = ruler._build_case(
        checkpoint, "long-" + mix, worlds, device
    )
    sync(device)
    setup_seconds = time.perf_counter() - start
    collection = ruler._collection(1, worlds, sim.device)
    baseline, timings, profiled = [], [], []

    def tick(timer: ExclusiveTiming | None = None) -> float:
        _check_deadline(deadline)
        sync(device)
        began = time.perf_counter()
        if timer is None:
            ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
        else:
            with instrument(sim, timer), timer.stage("unscoped"):
                with timer.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
        sync(device)
        return time.perf_counter() - began

    for _ in range(warmups):
        tick()
    for repeat in range(repeats):
        for _ in range(ticks):
            baseline.append(tick())
        for _ in range(ticks):
            timer = ExclusiveTiming(device)
            wall = tick(timer)
            timings.append({"repeat": repeat, "tick_wall_seconds": wall, **timer.result()})
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        timer = ExclusiveTiming(device, events=False, annotations=True)
        with torch.profiler.profile(activities=activities, profile_memory=True) as profiler:
            profiled_wall = tick(timer)
        summary = profiler_summary(profiler.events(), device)
        summary["profiled_tick_wall_seconds"] = profiled_wall
        profiled.append(summary)
        progress(repeat + 1, repeats, start, cell=f"{mix}-e{worlds}", phase="repeat")
        del profiler
        gc.collect()
    result = {
        "mix": mix,
        "worlds": worlds,
        "snakes": 6,
        "body_lengths": [800, 900, 1000, 1100, 1300, 1500],
        "repeats": repeats,
        "ticks_per_repeat": ticks,
        "warmups": warmups,
        "setup_seconds_including_initial_upload": setup_seconds,
        "baseline_tick_seconds": baseline,
        "timed_ticks": timings,
        "profiler_ticks": profiled,
        "memory": ruler._memory_now(device),
        "peak_memory": ruler.realistic._peak_memory(device),
        "metrics": metrics(baseline, timings, worlds),
        "rng_note": "Counter RNG in tick; random_safe SHA table construction/upload is setup only.",
        "transfer_note": (
            "Profiler records in-tick copies including implicit scalar DtoH; resident "
            "reset is DtoD."
        ),
    }
    sim = policy = template = policy_template = collection = None
    ruler._release_cell(device)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Skip exactly verified cells, publish complete units, retain diagnostic status."""
    if not 0 < args.seconds <= (900 if args.canary else 2400) or (
        args.device != "cuda" and not args.smoke
    ):
        raise ValueError("positive wall cap required; CPU is tiny preflight only")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and "A100" not in torch.cuda.get_device_name():
        raise ValueError("A1-PROF requires an A100")
    commit = source_commit()
    namespace = units.namespace(commit, args.smoke, args.canary)
    allocation = units.allocation(args.smoke, args.canary)
    verified = units.load_ledger(args.ledger, namespace, allocation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "a1-prof.json").exists():
        raise FileExistsError("A1-PROF immutable summary already exists")
    started = time.perf_counter()
    deadline = started + args.seconds
    progress(0, len(allocation), started, phase="start", device=args.device)
    complete = []
    for unit_id in allocation:
        if unit_id in verified:
            # Leave verified payloads in their immutable prior attempt; never remeasure.
            complete.append(unit_id)
            progress(
                len(complete), len(allocation), started, phase="verified_skip", unit_id=unit_id
            )
            continue
        _, mix, shape = unit_id.split("-")
        cell = measure_cell(
            args.checkpoint,
            mix,
            int(shape[1:]),
            args.device,
            1 if args.smoke or args.canary else 3,
            3 if args.smoke or not args.canary else 1,
            1 if args.smoke or args.canary else 3,
            deadline,
        )
        units.publish_unit(args.output_dir, namespace, allocation, unit_id, cell)
        complete.append(unit_id)
        progress(len(complete), len(allocation), started, phase="cell_complete", unit_id=unit_id)
    receipt = {
        "schema": "gpu-sim-spike-a1-prof/v1",
        "status": "PREFLIGHT" if args.smoke else "COMPLETE",
        "claim": (
            "Outcome-free timing diagnostics only; no labels, gate evidence or release " "authority"
        ),
        "commit": commit,
        "source_sha256": source_hashes(),
        "device": args.device,
        "namespace": namespace,
        "allocation": list(allocation),
        "completed_units": complete,
        "skipped_verified_units": [unit for unit in allocation if unit in verified],
        "elapsed_seconds": time.perf_counter() - started,
        "wall_cap_seconds": args.seconds,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "precision": {"dtype": "fp32", "matmul_tf32": False, "cudnn_tf32": False},
        "sealed_diagnostics": True,
        "aggregate_entrypoint": "research.gpu_sim_spike.a1_prof_analyze",
    }
    units.publish_json(args.output_dir / "a1-prof.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Preserve completed cells, but do not publish a successful summary.
        units.publish_json(
            args.output_dir / "a1-prof-failure.json",
            {"status": "INCOMPLETE", "error": str(exc), "type": type(exc).__name__},
        )
        raise


if __name__ == "__main__":
    main()
