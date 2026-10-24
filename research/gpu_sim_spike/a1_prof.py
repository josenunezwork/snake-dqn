"""A1-PROF: outcome-free full-tick attribution; never labels or gate evidence."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import platform
import re
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
UNATTRIBUTED_TOP_N = 30
TRACE_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
TRACE_MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
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


class NativeLaunchDiagnostics:
    """One opt-in event pass around direct CUDA-extension launches.

    Launch modules preallocate their event pools before the measured tick.
    This collector performs the sole device drain at tick end, then consumes
    events from the stream the native launch itself used.  It is deliberately
    separate from both ordinary plain ticks and the Kineto trace pass.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.modules: list[Any] = []
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "NativeLaunchDiagnostics":
        if self.device != "cuda":
            return self
        from research.gpu_sim_spike import event_cuda, flood_cuda

        self.modules = [event_cuda, flood_cuda]
        for module in self.modules:
            module.set_launch_diagnostics(True, reset=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.device != "cuda":
            return
        # A single fence makes every recorded same-stream end event complete.
        sync(self.device)
        try:
            for module in self.modules:
                self.records.extend(module.get_launch_diagnostics())
        finally:
            for module in self.modules:
                module.set_launch_diagnostics(False, reset=True)

    def summary(self) -> dict[str, Any]:
        """Return raw launch rows and stable per-kernel count/time totals."""
        table: dict[str, dict[str, Any]] = {}
        for row in self.records:
            name, count, elapsed = row["name"], row["count"], row["elapsed_ms"]
            if (
                not isinstance(name, str)
                or type(count) is not int
                or not isinstance(elapsed, float)
            ):
                raise RuntimeError("native launch diagnostic record is malformed")
            if count != 1 or not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("native launch diagnostic timing is invalid")
            total = table.setdefault(
                name, {"name": name, "count": 0, "elapsed_ms": 0.0, "streams": []}
            )
            total["count"] += count
            total["elapsed_ms"] += elapsed
            stream = row.get("stream")
            if stream is not None and stream not in total["streams"]:
                total["streams"].append(stream)
        rows = sorted(table.values(), key=lambda row: (-row["elapsed_ms"], row["name"]))
        for row in rows:
            row["mean_elapsed_ms"] = row["elapsed_ms"] / row["count"]
            row["streams"].sort()
        return {
            "clock": "same_current_stream_cuda_events_after_one_tick_end_fence",
            "launches": self.records,
            "kernel_table": rows,
            "total_launches": len(self.records),
            "total_elapsed_ms": sum(row["elapsed_ms"] for row in self.records),
        }


class NativeLaunchScopes:
    """Enable native RecordFunction/NVTX ancestry without recording CUDA events."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.modules: list[Any] = []

    def __enter__(self) -> "NativeLaunchScopes":
        if self.device != "cuda":
            return self
        from research.gpu_sim_spike import event_cuda, flood_cuda

        self.modules = [event_cuda, flood_cuda]
        for module in self.modules:
            module.set_launch_diagnostics(True, reset=True, capture_events=False)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for module in self.modules:
            module.set_launch_diagnostics(False, reset=True, capture_events=False)


def _publish_chrome_trace(profiler: Any, output_dir: Path, stem: str) -> dict[str, Any]:
    """Publish one bounded immutable raw Kineto Chrome trace for a measurement cell."""
    directory = output_dir / "a1-prof-traces"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{stem}.json.gz"
    if destination.exists():
        raise FileExistsError(f"A1-PROF Chrome trace already exists: {destination}")
    temporary = directory / f".{stem}.json.tmp"
    profiler.export_chrome_trace(str(temporary))
    try:
        raw = temporary.read_bytes()
        if not raw or len(raw) > TRACE_MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError(
                f"A1-PROF Chrome trace exceeds {TRACE_MAX_UNCOMPRESSED_BYTES} byte raw export bound"
            )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("A1-PROF Chrome trace must be a JSON object")
        payload = gzip.compress(raw, mtime=0)
        if len(payload) > TRACE_MAX_COMPRESSED_BYTES:
            raise RuntimeError(
                "A1-PROF Chrome trace exceeds compressed export bound: "
                f"{TRACE_MAX_COMPRESSED_BYTES} bytes"
            )
        units._atomic_create(destination, payload)
        digest = hashlib.sha256(payload).hexdigest()
        units._atomic_create(
            Path(str(destination) + ".sha256"), f"{digest}  {destination.name}\n".encode()
        )
        units._atomic_create(Path(str(destination) + ".complete"), b"complete\n")
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.relative_to(output_dir).as_posix(),
        "bytes": len(payload),
        "uncompressed_bytes": len(raw),
        "sha256": digest,
    }


class ExclusiveTiming:
    """Segment the default stream at every nested scope boundary without overlap.

    Device synchronization fences each exclusive segment, including work on
    auxiliary streams. Events measure elapsed intervals (including launch and
    synchronization gaps), not kernel-busy time. Kineto is diagnostic only.
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
            sync(self.device)
            self.first = torch.cuda.Event(enable_timing=True)
            self.first.record()
        self.started = time.perf_counter()

    def _finish(self) -> None:
        end = None
        elapsed = time.perf_counter() - self.started
        if self.first is not None:
            # Drain all streams before recording the end on the default stream.
            # Native/async launches must finish inside their exclusive segment.
            sync(self.device)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
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


def heuristic_stage(kernel_name: str) -> tuple[str | None, str | None]:
    """Suggest only source-identifiable native kernels; never guess generic ATen ops."""
    name = kernel_name.lower()
    if "fifo(" in name or name == "fifo":
        return "food_corpse", "event_cuda.cu::fifo (shared food/corpse insertion)"
    if any(
        token in name
        for token in (
            "sparse_frontier",
            "local_components",
            "boundary_components",
            "count_components",
            "gather_components",
        )
    ):
        return "b8_v8_features", "flood_cuda.cu native v8 flood (hero/opponent ambiguous)"
    return None, None


def profiler_summary(
    events: Any, device: str, kernel_links: dict[int, int] | None = None
) -> dict[str, Any]:
    """Inventory raw kernels once, independently of optional CPU scope correlation.

    Kineto links raw correlation IDs to launching CPU op IDs. FunctionEvent.kernels
    omits native launches without such links; it is never the inventory source.
    Missing links or scopes are diagnostic findings, with no coverage threshold.
    """
    events = list(events)
    kernel_links = kernel_links or {}
    # TorchOp external IDs and CUPTI runtime IDs have independent namespaces.
    # A numeric collision is uncertainty, never permission to overwrite an op.
    cpu_by_id: dict[int, list[Any]] = {}
    for event in events:
        if event.device_type == torch.autograd.DeviceType.CPU and hasattr(event, "id"):
            cpu_by_id.setdefault(event.id, []).append(event)

    def unique_cpu(event_id: int | None) -> Any:
        candidates = cpu_by_id.get(event_id, [])
        return candidates[0] if len(candidates) == 1 else None

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

    def scope(event: Any) -> str | None:
        while event is not None:
            if event.name.startswith(PREFIX):
                name = event.name[len(PREFIX) :]
                return name if name in STAGES else None
            event = getattr(event, "cpu_parent", None)
        return None

    copies: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    groups: dict[tuple[Any, ...], str] = {}
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        target = stages[scope(event) or "unscoped"]
        memory = getattr(event, "self_device_memory_usage", 0)
        target["self_device_memory_net_bytes"] += memory
        target["self_device_allocation_bytes"] += max(memory, 0)
        if event.name.startswith(PREFIX):
            target["python_gap_seconds"] += event.self_cpu_time_total / 1e6
        if event.name.startswith(("cudaLaunch", "cuLaunch")):
            target["launch_api_count"] += 1
            target["launch_api_seconds"] += event.cpu_time_total / 1e6
        # CPU-only preflight can still report copy activities if present.
        if device != "cuda":
            for kernel in event.kernels:
                if kernel.name.startswith(("Memcpy", "Memset")):
                    row = copies.setdefault(kernel.name, {"count": 0, "seconds": 0.0})
                    row["count"] += 1
                    row["seconds"] += kernel.duration / 1e6
    raw, raw_seconds, attributed, attributed_seconds = 0, 0.0, 0, 0.0
    linked_count = 0
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA or event.name == "[memory]":
            continue
        seconds = event.time_range.elapsed_us() / 1e6
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError(f"profiler invalid raw device duration: {event.name}: {seconds}")
        if event.name.startswith(("Memcpy", "Memset")):
            row = copies.setdefault(event.name, {"count": 0, "seconds": 0.0})
            row["count"] += 1
            row["seconds"] += seconds
            continue
        raw += 1
        raw_seconds += seconds
        event_id = getattr(event, "id", None)
        launch = unique_cpu(kernel_links.get(event_id))
        if launch is not None and re.match(r"^(cuda|cu)[A-Z]", launch.name):
            launch = None  # A linked frontend-op ID cannot name a CUPTI API.
        link_kind = "kineto_linked_op" if launch is not None else None
        if launch is None:
            # CUPTI may retain only the runtime/kernel correlation for native
            # extensions. Accept an exact ID match to a launch API, never names
            # or temporal overlap as proof of launching-stage ancestry.
            runtime = unique_cpu(event_id)
            if runtime is not None and runtime.name.startswith(("cudaLaunch", "cuLaunch")):
                launch, link_kind = runtime, "runtime_correlation_id"
        linked_count += launch is not None
        stage = scope(launch)
        # Walk the explicit launching op's ancestry for a captured stack.
        parent, stack = launch, []
        while parent is not None and not stack:
            stack = list(getattr(parent, "stack", None) or [])
            parent = getattr(parent, "cpu_parent", None)
        suggestion, reason = heuristic_stage(event.name) if stage is None else (None, None)
        name = stage or "unattributed"
        target = stages[name]
        target["kernel_count"] += 1
        target["kernel_seconds"] += seconds
        if stage is not None:
            attributed += 1
            attributed_seconds += seconds
        stream = getattr(event, "device_resource_id", None)
        device_index = getattr(event, "device_index", None)
        op = launch.name if launch is not None else None
        # Aggregate every raw kernel by all available provenance, without a
        # top-N truncation. Per-launch timestamps are unnecessary for this table.
        provenance = (event.name, name, device_index, stream, op, tuple(stack), link_kind)
        key = groups.setdefault(provenance, f"kernel_{len(groups):06d}")
        row = inventory.setdefault(
            key,
            {
                "name": event.name,
                "stage": name,
                "stream": stream,
                "device_index": device_index,
                "launching_op": op,
                "launching_stack": stack,
                "launch_correlation": link_kind,
                "heuristic_stage": suggestion,
                "heuristic_reason": reason,
                "count": 0,
                "seconds": 0.0,
            },
        )
        row["count"] += 1
        row["seconds"] += seconds
    if device == "cuda" and (raw == 0 or raw_seconds <= 0):
        raise RuntimeError("profiler raw CUDA kernel trace is missing or empty")
    missing = stages["unattributed"]["kernel_count"]
    missing_seconds = stages["unattributed"]["kernel_seconds"]
    heuristic_totals: dict[str, Any] = {}
    unmatched_names: dict[str, Any] = {}
    for key, row in inventory.items():
        row["total_device_seconds"] = row["seconds"]
        row["mean_device_seconds"] = row["seconds"] / row["count"]
        if row["heuristic_stage"]:
            target = heuristic_totals.setdefault(
                row["heuristic_stage"], {"count": 0, "seconds": 0.0}
            )
            target["count"] += row["count"]
            target["seconds"] += row["seconds"]
        if row["stage"] == "unattributed":
            total = unmatched_names.setdefault(
                row["name"],
                {
                    "name": row["name"],
                    "count": 0,
                    "seconds": 0.0,
                    "streams": [],
                    "inventory_keys": [],
                    "heuristic_stage": row["heuristic_stage"],
                    "heuristic_reason": row["heuristic_reason"],
                },
            )
            total["count"] += row["count"]
            total["seconds"] += row["seconds"]
            total["inventory_keys"].append(key)
            if row["stream"] not in total["streams"]:
                total["streams"].append(row["stream"])
    for row in unmatched_names.values():
        row["total_device_seconds"] = row["seconds"]
        row["mean_device_seconds"] = row["seconds"] / row["count"]
        row["streams"].sort(key=str)
    top = sorted(unmatched_names.values(), key=lambda row: (-row["seconds"], row["name"]))[
        :UNATTRIBUTED_TOP_N
    ]
    return {
        "stages": stages,
        "kernel_count": raw,
        "attributed_cuda_kernel_count": attributed,
        "raw_cuda_kernel_count": raw,
        "linked_cuda_kernel_count": linked_count,
        "unattributed_cuda_kernel_count": missing,
        "unattributed_cuda_kernel_count_share": missing / raw if raw else 0.0,
        "attributed_cuda_kernel_seconds": attributed_seconds,
        "raw_cuda_kernel_seconds": raw_seconds,
        "unattributed_cuda_kernel_seconds": missing_seconds,
        "unattributed_cuda_kernel_time_share": (
            missing_seconds / raw_seconds if raw_seconds else 0.0
        ),
        "attribution_gaps_are_fatal": False,
        "authoritative_stage_clock": "synchronized_exclusive_cuda_events",
        "device_activities": copies,
        "host_device_transfers": {
            key: value
            for key, value in copies.items()
            if any(token in key.lower() for token in ("htod", "dtoh", "h2d", "d2h"))
        },
        "kernel_inventory": inventory,
        "top_unattributed_kernels_by_device_time": top,
        "unattributed_top_n": UNATTRIBUTED_TOP_N,
        "heuristic_unattributed_stage_totals": heuristic_totals,
        "cuda_available": device == "cuda",
        "notes": (
            "Raw CUDA activities are counted once, including unlinked native kernels. "
            "Inventory includes every kernel name/provenance group, not just top-N. "
            "Missing stream/op/stack is null/empty. Scope correlation and name heuristics "
            "are diagnostic only; heuristics never alter measured stage totals. Native v8 "
            "flood names cannot distinguish hero and opponent bundles. Exclusive synchronized "
            "CUDA-event intervals are authoritative; kernel duration is a separate busy-time "
            "diagnostic. Python gaps and launch API time overlap device work."
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
    accounted = sum(attribution.values())
    accounting_discrepancy = abs(wall - accounted) / wall
    return {
        "complete_world_steps_per_second": worlds * len(baseline) / sum(baseline),
        "uninstrumented_mean_tick_seconds": statistics.mean(baseline),
        "instrumented_mean_tick_seconds": wall,
        "instrumentation_wall_ratio": wall / statistics.mean(baseline),
        "exclusive_host_seconds_per_tick": host,
        "exclusive_cuda_elapsed_seconds_per_tick": cuda,
        "boundary_and_sync_seconds_per_tick": wall - accounted,
        "host_boundary_seconds_per_tick": wall - sum(host.values()),
        "accounting_discrepancy_fraction": accounting_discrepancy,
        "attribution_status": "COMPLETE" if accounting_discrepancy <= 0.10 else "INCOMPLETE",
        "attribution_status_rule": (
            "COMPLETE only when exclusive-stage reconciliation is within 10%"
        ),
        "flood_fill_share_of_instrumented_tick": attribution["vector61_flood_fill"] / wall,
        "non_flood_remainder_seconds_per_tick": wall - attribution["vector61_flood_fill"],
        "non_flood_remainder_definition": (
            "Whole synchronized tick less the vector61 capped-free-space stage; it retains "
            "other observations, policies, simulator work, and boundary time."
        ),
        "attribution_clock": "cuda_event_elapsed" if cuda is not None else "cpu_wall",
        "flood_fill_host_enqueue_share": host["vector61_flood_fill"] / wall,
        "flood_fill_cuda_share_of_segment_elapsed": (
            cuda["vector61_flood_fill"] / sum(cuda.values())
            if cuda and sum(cuda.values()) > 0
            else None
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
    output_dir: Path,
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
    traces = []
    native_launch_ticks = []
    for repeat in range(repeats):
        for _ in range(ticks):
            baseline.append(tick())
        for _ in range(ticks):
            timer = ExclusiveTiming(device)
            wall = tick(timer)
            timings.append({"repeat": repeat, "tick_wall_seconds": wall, **timer.result()})
        # Same-stream launch events answer a different question from the
        # outer-stage event clock.  Allocate the pools before this tick and
        # fence only once in the collector's __exit__.
        native = NativeLaunchDiagnostics(device)
        with native:
            native_wall = tick()
        native_launch_ticks.append(
            {"repeat": repeat, "tick_wall_seconds": native_wall, **native.summary()}
        )
        # One minimal raw trace per cell gives native launch ancestry without
        # multiplying trace overhead or immutable export bytes by repeats.
        if repeat > 0:
            progress(repeat + 1, repeats, start, cell=f"{mix}-e{worlds}", phase="repeat")
            continue
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        timer = ExclusiveTiming(device, events=False, annotations=True)
        with (
            NativeLaunchScopes(device),
            torch.profiler.profile(
                activities=activities, profile_memory=False, with_stack=False
            ) as profiler,
        ):
            profiled_wall = tick(timer)
        # Preserve explicit raw-device -> CPU-op links discarded by FunctionEvent.
        links = {
            event.correlation_id(): event.linked_correlation_id()
            for event in profiler.profiler.kineto_results.events()
            if event.device_type() == torch.autograd.DeviceType.CUDA
            and event.linked_correlation_id() > 0
        }
        # Persist the raw trace before reducing it to summary tables.  If the
        # bounded immutable export fails, this cell is INCOMPLETE rather than
        # leaving a summary which cannot be independently inspected.
        trace = _publish_chrome_trace(profiler, output_dir, f"cell-{mix}-e{worlds}-trace")
        summary = profiler_summary(profiler.events(), device, links)
        summary["profiled_tick_wall_seconds"] = profiled_wall
        summary["chrome_trace"] = trace
        traces.append(trace)
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
        "native_launch_event_ticks": native_launch_ticks,
        "profiler_ticks": profiled,
        "chrome_traces": traces,
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
            args.output_dir,
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
