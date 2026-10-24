"""I0 reset-inclusive all-roster ruler for the admitted combined engine.

Plain timing is the sole scored clock. CUDA event attribution and a one-tick
trace are diagnostic only, and never repair a missing cell or continuous run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from research.gpu_sim_spike import a1_prof
from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import integration_native
from research.gpu_sim_spike import integration_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes

ARMS = engine.ARMS
ORDERS = (ARMS, ("I", "R"), ARMS)
WARMUPS = 3
REPEATS = 3
CONTINUOUS_TICKS = 32


class IntegrationTiming(a1_prof.ExclusiveTiming):
    """Exclusive default-stream intervals with explicit body/carry substages."""

    @contextmanager
    def stage(self, name: str):
        name = a1_prof.STAGE_MAP.get(name, name)
        if self.stack:
            self._finish()
        self.stack.append(name)
        self._start()
        try:
            yield
        finally:
            self._finish()
            self.stack.pop()
            if self.stack:
                self._start()

    def result(self) -> dict:
        names = set(a1_prof.STAGES) | {row[0] for row in self.segments}
        host = dict.fromkeys(names, 0.0)
        cuda = dict.fromkeys(names, 0.0) if self.device == "cuda" else None
        for name, elapsed, start, end in self.segments:
            host[name] += elapsed
            if cuda is not None:
                cuda[name] += start.elapsed_time(end) / 1000
        return {"exclusive_host_seconds": host, "exclusive_cuda_elapsed_seconds": cuda}


@contextmanager
def instrument(sim: Any, timer: IntegrationTiming):
    from research.gpu_sim_spike.v8_bench import _instrument
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

    def wrap(method: Any, stage: str):
        def call(*args: Any, **kwargs: Any):
            with timer.stage(stage):
                return method(*args, **kwargs)

        return call

    with ExitStack() as stack:
        stack.enter_context(_instrument(sim, timer))
        for cls, method, stage in (
            (DeviceVectorRuntime, "_body_view", "vector61_body_occupancy"),
            (IndexedVectorRuntime, "_indexed_body_view", "vector61_body_occupancy"),
            (DeviceVectorRuntime, "prepare", "vector61_carry"),
            (DeviceVectorRuntime, "observe_step", "vector61_carry"),
            (DeviceVectorRuntime, "features", "vector61_other"),
        ):
            stack.enter_context(patch.object(cls, method, wrap(getattr(cls, method), stage)))
        yield


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _memory(device: str) -> dict[str, int | None]:
    if device != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "active_segments": None,
            "free_bytes": None,
        }
    free, _total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "allocated_segments": int(stats["segment.all.current"]),
        "allocation_count": int(stats["allocation.all.allocated"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(free),
    }


def _tick(
    case: tuple, device: str, timer: a1_prof.ExclusiveTiming | None = None, *, digest: bool = True
) -> tuple[float, str]:
    sim, policy, template, policy_template, collection, name = case
    with engine._arm(name), torch.inference_mode():
        _sync(device)
        started = time.perf_counter()
        if timer is None:
            ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
        else:
            with instrument(sim, timer), timer.stage("unscoped"):
                with timer.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
        _sync(device)
    return time.perf_counter() - started, engine.digest(case) if digest else ""


def _continuous(cases: dict[str, tuple], device: str, deadline: float) -> dict[str, Any]:
    """Interleaved progressing blocks, reset once, with every collector position used."""
    rows: dict[str, list] = {name: [] for name in ARMS}
    collections = {
        name: ruler._collection(CONTINUOUS_TICKS, case[0].E, case[0].device)
        for name, case in cases.items()
    }
    for name, case in cases.items():
        sim, policy, template, policy_template = case[:4]
        with engine._arm(name), torch.inference_mode():
            ruler._reset(sim, policy, template, policy_template)
            for index in range(WARMUPS):
                ruler._full_tick(sim, policy, collections[name], index)
    for repeat, order in enumerate(ORDERS):
        for name in order:
            sim, policy, template, policy_template = cases[name][:4]
            if time.monotonic() >= deadline:
                raise TimeoutError("I0 deadline before continuous block")
            tick_seconds = []
            with engine._arm(name), torch.inference_mode():
                _sync(device)
                started = time.perf_counter()
                ruler._reset(sim, policy, template, policy_template)
                for index in range(CONTINUOUS_TICKS):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("I0 deadline during continuous block")
                    began = time.perf_counter()
                    ruler._full_tick(sim, policy, collections[name], index)
                    _sync(device)
                    tick_seconds.append(time.perf_counter() - began)
                _sync(device)
                elapsed = time.perf_counter() - started
            final_case = (*cases[name][:4], collections[name], name)
            rows[name].append(
                {
                    "repeat": repeat,
                    "completed_world_steps": sim.E * CONTINUOUS_TICKS,
                    "wall_seconds": elapsed,
                    "tick_seconds": tick_seconds,
                    "complete_world_steps_per_second": sim.E * CONTINUOUS_TICKS / elapsed,
                    "final_digest": engine.digest(final_case),
                    "collector_positions": list(range(CONTINUOUS_TICKS)),
                }
            )
    return {
        "status": "COMPLETE",
        "ticks": CONTINUOUS_TICKS,
        "orders": [list(order) for order in ORDERS],
        "arms": {name: {"repeats": values} for name, values in rows.items()},
        "reset_charged_once_per_repeat": True,
        "warmups": WARMUPS,
    }


def _trace(case: tuple, device: str, output: Path, unit: str) -> dict[str, Any]:
    """Export one bounded profiler trace outside scored samples."""
    if device != "cuda":
        return {"status": "NOT_RUN_CPU"}
    destination = output / "traces" / f"{unit}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    raw = destination.with_suffix(".json")
    try:
        from torch.utils._python_dispatch import TorchDispatchMode

        logical_copies = {"operations": 0, "logical_bytes": 0}

        class CopyBytes(TorchDispatchMode):
            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                result = func(*args, **(kwargs or {}))
                if str(func) == "aten.copy_.default" and isinstance(args[0], torch.Tensor):
                    logical_copies["operations"] += 1
                    logical_copies["logical_bytes"] += args[0].numel() * args[0].element_size()
                return result

        with (
            CopyBytes(),
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
            ) as prof,
        ):
            _tick(case, device, digest=False)
        events = list(prof.events())
        summary = a1_prof.profiler_summary(events, device)
        copies = []
        for event in events:
            if event.device_type == torch.autograd.DeviceType.CPU and event.name in (
                "aten::copy_",
                "aten::_to_copy",
            ):
                shapes = getattr(event, "input_shapes", [])
                copies.append(
                    {
                        "operation": event.name,
                        "input_shapes": shapes,
                        "cpu_seconds": event.cpu_time_total / 1e6,
                    }
                )
        prof.export_chrome_trace(str(raw))
        payload = gzip.compress(raw.read_bytes(), mtime=0)
        if len(payload) > 32 * 1024**2:
            raise RuntimeError("I0 trace exceeds compact export limit")
        from research.gpu_sim_spike.a1_prof_units import _atomic_create

        _atomic_create(destination, payload)
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_create(
            Path(str(destination) + ".sha256"), f"{digest}  {destination.name}\n".encode()
        )
        _atomic_create(Path(str(destination) + ".complete"), b"complete\n")
        return {
            "status": "PASS",
            "path": str(destination.relative_to(output)),
            "sha256": digest,
            "size": len(payload),
            "kernel_host_inventory": summary,
            "copy_operations": copies,
            "bytes_moved": {
                "logical_copy_bytes_observed": logical_copies["logical_bytes"],
                "copy_operations_observed": logical_copies["operations"],
                "scope": "runtime copy_ destinations in separate trace; not physical DRAM traffic",
            },
            "hardware_counters": "NOT_AVAILABLE; no counter-tool setup attempted",
            "bottleneck_interpretation": "source and trace inference; not HBM saturation proof",
        }
    finally:
        raw.unlink(missing_ok=True)


def measure(
    checkpoint: Path,
    mix: str,
    worlds: int,
    device: str,
    deadline: float,
    output: Path,
    native: dict[str, Any],
    trace: bool,
    repeats: int,
    warmups: int,
    continuous_ticks: int,
) -> dict[str, Any]:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cases = {name: engine._case(checkpoint, mix, worlds, device, name) for name in ARMS}
    engine.assert_oracle_separation(cases["R"], cases["I"])
    try:
        from research.gpu_sim_spike import integration_parity

        proof = integration_parity.prove(cases, device)
        if not (proof.get("frozen_device_identity") and proof.get("cpu_geometry_carry")):
            raise RuntimeError("combined I0 parity proof failed")
        arms = {name: {"plain_seconds": [], "digests": [], "stages": []} for name in ARMS}
        for case in cases.values():
            for _ in range(warmups):
                if time.monotonic() >= deadline:
                    raise TimeoutError("I0 deadline during warmup")
                _tick(case, device)
        orders = ORDERS[:repeats]
        for repeat, order in enumerate(orders):
            for name in order:
                if time.monotonic() >= deadline:
                    raise TimeoutError("I0 deadline during scored repeat")
                plain, digest = _tick(cases[name], device)
                arms[name]["plain_seconds"].append(plain)
                arms[name]["digests"].append(digest)
        # Diagnostic ticks begin only after every plain scored repeat. They
        # cannot perturb cache/allocator state before a later scored sample.
        for repeat, order in enumerate(orders):
            for name in order:
                timer = IntegrationTiming(device)
                event, event_digest = _tick(cases[name], device, timer)
                if arms[name]["digests"][repeat] != event_digest:
                    raise RuntimeError("plain and diagnostic tick state differs")
                attribution = timer.result()
                stages = (
                    attribution["exclusive_cuda_elapsed_seconds"]
                    if device == "cuda"
                    else attribution["exclusive_host_seconds"]
                )
                total = sum(stages.values())
                if not math.isfinite(total) or total <= 0 or abs(total - event) > event * 0.10:
                    raise RuntimeError(
                        "exclusive diagnostic stages do not reconcile within 10 percent"
                    )
                arms[name]["stages"].append(
                    {"repeat": repeat, "wall_seconds": event, **attribution}
                )
        continuous = (
            _continuous(cases, device, deadline) if continuous_ticks else {"status": "NOT_REQUIRED"}
        )
        return {
            "mix": mix,
            "worlds": worlds,
            "repeats": repeats,
            "warmups": warmups,
            "orders": [list(x) for x in orders],
            "arms": arms,
            "continuous": continuous,
            "body_inventory": {
                name: {
                    "active_body_segments": int(case[0].seg_count[case[0].alive].sum().cpu()),
                    "body_storage_bytes": case[0].bodies.numel() * case[0].bodies.element_size(),
                }
                for name, case in cases.items()
            },
            "capacity": int(cases["I"][0].cap),
            "lengths": [800, 900, 1000, 1100, 1300, 1500],
            "device": device,
            "memory": _memory(device),
            "native_parity": native["status"],
            "frozen_device_identity": proof["frozen_device_identity"],
            "cpu_geometry_carry": proof["cpu_geometry_carry"],
            "combined_parity": proof,
            "cpu_closed_loop_identity": "UNPROVED",
            "trace": (
                _trace(cases["I"], device, output, f"cell-{mix}-e{worlds}")
                if trace
                else {"status": "NOT_RUN"}
            ),
        }
    finally:
        cases.clear()
        ruler._release_cell(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("I0 requires A100")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= (1200 if args.canary else 5400):
        raise ValueError("I0 requires a finite positive bounded producer deadline")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("CPU is tiny smoke only")
    if args.smoke and args.canary:
        raise ValueError("smoke and canary are exclusive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + args.seconds
    commit = source_commit()
    cells = units.allocation(smoke=args.smoke, canary=args.canary)
    ns = units.namespace(commit, smoke=args.smoke, canary=args.canary)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(110):
            print(
                json.dumps(
                    {"event": "i0_heartbeat", "elapsed_seconds": time.monotonic() - started}
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    native = integration_native.run(output / "native-parity.json", device=args.device)
    from research.gpu_sim_spike import integration_parity

    discrepancies = integration_parity.cpu_discrepancy_register(args.checkpoint, args.device)
    units.publish_json(
        output / "discrepancy-register.json",
        {
            "schema": "gpu-sim-spike-i0-discrepancy-register/v1",
            "commit": commit,
            "source_sha256": source_hashes(),
            "native_status": native["status"],
            "cpu_component_register": discrepancies,
            "cpu_closed_loop_identity": "UNPROVED",
            "declared": "CPU geometry/carry component proof only; closed loop remains unproved",
        },
    )
    # Compile/native admission is intentionally before every potentially expensive cell.
    if native["status"] not in ("PASS", "PREFLIGHT"):
        result = {
            "schema": "gpu-sim-spike-i0-bench/v1",
            "status": "INCOMPLETE" if native["status"] != "INVALID" else "INVALID",
            "reason": "NATIVE_COMPILE_PARITY_REQUIRED",
            "commit": commit,
            "source_sha256": source_hashes(),
            "completed_units": [],
            "native_parity": "native-parity.json",
            "discrepancy_register": "discrepancy-register.json",
        }
        units.publish_json(output / "integration-bench.json", result)
        return result
    prior = units.load_ledger(args.ledger, ns, cells)
    completed = []
    skipped_units = []
    try:
        for index, unit in enumerate(cells):
            if unit in prior:
                # A verified outcome is immutable evidence from its original
                # attempt. Do not republish it into this fresh output tree.
                skipped_units.append(unit)
                completed.append(unit)
                print(json.dumps({"event": "unit_skipped_verified", "unit": unit}), flush=True)
                continue
            else:
                mix, worlds = unit.split("-")[1], int(unit.rsplit("e", 1)[1])
                repeats = 1 if args.smoke or args.canary else REPEATS
                warmups = 1 if args.smoke or args.canary else WARMUPS
                continuous_ticks = (
                    CONTINUOUS_TICKS if not (args.smoke or args.canary) and worlds == 1024 else 0
                )
                value = measure(
                    args.checkpoint,
                    mix,
                    worlds,
                    args.device,
                    deadline,
                    output,
                    native,
                    trace=index == 0,
                    repeats=repeats,
                    warmups=warmups,
                    continuous_ticks=continuous_ticks,
                )
            units.publish_unit(output, ns, cells, unit, value)
            completed.append(unit)
            print(
                json.dumps(
                    {
                        "event": "unit_done",
                        "unit": unit,
                        "done": len(completed),
                        "total": len(cells),
                    }
                ),
                flush=True,
            )
    except TimeoutError as exc:
        result = {
            "schema": "gpu-sim-spike-i0-bench/v1",
            "status": "INCOMPLETE",
            "reason": str(exc),
            "commit": commit,
            "source_sha256": source_hashes(),
            "completed_units": completed,
            "skipped_units": skipped_units,
            "native_parity": "native-parity.json",
            "discrepancy_register": "discrepancy-register.json",
        }
        units.publish_json(output / "integration-bench.json", result)
        stop_heartbeat.set()
        return result
    ledger = units.build_ledger(
        output, output / "verified-ledger.json", commit=commit, smoke=args.smoke, canary=args.canary
    )
    result = {
        "schema": "gpu-sim-spike-i0-bench/v1",
        "status": "MEASURED_PENDING_MAC_REVIEW",
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
        "precision": {"dtype": "fp32", "matmul_tf32": False, "cudnn_tf32": False},
        "commit": commit,
        "source_sha256": source_hashes(),
        "namespace": ledger["namespace"],
        "completed_units": completed,
        "skipped_units": skipped_units,
        "native_parity": "native-parity.json",
        "discrepancy_register": "discrepancy-register.json",
        "ledger": "verified-ledger.json",
        "component_flags": {
            "flood": "static",
            "v8_bundle": "off",
            "move": "indexed",
            "geometry": "native",
        },
        "cpu_closed_loop_identity": "UNPROVED",
    }
    units.publish_json(output / "integration-bench.json", result)
    stop_heartbeat.set()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    result = run(a)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] not in ("MEASURED_PENDING_MAC_REVIEW",):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
