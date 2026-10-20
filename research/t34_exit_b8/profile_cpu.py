"""Excluded, CPU-only runtime diagnostic for the T34 ExIt teacher.

This is deliberately a measurement entrypoint, not an A5 collector.  It uses
the public-input tree prepared by :mod:`public_inputs`, executes one resident
world at a time, and writes a diagnostic-only receipt.  It never changes a
policy, a replay/export payload, or a campaign result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

DIAGNOSTIC_NAMESPACE = "t34-exit-cpu-profile-diagnostic/v1"
MIXES = ("scripted", "frozen", "mixed")
BODY_REGIMES = ("short", "long")
STAGES = ("digest", "v8", "raster", "sim", "inference")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _write_completed(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable CLI-export payload, digest sidecar, then marker."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(hashlib.sha256(payload).hexdigest() + "\n")
    path.with_name(path.name + ".complete").touch(exist_ok=False)


def runtime_hardware() -> dict[str, Any]:
    """Capture scheduling and memory facts before constructing a policy."""
    ram = None
    try:
        ram = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        "nproc": max(1, int(os.cpu_count() or 1)),
        "ram_bytes": ram,
        "affinity_cpus": affinity,
        "cgroup_cpu_max": _read_text(Path("/sys/fs/cgroup/cpu.max")),
        "cgroup_cpuset_effective": _read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective")),
        "cgroup_membership": _read_text(Path("/proc/self/cgroup")),
    }


def provenance() -> dict[str, Any]:
    """Capture source and platform facts without loading policy modules."""
    source = Path(__file__).resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source.parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "argv": list(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "profile_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_revision": revision,
    }


def default_workers() -> int:
    """Use all visible CPUs remotely, while preserving the two-thread Mac rule."""
    visible = max(1, int(os.cpu_count() or 1))
    return min(visible, 2) if sys.platform == "darwin" else visible


def effective_workers(hardware: Mapping[str, Any]) -> int:
    """Bound fan-out by affinity and cgroup quota, not host CPU count alone."""
    limits = [int(hardware["nproc"])]
    affinity = hardware.get("affinity_cpus")
    if isinstance(affinity, list) and affinity:
        limits.append(len(affinity))
    quota = str(hardware.get("cgroup_cpu_max") or "").split()
    if len(quota) == 2 and quota[0] != "max":
        try:
            limits.append(max(1, int(quota[0]) // int(quota[1])))
        except (ValueError, ZeroDivisionError):
            pass
    return max(1, min(limits))


class ExclusiveStages:
    """Nested wall-clock accounting where each stage excludes named children."""

    def __init__(self) -> None:
        self.inclusive: dict[str, float] = defaultdict(float)
        self.exclusive: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._stack: list[list[Any]] = []

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        entry: list[Any] = [name, started, 0.0]
        self._stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._stack.pop()
            self.inclusive[name] += elapsed
            self.exclusive[name] += max(0.0, elapsed - float(entry[2]))
            self.calls[name] += 1
            if self._stack:
                self._stack[-1][2] += elapsed

    def receipt(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "inclusive_seconds": float(self.inclusive[name]),
                "exclusive_seconds": float(self.exclusive[name]),
                "calls": int(self.calls[name]),
            }
            for name in STAGES
        }


@contextmanager
def instrument(stages: ExclusiveStages) -> Iterator[None]:
    """Temporarily wrap the actual production hot-path functions.

    Wrappers nest, so a raster call below a module call is charged to raster and
    removed from inference's exclusive residual.  Same-category recursion is
    intentionally retained as an inclusive sum; its exclusive value remains
    meaningful at each call boundary.
    """
    import torch

    from research.t34_exit_b8 import teacher_v2
    from research.t34_shadow import shadow
    from src.simd_env import ego_raster
    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    patches: list[tuple[Any, str, Any]] = []

    def wrap(owner: Any, attribute: str, stage: str) -> None:
        original = getattr(owner, attribute)

        def timed(*args: Any, **kwargs: Any) -> Any:
            with stages.measure(stage):
                return original(*args, **kwargs)

        patches.append((owner, attribute, original))
        setattr(owner, attribute, timed)

    wrap(shadow, "full_digest", "digest")
    # teacher_v2 imports this symbol directly, so both references are needed.
    wrap(teacher_v2, "full_digest", "digest")
    wrap(Vector61SimdPolicy, "_apply_veto", "v8")
    wrap(ego_raster, "build_ego_raster", "raster")
    wrap(BatchSim, "step", "sim")
    wrap(torch.nn.Module, "_call_impl", "inference")
    try:
        yield
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)


def _cpu_ceiling() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _rows() -> list[dict[str, Any]]:
    # The generated row seeds stay outside the campaign's registered namespace.
    from research.t34_exit_b8.trajectory import make_world_rows

    return make_world_rows({"namespace": DIAGNOSTIC_NAMESPACE, "mixes": MIXES}, len(MIXES))


def _reset_world_builder(row: Mapping[str, Any], checkpoint: Path) -> Any:
    """Build then deep-clone/reset a snapshot before timing the hot rows."""
    from research.t34_exit_b8.trajectory import build_world, prepare_world
    from research.t34_shadow.shadow import clone_world_exact

    world = build_world(row, checkpoint)
    prepare_world(world)
    return clone_world_exact(world)


def _long_world_builder(row: Mapping[str, Any], checkpoint: Path) -> Any:
    from src.simd_env.grid_scenarios import inject_serpentines

    world = _reset_world_builder(row, checkpoint)
    # Six slots get long, deterministic bodies; this setup is intentionally
    # outside hot-row timing.  The helper refreshes grids/masks without RNG use.
    lengths = [900, 600, 400, 300, 150, 150]
    if int(world.sim.S) != len(lengths) or int(world.sim.cap) <= max(lengths) + 3:
        raise ValueError("long-body profile requires six slots and ring capacity above 903")
    inject_serpentines([world.sim], lengths, box_height=8)
    # Geometry injection invalidates decision-time Vector61 features. Rebind
    # this excluded diagnostic snapshot before selecting any opponent action.
    controlled = world.runtime.controlled.copy()
    world.runtime.bind(world.sim, controlled)
    world.sim._policy_phase_prepared = False
    from research.t34_exit_b8.trajectory import prepare_world

    prepare_world(world)
    return world


def _case_config(
    checkpoint: Path, output_dir: Path, seconds: float, mix: str, body: str
) -> dict[str, Any]:
    row = next(row for row in _rows() if row["mix"] == mix)
    config: dict[str, Any] = {
        "kernel": "v2",
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "rows": [row],
        "max_worlds": 1,
        "worlds": 1,
        "frames": 5000,
        "seconds": seconds,
        "steady_state_seconds": seconds,
        "horizon": 300,
        "threads": 1,
        "device": "cpu",
        "namespace": DIAGNOSTIC_NAMESPACE,
        "mixes": MIXES,
    }
    config["world_builder"] = _long_world_builder if body == "long" else _reset_world_builder
    return config


def _rss_bytes() -> int:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _profile_worker(
    rank: int, checkpoint: str, output_dir: str, seconds: float, result_queue: Any
) -> None:
    try:
        _cpu_ceiling()
        from research.t34_exit_b8.teacher_v2 import run_worlds

        warm_started = time.monotonic()
        # Compile/load and exercise every mix/body outside the measured window.
        for body in BODY_REGIMES:
            for mix in MIXES:
                warm = _case_config(
                    Path(checkpoint), Path(output_dir) / f"warm-{body}-{mix}", 180, mix, body
                )
                warm.update(frames=2, horizon=2, shadow_depth=2, steady_state_seconds=0)
                warmed = run_worlds(warm)
                if not warmed.get("complete") or int(warmed.get("active_row_steps", 0)) < 1:
                    raise RuntimeError(f"warmup produced no useful row for {body}/{mix}")
        warmup_seconds = time.monotonic() - warm_started
        started_wall, started_cpu = time.monotonic(), time.process_time()
        stages = ExclusiveStages()
        cases: list[dict[str, Any]] = []
        # A copied/reset world is built by the production builder for every
        # case.  Setup and export are reported separately from the hot timings.
        case_seconds = max(0.01, seconds / (len(MIXES) * len(BODY_REGIMES)))
        with instrument(stages):
            for body in BODY_REGIMES:
                for mix in MIXES:
                    case_started = time.monotonic()
                    receipt = run_worlds(
                        _case_config(
                            Path(checkpoint),
                            Path(output_dir) / f"{body}-{mix}",
                            case_seconds,
                            mix,
                            body,
                        )
                    )
                    cases.append(
                        {
                            "mix": mix,
                            "body_regime": body,
                            "inclusive_elapsed_seconds": time.monotonic() - case_started,
                            "active_row_steps": int(receipt.get("active_row_steps", 0)),
                            "complete": bool(receipt.get("complete", False)),
                        }
                    )
        ended_wall = time.monotonic()
        wall, process_cpu = ended_wall - started_wall, time.process_time() - started_cpu
        active = sum(case["active_row_steps"] for case in cases)
        result_queue.put(
            (
                rank,
                {
                    "rank": rank,
                    "warmup_seconds": warmup_seconds,
                    "wall_seconds": wall,
                    "phase_started_monotonic": started_wall,
                    "phase_ended_monotonic": ended_wall,
                    "process_cpu_seconds": process_cpu,
                    "wall_to_process_cpu_ratio": wall / process_cpu if process_cpu else None,
                    "rss_bytes": _rss_bytes(),
                    "active_row_steps": active,
                    "active_row_steps_per_second": active / wall if wall else None,
                    "exclusive_stage_seconds": stages.receipt(),
                    "cases": cases,
                },
                None,
            )
        )
    except BaseException as exc:
        result_queue.put((rank, None, f"{type(exc).__name__}: {exc}"))


def _run_group(
    checkpoint: Path, output_dir: Path, seconds: float, workers: int
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    results = context.Queue()
    children = [
        context.Process(
            target=_profile_worker,
            args=(rank, str(checkpoint), str(output_dir / f"worker-{rank:03d}"), seconds, results),
        )
        for rank in range(workers)
    ]
    reports: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    deadline = time.monotonic() + min(720.0, seconds + 180.0)
    try:
        for child in children:
            child.start()
        while len(reports) + len(failures) < workers:
            try:
                rank, report, failure = results.get(timeout=1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError("profile group exceeded its bounded wall deadline")
                dead = [
                    index
                    for index, child in enumerate(children)
                    if index not in reports and child.exitcode is not None
                ]
                if dead:
                    raise RuntimeError(f"profile worker crashed before reporting: {dead}")
                continue
            if failure:
                failures[int(rank)] = str(failure)
                raise RuntimeError(f"profile worker {rank} failed: {failure}")
            reports[int(rank)] = dict(report)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()  # Owned child only; never touches a developer process.
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
    if failures or len(reports) != workers:
        raise RuntimeError(f"profile group incomplete: {failures}")
    return [reports[index] for index in range(workers)]


def run_profile(
    checkpoint: Path, output_dir: Path, *, seconds: float, workers: int | None
) -> dict[str, Any]:
    """Run W1 alone, then the requested W1 fan-out with an identical window."""
    if not 0 < seconds <= 60:
        raise ValueError("profile seconds must be in (0, 60]")
    hardware_path = output_dir / "hardware-first.json"
    hardware_files = {
        hardware_path,
        hardware_path.with_name(hardware_path.name + ".sha256"),
        hardware_path.with_name(hardware_path.name + ".complete"),
    }
    if output_dir.exists() and set(output_dir.iterdir()) != hardware_files:
        raise ValueError(
            f"profile output must be fresh except for hardware-first.json: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = (
        json.loads(hardware_path.read_text())["hardware"]
        if hardware_path.exists()
        else runtime_hardware()
    )
    effective = effective_workers(hardware)
    if workers is None:
        workers = min(effective, 2) if sys.platform == "darwin" else effective
    if workers < 1:
        raise ValueError("workers must be positive")
    if sys.platform == "darwin" and workers > 2:
        raise ValueError("Mac diagnostic is capped at two worker processes")
    if workers > effective:
        raise ValueError(f"workers={workers} exceeds effective CPU allocation={effective}")
    alone = _run_group(checkpoint, output_dir / "alone", seconds, 1)
    fanout = _run_group(checkpoint, output_dir / "fanout", seconds, workers)
    phase_start = min((float(row["phase_started_monotonic"]) for row in fanout), default=0.0)
    phase_end = max((float(row["phase_ended_monotonic"]) for row in fanout), default=0.0)
    shared_wall = max(0.0, phase_end - phase_start)
    total_rows = sum(int(row["active_row_steps"]) for row in fanout)
    aggregate = total_rows / shared_wall if shared_wall else None
    return {
        "schema": "t34-exit-cpu-runtime-profile/v1",
        "authority": "excluded diagnostic; not A5, Round 0, gate, or serving evidence",
        "namespace": DIAGNOSTIC_NAMESPACE,
        "provenance": provenance(),
        "hardware": hardware,
        "recipe": {
            "device": "cpu",
            "workers": workers,
            "effective_cpu_workers": effective,
            "resident_worlds_per_process": 1,
            "threads_per_process": 1,
            "window_seconds": seconds,
            "throughput_denominator": "post-warmup phase wall includes per-case reset and setup",
            "mixes": list(MIXES),
            "body_regimes": list(BODY_REGIMES),
        },
        "alone": alone[0],
        "fanout": {
            "per_process": fanout,
            "shared_phase_wall_seconds": shared_wall,
            "shared_phase_started_monotonic": phase_start,
            "shared_phase_ended_monotonic": phase_end,
            "aggregate_active_row_steps_per_second": aggregate,
            "sum_of_per_process_rates": sum(
                float(row["active_row_steps_per_second"] or 0) for row in fanout
            ),
        },
        "profiler_overhead": {
            "separate_uninstrumented_estimate_seconds": None,
            "reason": "no duplicate workload is run; wrappers are included in wall time",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seconds", type=float, default=float(os.environ.get("T34_PROFILE_SECONDS", "60"))
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=(
            int(os.environ["T34_WORKERS"])
            if "T34_WORKERS" in os.environ
            else int(os.environ["T34_NPROC"]) if "T34_NPROC" in os.environ else None
        ),
    )
    parser.add_argument("--hardware-only", action="store_true")
    args = parser.parse_args()
    if args.hardware_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / "hardware-first.json"
        if target.exists():
            raise ValueError(f"hardware receipt already exists: {target}")
        _write_completed(
            target,
            {
                "schema": "t34-exit-cpu-runtime-hardware/v1",
                "hardware": runtime_hardware(),
                "provenance": provenance(),
            },
        )
        print(json.dumps({"hardware": str(target)}, sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --hardware-only")
    receipt = run_profile(
        args.checkpoint.resolve(), args.output_dir, seconds=args.seconds, workers=args.workers
    )
    target = args.output_dir / "profile-cpu.json"
    _write_completed(target, receipt)
    print(
        json.dumps(
            {
                "profile": str(target),
                "aggregate_rows_per_second": receipt["fanout"][
                    "aggregate_active_row_steps_per_second"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
