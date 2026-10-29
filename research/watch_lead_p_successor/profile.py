"""Bounded exact-P CPU profiling, exporting timing and opaque identity only."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import faulthandler
import json
import multiprocessing
import os
import platform
import signal
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psutil

from research.bl_served_search.exports import export_json
from research.watch_lead_p import engine
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p.run import lower_worker_priority
from research.watch_lead_p.uploaded import uploaded_checkpoints


def measurement(row: dict, wall: float, cpu: float, submit_s: float, calls: int) -> dict:
    """Explicit allowlist: never export death, mass, choices or search outcomes."""
    return {
        **{key: row[key] for key in ("seed", "arm", "frames", "trace_sha256", "initial_state")},
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "cpu_percent_one_core": 100 * cpu / wall,
        "seconds_per_frame": wall / row["frames"],
        "submit_seconds": submit_s,
        "submit_calls": calls,
        "non_submit_seconds": wall - submit_s,
    }


def root(seed: int) -> tuple[bytes | None, dict]:
    """Build each natural mature root once, independently of concurrency sweeps."""
    lower_worker_priority()
    faulthandler.dump_traceback_later(900, exit=True)
    try:
        with uploaded_checkpoints():
            started, cpu = time.monotonic(), time.process_time()
            world = LiveWorld(seed)
            while world.frame < 6000 and (not world.hero.is_alive or world.hero.length < 600):
                world.step()
            supported = world.hero.is_alive and world.hero.length >= 600
            payload = world.snapshot() if supported else None
            return payload, {
                "seed": seed,
                "prefix_frames": world.frame,
                "supported": supported,
                "prefix_seconds": time.monotonic() - started,
                "prefix_cpu_seconds": time.process_time() - cpu,
            }
    finally:
        faulthandler.cancel_dump_traceback_later()


def task(seed: int, arm: str, frames: int, payload: bytes | None) -> dict:
    """Run unmodified run_arm, optionally from a natural mature dev snapshot.

    Only the constructor seam and identity namespace are patched. No geometry,
    death event, proposal, search, delay, precision or delivery rule is patched.
    The natural prefix cost is measured separately and excluded from window rate.
    """
    lower_worker_priority()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    faulthandler.dump_traceback_later(900, exit=True)
    try:
        with uploaded_checkpoints():
            setup_wall, setup_cpu = time.monotonic(), time.process_time()
            if payload:
                # GameConfig has process-global feature settings. A spawned
                # worker must establish the real session config before restoring
                # a root; otherwise cold workers can featurize 58 instead of 61.
                LiveWorld(seed)
            expected_initial = LiveWorld.restore(payload).digest() if payload else None
            setup = {
                "setup_wall_seconds": time.monotonic() - setup_wall,
                "setup_cpu_seconds": time.process_time() - setup_cpu,
            }

            class RootWorld:
                """Fresh independent exact restore of the fixed dev root."""

                restore = staticmethod(LiveWorld.restore)

                def __new__(cls, value: int) -> LiveWorld:
                    if value != seed:
                        raise ValueError("foreign profile seed")
                    if payload is None:
                        raise ValueError("missing engineering root")
                    return LiveWorld.restore(payload)

            submit_s = 0.0
            calls = 0
            original_submit = engine.submit

            def timed_submit(*args: Any, **kwargs: Any) -> Any:
                nonlocal submit_s, calls
                start = time.monotonic()
                result = original_submit(*args, **kwargs)
                submit_s += time.monotonic() - start
                calls += 1
                return result

            def pulse(_row: dict) -> None:
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(900, exit=True)

            # Keep the natural-start smoke directly on the original constructor.
            constructor = (
                patch.object(engine, "LiveWorld", RootWorld)
                if payload
                else contextlib.nullcontext()
            )
            wall, cpu = time.monotonic(), time.process_time()
            with (
                constructor,
                patch.object(engine, "NAMESPACE", "a100-dev/p-successor-profile-v1/"),
                patch.object(engine, "submit", timed_submit),
            ):
                row = engine.run_arm(seed, arm, frames=frames, pulse=pulse)
            measured = measurement(
                row, time.monotonic() - wall, time.process_time() - cpu, submit_s, calls
            )
            if payload and measured["initial_state"] != expected_initial:
                raise ValueError("profile root identity mismatch")
            print(
                f"P_PROFILE_FINISH arm={arm} frames={frames} wall={measured['wall_seconds']:.6f}",
                flush=True,
            )
            return {**measured, **setup}
    finally:
        faulthandler.cancel_dump_traceback_later()


def host_record() -> dict:
    """Capture CPU shape and effective scheduling limits without assuming vCPUs."""
    cpuinfo = Path("/proc/cpuinfo")
    cgroup = {}
    for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        cgroup[name] = path.read_text() if path.is_file() else None
    proc = psutil.Process()
    return {
        "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "affinity": proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None,
        "nice": proc.nice(),
        "memory_bytes": psutil.virtual_memory().total,
        "cpuinfo": cpuinfo.read_text() if cpuinfo.is_file() else None,
        "cgroup": cgroup,
        "threads": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OMP_THREAD_LIMIT",
                "TORCHINDUCTOR_COMPILE_THREADS",
            )
        },
    }


def sweep(
    seeds: list[int],
    frames: int,
    workers: int,
    roots: dict[int, bytes | None],
    output: Path,
    deadline: float,
) -> dict:
    """Measure identical task sets at each concurrency; retain partial timing only."""
    pool = cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    started = time.monotonic()
    futures = {
        pool.submit(task, seed, arm, frames, roots[seed]): (seed, arm)
        for seed in seeds
        for arm in ("P0", "P48", "bare")
    }
    rows, samples = [], []
    proc = psutil.Process()
    last_print = 0.0
    try:
        while futures:
            if time.monotonic() >= deadline:
                raise TimeoutError("profile engineering deadline; no forecast qualification")
            done, _ = cf.wait(futures, timeout=1, return_when=cf.FIRST_COMPLETED)
            for future in done:
                seed, arm = futures.pop(future)
                row = future.result()
                export_json(output / f"w{workers}-{seed}-{arm}.json", row)
                rows.append(row)
            children = proc.children(recursive=True)
            rss = 0.0
            for child in children:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            # cpu_times (cumulative) survives psutil.Process instance recreation;
            # derive utilization below from successive CPU-second sums instead.
            cpu_seconds = 0.0
            for child in children:
                try:
                    times = child.cpu_times()
                    cpu_seconds += times.user + times.system
                except psutil.NoSuchProcess:
                    pass
            samples.append(
                {
                    "elapsed": time.monotonic() - started,
                    "child_cpu_seconds": cpu_seconds,
                    "child_rss_bytes": rss,
                    "host_cpu_percent": psutil.cpu_percent(),
                    "load": os.getloadavg(),
                }
            )
            if time.monotonic() - last_print >= 30:
                print(
                    f"P_PROFILE workers={workers} done={len(rows)}/{len(seeds)*3} "
                    f"elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
                last_print = time.monotonic()
        result = {
            "workers": workers,
            "wall_seconds": time.monotonic() - started,
            "rows": sorted(rows, key=lambda r: (r["seed"], r["arm"])),
            "samples": samples,
        }
        export_json(output / f"sweep-w{workers}.json", result)
        return result
    finally:
        reap(pool)


def reap(pool: cf.ProcessPoolExecutor) -> None:
    """Reap only this executor's children, with bounded forced cleanup."""
    processes = list((pool._processes or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2

    def running(process: multiprocessing.Process) -> bool:
        try:
            return psutil.Process(process.pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    alive = processes
    while alive and time.monotonic() < deadline:
        alive = [process for process in alive if running(process)]
        if alive:
            time.sleep(0.02)
    for process in alive:
        try:
            psutil.Process(process.pid).kill()
        except psutil.NoSuchProcess:
            pass
    pool.shutdown(wait=True, cancel_futures=True)
    for process in processes:
        process.join(timeout=1)
        if psutil.pid_exists(process.pid):
            raise RuntimeError("profile worker reaping failed")


def prepare_roots(seeds: list[int], output: Path, deadline: float) -> dict[int, bytes]:
    """Fixed four engineering seeds; no replacement of unsupported roots."""
    pool = cf.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    futures = {pool.submit(root, seed): seed for seed in seeds}
    roots = {}
    try:
        while futures:
            if time.monotonic() >= deadline:
                raise TimeoutError("natural root preparation deadline")
            done, _ = cf.wait(futures, timeout=30, return_when=cf.FIRST_COMPLETED)
            for future in done:
                seed = futures.pop(future)
                payload, meta = future.result()
                export_json(output / f"prefix-{seed}.json", meta)
                if payload is None:
                    raise ValueError("unsupported mature root; no replacement or forecast")
                roots[seed] = payload
            print(f"P_PROFILE_PREFIX done={len(roots)}/{len(seeds)}", flush=True)
        return roots
    finally:
        reap(pool)


def execute(bank: Path, output: Path, *, smoke: bool = False) -> None:
    """Fixed fresh bank; unsupported/partial sweeps cannot price the successor."""
    allocation = json.loads(bank.read_text())
    from research.watch_lead_p_successor.prepare import bank_record

    if allocation != bank_record():
        raise ValueError("foreign profile bank")
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "host.json", host_record())
    started = time.monotonic()
    results = []
    seeds = allocation["seeds"][:1] if smoke else allocation["seeds"]
    grid = (1, 2) if smoke else (1, 4, 8, 11)
    roots = {seed: None for seed in seeds} if smoke else prepare_roots(seeds, output, started + 900)
    if smoke:
        # Compare original construction with exact restore on a tiny real path.
        with uploaded_checkpoints():
            smoke_root = LiveWorld(seeds[0]).snapshot()
    for workers in grid:
        results.append(
            sweep(
                seeds,
                2 if smoke else 64,
                workers,
                {seeds[0]: smoke_root} if smoke and workers == 2 else roots,
                output,
                started + (120 if smoke else 900),
            )
        )
    baseline = results[0]["wall_seconds"]
    identities = [
        {
            (r["seed"], r["arm"]): (r.get("trace_sha256"), r.get("initial_state"))
            for r in result["rows"]
        }
        for result in results
    ]
    if any(record != identities[0] for record in identities[1:]):
        raise ValueError("serial/parallel trajectory identity differs")
    search_support = all(
        row.get("submit_calls", 0) > 0
        for result in results
        for row in result["rows"]
        if row["arm"] != "bare"
    )
    export_json(
        output / "profile.json",
        {
            "engineering_only": True,
            "smoke": smoke,
            "bank": allocation,
            "complete": True,
            "mature_search_supported": not smoke and search_support,
            "serial_parallel_identity": True,
            "successor_envelope_qualified": False,
            "reason": (
                "64-frame mature windows do not measure full-session long-body/late-branch tails"
            ),
            "sweeps": [
                {
                    "workers": r["workers"],
                    "wall_seconds": r["wall_seconds"],
                    "speedup": baseline / r["wall_seconds"],
                    "efficiency": baseline / r["wall_seconds"] / r["workers"],
                }
                for r in results
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    execute(args.bank, args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
