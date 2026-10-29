"""Bounded exact-P CPU profiling; no scientific outcomes cross the export boundary."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import psutil
import torch

from research.bl_served_search.exports import export_json
from research.watch_lead_p import engine
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p.run import lower_worker_priority
from research.watch_lead_p.uploaded import uploaded_checkpoints

WINDOWS = {"P0": 8, "P48": 64, "bare": 64}
ROOTS: dict[int, bytes | None] = {}


def measurement(row: dict, wall: float, cpu: float, submit_s: float, calls: int) -> dict:
    """Allow only timing, workload and opaque state/trajectory identities."""
    return {
        **{
            key: row[key]
            for key in ("seed", "arm", "frames", "trace_sha256", "initial_state", "final_state")
        },
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "cpu_percent_one_core": 100 * cpu / wall,
        "seconds_per_frame": wall / row["frames"],
        "submit_seconds": submit_s,
        "submit_calls": calls,
        "non_submit_seconds": wall - submit_s,
    }


def root(seed: int) -> tuple[bytes | None, dict]:
    """First natural alive L600 root, never replacement or artificial geometry."""
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
        "root_sha256": hashlib.sha256(payload).hexdigest() if payload else None,
    }


def task(seed: int, arm: str, frames: int, payload: bytes | None) -> dict:
    """Call unchanged P run_arm; only initial constructor and namespace differ."""

    class RootWorld:
        restore = staticmethod(LiveWorld.restore)

        def __new__(cls, value: int) -> LiveWorld:
            if value != seed or payload is None:
                raise ValueError("foreign/missing engineering root")
            return LiveWorld.restore(payload)

    submit_s = search_s = 0.0
    calls = searches = 0
    original_submit, original_search = engine.submit, engine.search

    def timed_submit(*args: Any, **kwargs: Any) -> Any:
        nonlocal submit_s, calls
        start = time.monotonic()
        result = original_submit(*args, **kwargs)
        submit_s += time.monotonic() - start
        calls += 1
        return result

    def timed_search(*args: Any, **kwargs: Any) -> Any:
        nonlocal search_s, searches
        start = time.monotonic()
        result = original_search(*args, **kwargs)
        search_s += time.monotonic() - start
        searches += 1
        return result

    def pulse(_row: dict) -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(900, exit=True)

    from research.watch_lead_p_successor.prepare import STUDY

    constructor = (
        patch.object(engine, "LiveWorld", RootWorld) if payload else contextlib.nullcontext()
    )
    started, cpu = time.monotonic(), time.process_time()
    with (
        constructor,
        patch.object(engine, "NAMESPACE", f"a100-dev/{STUDY}/"),
        patch.object(engine, "submit", timed_submit),
        patch.object(engine, "search", timed_search),
    ):
        row = engine.run_arm(seed, arm, frames=frames, pulse=pulse)
    ended = time.monotonic()
    return {
        **measurement(row, ended - started, time.process_time() - cpu, submit_s, calls),
        "search_seconds": search_s,
        "search_calls": searches,
        "started_monotonic": started,
        "ended_monotonic": ended,
        "pid": os.getpid(),
    }


def worker(tasks: Any, results: Any, ready: Any, roots: dict, seed: int, scratch: str) -> None:
    """Initialize each fixed worker once; there is no executor respawn thread."""
    global ROOTS
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGUSR1})
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, signal.SIG_DFL)
    lower_worker_priority()
    faulthandler.dump_traceback_later(900, exit=True)
    try:
        initialized = time.monotonic()
        with uploaded_checkpoints():
            # Real construction establishes process-global 61-feature config.
            primer = LiveWorld(seed)
            primer_started = time.monotonic()
            primer.step()  # Compile/warm actual features and opponent CPU paths.
            primer_seconds = time.monotonic() - primer_started
            if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
                raise ValueError("P worker requires one torch intra/interop thread")
            ROOTS = {}
            for key, value in roots.items():
                payload = Path(value["path"]).read_bytes() if value else None
                if payload and hashlib.sha256(payload).hexdigest() != value["sha256"]:
                    raise ValueError("engineering root transfer hash mismatch")
                ROOTS[key] = payload
            ready.put(
                {
                    "pid": os.getpid(),
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "initialization_seconds": time.monotonic() - initialized,
                    "primer_step_seconds": primer_seconds,
                }
            )
            while True:
                job_id, kind, args = tasks.get()
                try:
                    if kind == "root":
                        payload, metadata = root(*args)
                        path = Path(scratch) / f"root-{job_id}.bin"
                        if payload:
                            with path.open("xb") as handle:
                                handle.write(payload)
                        value = (str(path) if payload else None, metadata)
                    elif kind == "tail-scan":
                        from research.watch_lead_p_successor.followup import scan

                        value = scan(*args, scratch=Path(scratch))
                    elif kind == "tail-window":
                        from research.watch_lead_p_successor.followup import detail_task

                        value = detail_task(*args, ROOTS[args[0]])
                    elif kind == "tail-stress":
                        from research.watch_lead_p_successor.followup import stress_task

                        value = stress_task(*args, ROOTS[args[0]])
                    else:
                        value = task(*args, ROOTS[args[0]])
                    results.put((job_id, True, value))
                except BaseException as exc:
                    results.put((job_id, False, repr(exc)))
    except BaseException as exc:
        ready.put({"pid": os.getpid(), "error": repr(exc)})
    finally:
        faulthandler.cancel_dump_traceback_later()


def reap(processes: list[mp.Process]) -> None:
    """No private executor dictionaries, manager threads or replacement spawns."""
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.kill()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=2)
            if process.is_alive():
                raise RuntimeError("P profile failed to reap owned worker")


class Workers:
    """Fixed owned spawn processes; warm all workers before submitting work."""

    def __init__(self, count: int, roots: dict, seed: int, deadline: float) -> None:
        ctx = mp.get_context("spawn")
        self.tasks, self.results, self.ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.scratch = tempfile.TemporaryDirectory(prefix="p-profile-workers-")
        self.processes: list[mp.Process] = []
        self.boot = []
        started = time.monotonic()
        try:
            staged = {}
            for key, payload in roots.items():
                if payload is None:
                    staged[key] = None
                    continue
                path = Path(self.scratch.name) / f"input-{key}.bin"
                path.write_bytes(payload)
                staged[key] = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
            for _ in range(count):
                process = ctx.Process(
                    target=worker,
                    args=(self.tasks, self.results, self.ready, staged, seed, self.scratch.name),
                )
                self.processes.append(process)
                # Deferral covers the fork-to-PID registration window. Children
                # explicitly unblock these signals before initializing.
                previous = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGUSR1}
                )
                try:
                    process.start()
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            last_print = started
            while len(self.boot) < count:
                self.check(deadline)
                if time.monotonic() - last_print >= 30:
                    print(f"P_PROFILE_WARM ready={len(self.boot)}/{count}", flush=True)
                    last_print = time.monotonic()
                try:
                    record = self.ready.get(timeout=min(1, max(0, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if "error" in record:
                    raise RuntimeError(record["error"])
                self.boot.append(record)
            self.spawn_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise

    def check(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded engineering deadline")
        if any(not p.is_alive() for p in self.processes):
            raise RuntimeError("profile worker exited; no automatic replacement")

    def close(self) -> None:
        reap(self.processes)
        for channel in (self.tasks, self.results, self.ready):
            channel.cancel_join_thread()
            channel.close()
        self.scratch.cleanup()


def host_record() -> dict:
    """CPU shape, effective quota/affinity, load, memory and thread settings."""
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
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
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


def concurrency(rows: list[dict]) -> None:
    """Time-weighted overlapping measured windows, not nominal worker count."""
    for row in rows:
        start, end = row["start_elapsed"], row["end_elapsed"]
        row["mean_concurrent_tasks"] = sum(
            max(0, min(end, other["end_elapsed"]) - max(start, other["start_elapsed"]))
            for other in rows
        ) / (end - start)


def sample(pool: Workers, started: float) -> dict:
    cpu = rss = 0.0
    for process in pool.processes:
        p = psutil.Process(process.pid)
        times = p.cpu_times()
        cpu += times.user + times.system
        rss += p.memory_info().rss
    return {
        "elapsed": time.monotonic() - started,
        "child_cpu_seconds": cpu,
        "child_rss_bytes": rss,
        "host_cpu_percent": psutil.cpu_percent(),
        "load": os.getloadavg(),
    }


def sweep(
    jobs: list[tuple],
    roots: dict,
    workers: int,
    output: Path,
    deadline: float,
    label: str,
    *,
    kind: str = "window",
) -> dict:
    """Identical fixed task sets, warm worker barrier, independent spawn clock."""
    pool = Workers(workers, roots, jobs[0][0], deadline)
    try:
        psutil.cpu_percent()  # Prime after all workers have finished warming.
        started = time.monotonic()
        samples = [sample(pool, started)]
        samples[0]["elapsed"] = 0.0
        for i, args in enumerate(jobs):
            pool.tasks.put((i, kind, args))
        rows = []
        last_print = started
        while len(rows) < len(jobs):
            pool.check(deadline)
            try:
                index, success, row = pool.results.get(
                    timeout=min(1, max(0, deadline - time.monotonic()))
                )
            except queue.Empty:
                row = None
            if row is not None:
                if not success:
                    raise RuntimeError(row)
                row.update(
                    task_index=index,
                    start_elapsed=row.pop("started_monotonic") - started,
                    end_elapsed=row.pop("ended_monotonic") - started,
                )
                export_json(output / f"{label}-w{workers}-task{index}.json", row)
                rows.append(row)
                print(
                    f"P_PROFILE_FINISH cell={label} w={workers} task={index} "
                    f"arm={row['arm']} wall={row['wall_seconds']:.6f}",
                    flush=True,
                )
            samples.append(sample(pool, started))
            if time.monotonic() - last_print >= 30:
                print(
                    f"P_PROFILE cell={label} w={workers} done={len(rows)}/{len(jobs)}", flush=True
                )
                last_print = time.monotonic()
        concurrency(rows)
        result = {
            "workers": workers,
            "pool_spawn_seconds": pool.spawn_seconds,
            "worker_boot": pool.boot,
            "wall_seconds": time.monotonic() - started,
            "rows": sorted(rows, key=lambda r: r["task_index"]),
            "samples": samples,
            "label": label,
        }
        export_json(output / f"{label}-w{workers}.json", result)
        return result
    finally:
        pool.close()


def prepare_roots(
    seeds: list[int], output: Path, deadline: float, workers: int = 4, cache: bool = False
) -> dict[int, bytes]:
    """Fixed engineering bank; unsupported roots refuse, never replace a seed."""
    pool = Workers(workers, {}, seeds[0], deadline)
    roots = {}
    try:
        for seed in seeds:
            pool.tasks.put((seed, "root", (seed,)))
        last_print = time.monotonic()
        while len(roots) < len(seeds):
            pool.check(deadline)
            if time.monotonic() - last_print >= 30:
                print(f"P_PROFILE_PREFIX done={len(roots)}/{len(seeds)}", flush=True)
                last_print = time.monotonic()
            try:
                seed, success, value = pool.results.get(
                    timeout=min(1, max(0, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if not success:
                raise RuntimeError(value)
            path, metadata = value
            payload = Path(path).read_bytes() if path else None
            if payload and hashlib.sha256(payload).hexdigest() != metadata["root_sha256"]:
                raise ValueError("engineering root result hash mismatch")
            export_json(output / f"prefix-{seed}.json", metadata)
            if payload is None:
                raise ValueError("unsupported mature dev root; no replacement")
            roots[seed] = payload
            if cache:
                with (output / f"root-{seed}.bin").open("xb") as handle:
                    handle.write(payload)
            print(f"P_PROFILE_PREFIX done={len(roots)}/{len(seeds)}", flush=True)
        return roots
    finally:
        pool.close()


@contextlib.contextmanager
def coordinator_signals() -> Iterator[None]:
    """Caught parent signals unwind cleanup; workers reset both to default."""

    stopping = False

    def stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        raise InterruptedError(f"profile coordinator signal {signum}")

    old = {s: signal.signal(s, stop) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def execute(bank: Path, output: Path, *, smoke: bool = False, check_roots: bool = False) -> None:
    from research.watch_lead_p_successor.prepare import bank_record

    if json.loads(bank.read_text()) != bank_record():
        raise ValueError("foreign profile bank")
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "host.json", host_record())
    started = time.monotonic()
    deadline = started + (120 if smoke else 900)
    seeds = bank_record()["seeds"][:1] if smoke else bank_record()["seeds"]
    with coordinator_signals():
        if smoke:
            with uploaded_checkpoints():
                from research.t1_commitment_20261015.replay import configure_cpu

                configure_cpu()
                payload = LiveWorld(seeds[0]).snapshot()
            roots = {seeds[0]: payload}
        else:
            roots = prepare_roots(
                seeds, output, deadline, workers=1 if check_roots else 4, cache=check_roots
            )
        grid = (1,) if check_roots else ((1, 2) if smoke else (1, 4, 8, 11))
        results = []
        for count in grid:
            jobs = [
                (seed, arm, 2 if smoke else WINDOWS[arm])
                for seed in seeds
                for arm in ("P0", "P48", "bare")
            ]
            selected = {seeds[0]: None} if smoke and count == 1 else roots
            results.append(sweep(jobs, selected, count, output, deadline, "arms"))
        contention = []
        if not smoke and not check_roots:
            jobs = [(seeds[i % len(seeds)], "P0", 1) for i in range(11)]
            for count in grid:
                contention.append(sweep(jobs, roots, count, output, deadline, "contention"))
        identities = [
            [(r["seed"], r["arm"], r["trace_sha256"], r["initial_state"]) for r in result["rows"]]
            for result in results
        ]
        if any(value != identities[0] for value in identities[1:]):
            raise ValueError("serial/parallel trajectory identity differs")
        contention_identities = [
            [(r["seed"], r["trace_sha256"], r["initial_state"]) for r in result["rows"]]
            for result in contention
        ]
        if any(value != contention_identities[0] for value in contention_identities[1:]):
            raise ValueError("homogeneous contention trajectory identity differs")
        supported = not smoke and all(
            r["search_calls"] > 0
            for result in results
            for r in result["rows"]
            if r["arm"] != "bare"
        )
        export_json(
            output / "profile.json",
            {
                "engineering_only": True,
                "smoke": smoke,
                "check_roots": check_roots,
                "complete": True,
                "mature_search_supported": supported,
                "serial_parallel_identity": True,
                "successor_envelope_qualified": False,
                "reason": "short windows do not measure full-session long-body/late-search tails",
                "sweeps": [
                    {
                        "workers": r["workers"],
                        "label": r["label"],
                        "wall_seconds": r["wall_seconds"],
                        "spawn_seconds": r["pool_spawn_seconds"],
                        "speedup": (results[0] if r["label"] == "arms" else contention[0])[
                            "wall_seconds"
                        ]
                        / r["wall_seconds"],
                    }
                    for r in results + contention
                ],
            },
        )
        if not smoke and not supported:
            raise ValueError("actual H300 search unsupported; profile cannot qualify")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--check-roots", action="store_true")
    args = parser.parse_args()
    execute(args.bank, args.output, smoke=args.smoke, check_roots=args.check_roots)


if __name__ == "__main__":
    main()
