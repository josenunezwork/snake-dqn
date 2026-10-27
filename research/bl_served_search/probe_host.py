"""Off-tick Mac admission, exclusive lease and bounded host evidence for Build 0."""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
from research.bl_mac_latency import host
from research.bl_paced_identity.record import export_json
from research.mac_lease import held

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
INTERVAL_SECONDS = 10.0
HEAVY_PERCENT = 20.0


def environment() -> dict[str, Any]:
    """Require one-thread CPU execution and the requested low test priority."""
    value = {
        "platform": platform.platform(),
        "threads": {key: os.environ.get(key) for key in THREAD_ENV},
        "device": os.environ.get("SNAKE_DQN_DEVICE"),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "qos": "ordinary inherited service policy; no affinity or QoS mutation",
    }
    if value["device"] != "cpu" or any(v != "1" for v in value["threads"].values()):
        raise RuntimeError("require SNAKE_DQN_DEVICE=cpu and all five math thread caps=1")
    if value["nice"] < 15:
        raise RuntimeError("require nice >= 15")
    return value


def foreign_load(screen: dict, owner_pid: int) -> list[dict]:
    """Keep ordinary daemons, but refuse foreign processes using >=20% of one CPU.

    This coordinator, its descendants and ordinary system daemons are exempt.
    System identity follows the existing M1 executable-location rule. No process is stopped.
    Raw process inventory and the exact exemption set are retained in every sample.
    """
    import psutil

    owner = psutil.Process(owner_pid)
    allowed = {owner_pid, *(process.pid for process in owner.children(recursive=True))}
    screen["probe_exempt_pids"] = sorted(allowed)
    rows = []
    background = []
    for line in screen["processes"].splitlines()[1:]:
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            raise RuntimeError("unparseable process screen")
        pid, ppid, cpu = int(fields[0]), int(fields[1]), float(fields[2])
        row = {"pid": pid, "ppid": ppid, "cpu_percent": cpu, "command": fields[3]}
        if fields[3].startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(fields[3]).parts:
            background.append(row)
        # A point-sampled owned helper (notably ps itself) may have exited
        # before children() runs. Its sampled parent still proves ownership.
        elif pid not in allowed and ppid not in allowed and cpu >= HEAVY_PERCENT:
            rows.append(row)
    screen["ordinary_system_background"] = background
    screen["probe_threshold_percent_one_core"] = HEAVY_PERCENT
    return rows


def preflight() -> dict[str, Any]:
    """Read-only check: no lease, game imports, output creation or process changes."""
    env = environment()
    observation = host.snapshot(process_policy=lambda screen: foreign_load(screen, os.getpid()))
    return {"environment": env, "host": observation, "admitted": bool(observation["quiet"])}


def _owned_inventory(owner_pid: int) -> list[dict[str, Any]]:
    """Retain orderly owned-child exits during a point sample as lifecycle evidence."""
    import psutil

    rows = []
    for process in [psutil.Process(owner_pid), *psutil.Process(owner_pid).children()]:
        try:
            rows.append(
                {
                    "pid": process.pid,
                    "rss_bytes": process.memory_info().rss,
                    "threads": process.num_threads(),
                    "nice": process.nice(),
                    "argv": process.cmdline(),
                }
            )
        except psutil.NoSuchProcess:
            rows.append({"pid": process.pid, "status": "exited_during_sample"})
    return rows


def _monitor(
    owner_pid: int, output: str, seconds: float, ready: Any, stop: Any, failed: Any, heartbeat: Any
) -> None:
    """Sample in an independent interpreter; shell/RSS work never holds tick GIL."""
    observations = []
    reason = None
    started = time.monotonic()
    try:
        # Bounded by both lifetime and count, including final sample.
        limit = int(seconds / INTERVAL_SECONDS) + 3
        while len(observations) < limit:
            observation = host.snapshot(
                process_policy=lambda screen: foreign_load(screen, owner_pid)
            )
            observation["probe_processes"] = _owned_inventory(owner_pid)
            observations.append(observation)
            heartbeat.value = time.monotonic()
            ready.set()
            if not observation["quiet"]:
                raise RuntimeError("foreign heavy load; retain entire attempt as contaminated")
            if time.monotonic() - started >= seconds:
                raise TimeoutError("7200-second diagnostic cap reached")
            if stop.is_set():
                break
            stop.wait(INTERVAL_SECONDS)
        else:
            raise RuntimeError("host evidence capacity exhausted")
    except BaseException as exc:
        failed.set()
        reason = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, host.GuardError):
            observations.append(exc.observation)
        ready.set()
    finally:
        export_json(
            Path(output) / "host.json",
            {
                "schema": "served-tick-host/v1",
                "status": "INVALID" if reason else "COMPLETE",
                "reason": reason,
                "interval_seconds": INTERVAL_SECONDS,
                "heavy_percent_one_core": HEAVY_PERCENT,
                "elapsed_seconds": time.monotonic() - started,
                "observations": observations,
            },
        )


class HostMonitor:
    """Shared-event guard interface; check() performs no shell calls or disk I/O."""

    def __init__(self, output: Path, seconds: float = 7200) -> None:
        self.output, self.seconds = Path(output), seconds
        context = mp.get_context("spawn")
        self.ready, self.stop, self.failed = context.Event(), context.Event(), context.Event()
        self.heartbeat = context.Value("d", 0.0, lock=False)
        self.process = context.Process(
            target=_monitor,
            args=(
                os.getpid(),
                str(self.output),
                seconds,
                self.ready,
                self.stop,
                self.failed,
                self.heartbeat,
            ),
            name="served-tick-host-monitor",
        )
        self.receipt: dict[str, Any] = {}

    def check(self) -> None:
        """Refuse lost monitor or any contaminated sample; never restart a cell."""
        if self.failed.is_set():
            raise RuntimeError("host guard failed; inspect create-only host.json")
        if not self.process.is_alive() or time.monotonic() - self.heartbeat.value > 45:
            raise RuntimeError("host monitor unavailable or heartbeat stale")

    def __enter__(self) -> "HostMonitor":
        self.process.start()
        try:
            if not self.ready.wait(40):
                raise RuntimeError("host monitor startup timeout")
            self.check()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.process.join(40)
        if self.process.is_alive():
            self.process.terminate()  # Only our exact, still-owned monitor child.
            self.process.join(5)
            self.failed.set()
        self.receipt["host_path"] = str(self.output / "host.json")
        self.receipt["monitor_exitcode"] = self.process.exitcode
        if args[0] is None and (self.failed.is_set() or self.process.exitcode != 0):
            raise RuntimeError("host monitor failed; no uncontended timing claim")


def guarded_run(output: Path, callback: Callable[[HostMonitor], Any]) -> Any:
    """Acquire the shared lease; record all samples without retries or cherry-picking."""
    environment()
    with ExitStack() as stack:
        lease = stack.enter_context(
            held("sol-served-search-20261026", "Build 0 Watch6 timing diagnostic", seconds=7200)
        )
        monitor = HostMonitor(output)
        monitor.receipt["lease"] = lease.receipt
        stack.enter_context(monitor)
        return callback(monitor)
