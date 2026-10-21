"""E3 scheduling headroom, independent of reconstruction and student math."""

from __future__ import annotations

import os

RESERVED_CPUS = 2
MIN_NICE = 10
PROGRESS_SECONDS = 120.0


def available_cpus() -> int:
    """Count usable CPUs, respecting Linux affinity (as nproc does)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def worker_count(requested: int, cpus: int | None = None) -> int:
    """Leave two CPUs for the kernel, watchdog and coordinator; fail closed."""
    cpus = available_cpus() if cpus is None else cpus
    if not 1 <= requested <= 12 or cpus <= RESERVED_CPUS:
        raise ValueError("E3 needs valid workers and at least three usable CPUs")
    return min(requested, cpus - RESERVED_CPUS)


def lower_priority() -> int:
    """Use nice 10 or lower scheduling priority, including spawned children."""
    current = os.nice(0)
    if current < MIN_NICE:
        current = os.nice(MIN_NICE - current)
    return current
