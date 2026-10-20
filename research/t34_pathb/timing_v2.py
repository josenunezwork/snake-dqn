"""T15-style ON/OFF paced timing probes for Path B v2."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping


class ProbeFault(RuntimeError):
    """An unconditional controller fault encountered during a timing probe."""


def _step(session: Any) -> None:
    session.step()
    state = session.controller_state() if hasattr(session, "controller_state") else {}
    faults = state.get("unconditional_faults", [])
    if faults:
        fault = faults[-1]
        raise ProbeFault(str(fault.get("marker", fault)) if isinstance(fault, dict) else str(fault))
    counts = state.get("counts", {})
    if counts.get("worker_errors", 0) or counts.get("invalid_results", 0):
        raise RuntimeError("timing assistance identity is not verified after worker failure")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty timing distribution")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def probe(
    session: Any,
    *,
    fps: int,
    ticks: int,
    warmup: int = 200,
    contended: bool = False,
    check_cap: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run real paced steps; no GameSession queue is drained outside its tick path."""
    if fps not in (12, 24) or ticks < 1 or warmup < 0:
        raise ValueError("invalid T15 probe settings")
    for _ in range(warmup):
        if check_cap:
            check_cap()
        _step(session)
        time.sleep(1 / fps)
    updates: list[float] = []
    full: list[float] = []
    upkeep: list[float] = []
    first_result: list[float] = []
    overruns = 0
    cpu_total = 0.0
    paced_started = time.monotonic()
    for _ in range(ticks):
        if check_cap:
            check_cap()
        started = time.monotonic()
        before = time.process_time()
        _step(session)
        elapsed = (time.monotonic() - started) * 1000
        cpu_ms = (time.process_time() - before) * 1000
        cpu_total += cpu_ms
        update = float(getattr(session, "last_update_ms", elapsed))
        updates.append(update)
        full.append(elapsed)
        upkeep.append(float(getattr(session, "shadow_upkeep_ms", 0.0)))
        value = getattr(session, "first_result_latency_ms", None)
        if value is not None:
            first_result.append(float(value))
        overruns += int(elapsed >= 1000 / fps)
        time.sleep(max(0.0, 1 / fps - (time.monotonic() - started)))
    return {
        "fps": fps,
        "ticks": ticks,
        "warmup": warmup,
        "contended": bool(contended),
        "update_ms": _summary(updates),
        "full_tick_ms": _summary(full),
        "shadow_upkeep_ms": _summary(upkeep),
        "shadow_upkeep_core_share": sum(upkeep) / (ticks * 1000 / fps),
        "first_result_latency_ms": (
            _summary(first_result) if first_result else {"status": "NOT_OBSERVED", "n": 0}
        ),
        "parent_cpu_ms": cpu_ms,
        "parent_cpu_total_ms": cpu_total,
        "elapsed_seconds": time.monotonic() - paced_started,
        "achieved_fps": ticks / (time.monotonic() - paced_started),
        "overruns": overruns,
    }


def run_probes(
    open_session: Callable[[str, int], Any],
    *,
    ticks: int,
    warmup: int,
    contended: bool,
    check_cap: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Open and close four exact Watch6 sessions, ON/OFF at the two serving rates."""
    probes: dict[str, Mapping[str, Any]] = {}
    for mode in ("bare", "paced"):
        for fps in (12, 24):
            session = open_session(mode, fps)
            try:
                probes[f"{mode}-{fps}"] = probe(
                    session,
                    fps=fps,
                    ticks=ticks,
                    warmup=warmup,
                    contended=contended,
                    check_cap=check_cap,
                )
            finally:
                session.close(wait=True)
    return {"schema": "t34-pathb-v2-timing/v1", "probes": probes}
