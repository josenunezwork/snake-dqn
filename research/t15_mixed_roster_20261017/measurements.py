"""Serving-path latency measurements for T15's opt-in mixed Watch roster."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Sequence

from research.t15_mixed_roster_20261017.episodes import new_session, roster_identity

WARMUP = 200
SAMPLES = 1500
PACED_TICKS = 120
EXPECTED_CEILINGS = {6: 24.0, 8: 12.0, 12: 12.0}


def _percentile(ordered: Sequence[float], q: float) -> float:
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return float(
        ordered[low]
        if low == high
        else ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    )


def _distribution(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise RuntimeError("no latency samples")
    ordered = sorted(float(sample) for sample in samples)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _timed_tick(session: Any) -> tuple[float, float]:
    original = session.game.update
    update_ms = 0.0

    def timed(*args: Any, **kwargs: Any) -> Any:
        nonlocal update_ms
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            update_ms = (time.perf_counter() - started) * 1000.0

    session.game.update = timed
    started = time.perf_counter()
    try:
        before = int(session.game.frame)
        session.step()
        session.snapshot()
        if int(session.game.frame) != before + 1 or session.last_error:
            raise RuntimeError(f"mixed latency tick failed: {session.last_error!r}")
    finally:
        session.game.update = original
    return update_ms, (time.perf_counter() - started) * 1000.0


def _paced(seed: int, num_snakes: int, fps: int) -> dict[str, Any]:
    session = new_session(seed, num_snakes)
    session.set_speed(fps)
    requested = float(session.speed)
    if requested <= 0:
        raise RuntimeError(
            f"mixed server returned non-positive speed at Watch{num_snakes}: {requested}"
        )
    for _ in range(WARMUP):
        _timed_tick(session)
    # The 120 request deliberately reaches beyond lower roster ceilings.
    # ``set_speed`` is server authority, so the retained value is the rate
    # whose zero-overrun claim this probe can establish.
    budget = 1.0 / requested
    ticks, overruns = [], 0
    started = time.perf_counter()
    for _ in range(PACED_TICKS):
        _, tick_ms = _timed_tick(session)
        ticks.append(tick_ms)
        if tick_ms >= budget * 1000.0:
            overruns += 1
        time.sleep(budget)
    elapsed_seconds = time.perf_counter() - started
    return {
        "requested_fps": fps,
        "actual_speed": session.speed,
        "speed_ceiling": session.speed_ceiling,
        "total_ai_count": len(session.game.snakes),
        "warmup": WARMUP,
        "ticks": PACED_TICKS,
        "raw_tick_ms": ticks,
        "tick_ms": _distribution(ticks),
        "overruns": overruns,
        "zero_overrun": overruns == 0,
        "elapsed_seconds": elapsed_seconds,
        "actual_achieved_fps": PACED_TICKS / elapsed_seconds,
        "roster": roster_identity(session),
    }


def run_latency(seed: int, num_snakes: int = 6) -> dict[str, Any]:
    """Measure 1,500 real Watch ticks and paced 12/120 FPS ceiling probes."""
    session = new_session(seed, num_snakes)
    for _ in range(WARMUP):
        _timed_tick(session)
    updates, full_ticks = [], []
    for _ in range(SAMPLES):
        update_ms, tick_ms = _timed_tick(session)
        updates.append(update_ms)
        full_ticks.append(tick_ms)
    update, tick = _distribution(updates), _distribution(full_ticks)
    paced = [_paced(seed, num_snakes, fps) for fps in (12, 120)]
    errors = []
    if float(update["p95"]) >= 66.667:
        errors.append(f"update p95 {update['p95']:.3f} ms is not below 66.667 ms")
    default_probe, ceiling_probe = paced
    if float(default_probe["actual_speed"]) != 12.0:
        errors.append(f"default 12 fps was not accepted ({default_probe['actual_speed']})")
    expected_ceiling = EXPECTED_CEILINGS.get(int(num_snakes))
    if expected_ceiling is not None and float(ceiling_probe["actual_speed"]) != expected_ceiling:
        errors.append(
            f"Watch{num_snakes} ceiling {ceiling_probe['actual_speed']} != expected "
            f"{expected_ceiling}"
        )
    for probe in paced:
        if not probe["zero_overrun"]:
            errors.append(f"{probe['requested_fps']} fps request had {probe['overruns']} overruns")
    return {
        "schema_version": "t15-mixed-watch-latency/v1",
        "seed": int(seed),
        "num_snakes": int(num_snakes),
        "warmup": WARMUP,
        "samples": {"update_ms": updates, "tick_ms": full_ticks},
        "update_ms": update,
        "tick_ms": tick,
        "p95_threshold_ms": 66.667,
        "paced": paced,
        "roster": roster_identity(session),
        "pass": not errors,
        "errors": errors,
    }
