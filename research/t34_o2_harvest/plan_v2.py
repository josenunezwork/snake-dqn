"""Prospective conservative O2 chunk budgets; no provider calls."""

from __future__ import annotations

import math
from typing import Any

MIXES = ("scripted", "frozen", "mixed")
BASE_RATE_FLOOR = 200.0
BRANCH_RATE_FLOOR = 200.0
WORKERS = 10
RATE = 2.0
UNIT_CAP = 150.0


def chunk_budget(count: int) -> dict[str, Any]:
    """Price setup/export, a fixture wave, 3x slowdown and a twofold timeout."""
    if type(count) is not int or count < 1:
        raise ValueError("positive integer world count required")
    waves = math.ceil(count / WORKERS)
    seconds = (waves + 1) * 3 * (15000 / BASE_RATE_FLOOR + 48000 / BRANCH_RATE_FLOOR) + 600
    estimate = math.ceil(seconds / 60)
    timeout = 2 * estimate
    units = timeout * RATE / 60
    return {
        "forecast_seconds": seconds,
        "estimated_minutes": estimate,
        "timeout_minutes": timeout,
        "units_per_job": units,
        "reserve_units_per_job": units * 0.2,
        "ordinary_units_per_job": units * 0.8,
        "max_job_seconds": timeout * 60,
        "ordinary_job_seconds": timeout * 60 * 0.8,
    }


def optimal_capacity() -> dict[str, Any]:
    """Exact integer DP over worker waves and centi-minute-priced timeouts.

    One wave carries ten worlds. All costs are integer minutes of runtime;
    150 units at 2 units/hour permit 4500 minutes. Minimize cost for each wave
    count with jobs <=480 minutes, then choose the largest balanced N.
    """
    max_waves = max(w for w in range(1, 100) if chunk_budget(w * WORKERS)["timeout_minutes"] <= 480)
    budget_minutes = int(UNIT_CAP * 60 / RATE)
    costs = [0]
    packs: list[list[int]] = [[]]
    for waves in range(1, 1000):
        candidates = [
            (
                costs[waves - w] + chunk_budget(w * WORKERS)["timeout_minutes"],
                packs[waves - w] + [w],
            )
            for w in range(1, min(waves, max_waves) + 1)
        ]
        cost, pack = min(candidates, key=lambda item: (item[0], len(item[1]), item[1]))
        costs.append(cost)
        packs.append(pack)
        if cost > budget_minutes:
            break
    feasible = max(i for i, cost in enumerate(costs) if cost <= budget_minutes)
    total = feasible * WORKERS // 3 * 3
    pack = sorted(packs[feasible], reverse=True)
    chunks, start = [], 0
    for index, waves in enumerate(pack):
        count = min(waves * WORKERS, total - start)
        if count <= 0:
            break
        chunks.append(
            {
                "index": index,
                "start": start,
                "count": count,
                "mix": "balanced-bank-slice",
                "caps": chunk_budget(count),
            }
        )
        start += count
    return {
        "study_worlds": total,
        "mix_counts": dict.fromkeys(MIXES, total // 3),
        "chunks": chunks,
        "aggregate_unit_cap": sum(c["caps"]["units_per_job"] for c in chunks),
        "next_wave_minimum_timeout_minutes": costs[feasible + 1],
        "available_timeout_minutes": budget_minutes,
        "max_waves_per_job": max_waves,
    }


def caps() -> dict[str, Any]:
    """Aggregate and maximum per-runtime envelope, never allocation authority."""
    plan = optimal_capacity()
    maximum = max(plan["chunks"], key=lambda c: c["count"])["caps"]
    return {
        **maximum,
        "units": plan["aggregate_unit_cap"],
        "reserve_units": plan["aggregate_unit_cap"] * 0.2,
        "ordinary_units": plan["aggregate_unit_cap"] * 0.8,
        "study_ceiling_units": UNIT_CAP,
        "l4_rate_units_per_hour": RATE,
        "jobs": len(plan["chunks"]),
        "workers_per_job": WORKERS,
        "export_bytes_per_job": 67108864,
    }
