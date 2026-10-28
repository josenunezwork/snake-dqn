"""Excluded natural Watch6 fixture: full-inclusive snapshot, shadow and H300 costs."""

from __future__ import annotations

import argparse
import heapq
import math
import time
from pathlib import Path
from unittest.mock import patch

from research.bl_served_search.exports import export_json
from research.watch_lead_p.admission import forecast_envelope
from research.watch_lead_p.engine import submit
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p.protocol import FIXTURE_SEED, NAMESPACE
from research.watch_lead_p.run import WORKERS, task_schedule


def project(bare_tick: float, p0: float, p48: float, workers: int = WORKERS) -> dict:
    """Indivisible arm-task list scheduling, including every tail and setup/export."""
    costs = {
        "bare": 6000 * bare_tick,
        "P0": 6000 * bare_tick + 1200 * p0,
        "P48": 6000 * bare_tick + 123 * p48,
    }
    loads = [0.0] * workers
    finishes = {i: [] for i in range(32)}
    for index, arm in task_schedule(list(range(32)), workers):
        finish = heapq.heappop(loads) + costs[arm]
        finishes[index].append(finish)
        heapq.heappush(loads, finish)
    publications = sorted(max(times) for times in finishes.values())
    per_pair = sum(costs.values())
    return {
        "projected_seconds": max(loads) + 1200,
        "arm_seconds": costs,
        "worker_loads_seconds": sorted(loads),
        "session_publication_seconds": publications,
        "schedule": "session cohorts of workers; P0/P48/bare within each cohort",
        "old_ten_worker_pair_seconds_corrected": math.ceil(32 / 10) * per_pair + 1200,
        "pair_waves": math.ceil(32 / workers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = LiveWorld(FIXTURE_SEED)
    started = time.monotonic()
    while world.frame < 6000 and (not world.hero.is_alive or world.hero.length < 600):
        world.step()
    if world.hero.length < 600:
        raise RuntimeError("excluded fixture lacks natural L600 support")
    warmup_seconds = time.monotonic() - started
    warmup_frames = world.frame
    bare_started = time.monotonic()
    bare = world.clone()
    for _ in range(300):
        bare.step()
    bare_tick_seconds = (time.monotonic() - bare_started) / 300
    samples = []
    for index in range(3):
        for delay in (0, 48):
            before = world.digest()
            start = time.monotonic()
            cert, details = submit(world, f"{NAMESPACE}excluded-fixture", delay)
            seconds = time.monotonic() - start
            if world.digest() != before:
                raise RuntimeError("benchmark source mutation")
            samples.append(
                {
                    "index": index,
                    "delay": delay,
                    "seconds": seconds,
                    "frame": world.frame,
                    "length": world.hero.length,
                    "branches": details.get("result", {}).get("branches_evaluated", 0),
                    "root_bytes": details["root_bytes"],
                    "certificate": cert is not None,
                    "drop": details.get("drop"),
                }
            )
            print(f"P_BENCH {samples[-1]}", flush=True)
        for _ in range(5):
            world.step()
    # Excluded engineering fault: force only bare's first branch transition to
    # report death after its real Watch tick. Alternatives execute real H300.
    # This measures a non-surviving, multi-branch rescue seam without mining data.
    original_step = LiveWorld.step

    def fatal_bare(branch, forced=None):
        event = original_step(branch, forced)
        if forced is not None and int(forced) == int(event.get("bare", -1)):
            event["death"] = True
        return event

    for delay in (0, 48):
        start = time.monotonic()
        with patch.object(LiveWorld, "step", fatal_bare):
            cert, details = submit(world, f"{NAMESPACE}excluded-fault", delay)
        samples.append(
            {
                "index": 3,
                "delay": delay,
                "seconds": time.monotonic() - start,
                "branches": details["result"]["branches_evaluated"],
                "fixture": "engineering forced first-transition bare death; real H300 alternatives",
                "outcomes": details["result"]["outcomes"],
            }
        )
        print(f"P_BENCH {samples[-1]}", flush=True)
    p0 = max(s["seconds"] for s in samples if s["delay"] == 0)
    p48 = max(s["seconds"] for s in samples if s["delay"] == 48)
    projected = project(bare_tick_seconds, p0, p48)
    forecast = projected["projected_seconds"]
    envelope = forecast_envelope(forecast, 1.54)
    export_json(
        args.output,
        {
            "fixture_index": 32,
            "seed": FIXTURE_SEED,
            "excluded_from_all_32_sessions": True,
            "fixture": "natural actual Watch6 L>=600; no geometry or future action tape",
            "warmup_frames": warmup_frames,
            "warmup_seconds": warmup_seconds,
            "bare_tick_seconds": bare_tick_seconds,
            "samples": samples,
            "workers": WORKERS,
            "min_vcpu": 12,
            **projected,
            "late_six_branch_stress_seconds": project(bare_tick_seconds, 6 * p0, 6 * p48)[
                "projected_seconds"
            ],
            "projection": (
                "max observed inclusive root cost; P0 1200/P48 123 roots per session; "
                "96 indivisible arm tasks, cohort-11 P0/P48/bare /11 workers +1200s setup/export"
            ),
            "host_scaling": (
                "Mac measurements; L4 host speed and eleven-worker contention unmeasured"
            ),
            "base_seconds": envelope["base_minutes"] * 60,
            "effective_seconds": envelope["effective_minutes"] * 60,
            "effective_unit_cap": envelope["effective_units"],
            "margin_seconds": envelope["margin_seconds"],
        },
    )
    print(f"P_FORECAST seconds={forecast:.2f} hours={forecast/3600:.3f}", flush=True)


if __name__ == "__main__":
    main()
