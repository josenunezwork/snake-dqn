"""Excluded natural Watch6 fixture: full-inclusive snapshot, shadow and H300 costs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from research.bl_served_search.exports import export_json
from research.watch_lead_p.engine import submit
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p.protocol import FIXTURE_SEED, NAMESPACE


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
    # Worst schedule occupancy, no quiet-session or GPU speedup assumptions.
    p0 = max(s["seconds"] for s in samples if s["delay"] == 0)
    p48 = max(s["seconds"] for s in samples if s["delay"] == 48)
    per_pair = 18000 * bare_tick_seconds + 1200 * p0 + 123 * p48
    forecast = 32 * per_pair / 10 + 1200
    export_json(
        args.output,
        {
            "fixture_index": 32,
            "seed": FIXTURE_SEED,
            "excluded_from_all_32_sessions": True,
            "fixture": "natural actual Watch6 L>=600; no geometry or future action tape",
            "warmup_frames": world.frame - 15,
            "warmup_seconds": warmup_seconds,
            "bare_tick_seconds": bare_tick_seconds,
            "samples": samples,
            "workers": 10,
            "min_vcpu": 12,
            "projected_seconds": forecast,
            "projection": (
                "max observed inclusive root cost; P0 1200/P48 123 roots per session; "
                "32 pairs/10 workers +1200s setup/export"
            ),
            "host_scaling": "Mac measurements; L4 host speed and ten-worker contention unmeasured",
            "base_seconds": 14400,
            "effective_seconds": 28800,
        },
    )
    print(f"P_FORECAST seconds={forecast:.2f} hours={forecast/3600:.3f}", flush=True)


if __name__ == "__main__":
    main()
