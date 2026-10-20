"""One-thread shadow timing and empirical T1 timing projections, not efficacy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.t34_shadow.common import Budget, provenance, sha, write_once

T1 = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-commitment-20261015/rows.jsonl")


def quantiles(values: list[float]) -> dict[str, Any]:
    """Retain raw seconds and NumPy linear percentile estimates."""
    if not values:
        return {"n": 0, "p50_seconds": None, "p95_seconds": None, "p99_seconds": None}
    return {
        "n": len(values),
        **{f"p{p}_seconds": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "raw_seconds": values,
    }


def reachability(catchup: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    """Project work completion before t*=t_doom-H_LAD, retaining censored deaths."""
    rows = [json.loads(line) for line in T1.read_text().splitlines() if line.strip()]
    late = [row for row in rows if row["stratum"] == "late"]
    if len(late) != 49:
        raise ValueError("T1 late-death denominator drift")
    projections = []
    for percentile in (50, 95, 99):
        catch_s = catchup[f"p{percentile}_seconds"]
        search_s = search[f"p{percentile}_seconds"]
        if catch_s is None or search_s is None:
            continue
        for fps in (12, 24):
            horizons = [row.get("H_LAD") for row in late]
            # Include cold re-fork cost conservatively. A continuously maintained
            # shadow has already paid it; report that case separately.
            warm = sum(h is not None and (300 - int(h)) / fps > search_s for h in horizons)
            cold = sum(
                h is not None and (300 - int(h)) / fps > catch_s + search_s for h in horizons
            )
            projections.append(
                {
                    "fps": fps,
                    "latency_percentile": percentile,
                    "denominator": 49,
                    "warm_reachable": warm,
                    "warm_fraction": warm / 49,
                    "cold_reachable": cold,
                    "cold_fraction": cold / 49,
                    "search_seconds": search_s,
                    "catchup_seconds": catch_s,
                }
            )
    return {
        "T1_input": {"path": str(T1), "sha256": sha(T1)},
        "D": 300,
        "late_deaths": 49,
        "censored_failures": sum(row.get("H_LAD") is None for row in late),
        "condition": "(300-H_LAD)/fps > search_seconds [+ catchup_seconds for cold]",
        "interpretation": (
            "empirical timing projection; not measured online reachability or efficacy"
        ),
        "projections": projections,
    }


def benchmark(mix: str, warm_frames: int, samples: int, catchups: int, budget: Budget) -> dict:
    """Time full shadow operations beside an independently advanced T3 world."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step
    from research.t34_shadow.identity import _new_pair
    from research.t34_shadow.shadow import ShadowTrajectory, clone_world_exact

    # A fixed diagnostic seed, not a new efficacy bank. Record the exact roster.
    _session, source, specs, checkpoint = _new_pair(2026101934, mix)
    checkpoint.pop("live_rng")
    plan = {"seed": 2026101934, "mix": mix, "specs": specs}
    checkpoints = [clone_world_exact(source)]
    for index in range(warm_frames):
        budget.check()
        if index and index % 500 == 0:
            checkpoints.append(clone_world_exact(source))
        choose_and_step(source, capture_obs=False)
        if not bool(source.sim.alive[0, 0]):
            break
    observed_until = int(source.sim.frame[0]) - int(not bool(source.sim.alive[0, 0]))
    eligible = [s for s in checkpoints if int(s.sim.frame[0]) + 300 <= observed_until]
    source = eligible[-1] if eligible else checkpoints[0]
    source_frame = int(source.sim.frame[0])
    deadline = budget.started + budget.seconds
    # Exclude one warm-up operation. Include clone, catch-up and every digest
    # in re-fork timing; do not call the bare simulator a shadow catch-up.
    shadow = ShadowTrajectory(300, checkpoint=checkpoint)
    shadow.fork(source)
    shadow.catch_up(deadline=deadline)
    catchup_times, catchup_counts = [], []
    for _ in range(catchups):
        budget.check()
        start = time.perf_counter()
        shadow.fork(source)
        count = shadow.catch_up(deadline=deadline)
        catchup_times.append(time.perf_counter() - start)
        catchup_counts.append(count)
    step_times, confirm_times, tick_times, confirmations = [], [], [], 0
    for _ in range(samples):
        budget.check()
        if not bool(source.sim.alive[0, 0]) or not shadow.frames:
            break
        choose_and_step(source, capture_obs=False)
        tick_started = time.perf_counter()
        if not shadow.confirm(source):
            raise RuntimeError("independent source/shadow full-state confirmation failed")
        confirm_times.append(time.perf_counter() - tick_started)
        confirmations += 1
        start = time.perf_counter()
        made = shadow.catch_up(deadline=deadline)
        elapsed = time.perf_counter() - start
        if made == 1:
            step_times.append(elapsed)
            tick_times.append(time.perf_counter() - tick_started)
    complete_times = [t for t, n in zip(catchup_times, catchup_counts) if n == 300]
    return {
        "mix": mix,
        "plan": plan,
        "checkpoint": checkpoint,
        "state_frame_after_steps": int(source.sim.frame[0]),
        "hero_length_after_steps": int(source.sim.length[0, 0]),
        "requested_warm_frames": warm_frames,
        "timing_source_frame": source_frame,
        "selection": "latest fixed 500-frame checkpoint known to survive 300 frames, else initial",
        "confirmations": confirmations,
        "step": quantiles(step_times),
        "confirm": quantiles(confirm_times),
        "confirm_plus_step": quantiles(tick_times),
        "catchup_300": quantiles(complete_times),
        "all_catchup_seconds": catchup_times,
        "all_catchup_frames": catchup_counts,
        "incomplete_catchups": sum(n != 300 for n in catchup_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", choices=("scripted", "frozen", "mixed"), required=True)
    parser.add_argument("--warm-frames", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--catchups", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("receipt already exists")
    if not (
        0 <= args.warm_frames <= 5000 and 1 <= args.samples <= 5000 and 1 <= args.catchups <= 100
    ):
        parser.error("invalid bounded sample counts")
    budget = Budget(args.seconds)
    meta = provenance()
    result: dict[str, Any] = {"schema": "t34-shadow-m3/v1", "provenance": meta}
    try:
        result.update(benchmark(args.mix, args.warm_frames, args.samples, args.catchups, budget))
        result["status"] = "COMPLETE"
    except Exception as exc:
        result.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - budget.started
    write_once(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
