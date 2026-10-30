"""Excluded target-host contention measurements and mechanical main-shard sizing."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import publish_json

PROFILE_HORIZON = 300
BARRIER_TIMEOUT_SECONDS = 600  # W8 cold load/JIT skew, within the unchanged30min canary
MARGIN = 1.25
SETUP_MINUTES = 20
MAX_BASE_MINUTES = 660  # doubled guard22h, leaving two hours below provider lifetime


def branch_fixture(seed: int) -> dict:
    """Time a long-body full-H300 exact-stop batch with the actual product roster."""
    import numpy as np

    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch6_search_efficacy.engine import dense_batch
    from research.watch6_search_efficacy.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    session = open_watch({"seed": seed}, 24, "bare")
    try:
        world = session.world
        inject_serpentines([world.sim], (900, 600, 400, 300, 150, 150), box_height=8)
        _prepare_sim(world.sim)
        world.runtime.prepare(world.sim)
        obs = build_ego_raster(
            world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = prepared_q(world, obs)
        legal, _ = _legal(q, world.sim.get_advisory_action_mask()[0, 0])
        started = time.monotonic()
        result = dense_batch(world, 0, q, legal)
        seconds = time.monotonic() - started
        outcomes = [r for r in result["outcomes"].values() if "frames_executed" in r]
        if not any(r["frames_executed"] == 300 for r in outcomes):
            raise ValueError("long-body fixture lacks complete H300 branch")
        return {
            "seed": seed,
            "seconds": seconds,
            # Whole batch duration bounds every singleton, including cloning/checks.
            "branch_seconds_upper": seconds,
            "branches": result["branches_evaluated"],
            "frames_executed": [r["frames_executed"] for r in outcomes],
            "native_v8": result["native_v8"],
        }
    finally:
        session.close()


def profile_worker(index: int, barrier: object) -> dict:
    """Measure both main loops and long-body branches under simultaneous contention."""
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.watch6_search_efficacy.engine import run_arm

    configure_cpu()
    # Load/JIT warmup precedes the barrier so cold startup is not called throughput.
    seed = p.FIXTURE_SEEDS[index % len(p.FIXTURE_SEEDS)]
    run_arm(seed, False, lambda: None, 2)
    branch_fixture(seed)
    arms = {}
    for dense in (False, True):
        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        raw = run_arm(seed, dense, lambda: None, PROFILE_HORIZON)
        arms["exact_stop" if dense else "bare"] = {
            "main_seconds_per_frame": raw["timing"]["main_seconds"] / PROFILE_HORIZON,
            "mirror_identity_frames": raw["mirror_identity_frames"],
        }
    branches = []
    for fixture_seed in p.FIXTURE_SEEDS:
        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
        branches.append(branch_fixture(fixture_seed))
    return {"index": index, "arms": arms, "branches": branches}


def measure() -> dict:
    """Only excluded operational fixtures; no bank session or outcome is exported."""
    if (os.cpu_count() or 0) < 12:
        raise ValueError("W4/W6/W8 canary requires >=12 vCPU")
    context = multiprocessing.get_context("spawn")
    profiles = []
    for workers in p.WORKERS:
        with context.Manager() as manager:
            barrier = manager.Barrier(workers)
            with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context) as pool:
                futures = [pool.submit(profile_worker, i, barrier) for i in range(workers)]
                rows = [f.result(timeout=900) for f in futures]
        profiles.append(
            {
                "workers": workers,
                "branch_seconds_upper": max(
                    b["branch_seconds_upper"] for r in rows for b in r["branches"]
                ),
                "main_seconds_per_frame": {
                    arm: max(r["arms"][arm]["main_seconds_per_frame"] for r in rows)
                    for arm in ("bare", "exact_stop")
                },
                "rows": rows,
            }
        )
        print(f"WATCH6_PROFILE W{workers} complete", flush=True)
    return {
        "schema": "watch6-target-performance/v1",
        "science": p.identity(),
        "cpu_count": os.cpu_count(),
        "fixture": True,
        "horizon": PROFILE_HORIZON,
        "cost_limitations": (
            "Late-game main-loop cost is unmeasured; "
            "H300 fixtures are forecasts, not all-state bounds."
        ),
        "profiles": profiles,
    }


def forecast(report: dict, frozen_layout: dict | None = None) -> dict:
    """Use measured contended costs, six branches/root, both loops, 25% margin."""
    if (
        report.get("schema") != "watch6-target-performance/v1"
        or report.get("science") != p.identity()
        or report.get("fixture") is not True
        or report.get("horizon") != PROFILE_HORIZON
        or type(report.get("cpu_count")) is not int
        or report["cpu_count"] < 12
        or len(report.get("profiles", [])) != len(p.WORKERS)
        or {r.get("workers") for r in report["profiles"]} != set(p.WORKERS)
    ):
        raise ValueError("requires current measured W4/W6/W8 >=12vCPU canary")
    choices = []
    pair_costs = {}
    for profile in report["profiles"]:
        workers = profile["workers"]
        branch = profile["branch_seconds_upper"]
        loops = profile["main_seconds_per_frame"]
        costs = [branch, loops["bare"], loops["exact_stop"]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in costs):
            raise ValueError("nonpositive/nonfinite measured cost")
        rows = profile.get("rows", [])
        if len(rows) != workers or {r["index"] for r in rows} != set(range(workers)):
            raise ValueError("missing or duplicate contended profile workers")
        for row in rows:
            if any(
                row["arms"][arm]["mirror_identity_frames"] != PROFILE_HORIZON
                or row["arms"][arm]["main_seconds_per_frame"] <= 0
                for arm in ("bare", "exact_stop")
            ):
                raise ValueError("incomplete both-arm main-loop measurement")
            if [b["seed"] for b in row["branches"]] != p.FIXTURE_SEEDS:
                raise ValueError("incomplete long-body timing panel")
            for batch in row["branches"]:
                counters = batch["native_v8"]
                if (
                    300 not in batch["frames_executed"]
                    or batch["branch_seconds_upper"] != batch["seconds"]
                    or counters["kernel_rows"] <= 0
                    or any(
                        counters[k] != 0
                        for k in ("stock_rows", "fallback_rows", "noncandidate_rows")
                    )
                ):
                    raise ValueError("incomplete H300/native timing coverage")
        if branch != max(b["seconds"] for r in rows for b in r["branches"]) or any(
            loops[arm] != max(r["arms"][arm]["main_seconds_per_frame"] for r in rows)
            for arm in ("bare", "exact_stop")
        ):
            raise ValueError("timing summary differs from measured rows")
        # DenseTrigger first eligibility plus every5 decisions: ceil(H/5) roots.
        # All six legal actions is the physical maximum, not a prior mean rate.
        pair_seconds = p.HORIZON * sum(loops.values()) + math.ceil(p.HORIZON / 5) * 6 * branch
        pair_costs[workers] = pair_seconds
        for shards in (3, 6, 12, 24, 48, 96):
            pairs = math.ceil(p.COUNT / shards)
            waves = math.ceil(pairs / workers)
            base = math.ceil(SETUP_MINUTES + MARGIN * waves * pair_seconds / 60)
            if base <= MAX_BASE_MINUTES:
                choices.append(
                    {
                        "workers": workers,
                        "shards": shards,
                        "pairs_per_shard": pairs,
                        "pair_seconds_upper_forecast": pair_seconds,
                        "base_minutes_per_shard": base,
                        "effective_minutes": 2 * base,
                        "aggregate_expected_runtime_hours": shards * base / 60,
                    }
                )
                break
    if not choices:
        raise ValueError("NOT_READY measured forecast exceeds shard wall with margin")
    # Fewest shards first; then measured aggregate runtime, then fewer workers.
    chosen = min(
        choices, key=lambda r: (r["shards"], r["aggregate_expected_runtime_hours"], r["workers"])
    )
    if frozen_layout is not None:
        workers = frozen_layout["workers"]
        shards = frozen_layout["shards"]
        base = frozen_layout["base_minutes_per_shard"]
        if (
            workers not in p.WORKERS
            or shards not in (3, 6, 12, 24, 48, 96)
            or type(base) is not int
            or not 0 < base <= MAX_BASE_MINUTES
        ):
            raise ValueError("foreign frozen operational layout")
        pairs = math.ceil(p.COUNT / shards)
        required = math.ceil(
            SETUP_MINUTES + MARGIN * math.ceil(pairs / workers) * pair_costs[workers] / 60
        )
        if required > base:
            raise ValueError(
                "completion forecast exceeds frozen shard wall; never raise caps mid-series"
            )
        chosen = {
            "workers": workers,
            "shards": shards,
            "pairs_per_shard": pairs,
            "pair_seconds_upper_forecast": pair_costs[workers],
            "base_minutes_per_shard": base,
            "effective_minutes": 2 * base,
            "aggregate_expected_runtime_hours": shards * base / 60,
        }
    return {
        **chosen,
        "status": "MEASURED_CANARY_FORECAST",
        "margin": MARGIN,
        "setup_minutes": SETUP_MINUTES,
        "roots_per_session_upper_bound": math.ceil(p.HORIZON / 5),
        "branches_per_root_physical_maximum": 6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = measure()
    # Even an over-slow host retains its measured evidence before refusing main.
    publish_json(args.out, report)
    print(json.dumps(forecast(report)), flush=True)


if __name__ == "__main__":
    main()
