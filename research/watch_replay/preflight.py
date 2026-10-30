"""Excluded tiny real entrypoint and three matched long-body H300 timings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research.watch_replay.run import allocation, identity
from tools.ops.package_spec import empty_ledger, publish_json


def branches(out: Path) -> list[dict]:
    """Exercise actual exact-stop branches even though tiny worlds cannot trigger."""
    import numpy as np

    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch_replay.engine import dense_batch
    from research.watch_replay.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    configure_cpu()
    timings = []
    for seed in (91841001, 91841002, 91841003):
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
            timings.append(
                {
                    "seed": seed,
                    "seconds": time.monotonic() - started,
                    "branches": result["branches_evaluated"],
                    "native_v8": result["native_v8"],
                }
            )
        finally:
            session.close()
    publish_json(out, {"branch_timings": timings})
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--branches-only", action="store_true")
    args = parser.parse_args()
    if args.branches_only:
        branches(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    contract = {
        **allocation("t34-watch-replay-mac-fixture-v1", 3, "exact-stop-l2-prime", 12),
        "commit": commit,
    }
    publish_json(args.out / "contract.json", contract)
    publish_json(
        args.out / "ledger.json",
        empty_ledger(contract["namespace"], "sessions", identity(contract)),
    )
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch_replay.run",
            "--contract",
            str(args.out / "contract.json"),
            "--ledger",
            str(args.out / "ledger.json"),
            "--out",
            str(args.out / "tiny"),
            "--workers",
            "1",
        ],
        check=True,
    )
    elapsed = time.monotonic() - started
    timings = branches(args.out / "branches.json")
    publish_json(
        args.out / "benchmark.json",
        {
            "commit": commit,
            "tiny_seconds": elapsed,
            "tiny_units": 3,
            "tiny_horizon": 12,
            "branch_timings": timings,
            "cpu_count": os.cpu_count(),
            "scope": "excluded Mac loaded-host fixture",
        },
    )
    print(json.dumps({"tiny_seconds": elapsed, "branch_timings": timings}))


if __name__ == "__main__":
    main()
