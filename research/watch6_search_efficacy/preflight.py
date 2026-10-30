"""Excluded Mac actual-entrypoint fixture plus product-config long-body branch timing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.run import ledger_identity
from tools.ops.package_spec import empty_ledger, publish_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ledger = args.out / "verified-ledger.json"
    publish_json(ledger, empty_ledger(p.NAMESPACE, "watch6", ledger_identity()))
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.watch6_search_efficacy.run",
            "--ledger",
            str(ledger),
            "--out",
            str(args.out / "tiny"),
            "--fixture",
            "--fixture-horizon",
            "12",
        ],
        check=True,
    )
    tiny_seconds = time.monotonic() - started
    import numpy as np

    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t1_commitment_20261015.replay import configure_cpu
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch6_search_efficacy.engine import dense_batch
    from research.watch6_search_efficacy.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    configure_cpu()
    timings = []
    for seed in p.FIXTURE_SEEDS:
        session = open_watch({"seed": seed}, 24, "bare")
        try:
            world = session.world
            # Excluded profiling fixture only. Product configuration/opponents
            # retained; synthetic geometry is never a scientific initial state.
            inject_serpentines([world.sim], (900, 600, 400, 300, 150, 150), box_height=8)
            _prepare_sim(world.sim)
            world.runtime.prepare(world.sim)
            obs = build_ego_raster(
                world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            q = prepared_q(world, obs)
            legal, _ = _legal(q, world.sim.get_advisory_action_mask()[0, 0])
            start = time.monotonic()
            result = dense_batch(world, 0, q, legal)
            elapsed = time.monotonic() - start
            timings.append(
                {
                    "seed": seed,
                    "seconds": elapsed,
                    "branches": result["branches_evaluated"],
                    "native_v8": result["native_v8"],
                }
            )
        finally:
            session.close()
    publish_json(
        args.out / "benchmark.json",
        {
            "science": p.identity(),
            "tiny_pair_count": 3,
            "tiny_frames_per_arm": 12,
            "tiny_seconds": tiny_seconds,
            "branch_timings": timings,
            "cpu_count": os.cpu_count(),
            "scope": "Mac loaded-host throughput fixture; no product outcome or latency claim",
        },
    )
    print(json.dumps({"tiny_seconds": tiny_seconds, "branch_timings": timings}))


if __name__ == "__main__":
    main()
