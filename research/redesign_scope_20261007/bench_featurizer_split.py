"""ego2s featurizer cost per agent: NumPy reference vs numba backend (64 gate-size worlds).

All rows (every living snake) and hero rows only (slot 0 of every world), on ``fresh``
worlds (after 200 frames of greedy play) and ``big`` worlds (injected serpentines
900/600/400/300/150/150). The numba backend reads the sim's maintained coarse counts.
Run from the repo root with ``OMP_NUM_THREADS=1 nice -n 10``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simd_env.batch_sim import BatchSimConfig  # noqa: E402
from src.simd_env.ego_raster import build_ego_raster  # noqa: E402
from src.simd_env.grid_scenarios import (  # noqa: E402
    GreedySafePolicy,
    inject_serpentines,
)
from src.simd_env.grid_sim import GridBatchSim  # noqa: E402


def world(scenario: str) -> GridBatchSim:
    cfg = BatchSimConfig(
        num_envs=64,
        num_snakes=6,
        game_width=1450,
        game_height=830,
        initial_food=250,
        max_food=300,
        mechanics_version=2,
        max_capacity=1600,
    )
    sim = GridBatchSim(cfg, seeds=list(range(64)), train_mode=False, allow_respawn=True)
    if scenario == "big":
        inject_serpentines([sim], [900, 600, 400, 300, 150, 150], box_height=8)
    else:
        pol = GreedySafePolicy(1, boost_prob=0.1)
        for _ in range(200):
            sim.step(pol.actions(sim, sim.get_action_mask()))
    return sim


def per_agent_us(sim, backend: str, rows: np.ndarray, reps: int) -> float:
    build_ego_raster(sim, backend=backend, rows=rows)
    t0 = time.perf_counter()
    for _ in range(reps):
        build_ego_raster(sim, backend=backend, rows=rows)
    return (time.perf_counter() - t0) / reps / len(rows) * 1e6


def main() -> int:
    out = []
    for scenario in ("fresh", "big"):
        sim = world(scenario)
        all_rows = np.argwhere(sim.alive)
        hero_rows = np.argwhere(sim.alive[:, :1])
        for backend in ("numpy", "numba"):
            for name, rows in (("all", all_rows), ("hero", hero_rows)):
                reps = 3 if backend == "numpy" else 30
                us = per_agent_us(sim, backend, rows, reps)
                row = {"scenario": scenario, "backend": backend, "rows": name, "us_per_agent": us}
                row["n_rows"] = int(len(rows))
                row["mean_length"] = float(sim.length[sim.alive].mean())
                out.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({"loadavg": os.getloadavg()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
