"""Per-component ego2s featurizer cost (64 envs x 6 snakes, injected big bodies).

Run from the repo root under ``research/perf_sim_20261007/with_slots.py --slots 1``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simd_env import ego_raster as er  # noqa: E402
from src.simd_env.batch_sim import BatchSimConfig  # noqa: E402
from src.simd_env.grid_scenarios import inject_serpentines  # noqa: E402
from src.simd_env.grid_sim import GridBatchSim  # noqa: E402


def main() -> int:
    cfg = BatchSimConfig(num_envs=64, num_snakes=6, max_capacity=1600)
    sim = GridBatchSim(cfg, seeds=list(range(64)))
    inject_serpentines([sim], [900, 600, 400, 300, 150, 150], box_height=8)
    c = er.EgoRasterConfig(reach_steps=0)
    n = int(sim.alive.sum())
    zeros_ttl = np.zeros((64, 6, 31, 31), np.int32)
    zeros_wall = np.zeros((64, 6, 31, 31), bool)
    parts = [
        ("local", lambda: er._local_planes(sim, c)),
        ("global", lambda: er._global_planes(sim, c)),
        ("coarse", lambda: er._coarse_counts(sim, 8)),
        ("scalars", lambda: er._scalars(sim, c)),
        ("reach", lambda: er._reach_time(zeros_ttl, zeros_wall, er.EgoRasterConfig())),
    ]
    for name, fn in parts:
        fn()
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        print(name, round((time.perf_counter() - t0) / reps / n * 1e6, 2), "us/agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
