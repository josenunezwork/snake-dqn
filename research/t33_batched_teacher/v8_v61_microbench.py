"""One-thread matched microbenchmark for T33's v8 and Vector61 hot-path cuts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numba
import numpy as np

from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import v8_decide
from src.simd_env.vector61_featurizer import Vector61Params
from src.simd_env.vector61_nb import Vector61FeaturizerNB

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median_rows_per_second(call, rows: int, repeats: int, rounds: int) -> list[float]:
    values = []
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(repeats):
            call()
        values.append(rows * repeats / (time.perf_counter() - started))
    return values


def main() -> None:
    numba.set_num_threads(1)
    cfg = BatchSimConfig(
        num_envs=48,
        num_snakes=3,
        game_width=300,
        game_height=200,
        initial_food=12,
        max_food=16,
        max_capacity=100,
    )
    grid = GridBatchSim(cfg, seeds=list(range(cfg.num_envs)), train_mode=True, allow_respawn=True)
    plain = BatchSim(cfg, seeds=list(range(cfg.num_envs)), train_mode=True, allow_respawn=True)
    rows = np.column_stack((np.arange(grid.E), np.zeros(grid.E, dtype=np.int64)))
    q = np.tile(np.array([0.2, 0.1, 0.3, -0.1, 0.0, -0.2], dtype=np.float32), (grid.E, 1))
    mask = grid.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
    # Compile/cold allocation excluded from all timed rounds.
    v8_decide(grid, rows, q, mask, features=True, membership="painted")
    v8_decide(grid, rows, q, mask, features=True, membership="owner_pad")
    params = Vector61Params.from_game_config(cfg)
    old_food = Vector61FeaturizerNB(params)
    new_food = Vector61FeaturizerNB(params)
    envs = np.arange(grid.E, dtype=np.int64)
    old_food._food_arrays(plain, envs)
    new_food._food_arrays(grid, envs)

    result = {
        "schema": "t33-round2-v8-v61-microbench/v1",
        "host": platform.platform(),
        "python": platform.python_version(),
        "threads": 1,
        "nice_requested": 10,
        "shape": {"envs": grid.E, "snakes": grid.S, "v8_rows": len(rows)},
        "rounds": 3,
        "warmup_excluded": True,
        "rows_per_second": {
            "v8_painted_reference": _median_rows_per_second(
                lambda: v8_decide(grid, rows, q, mask, features=True, membership="painted"),
                len(rows),
                20,
                3,
            ),
            "v8_owner_pad": _median_rows_per_second(
                lambda: v8_decide(grid, rows, q, mask, features=True, membership="owner_pad"),
                len(rows),
                20,
                3,
            ),
            "v61_list_staging": _median_rows_per_second(
                lambda: old_food._food_arrays(plain, envs), len(envs), 1000, 3
            ),
            "v61_grid_array_staging": _median_rows_per_second(
                lambda: new_food._food_arrays(grid, envs), len(envs), 1000, 3
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (
                ROOT / "src/simd_env/v8_nb.py",
                ROOT / "src/simd_env/ego_raster_bv8.py",
                ROOT / "src/simd_env/vector61_nb.py",
                Path(__file__),
            )
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
