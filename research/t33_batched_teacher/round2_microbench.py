"""Frozen-source matched simulator/event and long-body v8 microbenchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numba
import numpy as np

from research.t33_batched_teacher.evidence import identity
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import V8Result, v8_decide

BASE = "f76820776e39a6a53c13197f6f0948aa7c6a95f4"
ROOT = Path(__file__).resolve().parents[2]


def frozen(name: str) -> tuple[types.ModuleType, str]:
    path = f"src/simd_env/{name}.py"
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], text=True)
    module = types.ModuleType(f"src.simd_env._t33_round2_frozen_{name}")
    module.__file__ = str(ROOT / path)
    sys.modules[module.__name__] = module
    exec(
        compile(source.replace("cache=True", "cache=False"), module.__file__, "exec"),
        module.__dict__,
    )
    return module, hashlib.sha256(source.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    numba.set_num_threads(1)
    old_grid, grid_sha = frozen("grid_sim")
    old_nb, nb_sha = frozen("grid_sim_nb")
    cfg = BatchSimConfig(
        num_envs=48,
        num_snakes=3,
        game_width=300,
        game_height=200,
        initial_food=12,
        max_food=16,
        max_capacity=100,
        mechanics_version=2,
    )

    def make(old: bool):
        sim = (old_grid.GridBatchSim if old else GridBatchSim)(
            cfg, seeds=list(range(48)), train_mode=True, allow_respawn=True, jit=True
        )
        if old:
            sim._kernels = old_nb
        return sim

    actions = np.ones((48, 3), dtype=np.int64)
    for old in (True, False):
        sim = make(old)
        for _ in range(10):
            sim.step(actions)
    grid, event = {"before": [], "after": []}, {"before": [], "after": []}
    cells = [(x, y) for y in range(2, 18) for x in range(2, 28)]
    cells += cells[:32]  # eviction followed by re-add and duplicate cases
    for _ in range(3):
        for old, label in ((True, "before"), (False, "after")):
            sim = make(old)
            start = time.perf_counter()
            for _step in range(80):
                sim.step(actions)
            grid[label].append(48 * 80 / (time.perf_counter() - start))
            before_rng = [r._rng.getstate() for r in sim._rngs]
            start = time.perf_counter()
            for env in range(sim.E):
                sim._drop_corpse_v2(env, cells)
            event[label].append(48 / (time.perf_counter() - start))
            assert before_rng == [r._rng.getstate() for r in sim._rngs]
    # Actual L800 timing fixture; compare current painted reference and owner
    # under the SAME native pool, with first-use JIT excluded.
    from research.phase0_search_20261019 import teacher
    from research.phase0_search_20261019.preflight import fixture
    from research.t33_batched_teacher.ablation import B8, calibration_rows
    from research.t33_batched_teacher.benchmark import prepared
    from research.t33_batched_teacher.teacher import merge_worlds

    ceiling = numba.config.NUMBA_NUM_THREADS
    teacher.configure_cpu()
    os.environ["NUMBA_NUM_THREADS"] = str(ceiling)
    numba.config.reload_config()
    world = prepared(fixture(calibration_rows(smoke=True)[0], B8, 800))
    merged = merge_worlds([world], 8, horizon=32)
    rows = np.column_stack((np.arange(48), np.zeros(48, dtype=np.int64)))
    mask = merged.sim.get_advisory_action_mask()[:, 0]
    q = np.zeros((48, 6), dtype=np.float32)
    v8 = {}
    reference = v8_decide(merged.sim, rows, q, mask, features=True)
    for threads in sorted({1, min(4, ceiling)}):
        numba.set_num_threads(threads)
        for membership in ("painted", "owner_pad"):
            actual = v8_decide(merged.sim, rows, q, mask, features=True, membership=membership)
            for name in V8Result.__dataclass_fields__:
                assert getattr(reference, name).tobytes() == getattr(actual, name).tobytes()
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                for _repeat in range(10):
                    v8_decide(merged.sim, rows, q, mask, features=True, membership=membership)
                samples.append(480 / (time.perf_counter() - start))
            v8[f"{membership}_{threads}_threads"] = samples
    value = {
        "schema": "t33-final-microbench/v1",
        "source": identity(),
        "baseline": BASE,
        "baseline_grid_sha256": grid_sha,
        "baseline_numba_sha256": nb_sha,
        "grid_rows_per_s": grid,
        "corpse_event_rows_per_s": event,
        "v8_L800_rows_per_s": v8,
        "shape": {
            "grid_envs": 48,
            "grid_snakes": 3,
            "grid_steps": 80,
            "corpse_cells_per_event": len(cells),
            "v8_rows": 48,
        },
        "notes": (
            "3 warmed rounds; both grid arms jit=True, one thread; "
            "nice10 caller; v8 pool explicit"
        ),
    }
    with args.out.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in value.items() if k != "source"}))


if __name__ == "__main__":
    main()
