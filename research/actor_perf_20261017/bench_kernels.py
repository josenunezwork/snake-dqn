#!/usr/bin/env python3
"""Bounded CPU-only, fixed-state actor kernel profile; no training or production edits.

Run with the project venv, nice -n 10, and one thread. JIT warm-up is excluded.
Synthetic serpentines expose length sensitivity, not a gameplay throughput estimate.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import argparse  # noqa: E402
import cProfile  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import pstats  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from numba import njit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.ego2s_network import Ego2sNet  # noqa: E402
from src.simd_env import ego_raster_b as eb  # noqa: E402
from src.simd_env import ego_raster_nb as enb  # noqa: E402
from src.simd_env import v8_nb as v8  # noqa: E402
from src.simd_env.batch_sim import BatchSimConfig  # noqa: E402
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster  # noqa: E402
from src.simd_env.ego_raster_bv8 import v8_scalars  # noqa: E402
from src.simd_env.grid_scenarios import inject_serpentines  # noqa: E402
from src.simd_env.grid_sim import GridBatchSim  # noqa: E402


def timed(fn, repeats: int) -> dict:
    """Warm once, then retain every sample in milliseconds."""
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(1000 * (time.perf_counter() - start))
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def profile(fn, repeats: int) -> list:
    """Compiled dispatchers are opaque; leaf probes below isolate landing work."""
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        fn()
    pr.disable()
    entries = []
    for (filename, line, name), (cc, nc, own, cumulative, _) in pstats.Stats(pr).stats.items():
        entries.append(
            {
                "file": filename,
                "line": line,
                "name": name,
                "calls": nc,
                "own_ms_per_call": 1000 * own / repeats,
                "cumulative_ms_per_call": 1000 * cumulative / repeats,
            }
        )
    return sorted(entries, key=lambda x: -x["own_ms_per_call"])[:20]


@njit(cache=True, nogil=True)
def landing_probe(
    bodies,
    hp,
    n,
    length,
    heading,
    gw,
    gh,
    min_boost,
    boost_cost,
    acap,
    others,
    ost,
    tmp,
    ws,
    seen,
    rel_st,
    rel_val,
    qx,
    qy,
    qd,
    repeats,
    paint_only,
):
    """Six actions, slot zero; production landing area vs its exact paint prefix.

    paint_only duplicates v8_nb._landing_area's release-map construction, including
    max on duplicate cells and deletion of the new head. No fastmath. Not a candidate
    implementation: it isolates a lower-level cost. The complete path calls production.
    """
    total = 0
    ring = bodies.shape[2]
    for _ in range(repeats):
        for action in range(6):
            if not paint_only:
                total += v8._landing_area(
                    bodies,
                    0,
                    0,
                    hp,
                    ring,
                    n,
                    length,
                    0,
                    heading,
                    action,
                    gw,
                    gh,
                    min_boost,
                    boost_cost,
                    acap,
                    others,
                    ost,
                    tmp,
                    ws,
                    seen,
                    rel_st,
                    rel_val,
                    qx,
                    qy,
                    qd,
                )
                continue
            hd = (heading + action % 3 - 1) % 4
            dx, dy = v8._DX[hd], v8._DY[hd]
            boosted, ln, lpost = v8._post_move(n, length, 0, action >= 3, min_boost, boost_cost)
            nh = 2 if boosted else 1
            hx, hy = bodies[0, 0, hp, 0], bodies[0, 0, hp, 1]
            ws[2] += 1
            stamp = ws[2]
            for i in range(ln):
                cx, cy = v8._seg_cell(bodies, 0, 0, hp, ring, nh, hx, hy, dx, dy, i)
                if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
                    continue
                idx = cx * gh + cy
                when = lpost - i + 1
                if rel_st[idx] != stamp:
                    rel_st[idx] = stamp
                    rel_val[idx] = when
                elif when > rel_val[idx]:
                    rel_val[idx] = when
            nx, ny = hx + dx * nh, hy + dy * nh
            if nx >= 0 and nx < gw and ny >= 0 and ny < gh:
                rel_st[nx * gh + ny] = 0
            total += ln
    return total


def scenario(worlds: int, length: int, repeats: int, forward: bool) -> dict:
    cfg = BatchSimConfig(num_envs=worlds, num_snakes=6, max_capacity=1600)
    sim = GridBatchSim(cfg, seeds=list(range(worlds)), train_mode=True, allow_respawn=False)
    inject_serpentines([sim], [length] * 6, box_height=9)
    sim.check_grid_invariants()
    rows = np.argwhere(sim.alive)
    mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]

    def ego() -> dict:
        return build_ego_raster(sim, EgoRasterConfig(version="b"), backend="numba", rows=rows)

    def features() -> tuple:
        return v8_scalars(sim, rows, mask)

    obs = ego()
    scalars, res = features()
    q = np.random.default_rng(17).normal(size=(len(rows), 6)).astype(np.float32)

    def rede() -> v8.V8Result:
        return v8.v8_redecide(res, q, mask)

    result = {
        "worlds": worlds,
        "rows": len(rows),
        "length": length,
        "v8": timed(features, repeats),
        "ego": timed(ego, repeats),
        "redecide": timed(rede, repeats),
        "ego_profile": profile(ego, repeats),
        "v8_profile": profile(features, repeats),
    }
    view = sim.ego_view()
    raster_cfg = EgoRasterConfig(version="b")
    result["ego_components"] = {
        "planes_local_global": timed(
            lambda: enb.build_planes_numba(view, raster_cfg, rows), repeats
        ),
        "enemy_next_plane": timed(lambda: eb.enemy_next_plane(view, raster_cfg, rows), repeats),
        "b_scalars_including_regions": timed(lambda: eb.b_scalars(view, rows, "numba"), repeats),
        "regions_only": timed(lambda: enb.region_sizes_numba(view, rows), repeats),
    }
    # Byte identity of repeated fixed-state calls (also catches scratch stamp leakage).
    obs2, scalars2 = ego(), features()[0]
    assert all(obs[k].tobytes() == obs2[k].tobytes() for k in obs)
    assert scalars.tobytes() == scalars2.tobytes()
    result["repeat_byte_identity"] = True
    result["observation_sha256"] = hashlib.sha256(
        b"".join(obs[k].tobytes() for k in sorted(obs)) + scalars.tobytes()
    ).hexdigest()
    # Independent numpy reference on a one-world view, outside timed work.
    if worlds == 1:
        ref = build_ego_raster(sim, EgoRasterConfig(version="b"), backend="numpy", rows=rows)
        assert all(obs[k].tobytes() == ref[k].tobytes() for k in obs)
        result["ego_numpy_byte_identity"] = True
    ws, others, hero, tmp, seen, rs, rv, qx, qy, qd = v8._workspace(sim.grid_w, sim.grid_h)
    ws[0] += 1
    ost = int(ws[0])
    for slot in range(1, sim.S):
        for x, y in sim.get_bodies(0, slot):
            others[x * sim.grid_h + y] = ost
    args = (
        sim.bodies,
        int(sim.head_ptr[0, 0]),
        length,
        length,
        int(sim.direction[0, 0]),
        sim.grid_w,
        sim.grid_h,
        cfg.min_boost_length,
        cfg.boost_length_cost_frames,
        int(res.area_cap[0]),
        others,
        ost,
        tmp,
        ws,
        seen,
        rs,
        rv,
        qx,
        qy,
        qd,
        100,
    )
    for paint in (False, True):
        key = "landing_paint_only_100x6" if paint else "landing_area_100x6"
        result[key] = timed(lambda: landing_probe(*args, paint), repeats)
    if forward:
        torch.manual_seed(0)
        net = Ego2sNet(7, 60).eval()
        inputs = [
            torch.from_numpy(obs["local"]),
            torch.from_numpy(obs["global"]),
            torch.from_numpy(np.concatenate([obs["scalars"], scalars], axis=1)),
        ]
        with torch.no_grad():
            result["cpu_forward_random_weights"] = timed(lambda: net(*inputs), repeats)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(1)
    paths = [
        "src/simd_env/v8_nb.py",
        "src/simd_env/ego_raster_nb.py",
        "src/simd_env/ego_raster.py",
        "src/simd_env/ego_raster_b.py",
        "research/redesign_m3c_20261008/train_m3c.py",
        str(Path(__file__).relative_to(ROOT)),
    ]
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numba": numba.__version__,
        "numpy": np.__version__,
        "threads": 1,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "cases": [],
    }
    start = time.perf_counter()
    for worlds, length in [
        (1, 5),
        (1, 150),
        (1, 600),
        (1, 1200),
        (16, 5),
        (16, 150),
        (16, 600),
        (16, 1200),
        (128, 5),
        (128, 1200),
    ]:
        cell = scenario(worlds, length, args.repeats, forward=worlds in (16, 128))
        result["cases"].append(cell)
        print(json.dumps({k: cell[k] for k in ("worlds", "length", "v8", "ego")}), flush=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    result["total_wall_s_including_setup_jit"] = time.perf_counter() - start
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
