"""Per-frame cost of the v8 shield: Python (live hook on SimRowSnake views) vs numba.

Development benchmark (not gate evidence). Builds an M3-C training segment of ``--worlds``
worlds, plays the Python v8 action with the frp3-s12 teacher's Q, and times, on the same
decision-time sim and inputs every frame: the Python shield (``make_shield(...).act``,
which includes the vector61 decision-time free-space featurization), ``v8_decide`` (lazy)
and ``v8_decide(features=True)`` (every per-action diagnostic). ``--grow-to`` raises the
target lengths so bodies get long (the veto's cost grows with the areas it floods).

    ./venv/bin/python research/redesign_m3c_20261008/bench_v8_nb.py --worlds 64 --frames 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))


def main() -> None:
    import torch
    from test_v8_nb import build_world

    from src.simd_env.v8_nb import v8_decide

    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grow-to", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    sim, runtime, policies, groups, shield, teacher = build_world(args.worlds, args.seed)
    if args.grow_to:
        rng = np.random.default_rng(args.seed)
        sim.length[:, :] = rng.integers(args.grow_to // 4, args.grow_to + 1, sim.length.shape)
    t_py, t_nb, t_nbf, rows, lens = [], [], [], 0, []
    for frame in range(args.warmup + args.frames):

        def choose(prepared):
            nonlocal rows
            alive = prepared.get_alive()
            masks = prepared.get_resolved_action_mask()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                states, sel_mask = runtime.selection(hero)
                with torch.no_grad():
                    q = teacher.q_values(states).float().cpu().numpy()
                t0 = time.perf_counter()
                a_py = shield.act(q, masks[hero[:, 0], 0], prepared, hero)
                t1 = time.perf_counter()
                res = v8_decide(prepared, hero, q, runtime.selection(hero)[1], 8.0)
                t2 = time.perf_counter()
                v8_decide(prepared, hero, q, runtime.selection(hero)[1], 8.0, features=True)
                t3 = time.perf_counter()
                assert np.array_equal(res.action, a_py)
                if frame >= args.warmup:
                    t_py.append(t1 - t0)
                    t_nb.append(t2 - t1)
                    t_nbf.append(t3 - t2)
                    rows += len(hero)
                    lens.append(float(prepared.seg_count[hero[:, 0], 0].mean()))
                actions[hero[:, 0], 0] = a_py
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    py, nb, nbf = (1e3 * np.mean(t) for t in (t_py, t_nb, t_nbf))
    print(
        f"worlds={args.worlds} frames={args.frames} rows={rows} grow_to={args.grow_to} "
        f"mean hero body={np.mean(lens):.0f}"
    )
    print(f"python v8      {py:8.3f} ms/frame")
    print(f"numba v8       {nb:8.3f} ms/frame  ({py / nb:6.1f}x)")
    print(f"numba v8+feats {nbf:8.3f} ms/frame  ({py / nbf:6.1f}x)")


if __name__ == "__main__":
    main()
