"""Throughput bench: live-game physics vs BatchSim vs GridBatchSim (+ featurizers).

Measures env-frames/s and agent-steps/s on ONE core (run under
``research/perf_sim_20261007/with_slots.py --slots 1``, which sets
``OMP_NUM_THREADS=1``) for the gate arena (1450x830, 6 snakes, 250/300 food,
mechanics v2, train-mode food, respawn on, frame_rate 1):

- ``live``  — ``parity.PyRefGame`` (the live ``Snake.move``, ``FoodManager``,
  ``GameLogic.check_collisions``/``handle_collisions`` primitives) plus the live
  per-snake mask (``simulate_relative_action_fatality``). This is the physics +
  mask share of a live actor frame; it excludes the 61-D ``get_state``, the
  forward pass and the v8 veto, which the perf-sim study measured separately.
- ``batch`` — :class:`BatchSim` (masks included in ``step``).
- ``grid``  — :class:`GridBatchSim` (masks included in ``step``).

Featurizer costs are timed separately on the same worlds:

- ``raster31v2`` — ``obs_inputs_from_batch_sim`` + ``build_observations``;
- ``ego`` / ``ego_noreach`` — ``build_ego_raster`` with / without the 24-step
  time-aware reach flood.

Scenarios: ``fresh`` (episode start, length-1 snakes growing) and ``big``
(injected serpentines of length 900/600/400/300/150/150 — the regime where the
champion dies). Actions come from per-env ``SurvivorPolicy`` on the current
mask and are computed outside the timed region.

Before timing, ``--verify`` frames of the big scenario are stepped in lockstep
live vs grid (cell-space snapshots compared) to prove the injected big-body
worlds are the same world in both engines.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.game_config import get_config, initialize_config  # noqa: E402
from src.simd_env import parity  # noqa: E402
from src.simd_env.batch_sim import BatchSim, BatchSimConfig  # noqa: E402
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster  # noqa: E402
from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim  # noqa: E402
from src.simd_env.grid_scenarios import inject_serpentines  # noqa: E402
from src.simd_env.grid_sim import GridBatchSim  # noqa: E402
from src.simd_env.parity import SurvivorPolicy  # noqa: E402

BIG = [900, 600, 400, 300, 150, 150]
CARD = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def gate_cfg(num_envs: int) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=num_envs,
        num_snakes=6,
        game_width=1450,
        game_height=830,
        initial_food=250,
        max_food=300,
        mechanics_version=2,
        max_capacity=1600,
        frame_rate=1,
    )


def policies(n: int, s: int):
    return [SurvivorPolicy(500 + e, s, boost_prob=0.15) for e in range(n)]


def sim_actions(pols, mask: np.ndarray) -> np.ndarray:
    return np.stack([pols[e].actions(mask[e]) for e in range(len(pols))])


# ---------------------------------------------------------------------------
# Live reference worlds
# ---------------------------------------------------------------------------
def live_worlds(cfg: BatchSimConfig, seeds, big: bool):
    """One PyRefGame per seed; for ``big`` copy an injected GridBatchSim world in."""
    import random

    one = replace(cfg, num_envs=1)
    worlds = []
    for seed in seeds:
        random.seed(seed)
        ref = parity.PyRefGame(one, allow_respawn=True)
        if big:
            src = GridBatchSim(one, seeds=[seed], train_mode=True, allow_respawn=True)
            inject_serpentines([src], BIG, box_height=8)
            copy_world_to_live(src, ref)
            worlds.append((ref, src))
        else:
            worlds.append((ref, None))
    return worlds


def copy_world_to_live(src: GridBatchSim, ref) -> None:
    s = src.s
    for sidx, snake in enumerate(ref.snakes):
        body = src.get_bodies(0, sidx)
        snake.segments = [(c * s, r * s) for c, r in body]
        snake.length = int(src.length[0, sidx])
        snake.direction = CARD[int(src.direction[0, sidx])]
        snake.is_alive = bool(src.alive[0, sidx])
        snake.boost_frames = int(src.boost_frames[0, sidx])
        snake._reward_prev_length = int(src.length[0, sidx])
    fm = ref.food_manager
    fm.food[:] = [(c * s, r * s) for c, r in src.food_cells[0]]
    if hasattr(fm, "_food_set"):
        fm._food_set = set(fm.food)


def verify_big_world(cfg: BatchSimConfig, frames: int) -> dict:
    """Lockstep the injected big world: live reference vs GridBatchSim."""
    import random

    one = replace(cfg, num_envs=1)
    random.seed(77)
    ref = parity.PyRefGame(one, allow_respawn=True)
    sim = GridBatchSim(one, seeds=[77], train_mode=True, allow_respawn=True)
    inject_serpentines([sim], BIG, box_height=8)
    copy_world_to_live(sim, ref)
    # Align RNG: the live reference consumes the global stream.
    random.setstate(sim._rngs[0]._rng.getstate())
    pol = SurvivorPolicy(9, cfg.num_snakes, boost_prob=0.3)
    deaths = 0
    for f in range(frames):
        mask = sim.get_action_mask()[0]
        live_mask = np.array(ref.action_masks(), dtype=bool)
        if not np.array_equal(mask, live_mask):
            return {"ok": False, "frame": f, "field": "mask"}
        a = pol.actions(mask)
        ref.step(a)
        sim.step(a.reshape(1, -1))
        snap = ref.snapshot()
        mine = parity._batch_snapshot(sim, 0)
        for key in ("bodies", "alive", "lengths", "food"):
            if snap[key] != mine[key]:
                return {"ok": False, "frame": f + 1, "field": key}
        deaths += int(sim.get_done().sum())
    return {"ok": True, "frames": frames, "deaths": deaths, "max_len": int(sim.length.max())}


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------
def time_live(cfg, n_envs: int, big: bool, budget: float, max_frames: int) -> dict:
    worlds = live_worlds(cfg, range(n_envs), big)
    pols = policies(n_envs, cfg.num_snakes)
    masks = [np.array(ref.action_masks(), dtype=bool) for ref, _ in worlds]
    frames = agents = 0
    spent = 0.0
    while spent < budget and frames < max_frames:
        for e, (ref, _) in enumerate(worlds):
            a = pols[e].actions(masks[e])
            alive = sum(1 for sn in ref.snakes if sn.is_alive)
            t0 = time.perf_counter()
            ref.step(a)
            masks[e] = np.array(ref.action_masks(), dtype=bool)
            spent += time.perf_counter() - t0
            frames += 1
            agents += alive
    return rates(frames, agents, spent, lengths=[sn.length for r, _ in worlds for sn in r.snakes])


def time_sim(cls, cfg, n_envs: int, big: bool, budget: float, max_frames: int, feat: list):
    sim = cls(cfg, seeds=list(range(n_envs)), train_mode=True, allow_respawn=True)
    if big:
        inject_serpentines([sim], BIG, box_height=8)
    pols = policies(n_envs, cfg.num_snakes)
    frames = agents = 0
    spent = 0.0
    feat_t = {k: 0.0 for k in feat}
    feat_n = 0
    steps = 0
    while spent < budget and frames < max_frames:
        a = sim_actions(pols, sim.get_action_mask())
        alive = int(sim.alive.sum())
        t0 = time.perf_counter()
        sim.step(a)
        spent += time.perf_counter() - t0
        frames += n_envs
        agents += alive
        steps += 1
        if feat and steps % 4 == 1:
            feat_n += int(sim.alive.sum())
            for k in feat:
                t0 = time.perf_counter()
                if k == "raster31v2":
                    build_observations(obs_inputs_from_batch_sim(sim), sim.get_action_mask())
                elif k == "ego":
                    build_ego_raster(sim)
                elif k == "ego_noreach":
                    build_ego_raster(sim, EgoRasterConfig(reach_steps=0))
                feat_t[k] += time.perf_counter() - t0
    out = rates(frames, agents, spent, lengths=sim.length[sim.alive].tolist())
    for k in feat:
        out[f"feat_{k}_agents_per_s"] = round(feat_n / feat_t[k], 1) if feat_t[k] else None
        out[f"feat_{k}_us_per_agent"] = round(1e6 * feat_t[k] / max(feat_n, 1), 2)
    return out


def rates(frames: int, agents: int, spent: float, lengths) -> dict:
    lens = np.asarray(lengths, dtype=float)
    return {
        "env_frames": frames,
        "agent_steps": agents,
        "seconds": round(spent, 3),
        "env_frames_per_s": round(frames / spent, 1),
        "agent_steps_per_s": round(agents / spent, 1),
        "mean_len_end": round(float(lens.mean()), 1) if lens.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=6.0, help="timed seconds per cell")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--verify", type=int, default=150)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument(
        "--big-steps",
        type=int,
        default=0,
        help="cap frames per world in the big scenario (0 = budget only); keeps it big-bodied",
    )
    ap.add_argument("--scenarios", type=str, default="fresh,big")
    args = ap.parse_args()

    saved = get_config()
    cfg = gate_cfg(1)
    parity._install_v2_config(cfg)
    try:
        result = {
            "host": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "budget_s": args.budget,
            "verify_big_world": verify_big_world(cfg, args.verify),
            "cells": [],
        }
        print(json.dumps(result["verify_big_world"]), flush=True)
        plan = []
        for scen in args.scenarios.split(","):
            big = scen == "big"
            plan.append(("live", 4, scen, big))
            plan.append(("batch", 1, scen, big))
            plan.append(("batch", 64, scen, big))
            plan.append(("grid", 1, scen, big))
            plan.append(("grid", 64, scen, big))
            plan.append(("grid", 256, scen, big))
        for rnd in range(args.rounds):
            for engine, n, scen, big in plan:
                c = gate_cfg(n)
                max_frames = 10**9
                if big and args.big_steps:
                    max_frames = args.big_steps * n
                if engine == "live":
                    r = time_live(c, n, big, args.budget, max_frames)
                else:
                    cls = BatchSim if engine == "batch" else GridBatchSim
                    feat = []
                    if n == 64 and rnd == 0:
                        feat = ["ego", "ego_noreach"]
                        if engine == "batch":
                            feat = ["raster31v2"]
                    r = time_sim(cls, c, n, big, args.budget, max_frames, feat)
                r.update({"engine": engine, "envs": n, "scenario": scen, "round": rnd})
                result["cells"].append(r)
                print(json.dumps(r), flush=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    finally:
        initialize_config(saved)
    return 0 if result["verify_big_world"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
