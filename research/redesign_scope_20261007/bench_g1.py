#!/usr/bin/env python3
"""Gate G1 bench: hero agent-steps/s per process, opponent rows included (redesign M1).

G1 (``docs/research/redesign_scope_2026-10-07.md`` section 8): at least 20k hero
agent-steps/s per process, ego2s observation and masks included, no learner, WITH the
opponent rows (5 of 6 slots) of the opponent mix.

One process, one thread (``OMP_NUM_THREADS=1``, ``torch.set_num_threads(1)``); run it under
``nice -n 10``. The world is the gate world: the pinned deployment config and the
``promotion-v2-watch-rect`` profile (1450x830, 6 snakes, mechanics v2, Watch food branch,
respawn on), on :class:`GridBatchSim` with ``E`` worlds in lockstep. Per frame, inside
``step_with_policy`` (the Watch decision point):

* sim: ``GridBatchSim(jit=--sim-jit)`` (numba mask / collision kernels on or off);
* hero (slot 0 of every world): ego2s observation for the living heroes
  (``build_ego_raster(rows=...)``, ``--featurizer numpy|numba``) and an action. The action
  is a uniform legal action unless ``--hero-forward cpu|mps``, which runs the draft
  ``Ego2sNet`` (bench_network.py) on the hero batch and takes the masked argmax (the
  acting forward, H2D copy and sync included; untrained weights, cost only);
* opponents (slots 1..5) by ``--opponents``:

  - ``random``: uniform legal actions (no policy cost; the sim + hero floor);
  - ``scripted``: the gate's scripted mix (``scripted-anchor/v1`` greedy_food /
    random_safe by the strict balanced roster), ``--anchors py`` (the gate's
    ``_ProfileAnchorSimdPolicy``) or ``fast`` (:class:`FastProfileAnchorSimdPolicy`,
    decision-identical);
  - ``frozen`` / ``mixed``: the gate's frozen / mixed rosters; checkpoint slots run the
    vector61 pool checkpoints through ``Vector61SimdPolicy(forward="batched")`` and its
    shared ``Vector61Runtime`` (prepare + post-step capture), scripted slots as above.

Scenarios: ``fresh`` (episode start), ``warm`` (after ``--warm-frames`` untimed frames of
the same play: longer bodies, corpses, respawns) and ``big`` (injected serpentines of length
900/600/400/300/150/150, the regime where the champion dies; capped at ``--big-frames``
frames per round so it stays big). ``big`` is a death-heavy stress case: the injected
bodies die at ~30x the gate-world rate (corpse drops and grid rebuilds dominate). Timing
excludes construction, JIT and warm-up frames.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

BIG = [900, 600, 400, 300, 150, 150]


class Timer:
    def __init__(self) -> None:
        self.t: Dict[str, float] = defaultdict(float)

    def add(self, key: str, started: float) -> float:
        now = time.perf_counter()
        self.t[key] += now - started
        return now


def build(args: argparse.Namespace, seed_offset: int) -> Dict[str, Any]:
    import torch

    from src.simd_env import eval_engine as ee
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.grid_scenarios import inject_serpentines
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

    ctx = args.ctx
    E = args.worlds
    seeds = gid.world_seeds(E, seed_offset)
    base = ee._config_from_game_config(6, 0.99, ctx["profile"])
    cfg = replace(base, num_envs=E, body_storage_capacity=max(2048, base.max_capacity + 8))
    sim = GridBatchSim(
        cfg, seeds=seeds, train_mode=False, allow_respawn=True, jit=args.sim_jit == "on"
    )
    if args.scenario == "big":
        inject_serpentines([sim], BIG, box_height=8)
    # Opponent assignment: the strict balanced roster of the mix (scripted for "random").
    mix = "scripted" if args.opponents in ("random", "scripted") else args.opponents
    rows = [r for r in ctx["ds"]._design_rows(seeds) if r["mix"] == mix]
    by_seed = {int(r["world_seed"]): r for r in rows}
    groups: Dict[Any, List[tuple]] = defaultdict(list)  # policy key -> (env, slot)
    for e, seed in enumerate(seeds):
        for slot_index, slot in enumerate(by_seed[int(seed)]["slots"], start=1):
            spec = ctx["lookup"][slot["member_sha256"]]
            groups[spec].append((e, slot_index))
    policies: Dict[Any, Any] = {}
    runtime = None
    controlled = np.zeros((E, 6), dtype=bool)
    for spec, cells in groups.items():
        if args.opponents == "random":
            policies[spec] = None
        elif spec[0] == "scripted":
            cls = FastProfileAnchorSimdPolicy if args.anchors == "fast" else None
            policies[spec] = (
                cls(spec[1], seeds) if cls else ee._ProfileAnchorSimdPolicy(spec[1], seeds)
            )
        else:
            if runtime is None:
                runtime = Vector61Runtime()
            policies[spec] = Vector61SimdPolicy(spec[1], runtime, forward="batched")
            for e, s in cells:
                controlled[e, s] = True
    if runtime is not None:
        runtime.bind(sim, controlled)
    net = None
    if args.hero_forward != "none":
        from research.redesign_scope_20261007.bench_network import Ego2sNet

        device = torch.device(args.hero_forward)
        torch.manual_seed(0)
        net = Ego2sNet().to(device).eval()
    return {
        "sim": sim,
        "groups": {k: np.array(v, dtype=np.int64) for k, v in groups.items()},
        "policies": policies,
        "runtime": runtime,
        "net": net,
        "rng": np.random.default_rng(seed_offset),
    }


def run_round(args: argparse.Namespace, world: Dict[str, Any], seconds: float, max_frames: int):
    import torch

    from src.simd_env.ego_raster import build_ego_raster

    sim = world["sim"]
    rng = world["rng"]
    runtime = world["runtime"]
    net = world["net"]
    timer = Timer()
    counts = {"frames": 0, "hero_steps": 0, "opp_rows": 0, "net_rows": 0, "deaths": 0}
    E = sim.E

    def random_legal(m: np.ndarray) -> np.ndarray:
        score = rng.random(m.shape) * m
        return np.where(m.any(axis=1), score.argmax(axis=1), 1)

    def choose(prepared):
        t = time.perf_counter()
        masks = prepared.get_resolved_action_mask()
        alive = prepared.get_alive()
        actions = np.ones((E, prepared.S), dtype=np.int64)
        hero_rows = np.argwhere(alive[:, :1])
        obs = build_ego_raster(prepared, backend=args.featurizer, rows=hero_rows)
        t = timer.add("hero_obs", t)
        hm = masks[hero_rows[:, 0], 0]
        if net is not None and len(hero_rows):
            dev = next(net.parameters()).device
            with torch.no_grad():
                q = net(
                    torch.from_numpy(obs["local"]).to(dev).float(),
                    torch.from_numpy(obs["global"]).to(dev).float(),
                    torch.from_numpy(obs["scalars"]).to(dev),
                )
                q = torch.where(torch.from_numpy(hm).to(dev), q, torch.full_like(q, -1e9))
                act = q.argmax(dim=1).cpu().numpy()
            actions[hero_rows[:, 0], 0] = act
            t = timer.add("hero_forward", t)
        else:
            actions[hero_rows[:, 0], 0] = random_legal(hm)
        counts["hero_steps"] += len(hero_rows)
        if runtime is not None:
            runtime.prepare(prepared)
            t = timer.add("opp_v61_prepare", t)
        for spec, cells in world["groups"].items():
            live = cells[alive[cells[:, 0], cells[:, 1]]]
            if not len(live):
                continue
            m = masks[live[:, 0], live[:, 1]]
            policy = world["policies"][spec]
            counts["opp_rows"] += len(live)
            if policy is None:
                actions[live[:, 0], live[:, 1]] = random_legal(m)
                t = timer.add("opp_random", t)
                continue
            actions[live[:, 0], live[:, 1]] = policy.actions(m, prepared, live)
            key = "opp_v61_forward" if spec[0] == "checkpoint" else "opp_scripted"
            if spec[0] == "checkpoint":
                counts["net_rows"] += len(live)
            t = timer.add(key, t)
        return actions

    started = time.perf_counter()
    while time.perf_counter() - started < seconds and counts["frames"] < max_frames:
        t0 = time.perf_counter()
        before = dict(timer.t)
        sim.step_with_policy(choose)
        inner = sum(timer.t.values()) - sum(before.values())
        t1 = time.perf_counter()
        timer.t["sim_step"] += (t1 - t0) - inner
        if runtime is not None:
            runtime.observe_step(sim)
            timer.add("opp_v61_capture", t1)
        counts["frames"] += 1
        counts["deaths"] += int(sim.get_done().sum())
    wall = time.perf_counter() - started
    return wall, counts, dict(timer.t), sim


def run_cell(args: argparse.Namespace) -> Dict[str, Any]:
    rounds = []
    for r in range(args.rounds):
        world = build(args, seed_offset=1000 * (r + 1))
        # Warm-up (JIT compile on first call; caches warm): 3 frames, untimed. The
        # "warm" scenario first plays --warm-frames untimed frames with the same policies.
        run_round(args, world, 1e9, args.warm_frames if args.scenario == "warm" else 3)
        cap = args.big_frames if args.scenario == "big" else 10**9
        wall, counts, parts, sim = run_round(args, world, args.seconds, cap)
        alive_len = sim.length[sim.alive]
        rounds.append(
            {
                "wall": wall,
                **counts,
                "hero_steps_per_s": counts["hero_steps"] / wall,
                "env_frames_per_s": counts["frames"] * args.worlds / wall,
                "us_per_world_frame": {
                    k: 1e6 * v / max(1, counts["frames"] * args.worlds) for k, v in parts.items()
                },
                "hero_obs_us_per_agent": 1e6
                * parts.get("hero_obs", 0.0)
                / max(1, counts["hero_steps"]),
                "mean_length_end": float(alive_len.mean()) if alive_len.size else 0.0,
                "max_length_end": int(alive_len.max()) if alive_len.size else 0,
                "deaths_per_world_frame": counts["deaths"] / max(1, counts["frames"] * args.worlds),
            }
        )
    med = float(np.median([r["hero_steps_per_s"] for r in rounds]))
    return {
        "scenario": args.scenario,
        "warm_frames": args.warm_frames if args.scenario == "warm" else 0,
        "worlds": args.worlds,
        "opponents": args.opponents,
        "anchors": args.anchors,
        "featurizer": args.featurizer,
        "hero_forward": args.hero_forward,
        "sim_jit": args.sim_jit,
        "median_hero_steps_per_s": med,
        "g1_met": med >= 20000,
        "rounds": rounds,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--scenario", choices=("fresh", "warm", "big"), default="fresh")
    ap.add_argument("--warm-frames", type=int, default=1500)
    ap.add_argument(
        "--opponents", choices=("random", "scripted", "frozen", "mixed"), default="scripted"
    )
    ap.add_argument("--anchors", choices=("py", "fast"), default="fast")
    ap.add_argument("--featurizer", choices=("numpy", "numba"), default="numba")
    ap.add_argument("--hero-forward", choices=("none", "cpu", "mps"), default="none")
    ap.add_argument("--sim-jit", choices=("on", "off"), default="on")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--big-frames", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None, help="append one JSON line")
    args = ap.parse_args()
    import numba
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.ctx = gid._context(1)
    result = run_cell(args)
    result["env"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "torch": torch.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "nice": os.nice(0),
        "loadavg": os.getloadavg(),
    }
    line = json.dumps(result, sort_keys=True)
    print(line, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
