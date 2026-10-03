"""Single-threaded micro-benchmarks for the serving-time search feasibility study.

Design input only (Tier 0). It measures the cost of the primitives a search would be
built from; it plays no gate episode, writes no artifact and changes no default.

Run (one process, one thread, each section well under 60 s)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
        /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
        research/serving_time_search_20261003/bench.py --section all

Sections: ``q`` (ApexNetwork forward, torch and NumPy), ``live`` (GameState frame,
hero decision pieces, world fork + clone stepping, veto primitives), ``proto`` (the
research rollout prototype, 6 actions x horizons 8-60), ``simd``
(BatchSim step at E=1/16/64, deepcopy, vector61 featurize). Worlds use the strict gate's
deployment config (``research/apex_safety_20260926/deployment.yaml``) with the champion
on every live slot. Numbers are wall-clock medians on a shared, loaded machine: they are
order-of-magnitude design inputs, never gate evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG = REPO / "research" / "apex_safety_20260926" / "deployment.yaml"
CHAMPION = Path(
    os.environ.get(
        "SNAKE_CHAMPION",
        "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/champion_a5_freespace_20260621.pth",
    )
)


def _ms(fn: Callable[[], object], repeat: int, inner: int = 1) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) * 1000.0 / inner)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p10_ms": round(sorted(samples)[len(samples) // 10], 4),
        "n": repeat * inner,
    }


def _setup() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from src.core.config_loader import load_and_initialize_config

    load_and_initialize_config(str(CONFIG))


def bench_q() -> Dict[str, object]:
    import numpy as np
    import torch

    from src.scripts.tournament_eval import build_policy_from_checkpoint

    net = build_policy_from_checkpoint(str(CHAMPION)).dqn.eval()
    out: Dict[str, object] = {}
    with torch.no_grad():
        for batch in (1, 6, 18, 64, 256, 1024):
            x = torch.randn(batch, 61)
            net(x)
            out[f"torch_b{batch}"] = _ms(lambda: net(x), repeat=30, inner=20)
    # NumPy float32 forward (same arithmetic graph; not bit-identical to torch).
    params = [p.detach().numpy().astype(np.float32) for p in net.parameters()]
    w1, b1, w2, b2, wv1, bv1, wv2, bv2, wa1, ba1, wa2, ba2 = params

    def np_forward(x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ w1.T + b1, 0)
        f = np.maximum(h @ w2.T + b2, 0)
        v = np.maximum(f @ wv1.T + bv1, 0) @ wv2.T + bv2
        a = np.maximum(f @ wa1.T + ba1, 0) @ wa2.T + ba2
        return v + a - a.mean(axis=1, keepdims=True)

    for batch in (1, 6, 64):
        xn = np.random.default_rng(0).standard_normal((batch, 61)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(xn)).numpy()
        out[f"numpy_b{batch}"] = _ms(lambda: np_forward(xn), repeat=30, inner=20)
        out[f"numpy_b{batch}_max_abs_diff_vs_torch"] = float(np.abs(np_forward(xn) - ref).max())
    # Batch-shape determinism: row 0 of a batch-64 forward vs a batch-1 forward.
    with torch.no_grad():
        x64 = torch.randn(64, 61)
        diff = (net(x64[:1]) - net(x64)[:1]).abs().max().item()
    out["torch_row0_b1_vs_b64_max_abs_diff"] = diff
    return out


def _live_world(seed: int, warm_frames: int):
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto
    from src.game.game_state_factory import (
        configure_eval_game_state,
        create_training_game_state,
    )
    from src.scripts.eval_cli import set_seed
    from src.scripts.tournament_eval import _attach_agent, parse_agent_spec

    set_seed(seed)
    gs = create_training_game_state(eval_mode=False)
    cache: Dict[str, object] = {}
    for slot in range(len(gs.snakes)):
        _attach_agent(gs, slot, parse_agent_spec(str(CHAMPION)), seed, cache, None)
    gs._shared_policy = None
    configure_eval_game_state(gs)
    hero = gs.snakes[0]
    veto = install_space_and_head_veto(hero, 8.0)
    hero.auto_respawn = False
    t0 = time.perf_counter()
    for _ in range(warm_frames):
        gs.update(train_mode=True, learn=False, allow_respawn=True)
        gs.food_manager.trim_ambient(300)
        if not hero.is_alive:
            break
    return gs, hero, veto, cache, (time.perf_counter() - t0) * 1000.0


def bench_live(seed: int, warm_frames: int) -> Dict[str, object]:
    import torch

    from src.evaluation.safety_veto_v3 import grid_for, static_blocked
    from src.evaluation.safety_veto_v5 import simulate_action
    from src.evaluation.safety_veto_v7 import area_cap, landing_area

    gs, hero, veto, cache, warm_ms = _live_world(seed, warm_frames)
    out: Dict[str, object] = {
        "seed": seed,
        "warm_frames": gs.frame,
        "hero_alive": bool(hero.is_alive),
        "hero_length": int(hero.length),
        "lengths": [int(s.length) for s in gs.snakes],
        "warm_ms_per_frame": round(warm_ms / max(gs.frame, 1), 3),
    }
    if not hero.is_alive:
        return out
    food = list(gs.food_manager.food) if hasattr(gs.food_manager, "food") else []

    # Hero decision pieces (fresh featurization path; the live loop usually carries).
    out["hero_get_state"] = _ms(
        lambda: hero.get_state(gs.snakes, food, update_enemy_memory=False), repeat=20
    )
    out["hero_safe_actions"] = _ms(lambda: hero._get_safe_actions(gs.snakes), repeat=20)
    state = hero.get_state(gs.snakes, food, update_enemy_memory=False)
    with torch.no_grad():
        q = hero.policy.dqn(state.unsqueeze(0)).squeeze()
    mask = torch.ones(6, dtype=torch.bool)
    base = int(q.argmax().item())
    v8_cold = copy.deepcopy(veto)
    out["hero_v8_apply"] = _ms(lambda: v8_cold.apply(hero, gs.snakes, q, mask, base), repeat=20)

    # Veto primitives a hand-written forward model would reuse.
    grid = grid_for(hero)
    blocked = static_blocked(hero, gs.snakes, grid)
    cap = area_cap(hero.length, grid, blocked)
    out["area_cap"] = cap
    out["simulate_action_x6"] = _ms(lambda: [simulate_action(hero, a) for a in range(6)], 20)
    out["landing_area_x6"] = _ms(
        lambda: [landing_area(hero, a, blocked, grid, cap) for a in range(6)], repeat=10
    )
    out["static_blocked"] = _ms(lambda: static_blocked(hero, gs.snakes, grid), repeat=20)

    # Fork: deepcopy the world with policies/networks shared via the memo (they are
    # frozen at eval), then step the clone with the global RNG saved/restored.
    def fork():
        memo = {}
        for snake in gs.snakes:
            pol = getattr(snake, "policy", None)
            if pol is not None:
                memo[id(pol)] = pol
        for pol in cache.values():
            memo[id(pol)] = pol
        return copy.deepcopy(gs, memo)

    out["fork_deepcopy"] = _ms(fork, repeat=10)
    out["rng_save_restore"] = _ms(lambda: random.setstate(random.getstate()), repeat=50)

    def clone_rollout(n: int):
        def run():
            state = random.getstate()
            clone = fork()
            for _ in range(n):
                clone.update(train_mode=True, learn=False, allow_respawn=True)
            random.setstate(state)

        return run

    for n in (1, 8, 32):
        out[f"fork_plus_{n}_frames"] = _ms(clone_rollout(n), repeat=3 if n == 32 else 5)

    # The naive deepcopy keeps GameState's ``get_frame=lambda: self.frame`` closure
    # pointing at the ORIGINAL world, so clone snakes miss the carry-forward frame guard
    # (fresh featurization every frame, and a different selection path from live).
    # Rebinding the closure to the clone restores live semantics.
    def fork_rebound():
        clone = fork()
        for snake in clone.snakes:
            if hasattr(snake, "_get_frame"):
                snake._get_frame = lambda c=clone: c.frame
        return clone

    def rebound_rollout(n: int):
        def run():
            state = random.getstate()
            clone = fork_rebound()
            for _ in range(n):
                clone.update(train_mode=True, learn=False, allow_respawn=True)
            random.setstate(state)

        return run

    for n in (8, 32):
        out[f"fork_rebound_plus_{n}_frames"] = _ms(rebound_rollout(n), repeat=3)

    # Fidelity: does a rebound fork predict the live world exactly (same opponents,
    # same RNG state)? And is the live world untouched by forking?
    frame_before = gs.frame
    rng_before = random.getstate()
    clone = fork_rebound()
    for _ in range(16):
        clone.update(train_mode=True, learn=False, allow_respawn=True)
    random.setstate(rng_before)
    out["live_frame_untouched_by_fork"] = gs.frame == frame_before
    naive = fork()
    for _ in range(16):
        naive.update(train_mode=True, learn=False, allow_respawn=True)
    random.setstate(rng_before)
    for _ in range(16):
        gs.update(train_mode=True, learn=False, allow_respawn=True)
    out["rebound_fork_16f_matches_live_bodies"] = all(
        list(a.segments) == list(b.segments) for a, b in zip(clone.snakes, gs.snakes)
    )
    out["naive_fork_16f_matches_live_bodies"] = all(
        list(a.segments) == list(b.segments) for a, b in zip(naive.snakes, gs.snakes)
    )
    out["rebound_fork_16f_matches_live_food"] = list(clone.food_manager.food) == list(
        gs.food_manager.food
    )

    # Live frame cost (6 champion snakes, v8 on the hero), measured last so the
    # world above is the warm-up state.
    def frame():
        gs.update(train_mode=True, learn=False, allow_respawn=True)
        gs.food_manager.trim_ambient(300)

    out["live_frame"] = _ms(frame, repeat=100)
    out["hero_length_end"] = int(hero.length)
    return out


def bench_proto(seed: int, warm_frames: int) -> Dict[str, object]:
    """Cost of the hero-only rollout prototype (all 6 actions) on a live world."""
    from research.serving_time_search_20261003.rollout_search import (
        RolloutParams,
        score_actions,
    )

    gs, hero, _veto, _cache, _ = _live_world(seed, warm_frames)
    out: Dict[str, object] = {"hero_alive": bool(hero.is_alive), "hero_length": int(hero.length)}
    if not hero.is_alive:
        return out
    rng_before = random.getstate()
    for horizon in (8, 20, 40, 60):
        params = RolloutParams(horizon=horizon)
        scores = score_actions(hero, gs.snakes, range(6), params)
        out[f"h{horizon}_all6"] = _ms(
            lambda: score_actions(hero, gs.snakes, range(6), params), repeat=5
        )
        out[f"h{horizon}_nodes"] = sum(s.nodes for s in scores.values())
        out[f"h{horizon}_passes"] = [bool(scores[a].passes) for a in range(6)]
    for cap in (16, 160):
        params = RolloutParams(horizon=40, policy_cap=cap)
        out[f"h40_policy_cap{cap}_all6"] = _ms(
            lambda: score_actions(hero, gs.snakes, range(6), params), repeat=5
        )
    out["global_rng_untouched"] = random.getstate() == rng_before
    return out


def bench_simd(warm_frames: int) -> Dict[str, object]:
    import numpy as np

    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.eval_engine import GreedyFoodSimdPolicy, _config_from_game_config
    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Params

    out: Dict[str, object] = {}
    policy = GreedyFoodSimdPolicy()
    base_cfg = _config_from_game_config(6, 0.99)
    from dataclasses import replace

    def make(e: int) -> BatchSim:
        return BatchSim(replace(base_cfg, num_envs=e), seeds=list(range(e)), train_mode=False)

    def act(sim: BatchSim) -> np.ndarray:
        rows = np.argwhere(np.ones((sim.E, sim.S), dtype=bool))
        masks = sim.get_action_mask().reshape(-1, 6)
        return policy.actions(masks, sim, rows).reshape(sim.E, sim.S)

    sim1 = make(1)
    t0 = time.perf_counter()
    for _ in range(warm_frames):
        sim1.step(act(sim1))
    out["E1_warm_ms_per_frame_incl_greedy"] = round(
        (time.perf_counter() - t0) * 1000 / warm_frames, 3
    )
    out["E1_lengths"] = [int(x) for x in sim1.get_lengths()[0]]
    out["E1_deepcopy"] = _ms(lambda: copy.deepcopy(sim1), repeat=20)

    def clone_steps(n: int):
        def run():
            c = copy.deepcopy(sim1)
            for _ in range(n):
                c.step(act(c))

        return run

    for n in (1, 8, 32):
        out[f"E1_clone_plus_{n}_steps"] = _ms(clone_steps(n), repeat=5 if n == 32 else 10)

    params = Vector61Params.from_game_config(sim1.cfg)
    feat = Vector61Featurizer(params)
    rows6 = np.array([[0, s] for s in range(6) if sim1.get_alive()[0, s]], dtype=np.int64)
    out["E1_featurize_alive_rows"] = len(rows6)
    out["E1_featurize"] = _ms(
        lambda: feat.featurize(sim1, rows6, None, update_memory=False), repeat=20
    )

    for e in (16, 64):
        sim = make(e)
        for _ in range(200):
            sim.step(act(sim))
        acts = act(sim)
        out[f"E{e}_step"] = _ms(lambda: sim.step(acts), repeat=10)
        out[f"E{e}_step_per_env_ms"] = round(out[f"E{e}_step"]["median_ms"] / e, 4)
        rows = np.argwhere(sim.get_alive())
        out[f"E{e}_featurize_rows"] = int(len(rows))
        out[f"E{e}_featurize"] = _ms(
            lambda: feat.featurize(sim, rows, None, update_memory=False), repeat=5
        )
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--section", choices=("q", "live", "simd", "proto", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--live-warm-frames", type=int, default=600)
    parser.add_argument("--simd-warm-frames", type=int, default=600)
    args = parser.parse_args(argv)
    _setup()
    result: Dict[str, object] = {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_threads": __import__("torch").get_num_threads(),
    }
    sections = ("q", "live", "simd", "proto") if args.section == "all" else (args.section,)
    for name in sections:
        t0 = time.perf_counter()
        if name == "q":
            result["q"] = bench_q()
        elif name == "live":
            result["live"] = bench_live(args.seed, args.live_warm_frames)
        elif name == "proto":
            result["proto"] = bench_proto(args.seed, args.live_warm_frames)
        else:
            result["simd"] = bench_simd(args.simd_warm_frames)
        result[f"{name}_section_wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
