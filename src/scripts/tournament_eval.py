#!/usr/bin/env python3
"""Absolute-skill evaluation: a candidate "hero" snake vs a pool of FROZEN opponents.

The existing `evaluate_checkpoints.py` is self-play — every snake shares one policy,
so cross-stage comparisons are only *relative* (a Red Queen effect: the opponents
improve too) and survival saturates at the frame cap. This harness instead pits the
candidate (snake 0) against opponents (snakes 1..N-1) that all run a FIXED frozen
checkpoint, so the score measures absolute skill against a stable reference.

Design choices for scientific validity:
  * Frozen opponents  -> absolute, not relative, skill.
  * Paired seeds      -> every candidate faces identical food/spawn RNG, so the only
                         difference between two candidates is the hero policy.
  * Greedy (eps=0)    -> fully deterministic rollouts; variance comes only from seeds.
  * Uncensored mass   -> we track len(segments) (slither.io "mass"), which does NOT
                         saturate the way episode-length does under respawn.
  * Mean +/- 95% CI   -> over many seeds; promote only on a CI-separated win.

Hero metrics (vs the frozen pool):
  max_mass   peak length reached            (dominance / ceiling)
  mean_mass  time-averaged length alive     (sustained dominance)
  kills      victims attributed to the hero (aggression skill)
  deaths     hero alive->dead transitions   (robustness; lower is better)
  survival   fraction of frames alive       (robustness)

Usage:
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
    saved_snakes/best_apex_pre_slither_20260615.bak.pth \
    saved_snakes/best_apex_stage1_20260615.bak.pth \
    saved_snakes/best_apex.pth \
    --opponent saved_snakes/best_apex_stage1_20260615.bak.pth \
    --config configs/goal_slither_20260615.yaml \
    --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 \
    --json-output logs/tournament_vs_stage1.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import apply_config_to_game_config, load_config  # noqa: E402
from src.core.game_config import GameConfig, initialize_config  # noqa: E402
from src.game.snake_factory import SnakeFactory  # noqa: E402
from src.main import (  # noqa: E402
    configure_eval_game_state,
    create_training_game_state,
    load_checkpoint_into_game_state,
)


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(s) for s in value.split(",") if s.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ckpt_value(ck: dict, key, default):
    """Read a contract value from a checkpoint (top-level or apex_config/config)."""
    if key in ck:
        return ck[key]
    for sub in ("apex_config", "config"):
        d = ck.get(sub)
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def build_policy_from_checkpoint(checkpoint_path: str):
    """Build an inference policy matching the checkpoint's OWN architecture.

    Architecture-aware so a recurrent (GRU/DRQN) candidate can play against a
    feedforward opponent and vice-versa. Inference (training=False) skips the
    TD/reward contract; the use_gru/hidden_size shape checks still run.
    """
    import torch

    from src.training.apex_policy import ApexPolicy

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    use_gru = bool(_ckpt_value(ck, "use_gru", False))
    hidden = int(_ckpt_value(ck, "hidden_size", GameConfig.HIDDEN_SIZE))
    policy = ApexPolicy(
        GameConfig.INPUT_SIZE,
        hidden,
        GameConfig.OUTPUT_SIZE,
        use_gru=use_gru,
        training=False,
    )
    policy.load_state_dict(ck)
    if hasattr(policy, "epsilon"):
        policy.epsilon = 0.0
    return policy


def rollout(candidate_path: str, opponent_path: str, frames: int, seed: int) -> Dict[str, float]:
    """One paired rollout: hero=candidate (snake 0) vs frozen opponents."""
    set_seed(seed)
    gs = create_training_game_state(eval_mode=False)

    # Build hero + opponent policies from their OWN checkpoints (architecture-aware),
    # then attach: snake 0 = candidate, snakes 1.. = one shared frozen opponent.
    hero_policy = build_policy_from_checkpoint(candidate_path)
    opp_policy = build_policy_from_checkpoint(opponent_path)
    gs.snakes[0].policy = hero_policy
    for snake in gs.snakes[1:]:
        snake.policy = opp_policy
    # Disable centralized training so no policy is ever updated.
    gs._shared_policy = None
    # Greedy, inference-only for every policy now attached to the roster.
    configure_eval_game_state(gs)

    hero = gs.snakes[0]
    hero_id = hero.id

    max_mass = len(hero.segments)
    mass_sum = 0.0
    alive_frames = 0
    deaths = 0
    kills = 0
    prev_alive = hero.is_alive

    for _ in range(frames):
        gs.update(train_mode=True, learn=False)
        alive = hero.is_alive
        if alive:
            m = len(hero.segments)
            max_mass = max(max_mass, m)
            mass_sum += m
            alive_frames += 1
        # alive -> dead transition counts one death
        if prev_alive and not alive:
            deaths += 1
        prev_alive = alive
        # kills attributed to the hero this frame
        kills += len(gs.frame_kills.get(hero_id, []))

    gs.full_cleanup()
    return {
        "seed": seed,
        "max_mass": float(max_mass),
        "mean_mass": float(mass_sum / alive_frames) if alive_frames else 0.0,
        "kills": float(kills),
        "deaths": float(deaths),
        "survival": float(alive_frames / frames),
    }


def ci95(values: Sequence[float]) -> float:
    """Half-width of the 95% confidence interval (normal approx)."""
    n = len(values)
    if n < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    return 1.96 * sd / math.sqrt(n)


def summarize(candidate: str, runs: List[Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"candidate": candidate, "n": len(runs), "runs": runs}
    for key in ("max_mass", "mean_mass", "kills", "deaths", "survival"):
        vals = [r[key] for r in runs]
        out[key] = float(np.mean(vals))
        out[f"{key}_ci"] = ci95(vals)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="+", help="Candidate checkpoints to evaluate as the hero")
    p.add_argument("--opponent", required=True, help="Frozen checkpoint used by ALL opponents")
    p.add_argument("--config", default="configs/goal_slither_20260615.yaml")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0,1,2,3,4,5,6,7,8,9"))
    p.add_argument("--json-output", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    initialize_config(cfg)
    apply_config_to_game_config(cfg)

    print(f"Opponents (frozen): {args.opponent}")
    print(f"Arena: {args.config} | frames={args.frames} | seeds={args.seeds}\n")

    summaries = []
    for cand in args.candidates:
        runs = [rollout(cand, args.opponent, args.frames, s) for s in args.seeds]
        summaries.append(summarize(cand, runs))
        s = summaries[-1]
        print(
            f"  done: {Path(cand).name:42s} "
            f"max_mass={s['max_mass']:.1f}+/-{s['max_mass_ci']:.1f} "
            f"mean_mass={s['mean_mass']:.1f} kills={s['kills']:.2f} "
            f"deaths={s['deaths']:.2f} survival={s['survival']:.2f}"
        )

    # Rank by sustained dominance (mean_mass) then peak then kills then fewer deaths.
    summaries.sort(key=lambda s: (s["mean_mass"], s["max_mass"], s["kills"], -s["deaths"]), reverse=True)

    print("\nrank | mean_mass (95% CI) | max_mass (95% CI) |  kills |  deaths | surv | candidate")
    print("-----|--------------------|-------------------|--------|---------|------|----------")
    for i, s in enumerate(summaries, 1):
        print(
            f"{i:>4} | {s['mean_mass']:>7.1f} +/- {s['mean_mass_ci']:>5.1f} | "
            f"{s['max_mass']:>7.1f} +/- {s['max_mass_ci']:>5.1f} | "
            f"{s['kills']:>6.2f} | {s['deaths']:>7.2f} | {s['survival']:>4.2f} | "
            f"{Path(s['candidate']).name}"
        )

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"opponent": args.opponent, "config": args.config,
                                   "frames": args.frames, "seeds": args.seeds,
                                   "summaries": summaries}, indent=2))
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
