#!/usr/bin/env python3
"""Actor hot-path throughput benchmark (blueprint P0 "Refit actor fixes").

Steps a seeded actor-shaped environment (default: 6 snakes on the 0.2-scale
actor board) through the REAL policy path — ``GameState.update(train_mode=True,
learn=False)``, exactly what an ``ApexActor`` process runs per frame — and
reports frames/s and transitions/s. Untrained network weights are fine: the
cost profile is feature extraction + safe-action simulation + collision
handling, not the tiny MLP forward.

The ``--no-carry`` flag disables the AISnake carry-forward selection cache so
the legacy duplicate-work path (get_state and _get_safe_actions computed twice
per snake per frame) can be measured for before/after comparison.

Usage:
  ./venv/bin/python src/scripts/bench_actor.py                # new cached path
  ./venv/bin/python src/scripts/bench_actor.py --no-carry     # legacy path
  ./venv/bin/python src/scripts/bench_actor.py --frames 2000 --snakes 6
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.game.ai_snake import AISnake  # noqa: E402
from src.game.game_state import GameState  # noqa: E402
from src.training.apex_actor import compute_actor_epsilon  # noqa: E402


def _configure_snakes_like_actor(env: GameState, epsilon: float, carry_forward: bool) -> None:
    """Apply the ApexActor per-snake overrides (epsilon, collection-only policy).

    Args:
        env: The benchmark environment.
        epsilon: Fixed actor epsilon override for every snake.
        carry_forward: Whether the carry-forward selection cache is enabled.
    """
    configured_policies = set()
    for snake in env.snakes:
        if not isinstance(snake, AISnake):
            continue
        snake.actor_epsilon = epsilon
        snake.current_epsilon = epsilon
        snake.carry_forward_selection = carry_forward
        policy = getattr(snake, "policy", None)
        if policy is not None and id(policy) not in configured_policies:
            configured_policies.add(id(policy))
            # Actors only collect experiences; the learner owns optimization.
            policy.training = False
            if getattr(policy, "memory", None) is not None:
                policy.memory.clear()


def run_benchmark(
    frames: int,
    num_snakes: int,
    board_scale: float,
    food_multiplier: float,
    seed: int,
    carry_forward: bool,
    warmup_frames: int,
) -> Dict[str, float]:
    """Run the actor-path benchmark and return throughput metrics.

    Args:
        frames: Number of timed environment frames to step.
        num_snakes: Snakes per environment.
        board_scale: Actor arena width/height multiplier.
        food_multiplier: Actor food-count multiplier.
        seed: RNG seed (random/numpy/torch).
        carry_forward: Enable the AISnake carry-forward selection cache.
        warmup_frames: Untimed frames stepped before measurement.

    Returns:
        Dict with frames/transitions counts, elapsed seconds, and rates.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GameState(
        headless=True,
        num_snakes=num_snakes,
        board_scale=board_scale,
        food_multiplier=food_multiplier,
    )
    epsilon = compute_actor_epsilon(0, num_actors=1)
    _configure_snakes_like_actor(env, epsilon, carry_forward)

    max_frames = GameConfig.MAX_FRAMES
    episode_steps = 0

    def step_once() -> int:
        """Step one frame (resetting episodes like the actor loop); return transitions."""
        nonlocal episode_steps
        if episode_steps >= max_frames or not any(s.is_alive for s in env.snakes):
            env.reset()
            _configure_snakes_like_actor(env, epsilon, carry_forward)
            episode_steps = 0
        env.update(train_mode=True, learn=False)
        episode_steps += 1
        frame = env.frame
        return sum(
            1 for snake in env.snakes if getattr(snake, "last_transition_frame", None) == frame
        )

    for _ in range(warmup_frames):
        step_once()

    transitions = 0
    start = time.perf_counter()
    for _ in range(frames):
        transitions += step_once()
    elapsed = time.perf_counter() - start

    return {
        "frames": float(frames),
        "transitions": float(transitions),
        "elapsed_s": elapsed,
        "frames_per_s": frames / elapsed if elapsed > 0 else float("nan"),
        "transitions_per_s": transitions / elapsed if elapsed > 0 else float("nan"),
    }


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark the actor env hot path.")
    parser.add_argument("--frames", type=int, default=2000, help="Timed frames (default 2000)")
    parser.add_argument("--warmup", type=int, default=100, help="Untimed warmup frames")
    parser.add_argument("--snakes", type=int, default=6, help="Snakes per env (default 6)")
    parser.add_argument(
        "--board-scale",
        type=float,
        default=None,
        help="Arena scale (default: config apex.actor_board_scale)",
    )
    parser.add_argument(
        "--food-multiplier",
        type=float,
        default=None,
        help="Food multiplier (default: config apex.actor_food_multiplier)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument(
        "--no-carry",
        action="store_true",
        help="Disable the carry-forward selection cache (legacy duplicate-work path)",
    )
    args = parser.parse_args(argv)

    load_and_initialize_config(args.config)
    torch.set_num_threads(1)

    board_scale = (
        args.board_scale if args.board_scale is not None else GameConfig.APEX_ACTOR_BOARD_SCALE
    )
    food_multiplier = (
        args.food_multiplier
        if args.food_multiplier is not None
        else GameConfig.APEX_ACTOR_FOOD_MULTIPLIER
    )

    print(
        f"bench_actor | snakes={args.snakes} board_scale={board_scale:.2f} "
        f"food_multiplier={food_multiplier:.2f} frames={args.frames} seed={args.seed} "
        f"carry_forward={not args.no_carry}"
    )
    results = run_benchmark(
        frames=args.frames,
        num_snakes=args.snakes,
        board_scale=board_scale,
        food_multiplier=food_multiplier,
        seed=args.seed,
        carry_forward=not args.no_carry,
        warmup_frames=args.warmup,
    )
    print(
        f"frames={int(results['frames'])} transitions={int(results['transitions'])} "
        f"elapsed_s={results['elapsed_s']:.3f}"
    )
    print(
        f"frames_per_s={results['frames_per_s']:.1f} "
        f"transitions_per_s={results['transitions_per_s']:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
