#!/usr/bin/env python3
"""Reproduce vector61 hidden-state aliases and simultaneous-mask errors.

Run from any directory:

    SNAKE_DQN_DEVICE=cpu /path/to/venv/bin/python \
      runs/state_system_audit_20260906/vector/vector_contract_probe.py

The probe uses one Torch CPU thread, the real ``free_space_v2`` observation
builder, the real static action-safety helper, and the real post-move collision
detector. It performs no training and does not load a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.device_manager import DeviceManager  # noqa: E402
from src.game.game_logic import GameLogic  # noqa: E402
from src.game.scripted_snake import ScriptedSnake  # noqa: E402
from src.game.snake import Snake  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_PATH.with_name("vector_contract_probe.json")


def _state_hash(state: torch.Tensor) -> str:
    """Return a stable digest of a CPU float32 observation."""
    values = state.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _base_snake() -> Snake:
    """Build the common six-cell physical state used by alias probes."""
    snake = Snake(0, (255, 0, 0), (100, 100), 10, 1450, 830, food_capacity=300)
    snake.direction = (1, 0)
    snake.segments = [(100 - 10 * index, 100) for index in range(6)]
    snake.length = 6
    return snake


def _scripted_snake(
    snake_id: int,
    position: tuple[int, int],
    direction: tuple[int, int],
) -> ScriptedSnake:
    """Build a length-one snake for the simultaneous-motion mask probes."""
    snake = ScriptedSnake(
        snake_id,
        (255, 0, 0),
        position,
        10,
        1450,
        830,
        "greedy_food",
        seed=snake_id,
        food_capacity=300,
    )
    snake.segments = [position]
    snake.length = 1
    snake.direction = direction
    return snake


def _collision_records(snakes: list[Snake]) -> list[dict[str, Any]]:
    """Serialize collision identities and types after every snake has moved."""
    return [
        {
            "snake_id": int(snake.id),
            "other_id": None if other is None else int(other.id),
            "type": collision_type,
        }
        for snake, other, collision_type in GameLogic.check_collisions(snakes)
    ]


def hidden_state_aliases() -> dict[str, Any]:
    """Show equal observations with different boost dynamics and v1 reward."""
    food = [(200, 100)]

    boost_early = _base_snake()
    boost_due = _base_snake()
    boost_early.boost_frames = 0
    boost_due.boost_frames = 2
    early_state = boost_early.get_state([], food)
    due_state = boost_due.get_state([], food)
    boost_early.is_boosting = True
    boost_due.is_boosting = True
    boost_early.move()
    boost_due.move()

    fed = _base_snake()
    hungry = _base_snake()
    fed.frames_since_food = 0
    hungry.frames_since_food = 600
    fed_state = fed.get_state([], food)
    hungry_state = hungry.get_state([], food)
    fed_reward = fed.calculate_reward(False, False, fed_state, fed_state, [], food)
    fed_breakdown = dict(fed.last_reward_breakdown or {})
    hungry_reward = hungry.calculate_reward(
        False,
        False,
        hungry_state,
        hungry_state,
        [],
        food,
    )
    hungry_breakdown = dict(hungry.last_reward_breakdown or {})

    return {
        "boost_phase": {
            "observations_equal": bool(torch.equal(early_state, due_state)),
            "observation_sha256": [_state_hash(early_state), _state_hash(due_state)],
            "hidden_boost_frames_before": [0, 2],
            "length_after_same_boost_action": [boost_early.length, boost_due.length],
            "hidden_boost_frames_after": [boost_early.boost_frames, boost_due.boost_frames],
        },
        "hunger_reward_v1": {
            "observations_equal": bool(torch.equal(fed_state, hungry_state)),
            "observation_sha256": [_state_hash(fed_state), _state_hash(hungry_state)],
            "hidden_frames_since_food_before": [0, 600],
            "reward_from_identical_observation": [fed_reward, hungry_reward],
            "reward_breakdown": [fed_breakdown, hungry_breakdown],
        },
    }


def vacating_head_mask_error() -> dict[str, Any]:
    """Show a safe simultaneous move rejected against an enemy's current head."""
    hero = _scripted_snake(0, (100, 100), (1, 0))
    enemy = _scripted_snake(1, (110, 100), (1, 0))
    snakes = [hero, enemy]
    safe_actions = hero._get_safe_actions(snakes, allow_fallback=False)
    per_action_danger = hero._get_per_action_danger(snakes)

    hero.move()
    enemy.move()
    collisions = _collision_records(snakes)
    hero_collided = any(record["snake_id"] == hero.id for record in collisions)

    return {
        "scenario": "hero moves into a length-one enemy head as that head vacates",
        "safe_actions_before": safe_actions,
        "straight_action": 1,
        "straight_marked_safe": 1 in safe_actions,
        "per_action_danger": per_action_danger,
        "heads_after_all_snakes_move": [list(hero.head), list(enemy.head)],
        "hero_collision_after_all_snakes_move": hero_collided,
        "collisions": collisions,
    }


def converging_head_mask_error() -> dict[str, Any]:
    """Show a fatal simultaneous head-on approved by the static safety helper."""
    hero = _scripted_snake(0, (100, 100), (1, 0))
    enemy = _scripted_snake(1, (120, 100), (-1, 0))
    snakes = [hero, enemy]
    safe_actions = hero._get_safe_actions(snakes, allow_fallback=False)
    per_action_danger = hero._get_per_action_danger(snakes)

    hero.move()
    enemy.move()
    collisions = _collision_records(snakes)
    hero_collided = any(record["snake_id"] == hero.id for record in collisions)

    return {
        "scenario": "hero and enemy converge on the same head cell",
        "safe_actions_before": safe_actions,
        "straight_action": 1,
        "straight_marked_safe": 1 in safe_actions,
        "per_action_danger": per_action_danger,
        "heads_after_all_snakes_move": [list(hero.head), list(enemy.head)],
        "hero_collision_after_all_snakes_move": hero_collided,
        "collisions": collisions,
    }


def main() -> int:
    """Run every deterministic probe and write its JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    DeviceManager.override_device(torch.device("cpu"))
    load_and_initialize_config(str(REPO_ROOT / "configs/free_space_v2.yaml"))

    result = {
        "probe": "vector61 contract aliases and static-mask errors",
        "config": "configs/free_space_v2.yaml",
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "training_performed": False,
        "checkpoint_loaded": False,
        "hidden_state_aliases": hidden_state_aliases(),
        "static_mask_errors": {
            "safe_move_rejected": vacating_head_mask_error(),
            "fatal_move_approved": converging_head_mask_error(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    DeviceManager.reset_for_testing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
