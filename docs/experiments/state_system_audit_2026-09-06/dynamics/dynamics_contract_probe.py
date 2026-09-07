#!/usr/bin/env python3
"""Deterministic dynamics probes for the 2026-09-06 state-system audit.

This is an evidence artifact, not a product test.  It records four observed
edge cases and labels each one according to the written SIMD dynamics contract.
Run from any directory with the repository venv's Python.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Keep this small audit process CPU-only and single-threaded even on a Mac with MPS.
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.game_config import AppConfig, initialize_config  # noqa: E402
from src.game.ai_snake import simulate_relative_action_fatality  # noqa: E402
from src.game.food_manager import FoodManager  # noqa: E402
from src.game.game_logic import GameLogic  # noqa: E402
from src.game.game_state import GameState  # noqa: E402
from src.game.scripted_snake import ScriptedSnake  # noqa: E402
from src.game.snake import Snake  # noqa: E402
from src.simd_env.batch_sim import BatchSim, BatchSimConfig  # noqa: E402
from src.simd_env.parity import (  # noqa: E402
    PyRefGame,
    _batch_snapshot,
    _install_v2_config,
)

OUTPUT_PATH = Path(__file__).with_name("dynamics_contract_probe.json")
WIDTH = 300
HEIGHT = 200
SEGMENT = 10


class MoveOnlySnake(Snake):
    """Snake whose update performs the already-selected move and no learning."""

    @property
    def total_reward(self) -> float:
        return 0.0

    def update(self, other_snakes: Sequence[Snake], food: Sequence[Tuple[int, int]]) -> None:
        del other_snakes, food
        self.move()


def install_mechanics_v2() -> None:
    """Install a small rectangular mechanics-v2 world in global GameConfig."""
    base = AppConfig.from_defaults()
    game = replace(
        base.game,
        mechanics_version=2,
        width=WIDTH,
        height=HEIGHT,
        num_snakes=3,
        segment_size=SEGMENT,
        wall_thickness=10,
        initial_food=0,
        max_food=0,
        min_boost_length=5,
        boost_length_cost_frames=3,
        arena_type="rectangular",
        frame_rate=1,
    )
    initialize_config(replace(base, game=game, rewards=replace(base.rewards, version=2)))


def food_manager(food: Sequence[Tuple[int, int]] = ()) -> FoodManager:
    manager = FoodManager(
        game_width=WIDTH,
        game_height=HEIGHT,
        max_food=0,
        initial_food=0,
        segment_size=SEGMENT,
        wall_thickness=10,
    )
    manager.food = list(food)
    return manager


def bare_game(snakes: Sequence[Snake], manager: FoodManager | None = None) -> GameState:
    """Construct only the GameState fields exercised by update/collision handling."""
    game = GameState.__new__(GameState)
    game.snakes = list(snakes)
    game.num_snakes = len(snakes)
    game.food_manager = manager if manager is not None else food_manager()
    game._game_width = WIDTH
    game._game_height = HEIGHT
    game.frame = 0
    game.alive_snakes = len(snakes)
    game.headless = True
    game.frame_collisions = {}
    game.frame_kills = {}
    game.frame_death_causes = {}
    game.frame_ate_food = {}
    game.episode_food_eaten = 0
    game.episode_best_length = max((len(s.segments) for s in snakes), default=1)
    game.episode_current_reward = 0.0
    game.episode_best_reward = 0.0
    game.episode_deaths = 0
    game.episode_kills = 0
    game.episode_collision_counts = {"wall": 0, "self": 0, "head": 0, "body": 0}
    game._shared_policy = None
    return game


def contested_food_once(list_order: Tuple[int, int]) -> Dict[str, object]:
    """Move two equal snakes through one shared head/food cell."""
    snakes: Dict[int, MoveOnlySnake] = {
        0: MoveOnlySnake(0, (255, 0, 0), (120, 50), SEGMENT, WIDTH, HEIGHT),
        1: MoveOnlySnake(1, (0, 0, 255), (140, 50), SEGMENT, WIDTH, HEIGHT),
    }
    snakes[0].segments = [(120 - i * SEGMENT, 50) for i in range(6)]
    snakes[1].segments = [(140 + i * SEGMENT, 50) for i in range(6)]
    for snake in snakes.values():
        snake.length = 6
    snakes[0].direction = (1, 0)
    snakes[1].direction = (-1, 0)

    ordered = [snakes[snake_id] for snake_id in list_order]
    game = bare_game(ordered, food_manager([(130, 50)]))
    game.update(train_mode=True, learn=False)

    alive_ids = [snake_id for snake_id, snake in sorted(snakes.items()) if snake.is_alive]
    ate_ids = [snake_id for snake_id, ate in game.frame_ate_food.items() if ate]
    return {
        "list_order": list(list_order),
        "ate_ids": ate_ids,
        "alive_ids": alive_ids,
        "length_by_id": {str(i): int(s.length) for i, s in sorted(snakes.items())},
        "head_by_id": {str(i): list(s.head) for i, s in sorted(snakes.items())},
        "frame_collisions": game.frame_collisions,
        "frame_kills": game.frame_kills,
    }


def probe_contested_food() -> Dict[str, object]:
    runs = [contested_food_once(order) for order in ((0, 1), (1, 0))]
    checks = {
        "first_listed_snake_always_claims_food": all(
            run["ate_ids"] == [run["list_order"][0]] for run in runs
        ),
        "food_growth_changes_head_on_winner": all(
            run["alive_ids"] == [run["list_order"][0]] for run in runs
        ),
        "winner_changes_when_only_list_order_changes": runs[0]["alive_ids"] != runs[1]["alive_ids"],
    }
    return {
        "title": "Contested food follows snake-list order before head-on size resolution",
        "classification": "proven index dependence; gameplay-fairness concern, not a written-contract violation",
        "written_contract_status": (
            "Conforms to docs/simd_env_spec.md section 1 step 7 (alive snakes consume in "
            "list order) followed by step 8 collision resolution, and section 7.6 "
            "(logical length is authoritative after food growth). The document specifies "
            "the order but does not state a fairness or contested-pellet rule."
        ),
        "source_refs": [
            "docs/simd_env_spec.md:107-130",
            "docs/simd_env_spec.md:623-627",
            "src/game/game_state.py:369-400",
            "src/game/game_state.py:646-652",
            "src/simd_env/batch_sim.py:403-407",
        ],
        "setup": "Equal logical length 6; both heads move to food cell (130,50); mechanics v2.",
        "runs": runs,
        "checks": checks,
        "observed": all(checks.values()),
    }


def three_way_once(list_order: Tuple[int, int, int]) -> Dict[str, object]:
    snakes = {i: Snake(i, (255, 0, 0), (130, 50), SEGMENT, WIDTH, HEIGHT) for i in range(3)}
    ordered = [snakes[snake_id] for snake_id in list_order]
    detected = [
        {"snake": snake.id, "other": None if other is None else other.id, "type": kind}
        for snake, other, kind in GameLogic.check_collisions(ordered)
    ]
    game = bare_game(ordered)
    frame_collisions = game.handle_collisions()
    survivors = [snake_id for snake_id, snake in sorted(snakes.items()) if snake.is_alive]
    return {
        "list_order": list(list_order),
        "detected": detected,
        "survivor_ids": survivors,
        "frame_collisions": frame_collisions,
        "frame_kills": game.frame_kills,
    }


def probe_three_way_head_on() -> Dict[str, object]:
    runs = [three_way_once(order) for order in itertools.permutations((0, 1, 2))]
    checks = {
        "exactly_one_equal_snake_survives_each_order": all(
            len(run["survivor_ids"]) == 1 for run in runs
        ),
        "last_listed_snake_survives_each_order": all(
            run["survivor_ids"] == [run["list_order"][-1]] for run in runs
        ),
        "survivor_identity_depends_on_list_order": len({tuple(run["survivor_ids"]) for run in runs})
        == 3,
    }
    return {
        "title": "Pairwise resolution leaves one arbitrary survivor in an equal three-head contact",
        "classification": "proven index dependence; underspecified multi-head gameplay semantics, not live/SIMD drift",
        "written_contract_status": (
            "Conforms to docs/simd_env_spec.md section 3.3, which requires consuming the "
            "ordered pair list and skipping pairs whose participant was already marked dead. "
            "The contract does not define an atomic three-or-more-head outcome."
        ),
        "source_refs": [
            "docs/simd_env_spec.md:239-240",
            "docs/simd_env_spec.md:270-298",
            "src/game/game_logic.py:148-167",
            "src/game/game_state.py:629-670",
            "src/simd_env/batch_sim.py:878-942",
        ],
        "setup": "Three equal length-1 snakes occupy one head cell; all six list permutations.",
        "runs": runs,
        "checks": checks,
        "observed": all(checks.values()),
    }


def probe_boost_self_tunnel() -> Dict[str, object]:
    snake = ScriptedSnake(
        0,
        (255, 0, 0),
        (20, 20),
        SEGMENT,
        WIDTH,
        HEIGHT,
        kind="random_safe",
        seed=1,
    )
    snake.segments = [(20, 20), (20, 30), (10, 30), (10, 20), (10, 10)]
    snake.length = 5
    snake.direction = (-1, 0)
    snake.is_boosting = True
    initial = list(snake.segments)

    normal_fatal, boost_fatal = simulate_relative_action_fatality(snake, [snake])

    # Explicitly materialize the body after only the first substep. The destination
    # is still present at index 4 then, before the second substep pops it.
    first_head = (initial[0][0] - SEGMENT, initial[0][1])
    post_first = [first_head] + initial
    if len(post_first) > snake.length:
        post_first.pop()
    first_substep_hits_body = first_head in post_first[3:]

    snake.move()
    reported_self_collision = GameLogic.check_self_collision(snake)
    checks = {
        "first_substep_enters_body_present_at_that_substep": first_substep_hits_body,
        "final_body_collision_check_reports_safe": not reported_self_collision,
        "action_mask_simulation_conservatively_marks_straight_boost_fatal": bool(boost_fatal[1]),
        "normal_straight_is_also_marked_fatal": bool(normal_fatal[1]),
    }
    return {
        "title": "Two-step boost can pass through a body cell removed by the second tail pop",
        "classification": (
            "contract-consistent execution with a conservative mask/execution mismatch; "
            "whether per-substep self collision is desired is a gameplay decision"
        ),
        "written_contract_status": (
            "Conforms literally to docs/simd_env_spec.md sections 2 and 3.1: movement "
            "finishes both insert/pop substeps, then all traversed heads are tested against "
            "the final segments[3:]. The action mask is outside the recommended pure-dynamics "
            "parity surface and correctly warns against this forced action via its normal-step check."
        ),
        "source_refs": [
            "docs/simd_env_spec.md:183-201",
            "docs/simd_env_spec.md:242-254",
            "docs/simd_env_spec.md:649-651",
            "src/game/snake.py:143-164",
            "src/game/game_logic.py:245-250",
            "src/game/ai_snake.py:56-96",
        ],
        "initial_segments": initial,
        "post_first_substep_segments": post_first,
        "traversed_heads": list(snake.last_move_positions),
        "final_segments": list(snake.segments),
        "reported_self_collision": reported_self_collision,
        "normal_fatal_by_relative_action": normal_fatal,
        "boost_fatal_by_relative_action": boost_fatal,
        "checks": checks,
        "observed": all(checks.values()),
    }


def parity_core(snapshot: Dict[str, object]) -> Dict[str, object]:
    return {key: snapshot[key] for key in ("heads", "bodies", "alive", "lengths", "food")}


def probe_consecutive_reset() -> Dict[str, object]:
    seed = 314159
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=4,
        game_width=300,
        game_height=200,
        segment_size=10,
        wall_thickness=10,
        initial_food=4,
        max_food=6,
        mechanics_version=2,
        max_capacity=32,
        frame_rate=1,
    )
    _install_v2_config(cfg)
    random.seed(seed)
    reference = PyRefGame(cfg)
    batch = BatchSim(cfg, seeds=[seed], train_mode=True)

    initial_ref = parity_core(reference.snapshot())
    initial_batch = parity_core(_batch_snapshot(batch))
    initial_equal = initial_ref == initial_batch

    reference._reset_world()
    batch.reset()
    second_ref = parity_core(reference.snapshot())
    second_batch = parity_core(_batch_snapshot(batch))
    second_equal = second_ref == second_batch
    differing_fields = [key for key in second_ref if second_ref[key] != second_batch[key]]
    checks = {
        "initial_post_construction_worlds_match": initial_equal,
        "second_reset_worlds_diverge": not second_equal,
        "second_reset_head_positions_diverge": second_ref["heads"] != second_batch["heads"],
    }
    return {
        "title": "BatchSim replays constructor-only discarded-food RNG draws on every reset",
        "classification": "written live/SIMD multi-episode parity violation",
        "written_contract_status": (
            "Violates the multi-episode live contract in docs/simd_env_spec.md section 5.4. "
            "FoodManager construction consumes the discarded-food draws once; later live resets "
            "only reposition snakes and reset food, while BatchSim.reset repeats those draws."
        ),
        "source_refs": [
            "docs/simd_env_spec.md:485-496",
            "src/game/game_state.py:117-155",
            "src/simd_env/batch_sim.py:247-287",
            "src/simd_env/parity.py:947-1000",
        ],
        "seed": seed,
        "initial_equal": initial_equal,
        "second_reset_equal": second_equal,
        "second_reset_differing_fields": differing_fields,
        "initial": {"reference": initial_ref, "batch": initial_batch},
        "second_reset": {"reference": second_ref, "batch": second_batch},
        "checks": checks,
        "observed": all(checks.values()),
    }


def source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    install_mechanics_v2()
    cases = {
        "contested_food_slot_order": probe_contested_food(),
        "three_way_head_on": probe_three_way_head_on(),
        "boost_self_tunnel": probe_boost_self_tunnel(),
        "consecutive_reset": probe_consecutive_reset(),
    }
    receipt = {
        "schema_version": 1,
        "probe": "dynamics_contract_edge_cases",
        "source_revision": source_revision(),
        "resource_contract": "CPU only; one process; one thread; no training",
        "all_expected_observations_reproduced": all(case["observed"] for case in cases.values()),
        "cases": cases,
    }
    OUTPUT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["all_expected_observations_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
