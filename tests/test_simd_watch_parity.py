"""Watch regressions for actual non-training GameState food replacement."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import _install_v2_config


def _cfg(num_snakes: int, frame_rate: int) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=1,
        num_snakes=num_snakes,
        game_width=200,
        game_height=160,
        segment_size=10,
        initial_food=0,
        max_food=2,
        mechanics_version=2,
        frame_rate=frame_rate,
        max_capacity=16,
    )


def _set_batch_snake(sim: BatchSim, slot: int, head: tuple[int, int]) -> None:
    sim.bodies[0, slot, 0] = (head[0] // 10, head[1] // 10)
    sim.head_ptr[0, slot] = 0
    sim.seg_count[0, slot] = 1
    sim.length[0, slot] = 1
    sim.direction[0, slot] = 1  # right


def _set_live_snake(snake, head: tuple[int, int]) -> None:
    snake.segments = [head]
    snake.length = 1
    snake.direction = (1, 0)
    snake.is_alive = True
    snake.update = lambda _others, _food, snake=snake, **_kwargs: snake.move()


def _food_cells(game) -> list[tuple[int, int]]:
    return [(x // 10, y // 10) for x, y in game.food_manager.food]


@pytest.mark.parametrize("frame_rate", [1, 3, 100])
def test_actual_game_state_corpse_at_ambient_cap_matches_batch_without_rng_draw(frame_rate):
    """Eating cap-exempt corpse food does not spawn replacement ambient food."""
    from src.game.game_state import GameState

    cfg = _cfg(1, frame_rate)
    saved = get_config()
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=1)
        sim = BatchSim(cfg, seeds=[19], train_mode=False)
        _set_live_snake(game.snakes[0], (20, 20))
        _set_batch_snake(sim, 0, (20, 20))
        # Two ambient pellets already meet the cap; the third is corpse-class.
        game.food_manager.food = [(30, 20), (50, 50), (60, 50)]
        game.food_manager._corpse_positions = {(30, 20)}
        sim.food_cells[0] = [(3, 2), (5, 5), (6, 5)]
        sim.food_set[0] = set(sim.food_cells[0])
        sim.corpse_cells[0] = {(3, 2)}
        random.seed(917)
        sim._rngs[0]._rng.seed(917)
        before_live, before_batch = random.getstate(), sim._rngs[0]._rng.getstate()

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.array([[1]], dtype=np.int64))

        assert _food_cells(game) == sim.get_food(0) == [(5, 5), (6, 5)]
        assert random.getstate() == before_live
        assert sim._rngs[0]._rng.getstate() == before_batch
    finally:
        initialize_config(saved)


def test_actual_game_state_multi_eater_preserves_ordered_food_and_rng():
    """An ambient eater draws once before a corpse eater at cap draws zero times."""
    from src.game.game_state import GameState

    cfg = _cfg(2, 1)
    saved = get_config()
    _install_v2_config(cfg)
    try:
        game = GameState(headless=True, num_snakes=2)
        sim = BatchSim(cfg, seeds=[37], train_mode=False)
        for slot, head in enumerate(((20, 20), (20, 40))):
            _set_live_snake(game.snakes[slot], head)
            _set_batch_snake(sim, slot, head)
        # Snake 0 eats ambient then maintain_count replaces it. Snake 1 eats a
        # corpse while ambient is again at cap, so it must not consume RNG.
        game.food_manager.food = [(30, 20), (30, 40), (60, 50)]
        game.food_manager._corpse_positions = {(30, 40)}
        sim.food_cells[0] = [(3, 2), (3, 4), (6, 5)]
        sim.food_set[0] = set(sim.food_cells[0])
        sim.corpse_cells[0] = {(3, 4)}
        random.seed(2718)
        sim._rngs[0]._rng.seed(2718)

        game.update(train_mode=False, learn=False, allow_respawn=False)
        sim.step(np.array([[1, 1]], dtype=np.int64))

        assert _food_cells(game) == sim.get_food(0)
        assert tuple(game.food_manager._corpse_positions) == tuple(sim.get_corpse_food(0))
        assert random.getstate() == sim._rngs[0]._rng.getstate()
    finally:
        initialize_config(saved)
