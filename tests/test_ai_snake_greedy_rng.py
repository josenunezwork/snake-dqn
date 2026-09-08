"""Actual greedy actions must leave the live world's Python RNG untouched."""

import random

import pytest

from src.core.game_config import GameConfig
from src.game.ai_snake import AISnake
from src.training.apex_policy import ApexPolicy


@pytest.mark.usefixtures("setup_config")
@pytest.mark.parametrize("epsilon", [0.0, 0.25, 1.0])
def test_action_preserves_expected_world_rng_consumption(epsilon):
    policy = ApexPolicy(
        GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE, training=False
    )
    snake = AISnake(0, (255, 0, 0), (100, 100), 10, 800, 600, policy)
    snake.actor_epsilon = epsilon
    allowed = snake._get_safe_actions([snake])
    assert allowed == [0, 1, 2]
    seed = 2  # First draw is > 0.25: one greedy coin flip or full exploration.
    expected = random.Random(seed)
    expected_action = None
    if epsilon > 0 and expected.random() < epsilon:
        expected_action = expected.choice(allowed)
    previous = random.getstate()
    try:
        random.seed(seed)
        snake.update([snake], [])
        assert random.getstate() == expected.getstate()
        if expected_action is not None:
            assert snake._pre_collision_action == expected_action
    finally:
        random.setstate(previous)
