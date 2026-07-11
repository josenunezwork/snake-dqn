"""Tests for reward v2 wiring in SnakeRewardMixin (blueprint §3.2) and v1 bit-identity."""

from dataclasses import replace

import pytest

from src.core.game_config import AppConfig, GameConfig, initialize_config
from src.game.snake_reward import REWARD_TERM_KEYS
from tests.conftest import make_test_snake

pytestmark = pytest.mark.usefixtures("setup_config")

GAMMA = 0.997

LEGACY_ONLY_KEYS = (
    "food",
    "food_shaping",
    "wall",
    "danger",
    "starvation",
    "survival",
    "boost",
    "clamp_delta",
)


def _init_rewards(version: int, gamma: float = GAMMA) -> None:
    """Initialize the global config with a rewards version and apex gamma."""
    config = AppConfig.from_defaults()
    config = replace(
        config,
        rewards=replace(config.rewards, version=version),
        apex=replace(config.apex, gamma=gamma),
    )
    initialize_config(config)


def _snake_at_length(snake_id: int, length: int):
    snake = make_test_snake(snake_id, (100, 100))
    snake.length = length
    snake._reward_prev_length = length
    return snake


class TestRewardV2:
    """rewards.version == 2 delegates to the pure event-based reward."""

    def test_eat_one_food_is_pure_potential(self):
        _init_rewards(2)
        snake = _snake_at_length(0, 10)
        snake.length = 11  # grew this frame
        reward = snake.calculate_reward(True, False, None, None, [snake], [], frame_kills={})
        assert reward == GAMMA * (11 / 10) - (10 / 10)
        breakdown = snake.last_reward_breakdown
        assert tuple(breakdown.keys()) == REWARD_TERM_KEYS
        assert breakdown["potential"] == reward
        for key in LEGACY_ONLY_KEYS:
            assert breakdown[key] == 0.0
        # Counters advance like v1's non-death path.
        assert snake.frames_since_food == 0
        assert snake._reward_prev_length == 11

    def test_death_at_length_40_forfeits_potential_plus_death_event(self):
        _init_rewards(2)
        snake = _snake_at_length(0, 40)
        reward = snake.calculate_reward(False, True, None, None, [snake], [], frame_kills={})
        assert reward == -7.0  # -Phi(40) - 3.0
        breakdown = snake.last_reward_breakdown
        assert breakdown["potential"] == -4.0
        assert breakdown["death"] == -3.0
        assert breakdown["kill"] == 0.0
        # Death leaves the baseline alone (respawn resets it), mirroring v1.
        assert snake._reward_prev_length == 40

    def test_same_frame_killer_death_still_pays_kill(self):
        _init_rewards(2)
        killer = _snake_at_length(0, 30)
        victim = _snake_at_length(1, 20)
        reward = killer.calculate_reward(
            False, True, None, None, [killer, victim], [], frame_kills={0: [1]}
        )
        breakdown = killer.last_reward_breakdown
        assert breakdown["kill"] == 6.0  # 0.3 x 20, NOT dropped by the death
        assert breakdown["death"] == -3.0
        assert breakdown["potential"] == -3.0
        assert reward == 0.0

    def test_kill_reward_is_unclamped_past_reward_max(self):
        _init_rewards(2)
        killer = _snake_at_length(0, 10)
        victim = _snake_at_length(1, 100)
        reward = killer.calculate_reward(
            False, False, None, None, [killer, victim], [], frame_kills={0: [1]}
        )
        assert killer.last_reward_breakdown["kill"] == 30.0
        assert reward > GameConfig.REWARD_MAX

    def test_no_legacy_shaping_terms_fire(self):
        _init_rewards(2)
        snake = _snake_at_length(0, 10)
        snake.frames_since_food = 10_000  # would trigger a big v1 starvation penalty
        reward = snake.calculate_reward(False, False, None, None, [snake], [], frame_kills={})
        breakdown = snake.last_reward_breakdown
        for key in LEGACY_ONLY_KEYS:
            assert breakdown[key] == 0.0
        # Plain move at constant length: pure potential decay, nothing else.
        assert reward == breakdown["potential"] == GAMMA * 1.0 - 1.0
        assert snake.frames_since_food == 10_001

    def test_breakdown_sums_exactly_in_insertion_order(self):
        _init_rewards(2)
        killer = _snake_at_length(0, 17)
        killer.length = 16  # lost a segment this frame
        victim = _snake_at_length(1, 13)
        reward = killer.calculate_reward(
            False, True, None, None, [killer, victim], [], frame_kills={0: [1]}
        )
        running = 0.0
        for value in killer.last_reward_breakdown.values():
            running += value
        assert running == reward


class TestRewardV1BitIdentity:
    """rewards.version == 1 must keep legacy behavior, including the death early return."""

    def test_death_returns_flat_death_reward(self):
        _init_rewards(1)
        snake = _snake_at_length(0, 40)
        reward = snake.calculate_reward(False, True, None, None, [snake], [], frame_kills={})
        assert reward == GameConfig.REWARD_DEATH
        assert snake.last_reward_breakdown["death"] == GameConfig.REWARD_DEATH

    def test_death_still_drops_same_frame_kill_credit(self):
        # The v1 early return drops kill credit for same-frame-dead killers;
        # that (known-bad) behavior is part of the v1 bit-identity contract.
        _init_rewards(1)
        killer = _snake_at_length(0, 30)
        victim = _snake_at_length(1, 20)
        reward = killer.calculate_reward(
            False, True, None, None, [killer, victim], [], frame_kills={0: [1]}
        )
        assert reward == GameConfig.REWARD_DEATH
        assert killer.last_reward_breakdown["kill"] == 0.0

    def test_eating_returns_food_base(self):
        _init_rewards(1)
        killer = _snake_at_length(0, 10)
        other = _snake_at_length(1, 10)
        reward = killer.calculate_reward(True, False, None, None, [killer, other], [], {})
        assert reward == GameConfig.REWARD_FOOD_BASE
        assert killer.last_reward_breakdown["food"] == GameConfig.REWARD_FOOD_BASE

    def test_plain_step_returns_survival_only(self):
        _init_rewards(1)
        snake = _snake_at_length(0, 10)
        other = _snake_at_length(1, 10)
        reward = snake.calculate_reward(False, False, None, None, [snake, other], [], {})
        assert reward == GameConfig.REWARD_SURVIVAL

    def test_v1_breakdown_keeps_potential_key_at_zero(self):
        _init_rewards(1)
        snake = _snake_at_length(0, 10)
        snake.calculate_reward(False, False, None, None, [snake], [], {})
        breakdown = snake.last_reward_breakdown
        assert tuple(breakdown.keys()) == REWARD_TERM_KEYS
        assert REWARD_TERM_KEYS[0] == "potential"
        assert breakdown["potential"] == 0.0

    def test_default_config_is_version_1(self):
        initialize_config()
        assert GameConfig.REWARD_VERSION == 1
