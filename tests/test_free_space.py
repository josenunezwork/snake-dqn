"""Tests for the free-space ("don't trap yourself") state features."""

import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import (
    AppConfig,
    GameConfig,
    GameSettings,
    NetworkSettings,
    initialize_config,
)
from src.game.snake import Snake


@pytest.fixture(autouse=True)
def _cpu_and_reset():
    DeviceManager.override_device(torch.device("cpu"))
    yield
    initialize_config(AppConfig.from_defaults())  # restore the 58-D default
    DeviceManager.reset_for_testing()


def _enable_free_space():
    cfg = AppConfig(
        network=NetworkSettings(input_size=61, use_free_space=True),
        game=GameSettings(width=800, height=600),
    )
    initialize_config(cfg)


def _snake(segments, direction=(1, 0), length=None, ss=10):
    s = Snake(0, (255, 0, 0), segments[0], ss, 800, 600)
    s.segments = list(segments)
    s.direction = direction
    s.length = length if length is not None else len(segments)
    s.is_alive = True
    return s


def test_disabled_by_default():
    initialize_config(AppConfig.from_defaults())
    assert GameConfig.INPUT_SIZE == 58
    assert GameConfig.USE_FREE_SPACE is False
    s = _snake([(100, 100), (90, 100)])
    assert s.get_state([], [(200, 200)]).shape[0] == 58


def test_enabled_state_is_61():
    _enable_free_space()
    assert GameConfig.INPUT_SIZE == 61
    assert GameConfig.USE_FREE_SPACE is True
    s = _snake([(400 - 10 * i, 300) for i in range(20)])
    assert s.get_state([], [(500, 300)]).shape[0] == 61


def test_open_field_all_high():
    _enable_free_space()
    s = _snake([(400 - 10 * i, 300) for i in range(30)])
    f = s._get_free_space_features([])
    assert all(v > 0.9 for v in f), f


def test_sealed_pocket_detected():
    _enable_free_space()
    # direction (1,0): relative-left turn -> up (400,290). Seal a 3-cell pocket there.
    s = _snake([(400 - 10 * i, 300) for i in range(30)])
    wall = []
    for cy in (290, 280, 270):
        wall += [(390, cy), (410, cy)]
    wall += [(400, 260)]
    enemy = Snake(2, (0, 0, 255), (390, 290), 10, 800, 600)
    enemy.segments = wall
    enemy.length = len(wall)
    enemy.is_alive = True
    f = s._get_free_space_features([enemy])
    assert f[0] < 0.3, f  # left (up) is a tiny pocket -> low
    assert f[1] > 0.9 and f[2] > 0.9, f  # straight/right open


def test_immediate_block_zero_and_aligned_with_danger():
    _enable_free_space()
    # direction (1,0): relative-right turn -> down (400,310). Block that cell.
    s = _snake([(400, 300), (390, 300), (380, 300), (370, 300), (360, 300)])
    blk = Snake(3, (0, 0, 255), (400, 310), 10, 800, 600)
    blk.segments = [(400, 310)]
    blk.length = 1
    blk.is_alive = True
    f = s._get_free_space_features([blk])
    danger = s._get_per_action_danger([blk])
    assert f[2] == 0.0, f  # right move has no reachable space
    assert danger[2] == 1.0  # and per-action danger agrees on the same index


def test_free_space_near_right_wall_not_collapsed():
    """Regression: round-based cell mapping pushed a head in the rightmost
    half-cell out of bounds, collapsing all free-space to 0 along that edge."""
    _enable_free_space()
    s = Snake(0, (255, 0, 0), (795, 300), 10, 800, 600)  # x in the right half-cell
    s.direction = (0, -1)
    s.segments = [(795, 300 + 10 * i) for i in range(5)]
    s.length = 5
    s.is_alive = True
    f = s._get_free_space_features([])
    assert max(f) > 0.0, f  # open board near the wall must report reachable space
