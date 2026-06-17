"""Pytest configuration and shared fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def setup_config():
    """Ensure config and device are initialized for every test.

    Shared across test_curriculum, test_enemy_features, test_kill_attribution,
    test_per_action_danger, test_relative_actions, test_speed_boost.
    """
    import torch

    from src.core.device_manager import DeviceManager
    from src.core.game_config import initialize_config

    initialize_config()
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()


def make_test_snake(sid, pos, direction=(1, 0), segments=None):
    """Helper to create a snake at a known position.

    Shared across test_enemy_features, test_kill_attribution.
    Use via: ``from tests.conftest import make_test_snake``.
    """
    from src.game.snake import Snake

    snake = Snake(sid, (255, 0, 0), pos, 10, 800, 600)
    snake.direction = direction
    if segments is not None:
        snake.segments = list(segments)
        snake.length = len(segments)
    return snake
