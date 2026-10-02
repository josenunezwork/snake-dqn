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

    from src.core import game_config
    from src.core.device_manager import DeviceManager
    from src.core.game_config import initialize_config

    # Capture and restore the global config singleton so a test that initializes a
    # non-default config here cannot leak it into a later test that reads the global
    # without going through this fixture.
    prev_config = game_config._current_config
    initialize_config()
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()
    game_config._current_config = prev_config


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


@pytest.fixture(autouse=True)
def _restore_torch_intraop_threads():
    """Undo torch.set_num_threads calls made by research harness helpers under test.

    Several research runners pin torch to 2 intra-op threads when configured; without this,
    that process-global setting leaks into later tests (e.g. the web app's 1-thread pin check).
    """
    import torch

    before = torch.get_num_threads()
    yield
    if torch.get_num_threads() != before:
        torch.set_num_threads(before)
