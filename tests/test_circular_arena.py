"""Tests for circular arena mode.

Tests cover:
- Circular wall collision detection
- Random position generation within circle
- Boundary feature computation (4 features at indices 40-43)
- Food spawning within circle
- Per-action danger with circular walls
- Danger map with circular boundary
- Backward compatibility (rectangular unchanged)
"""

import math

import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import (
    AppConfig,
    GameConfig,
    GameSettings,
    StateIndices,
    get_config,
    initialize_config,
)
from src.game.food_manager import FoodManager
from src.game.game_logic import GameLogic
from src.game.snake import Snake

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset global config to defaults after each test."""
    yield
    initialize_config(AppConfig.from_defaults())


def _init_circular(radius=200, cx=300, cy=300, width=600, height=600):
    """Initialize config with circular arena."""
    config = AppConfig(
        game=GameSettings(
            width=width,
            height=height,
            arena_type="circular",
            arena_radius=radius,
            arena_center_x=cx,
            arena_center_y=cy,
            num_snakes=1,
        )
    )
    initialize_config(config)
    return config


def _init_rectangular():
    """Initialize config with rectangular arena (default)."""
    config = AppConfig(
        game=GameSettings(
            width=600,
            height=600,
            arena_type="rectangular",
            num_snakes=1,
        )
    )
    initialize_config(config)
    return config


def _make_snake(pos, direction=(1, 0), segments=None):
    """Create a minimal Snake at *pos* with optional extra segments."""
    DeviceManager.override_device(torch.device("cpu"))
    s = Snake(
        id=0,
        color=(255, 0, 0),
        start_pos=pos,
        segment_size=10,
        game_width=get_config().game.width,
        game_height=get_config().game.height,
    )
    s.direction = direction
    if segments:
        s.segments = list(segments)
        s.length = len(s.segments)
    return s


# ===========================================================================
# 1. Circular wall collision
# ===========================================================================


class TestCircularWallCollision:
    """Tests for check_wall_collision with circular arena."""

    def test_inside_circle_no_collision(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300, 300))  # at center
        assert not GameLogic.check_wall_collision(snake)

    def test_inside_near_edge_no_collision(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300 + 199, 300))  # just inside
        assert not GameLogic.check_wall_collision(snake)

    def test_outside_circle_collision(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300 + 201, 300))  # just outside
        assert GameLogic.check_wall_collision(snake)

    def test_exactly_on_edge_no_collision(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((500, 300))  # 300+200 = on boundary
        assert not GameLogic.check_wall_collision(snake)

    def test_diagonal_outside(self):
        _init_circular(radius=200, cx=300, cy=300)
        # Distance = sqrt(150^2 + 150^2) ≈ 212 > 200
        snake = _make_snake((300 + 150, 300 + 150))
        assert GameLogic.check_wall_collision(snake)

    def test_diagonal_inside(self):
        _init_circular(radius=200, cx=300, cy=300)
        # Distance = sqrt(100^2 + 100^2) ≈ 141 < 200
        snake = _make_snake((300 + 100, 300 + 100))
        assert not GameLogic.check_wall_collision(snake)


# ===========================================================================
# 2. Rectangular unchanged (backward compatibility)
# ===========================================================================


class TestRectangularBackwardCompat:
    """Verify rectangular mode is unchanged."""

    def test_inside_rect_no_collision(self):
        _init_rectangular()
        snake = _make_snake((300, 300))
        assert not GameLogic.check_wall_collision(snake)

    def test_outside_rect_collision(self):
        _init_rectangular()
        snake = _make_snake((-1, 300))
        assert GameLogic.check_wall_collision(snake)

    def test_rect_boundary_features_unchanged(self):
        """Verify rectangular boundary features are still L/R/T/B distances."""
        _init_rectangular()
        snake = _make_snake((300, 300))
        state = snake.get_state([], [(100, 100)])
        w = get_config().game.width
        h = get_config().game.height
        assert abs(float(state[StateIndices.BOUNDARY_LEFT]) - 300 / w) < 0.01
        assert abs(float(state[StateIndices.BOUNDARY_RIGHT]) - (w - 300) / w) < 0.01
        assert abs(float(state[StateIndices.BOUNDARY_TOP]) - 300 / h) < 0.01
        assert abs(float(state[StateIndices.BOUNDARY_BOTTOM]) - (h - 300) / h) < 0.01


# ===========================================================================
# 3. Random position generation within circle
# ===========================================================================


class TestCircularPositionGeneration:
    """Tests for position generation within circular arena."""

    def test_find_empty_position_within_circle(self):
        _init_circular(radius=200, cx=300, cy=300)
        for _ in range(50):
            pos = GameLogic.find_empty_position(600, 600, [])
            assert pos is not None
            dx = pos[0] - 300
            dy = pos[1] - 300
            dist = math.sqrt(dx**2 + dy**2)
            # Should be within radius - wall_thickness margin
            assert dist <= 200, f"Position {pos} outside circle (dist={dist:.1f})"

    def test_game_state_random_position_within_circle(self):
        """GameState.get_random_position should return points inside circle."""
        _init_circular(radius=200, cx=300, cy=300)
        from src.game.game_state import GameState

        gs = GameState.__new__(GameState)
        # Manually set up just enough for get_random_position
        for _ in range(50):
            pos = gs.get_random_position()
            dx = pos[0] - 300
            dy = pos[1] - 300
            dist = math.sqrt(dx**2 + dy**2)
            assert dist <= 200

    def test_effective_board_size_scales_circular_position_generation(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        from src.game.game_state import GameState

        gs = GameState.__new__(GameState)
        gs._game_width = 400
        gs._game_height = 300

        for _ in range(50):
            pos = gs.get_random_position()
            dx = pos[0] - 200
            dy = pos[1] - 150
            dist = math.sqrt(dx**2 + dy**2)
            assert dist <= 100

    def test_find_empty_position_uses_effective_circular_geometry(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        for _ in range(50):
            pos = GameLogic.find_empty_position(400, 300, [])
            assert pos is not None
            dx = pos[0] - 200
            dy = pos[1] - 150
            dist = math.sqrt(dx**2 + dy**2)
            assert dist <= 100


# ===========================================================================
# 4. Boundary feature computation (circular)
# ===========================================================================


class TestCircularBoundaryFeatures:
    """Tests for state vector boundary features in circular mode."""

    def test_at_center(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300, 300))
        state = snake.get_state([], [(100, 100)])
        # At center, all cardinal distances to the circular boundary are half
        # the diameter.
        assert float(state[StateIndices.BOUNDARY_LEFT]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_RIGHT]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_TOP]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_BOTTOM]) == pytest.approx(0.5, abs=0.01)

    def test_near_edge(self):
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((490, 300))  # 190 px to the right of center
        state = snake.get_state([], [(100, 100)])
        # Right edge is 10 px away, normalized by diameter 400.
        assert float(state[StateIndices.BOUNDARY_LEFT]) == pytest.approx(0.975, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_RIGHT]) == pytest.approx(0.025, abs=0.01)
        # Vertical cardinal distance is still positive because the point is
        # inside the circle, but much shorter near the right edge.
        assert float(state[StateIndices.BOUNDARY_TOP]) == pytest.approx(0.156, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_BOTTOM]) == pytest.approx(0.156, abs=0.01)

    def test_state_size_unchanged(self):
        """State vector must still be INPUT_SIZE (58) in circular mode."""
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300, 300))
        state = snake.get_state([], [(100, 100)])
        assert state.shape == (GameConfig.INPUT_SIZE,)

    def test_scaled_snake_dimensions_scale_circular_boundary_features(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        snake = _make_snake((200, 150))
        snake.game_width = 400
        snake.game_height = 300

        state = snake.get_state([], [(250, 150)])

        assert float(state[StateIndices.BOUNDARY_LEFT]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_RIGHT]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_TOP]) == pytest.approx(0.5, abs=0.01)
        assert float(state[StateIndices.BOUNDARY_BOTTOM]) == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# 4b. Circular reward interpretation
# ===========================================================================


class TestCircularRewardSignals:
    """Tests for reward shaping that consumes circular boundary features."""

    def test_wall_reward_uses_nearest_circular_boundary_distance(self):
        """Circular center-ish states should not trigger wall penalties."""
        _init_circular(radius=500, cx=600, cy=600)
        config = AppConfig(
            game=GameSettings(
                width=1200,
                height=1200,
                arena_type="circular",
                arena_radius=500,
                arena_center_x=600,
                arena_center_y=600,
                num_snakes=1,
            )
        )
        initialize_config(config)
        snake = _make_snake((550, 600), direction=(1, 0))
        snake.game_width = 1200
        snake.game_height = 1200
        state = snake.get_state([], [(800, 600)])

        reward = snake.calculate_reward(
            ate_food=False,
            collided=False,
            old_state=state,
            new_state=state,
            other_snakes=[],
            food=[(800, 600)],
        )

        assert reward == pytest.approx(GameConfig.REWARD_SURVIVAL)


# ===========================================================================
# 5. Food spawning within circle
# ===========================================================================


class TestCircularFoodSpawning:
    """Tests for food spawning in circular arena."""

    def test_food_manager_positions_in_circle(self):
        _init_circular(radius=200, cx=300, cy=300)
        fm = FoodManager(
            game_width=600,
            game_height=600,
            max_food=50,
            initial_food=50,
            segment_size=10,
            wall_thickness=10,
        )
        for fx, fy in fm.food:
            dx = fx - 300
            dy = fy - 300
            dist = math.sqrt(dx**2 + dy**2)
            assert dist <= 200, f"Food at ({fx},{fy}) outside circle (dist={dist:.1f})"

    def test_food_manager_positions_use_effective_circular_geometry(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        fm = FoodManager(
            game_width=400,
            game_height=300,
            max_food=50,
            initial_food=50,
            segment_size=10,
            wall_thickness=10,
        )
        for fx, fy in fm.food:
            dx = fx - 200
            dy = fy - 150
            dist = math.sqrt(dx**2 + dy**2)
            assert dist <= 100, f"Food at ({fx},{fy}) outside scaled circle (dist={dist:.1f})"


# ===========================================================================
# 6. Per-action danger with circular walls
# ===========================================================================


class TestCircularPerActionDanger:
    """Tests for per-action danger signals in circular mode."""

    def test_danger_at_edge_heading_out(self):
        """Snake heading outward at edge should see danger=1 for straight."""
        _init_circular(radius=200, cx=300, cy=300)
        # Place snake near right edge heading right
        snake = _make_snake((495, 300), direction=(1, 0))
        dangers = snake._get_per_action_danger([])
        # Straight (index 1) should be danger=1.0 (next step goes to 505 > 200 from center)
        assert dangers[1] == 1.0

    def test_no_danger_at_center(self):
        """Snake at center should have low danger for all actions."""
        _init_circular(radius=200, cx=300, cy=300)
        snake = _make_snake((300, 300), direction=(1, 0))
        dangers = snake._get_per_action_danger([])
        # All danger values should be well below 1.0
        for d in dangers:
            assert d < 0.5

    def test_scaled_snake_dimensions_drive_circular_action_danger(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        snake = _make_snake((295, 150), direction=(1, 0))
        snake.game_width = 400
        snake.game_height = 300

        dangers = snake._get_per_action_danger([])

        assert dangers[1] == 1.0


class TestScaledCircularWallCollision:
    """Tests for circular wall collision with effective board dimensions."""

    def test_scaled_snake_dimensions_drive_wall_collision(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        snake = _make_snake((301, 150))
        snake.game_width = 400
        snake.game_height = 300

        assert GameLogic.check_wall_collision(snake)

    def test_scaled_snake_dimensions_keep_scaled_center_safe(self):
        _init_circular(radius=200, cx=400, cy=300, width=800, height=600)
        snake = _make_snake((200, 150))
        snake.game_width = 400
        snake.game_height = 300

        assert not GameLogic.check_wall_collision(snake)


# ===========================================================================
# 7. Danger map with circular boundary
# ===========================================================================


class TestCircularDangerMap:
    """Tests for danger map with circular boundary."""

    def test_danger_near_circular_edge(self):
        """Near the edge, danger map should show wall danger in relevant sectors."""
        _init_circular(radius=200, cx=300, cy=300)
        # Place snake near right edge
        snake = _make_snake((490, 300), direction=(1, 0))
        danger_map = snake._get_danger_map([])
        # There should be nonzero danger in at least one sector (the right side)
        assert max(danger_map) > 0

    def test_no_wall_danger_at_center(self):
        """At center of large circle, no wall danger should be present."""
        # Need radius > danger_max_distance * segment_size (30*10=300) + margin
        _init_circular(radius=500, cx=600, cy=600)
        config = AppConfig(
            game=GameSettings(
                width=1200,
                height=1200,
                arena_type="circular",
                arena_radius=500,
                arena_center_x=600,
                arena_center_y=600,
                num_snakes=1,
            )
        )
        initialize_config(config)
        snake = _make_snake((600, 600), direction=(1, 0))
        snake.game_width = 1200
        snake.game_height = 1200
        danger_map = snake._get_danger_map([])
        # All sectors should be 0 (no obstacles, walls far away)
        assert all(d == 0.0 for d in danger_map)


# ===========================================================================
# 8. Config properties
# ===========================================================================


class TestArenaConfigProperties:
    """Tests for GameConfig property accessors."""

    def test_default_arena_type(self):
        initialize_config(AppConfig.from_defaults())
        assert GameConfig.ARENA_TYPE == "rectangular"

    def test_circular_arena_config(self):
        _init_circular(radius=250, cx=400, cy=350)
        assert GameConfig.ARENA_TYPE == "circular"
        assert GameConfig.ARENA_RADIUS == 250
        assert GameConfig.ARENA_CENTER_X == 400
        assert GameConfig.ARENA_CENTER_Y == 350

    def test_yaml_roundtrip(self, tmp_path):
        """Config with arena settings should survive YAML save/load."""
        _init_circular(radius=300, cx=500, cy=400)
        config = get_config()
        path = str(tmp_path / "test_config.yaml")
        config.save_yaml(path)
        loaded = AppConfig.from_yaml(path)
        assert loaded.game.arena_type == "circular"
        assert loaded.game.arena_radius == 300
        assert loaded.game.arena_center_x == 500
        assert loaded.game.arena_center_y == 400
