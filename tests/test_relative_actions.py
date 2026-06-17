"""Tests for relative action space (§2.1)."""
import pytest
import torch

from src.game.game_logic import GameLogic, TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT
from src.game.snake import Snake
from src.core.game_config import GameConfig

pytestmark = pytest.mark.usefixtures("setup_config")


class TestRelativeToAbsoluteDirection:
    """Test GameLogic.relative_to_absolute_direction() — all 12 combos."""

    # Cardinal ordering: up=(0,-1), right=(1,0), down=(0,1), left=(-1,0)
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    # ── From UP ──────────────────────────────────────────────────────────
    def test_from_up_turn_left(self):
        result = GameLogic.relative_to_absolute_direction(self.UP, TURN_LEFT)
        assert result == self.LEFT

    def test_from_up_go_straight(self):
        result = GameLogic.relative_to_absolute_direction(self.UP, TURN_STRAIGHT)
        assert result == self.UP

    def test_from_up_turn_right(self):
        result = GameLogic.relative_to_absolute_direction(self.UP, TURN_RIGHT)
        assert result == self.RIGHT

    # ── From RIGHT ───────────────────────────────────────────────────────
    def test_from_right_turn_left(self):
        result = GameLogic.relative_to_absolute_direction(self.RIGHT, TURN_LEFT)
        assert result == self.UP

    def test_from_right_go_straight(self):
        result = GameLogic.relative_to_absolute_direction(self.RIGHT, TURN_STRAIGHT)
        assert result == self.RIGHT

    def test_from_right_turn_right(self):
        result = GameLogic.relative_to_absolute_direction(self.RIGHT, TURN_RIGHT)
        assert result == self.DOWN

    # ── From DOWN ────────────────────────────────────────────────────────
    def test_from_down_turn_left(self):
        result = GameLogic.relative_to_absolute_direction(self.DOWN, TURN_LEFT)
        assert result == self.RIGHT

    def test_from_down_go_straight(self):
        result = GameLogic.relative_to_absolute_direction(self.DOWN, TURN_STRAIGHT)
        assert result == self.DOWN

    def test_from_down_turn_right(self):
        result = GameLogic.relative_to_absolute_direction(self.DOWN, TURN_RIGHT)
        assert result == self.LEFT

    # ── From LEFT ────────────────────────────────────────────────────────
    def test_from_left_turn_left(self):
        result = GameLogic.relative_to_absolute_direction(self.LEFT, TURN_LEFT)
        assert result == self.DOWN

    def test_from_left_go_straight(self):
        result = GameLogic.relative_to_absolute_direction(self.LEFT, TURN_STRAIGHT)
        assert result == self.LEFT

    def test_from_left_turn_right(self):
        result = GameLogic.relative_to_absolute_direction(self.LEFT, TURN_RIGHT)
        assert result == self.UP

    # ── Edge cases ───────────────────────────────────────────────────────
    def test_invalid_direction_fallback(self):
        """Invalid direction returns itself as fallback."""
        result = GameLogic.relative_to_absolute_direction((2, 3), TURN_LEFT)
        assert result == (2, 3)


class TestConstants:
    """Verify action constants match expected values."""

    def test_turn_left_value(self):
        assert TURN_LEFT == 0

    def test_turn_straight_value(self):
        assert TURN_STRAIGHT == 1

    def test_turn_right_value(self):
        assert TURN_RIGHT == 2

    def test_output_size_is_six(self):
        assert GameConfig.OUTPUT_SIZE == 6  # 3 dirs × 2 speed modes


class TestNo180Turns:
    """Relative actions inherently prevent 180-degree turns."""

    @pytest.mark.parametrize("current_dir", [
        (0, -1), (1, 0), (0, 1), (-1, 0)
    ])
    def test_no_reverse_for_any_action(self, current_dir):
        """No relative action can produce the reverse of current_dir."""
        reverse = (-current_dir[0], -current_dir[1])
        for action in [TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT]:
            result = GameLogic.relative_to_absolute_direction(current_dir, action)
            assert result != reverse, (
                f"Action {action} from {current_dir} produced reverse {reverse}"
            )


class TestSafeActionMasking:
    """Test _get_safe_actions with relative actions on Snake base class.

    Since _get_safe_actions is on AISnake, we test the underlying logic
    by simulating what it does: convert each relative action to absolute
    direction and check bounds.
    """

    def _simulate_safe_actions(self, snake):
        """Reproduce AISnake._get_safe_actions logic on a base Snake."""
        head_x, head_y = snake.head
        safe = []
        for relative_action in range(3):
            abs_dir = GameLogic.relative_to_absolute_direction(
                snake.direction, relative_action
            )
            new_x = head_x + abs_dir[0] * snake.segment_size
            new_y = head_y + abs_dir[1] * snake.segment_size
            if 0 <= new_x < snake.game_width and 0 <= new_y < snake.game_height:
                safe.append(relative_action)
        return safe if safe else list(range(3))

    def test_open_area_all_safe(self):
        """All 3 actions safe in open area."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right
        safe = self._simulate_safe_actions(snake)
        assert safe == [0, 1, 2]

    def test_near_top_wall_facing_up(self):
        """Near top wall facing up: straight is unsafe."""
        snake = Snake(0, (255, 0, 0), (400, 5), 10, 800, 600)
        snake.direction = (0, -1)  # facing up
        safe = self._simulate_safe_actions(snake)
        # Straight (up) would go to y=-5 (unsafe)
        # Left (left) would go to x=390 (safe)
        # Right (right) would go to x=410 (safe)
        assert TURN_STRAIGHT not in safe
        assert TURN_LEFT in safe
        assert TURN_RIGHT in safe

    def test_corner_facing_wall(self):
        """In top-left corner facing up: left and straight unsafe."""
        snake = Snake(0, (255, 0, 0), (5, 5), 10, 800, 600)
        snake.direction = (0, -1)  # facing up
        safe = self._simulate_safe_actions(snake)
        # Straight (up) → y=-5 unsafe
        # Left (left) → x=-5 unsafe
        # Right (right) → x=15 safe
        assert safe == [TURN_RIGHT]

    def test_all_unsafe_returns_all(self):
        """Fallback: if all actions are unsafe, return all anyway."""
        # Place snake at (5, 5) facing up-left corner,
        # with game size so small all moves go out of bounds
        snake = Snake(0, (255, 0, 0), (0, 0), 10, 5, 5)
        snake.direction = (0, -1)  # facing up
        safe = self._simulate_safe_actions(snake)
        # All 3 actions lead out of bounds → fallback returns [0,1,2]
        assert safe == [0, 1, 2]


class TestStateDirectionEncoding:
    """Direction one-hot uses absolute 4D encoding despite 3-action output."""

    def test_direction_onehot_is_4d(self):
        """State vector direction slice is always 4D one-hot."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # right

        food = [(500, 300)]
        state = snake.get_state([], food)

        direction_slice = state[0:4]
        assert direction_slice.shape == (4,)
        # Right = index 1 in ACTIONS = [(0,-1), (1,0), (0,1), (-1,0)]
        assert float(direction_slice[1]) == 1.0
        assert float(direction_slice.sum()) == 1.0

    def test_state_size_matches_config(self):
        """State vector length matches GameConfig.INPUT_SIZE."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        food = [(500, 300)]
        state = snake.get_state([], food)
        assert state.shape == (GameConfig.INPUT_SIZE,)
