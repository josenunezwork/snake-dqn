"""Tests for per-action danger signals (§2.2)."""

import pytest

from src.core.game_config import GameConfig, StateIndices
from src.game.game_logic import TURN_LEFT, TURN_STRAIGHT
from src.game.snake import Snake
from src.training.action_mask import valid_action_mask_from_states

pytestmark = pytest.mark.usefixtures("setup_config")


class TestPerActionDangerOutput:
    """Test _get_per_action_danger() output format and basic values."""

    def test_returns_three_values(self):
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right
        dangers = snake._get_per_action_danger([])
        assert len(dangers) == 3

    def test_open_space_all_dangers_low(self):
        """Snake in open space: all dangers should be low (< 0.5)."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right, center of board
        dangers = snake._get_per_action_danger([])
        for i, d in enumerate(dangers):
            assert d < 0.5, f"Action {i} danger {d} too high in open space"

    def test_values_in_zero_one_range(self):
        """All danger values should be in [0, 1] range."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        dangers = snake._get_per_action_danger([])
        for d in dangers:
            assert 0.0 <= d <= 1.0


class TestWallDanger:
    """Test danger values when near walls."""

    def test_facing_wall_straight_danger_is_one(self):
        """Snake facing wall: straight danger = 1.0."""
        # Place snake 5px from right wall, facing right
        snake = Snake(0, (255, 0, 0), (795, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right
        dangers = snake._get_per_action_danger([])
        # straight (index 1) → moving right → new_x = 805 >= 800 → wall
        assert dangers[1] == 1.0

    def test_facing_wall_left_danger_is_one(self):
        """Snake at top wall facing right: left (=up) danger is 1.0."""
        # Place snake near top wall, facing right; left = up
        snake = Snake(0, (255, 0, 0), (400, 5), 10, 800, 600)
        snake.direction = (1, 0)  # facing right
        dangers = snake._get_per_action_danger([])
        # left (index 0) → up → new_y = 5-10 = -5 < 0 → wall
        assert dangers[0] == 1.0

    def test_near_left_wall_facing_left_straight_danger(self):
        """Snake near left wall facing left: straight danger = 1.0."""
        snake = Snake(0, (255, 0, 0), (5, 300), 10, 800, 600)
        snake.direction = (-1, 0)  # facing left
        dangers = snake._get_per_action_danger([])
        # straight (index 1) → left → new_x = 5-10 = -5 < 0
        assert dangers[1] == 1.0

    def test_near_bottom_wall_facing_down(self):
        """Snake near bottom wall facing down: straight danger = 1.0."""
        snake = Snake(0, (255, 0, 0), (400, 595), 10, 800, 600)
        snake.direction = (0, 1)  # facing down
        dangers = snake._get_per_action_danger([])
        # straight → down → new_y = 605 >= 600
        assert dangers[1] == 1.0


class TestProximityDanger:
    """Test soft proximity danger values."""

    def test_nearby_wall_has_nonzero_danger(self):
        """Wall within 3 segments: danger > 0 but < 1.0."""
        # Snake 25px from right wall (within 3*10=30), facing up
        snake = Snake(0, (255, 0, 0), (775, 300), 10, 800, 600)
        snake.direction = (0, -1)  # facing up
        dangers = snake._get_per_action_danger([])
        # right (index 2) → facing up, turn right → facing right
        # new_x = 775 + 10 = 785, within 30px of wall (800-785=15)
        assert dangers[2] > 0.0, "Should have proximity danger near wall"
        assert dangers[2] < 1.0, "Should not be full danger (not at wall)"

    def test_wall_parallel_safe_moves_stay_below_collision_danger(self):
        """Soft wall proximity should not look like an immediate collision."""
        snake = Snake(0, (255, 0, 0), (0, 10), 10, 800, 600)
        snake.direction = (1, 0)

        dangers = snake._get_per_action_danger([snake])
        state = snake.get_state([snake], [])
        action_mask = valid_action_mask_from_states(state.unsqueeze(0))

        assert all(0.0 < danger < 1.0 for danger in dangers)
        assert action_mask.tolist()[0][:3] == [True, True, True]

    def test_far_from_obstacles_danger_near_zero(self):
        """Snake far from all obstacles: danger approaches 0."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right, center
        dangers = snake._get_per_action_danger([])
        # All directions are far from walls (~300+ px away)
        for d in dangers:
            assert d < 0.1, f"Danger {d} too high far from obstacles"


class TestOtherSnakeDanger:
    """Test danger when other snakes are nearby."""

    def test_other_snake_body_causes_danger(self):
        """Snake body directly ahead → danger = 1.0."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # facing right

        # Place enemy snake body segment directly ahead (at 410, 300)
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True

        dangers = snake._get_per_action_danger([snake, enemy])
        # straight (index 1) → right → new_x=410 → dist to enemy head (410,300) = 0
        assert dangers[1] == 1.0

    def test_dead_snake_ignored(self):
        """Dead snakes should not cause danger."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)

        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = False

        dangers = snake._get_per_action_danger([snake, enemy])
        # Dead enemy shouldn't contribute danger
        assert dangers[1] < 1.0

    def test_enemy_tail_is_not_collision_danger_when_tail_will_move(self):
        """The current enemy tail is safe when its next move will pop it."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (430, 300), 10, 800, 600)
        enemy.direction = (1, 0)
        enemy.length = 3
        enemy.segments = [(430, 300), (420, 300), (410, 300)]

        dangers = snake._get_per_action_danger([snake, enemy])

        assert dangers[TURN_STRAIGHT] < 1.0

    def test_growing_enemy_tail_is_collision_danger(self):
        """A growing enemy's tail persists and remains an immediate obstacle."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (430, 300), 10, 800, 600)
        enemy.direction = (1, 0)
        enemy.length = 4
        enemy.segments = [(430, 300), (420, 300), (410, 300)]

        dangers = snake._get_per_action_danger([snake, enemy])

        assert dangers[TURN_STRAIGHT] == 1.0


class TestSelfBodyDanger:
    """Test danger from the snake's own body."""

    def test_safe_turns_ignore_non_collidable_neck(self):
        """The old head/neck adjacency should not make ordinary turns dangerous."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        snake.length = 2
        snake.segments = [(400, 300), (390, 300)]

        dangers = snake._get_per_action_danger([snake])

        assert all(d < 0.1 for d in dangers)

    def test_moving_into_tail_is_not_danger_when_tail_will_move(self):
        """The current tail is safe when move() will pop it before collision checks."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        snake.length = 3
        snake.segments = [(400, 300), (390, 300), (400, 290)]

        dangers = snake._get_per_action_danger([snake])

        assert dangers[TURN_LEFT] < 0.1

    def test_moving_into_persistent_body_segment_is_danger(self):
        """Old segment 2 remains after movement and can be a self-collision target."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        snake.length = 4
        snake.segments = [(400, 300), (390, 300), (400, 290), (380, 300)]

        dangers = snake._get_per_action_danger([snake])

        assert dangers[TURN_LEFT] == 1.0


class TestStateVectorIndices:
    """Test per-action danger at correct indices in state vector."""

    def test_per_action_danger_indices(self):
        """Per-action danger should be at indices 54-56."""
        assert StateIndices.PER_ACTION_DANGER_START == 54
        assert StateIndices.PER_ACTION_DANGER_END == 57

    def test_state_vector_contains_per_action_danger(self):
        """State vector should include per-action danger values."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        food = [(500, 300)]
        state = snake.get_state([], food)

        danger_slice = state[54:57]
        assert danger_slice.shape == (3,)
        # All values in [0, 1]
        for v in danger_slice:
            assert 0.0 <= float(v) <= 1.0

    def test_total_state_size_is_58(self):
        """State vector size should be 58 (with boost at index 57)."""
        assert GameConfig.INPUT_SIZE == 58
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        state = snake.get_state([], [(500, 300)])
        assert state.shape == (58,)


class TestRelativeActionDanger:
    """Test that danger is computed relative to current heading."""

    def test_danger_matches_relative_direction(self):
        """Danger for each action corresponds to left/straight/right relative to heading."""
        # Place snake near top-right corner facing right
        snake = Snake(0, (255, 0, 0), (795, 5), 10, 800, 600)
        snake.direction = (1, 0)  # facing right

        dangers = snake._get_per_action_danger([])

        # left (index 0) = up → new_y = 5-10 = -5 → wall collision → 1.0
        assert dangers[0] == 1.0
        # straight (index 1) = right → new_x = 805 → wall collision → 1.0
        assert dangers[1] == 1.0
        # right (index 2) = down → new_y = 15 → safe
        assert dangers[2] < 1.0

    def test_different_heading_different_dangers(self):
        """Same position but different heading gives different danger pattern."""
        # Near right wall
        pos = (795, 300)

        # Facing right: straight should be dangerous
        snake_right = Snake(0, (255, 0, 0), pos, 10, 800, 600)
        snake_right.direction = (1, 0)
        dangers_right = snake_right._get_per_action_danger([])

        # Facing up: right turn → facing right → dangerous
        snake_up = Snake(1, (0, 255, 0), pos, 10, 800, 600)
        snake_up.direction = (0, -1)
        dangers_up = snake_up._get_per_action_danger([])

        # Facing right, straight → danger
        assert dangers_right[1] == 1.0
        # Facing up, right turn → right → danger
        assert dangers_up[2] == 1.0
        # Facing up, straight → up → safe (far from top wall)
        assert dangers_up[1] < 1.0
