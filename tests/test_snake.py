"""Tests for snake module."""

import pytest
import torch

from src.core.game_config import GameConfig, StateIndices
from src.game.snake import Snake


class TestSnake:
    """Test suite for Snake class."""

    def test_snake_initialization(self):
        """Test snake is initialized correctly."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)

        assert snake.id == 0
        assert snake.color == (255, 0, 0)
        assert snake.segments == [(100, 100)]
        assert snake.direction == (1, 0)
        assert snake.is_alive is True
        assert snake.length == 1
        assert snake.segment_size == 10

    def test_snake_head_property(self):
        """Test head property returns correct position."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        assert snake.head == (100, 100)

    def test_snake_move_right(self):
        """Test snake movement to the right."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (1, 0)
        snake.move()

        assert snake.head == (110, 100)
        assert len(snake.segments) == 1

    def test_snake_move_left(self):
        """Test snake movement to the left."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (-1, 0)
        snake.move()

        assert snake.head == (90, 100)

    def test_snake_move_up(self):
        """Test snake movement upward."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (0, -1)
        snake.move()

        assert snake.head == (100, 90)

    def test_snake_move_down(self):
        """Test snake movement downward."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (0, 1)
        snake.move()

        assert snake.head == (100, 110)

    def test_snake_grow(self):
        """Test snake growth."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        initial_length = snake.length

        snake.grow(3)

        assert snake.length == initial_length + 3

    def test_snake_grow_default_amount(self):
        """Test snake grows by 1 by default."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        initial_length = snake.length

        snake.grow()

        assert snake.length == initial_length + 1

    def test_state_length_uses_logical_length_while_body_is_filling_in(self):
        """The length feature should match growth immediately after eating."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.length = GameConfig.MIN_BOOST_LENGTH

        state = snake.get_state([], [(200, 100)])

        assert len(snake.segments) == 1
        assert float(state[StateIndices.LENGTH_NORMALIZED]) == pytest.approx(
            GameConfig.MIN_BOOST_LENGTH / GameConfig.MAX_LENGTH
        )
        assert float(state[StateIndices.BOOST_AVAILABLE]) == pytest.approx(1.0)

    def test_food_density_uses_environment_food_capacity(self):
        """Scaled actor/curriculum worlds should not understate local food density."""
        snake = Snake(
            0,
            (255, 0, 0),
            (100, 100),
            10,
            800,
            600,
            food_capacity=GameConfig.NUM_SECTORS,
        )
        food = [(110, 100)]

        state = snake.get_state([snake], food)
        sector = Snake._angle_to_sector(10, 0, GameConfig.NUM_SECTORS)
        density_index = StateIndices.FOOD_DENSITY_START + sector

        assert float(state[density_index]) == pytest.approx(1.0)

    def test_snake_change_direction_valid(self):
        """Test valid direction change."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (1, 0)  # Moving right

        snake.change_direction((0, 1))  # Try to move down

        assert snake.direction == (0, 1)

    def test_snake_change_direction_invalid_180(self):
        """Test invalid 180-degree turn is rejected."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.direction = (1, 0)  # Moving right

        snake.change_direction((-1, 0))  # Try to move left (180 degrees)

        assert snake.direction == (1, 0)  # Direction unchanged

    def test_snake_die(self):
        """Test snake death."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)

        snake.die()

        assert snake.is_alive is False
        assert snake.respawn_timer == GameConfig.FRAME_RATE

    def test_snake_respawn(self):
        """Test snake respawn."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.die()
        snake.grow(5)

        snake.respawn((200, 200))

        assert snake.is_alive is True
        assert snake.segments == [(200, 200)]
        assert snake.length == 1
        assert snake.direction == (1, 0)
        assert snake.respawn_timer == 0

    def test_snake_move_past_left_wall(self):
        """Test move() does NOT kill snake — collision handled centrally."""
        snake = Snake(0, (255, 0, 0), (5, 100), 10, 800, 600)
        snake.direction = (-1, 0)

        snake.move()

        # Snake is still alive — move() only moves, no collision check
        assert snake.is_alive is True
        assert snake.head == (-5, 100)  # Head moved out of bounds

    def test_snake_move_past_right_wall(self):
        """Test move() does NOT kill snake — collision handled centrally."""
        snake = Snake(0, (255, 0, 0), (795, 100), 10, 800, 600)
        snake.direction = (1, 0)

        snake.move()

        assert snake.is_alive is True
        assert snake.head == (805, 100)

    def test_snake_segment_tail_removed_when_not_growing(self):
        """Test that tail segment is removed when not growing."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.grow(3)
        snake.move()  # Add segment at (110, 100), segments=2, length=4
        snake.move()  # Add segment at (120, 100), segments=3, length=4
        snake.move()  # Add segment at (130, 100), segments=4, length=4 (fully grown)

        initial_length = len(snake.segments)
        snake.move()  # Should add new head and remove tail

        assert len(snake.segments) == initial_length

    def test_open_movement_after_growth_not_punished_as_critical_danger(self):
        """Own adjacent neck in the danger map should not dominate reward."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.length = 2
        snake.segments = [(400, 300), (390, 300)]
        snake.direction = (1, 0)
        food = [(700, 300)]

        old_state = snake.get_state([snake], food)
        snake.move()
        new_state = snake.get_state([snake], food)
        reward = snake.calculate_reward(False, False, old_state, new_state, [snake], food)

        danger_map = new_state[slice(StateIndices.DANGER_MAP_START, StateIndices.DANGER_MAP_END)]
        action_danger = new_state[
            slice(StateIndices.PER_ACTION_DANGER_START, StateIndices.PER_ACTION_DANGER_END)
        ]
        assert float(danger_map.max()) < 0.1
        assert float(action_danger.max()) < 0.1
        assert reward > 0.0

    def test_danger_map_ignores_self_tail_when_tail_will_move(self):
        """The sector map should not mark a vacating self tail as danger."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.length = 3
        snake.segments = [(400, 300), (390, 300), (380, 300)]
        state = snake.get_state([snake], [(700, 300)])

        danger_map = state[slice(StateIndices.DANGER_MAP_START, StateIndices.DANGER_MAP_END)]

        assert float(danger_map.max()) < 0.1

    def test_danger_map_keeps_growing_self_tail_as_obstacle(self):
        """A growing self tail persists, so it should remain in the danger map."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.length = 4
        snake.segments = [(400, 300), (390, 300), (380, 300)]
        state = snake.get_state([snake], [(700, 300)])

        danger_map = state[slice(StateIndices.DANGER_MAP_START, StateIndices.DANGER_MAP_END)]

        assert float(danger_map.max()) > 0.9

    def test_danger_map_ignores_enemy_tail_when_tail_will_move(self):
        """The sector map should match exact masks for non-growing enemy tails."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        enemy = Snake(1, (0, 255, 0), (430, 330), 10, 800, 600)
        enemy.length = 3
        enemy.segments = [(430, 330), (420, 330), (410, 300)]
        state = snake.get_state([snake, enemy], [(700, 300)])

        danger_map = state[slice(StateIndices.DANGER_MAP_START, StateIndices.DANGER_MAP_END)]
        tail_sector = Snake._angle_to_sector(10, 0, GameConfig.NUM_SECTORS)

        assert float(danger_map[tail_sector]) < 0.1

    def test_danger_map_keeps_growing_enemy_tail_as_obstacle(self):
        """A growing enemy tail persists and should stay visible as danger."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        enemy = Snake(1, (0, 255, 0), (430, 330), 10, 800, 600)
        enemy.length = 4
        enemy.segments = [(430, 330), (420, 330), (410, 300)]
        state = snake.get_state([snake, enemy], [(700, 300)])

        danger_map = state[slice(StateIndices.DANGER_MAP_START, StateIndices.DANGER_MAP_END)]
        tail_sector = Snake._angle_to_sector(10, 0, GameConfig.NUM_SECTORS)

        assert float(danger_map[tail_sector]) > 0.9

    def test_action_danger_pressure_still_penalizes_constrained_states(self):
        """Danger reward should still fire when immediate actions are risky."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        state = torch.zeros(GameConfig.INPUT_SIZE)
        state[StateIndices.FOOD_DISTANCE] = 0.5
        state[slice(StateIndices.BOUNDARY_LEFT, StateIndices.BOUNDARY_BOTTOM + 1)] = 1.0
        state[slice(StateIndices.PER_ACTION_DANGER_START, StateIndices.PER_ACTION_DANGER_END)] = (
            torch.tensor([1.0, 1.0, 1.0])
        )

        reward = snake.calculate_reward(False, False, state, state, [], [])

        expected = GameConfig.REWARD_DANGER_CRITICAL + GameConfig.REWARD_SURVIVAL
        assert reward == pytest.approx(expected)
