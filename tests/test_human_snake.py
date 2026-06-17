"""Tests for human_snake module."""
import pytest
import torch
from src.game.human_snake import HumanSnake
from src.core.game_config import GameConfig
from src.game.game_logic import TURN_LEFT, TURN_STRAIGHT
from PyQt5.QtCore import Qt


class TestHumanSnake:
    """Test suite for HumanSnake class."""

    def test_human_snake_initialization(self):
        """Test HumanSnake initializes correctly."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        assert snake.id == 0
        assert snake.is_alive is True
        assert len(snake.experience_buffer) == 0
        assert snake.current_epsilon == 0  # Human doesn't use epsilon

    def test_human_snake_color_name(self):
        """Test color name conversion."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        assert snake.color_name == "Red"

        snake2 = HumanSnake(1, (0, 255, 0), (400, 300), 10, 800, 600)
        assert snake2.color_name == "Green"

    def test_set_direction_from_key_up(self):
        """Test setting direction from up arrow key."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # Moving right

        result = snake.set_direction_from_key(Qt.Key_Up)

        assert result is True
        assert snake.direction == (0, -1)

    def test_set_direction_from_key_down(self):
        """Test setting direction from down arrow key."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)

        result = snake.set_direction_from_key(Qt.Key_Down)

        assert result is True
        assert snake.direction == (0, 1)

    def test_set_direction_from_key_left(self):
        """Test setting direction from left arrow key."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (0, 1)  # Moving down

        result = snake.set_direction_from_key(Qt.Key_Left)

        assert result is True
        assert snake.direction == (-1, 0)

    def test_set_direction_from_key_right(self):
        """Test setting direction from right arrow key."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (0, 1)

        result = snake.set_direction_from_key(Qt.Key_Right)

        assert result is True
        assert snake.direction == (1, 0)

    def test_set_direction_prevents_180_turn(self):
        """Test that 180-degree turns are prevented."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)  # Moving right

        result = snake.set_direction_from_key(Qt.Key_Left)  # Try to go left

        assert result is False
        assert snake.direction == (1, 0)  # Direction unchanged

    def test_add_experience(self):
        """Test adding experience to buffer."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        state = torch.randn(GameConfig.INPUT_SIZE)
        next_state = torch.randn(GameConfig.INPUT_SIZE)

        snake.add_experience(state, 1, 10.0, next_state, False)

        assert len(snake.experience_buffer) == 1
        exp = snake.experience_buffer[0]
        assert exp['action'] == 1
        assert exp['reward'] == 10.0
        assert exp['done'] is False

    def test_human_turn_records_current_relative_transition(self):
        """Human replay should match Apex relative actions, not absolute directions."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)

        assert snake.set_direction_from_key(Qt.Key_Up) is True
        snake.update([snake], [])
        snake.compute_reward_and_train([snake], [], ate_food=False, collided=False)

        assert len(snake.experience_buffer) == 1
        exp = snake.experience_buffer[0]
        assert exp["action"] == TURN_LEFT
        assert exp["state"][1] == pytest.approx(1.0)  # Previous direction: right
        assert exp["next_state"][0] == pytest.approx(1.0)  # New direction: up

    def test_human_no_key_records_straight_transition_immediately(self):
        """The first human transition should not be delayed until the next reward."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (0, -1)

        snake.update([snake], [])
        snake.compute_reward_and_train([snake], [], ate_food=False, collided=False)

        assert len(snake.experience_buffer) == 1
        assert snake.experience_buffer[0]["action"] == TURN_STRAIGHT

    def test_get_experiences_clears_buffer(self):
        """Test that getting experiences clears the buffer."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        state = torch.randn(GameConfig.INPUT_SIZE)
        snake.add_experience(state, 0, 1.0, state, False)
        snake.add_experience(state, 1, 2.0, state, False)

        experiences = snake.get_experiences()

        assert len(experiences) == 2
        assert len(snake.experience_buffer) == 0

    def test_should_save_experiences_threshold(self):
        """Test experience saving threshold."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        # Buffer not full
        assert snake.should_save_experiences() is False

        # Fill buffer to threshold
        state = torch.randn(GameConfig.INPUT_SIZE)
        for _ in range(1000):
            snake.add_experience(state, 0, 1.0, state, False)

        assert snake.should_save_experiences() is True

    def test_get_state_returns_correct_size(self):
        """Test that get_state returns correct tensor size."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        state = snake.get_state([], [(100, 100)])

        assert state.shape == (GameConfig.INPUT_SIZE,)
        assert isinstance(state, torch.Tensor)

    def test_calculate_reward_collision(self):
        """Test reward calculation on collision."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        reward = snake.calculate_reward(False, True, None, None, [], [])

        assert reward == GameConfig.REWARD_DEATH

    def test_calculate_reward_eating_food(self):
        """Test reward calculation when eating food."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.grow(5)  # Length = 6

        state = snake.get_state([], [(100, 100)])
        reward = snake.calculate_reward(True, False, state, state, [], [])

        assert reward == GameConfig.REWARD_FOOD_BASE

    def test_total_reward_property(self):
        """Test total_reward property."""
        snake = HumanSnake(0, (255, 0, 0), (400, 300), 10, 800, 600)

        assert snake.total_reward == 0.0

        snake._total_reward = 100.5
        assert snake.total_reward == 100.5
