"""Tests for game_logic module."""

import pytest

from src.game.game_logic import GameLogic
from src.game.snake import Snake


class TestGameLogic:
    """Test suite for GameLogic class."""

    def test_distance_calculation(self):
        """Test distance calculation between two points."""
        point1 = (0, 0)
        point2 = (3, 4)
        distance = GameLogic.distance(point1, point2)
        assert distance == 5.0

    def test_distance_same_point(self):
        """Test distance when points are the same."""
        point = (10, 20)
        distance = GameLogic.distance(point, point)
        assert distance == 0.0

    def test_wall_collision_left(self):
        """Test collision with left wall."""
        snake = Snake(0, (255, 0, 0), (-10, 50), 10, 800, 600)
        assert GameLogic.check_wall_collision(snake) is True

    def test_wall_collision_right(self):
        """Test collision with right wall."""
        snake = Snake(0, (255, 0, 0), (1460, 50), 10, 1450, 830)
        assert GameLogic.check_wall_collision(snake) is True

    def test_wall_collision_top(self):
        """Test collision with top wall."""
        snake = Snake(0, (255, 0, 0), (50, -10), 10, 800, 600)
        assert GameLogic.check_wall_collision(snake) is True

    def test_wall_collision_bottom(self):
        """Test collision with bottom wall."""
        snake = Snake(0, (255, 0, 0), (50, 840), 10, 1450, 830)
        assert GameLogic.check_wall_collision(snake) is True

    def test_no_wall_collision(self):
        """Test no collision when snake is within bounds."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        assert GameLogic.check_wall_collision(snake) is False

    def test_wall_collision_uses_snake_dimensions(self):
        """Wall checks should respect the arena dimensions stored on the snake."""
        snake = Snake(0, (255, 0, 0), (805, 100), 10, 800, 600)
        assert GameLogic.check_wall_collision(snake) is True

    def test_head_collision(self):
        """Test head-on collision detection."""
        snake1 = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake2 = Snake(1, (0, 255, 0), (105, 100), 10, 800, 600)
        assert GameLogic.check_head_collision(snake1, snake2) is True

    def test_no_head_collision(self):
        """Test no head collision when snakes are far apart."""
        snake1 = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake2 = Snake(1, (0, 255, 0), (200, 200), 10, 800, 600)
        assert GameLogic.check_head_collision(snake1, snake2) is False

    def test_head_swap_collision(self):
        """Adjacent snakes swapping head cells in one frame is a head-on collision."""
        snake1 = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake2 = Snake(1, (0, 255, 0), (110, 100), 10, 800, 600)
        snake1.direction = (1, 0)
        snake2.direction = (-1, 0)

        snake1.move()
        snake2.move()

        assert snake1.head == (110, 100)
        assert snake2.head == (100, 100)
        assert GameLogic.check_head_collision(snake1, snake2) is True
        collisions = GameLogic.check_collisions([snake1, snake2])
        assert any(
            snake is snake1 and other_snake is snake2 and collision_type == "head"
            for snake, other_snake, collision_type in collisions
        )

    def test_body_collision(self):
        """Test body collision detection."""
        snake1 = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake2 = Snake(1, (0, 255, 0), (200, 200), 10, 800, 600)
        snake2.segments = [(200, 200), (100, 100), (100, 110)]  # Add body segments

        assert GameLogic.check_body_collision(snake1, snake2) is True

    def test_boost_path_body_collision(self):
        """Boosted movement should collide with bodies crossed between start and end."""
        snake1 = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake1.length = 5
        snake1.segments = [(400 - i * 10, 300) for i in range(snake1.length)]
        snake1.is_boosting = True
        snake2 = Snake(1, (0, 255, 0), (500, 300), 10, 800, 600)
        snake2.segments = [(500, 300), (410, 300)]

        snake1.move()

        assert snake1.head == (420, 300)
        assert snake1.last_move_positions == [(410, 300), (420, 300)]
        assert GameLogic.check_body_collision(snake1, snake2) is True
        collisions = GameLogic.check_collisions([snake1, snake2])
        assert any(
            snake is snake1 and other_snake is snake2 and collision_type == "body"
            for snake, other_snake, collision_type in collisions
        )

    def test_boost_path_head_collision(self):
        """Boosted movement should collide with heads crossed on the first boost step."""
        snake1 = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake1.length = 5
        snake1.segments = [(400 - i * 10, 300) for i in range(snake1.length)]
        snake1.is_boosting = True
        snake2 = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)

        snake1.move()

        assert snake1.head == (420, 300)
        assert GameLogic.check_head_collision(snake1, snake2) is True
        collisions = GameLogic.check_collisions([snake1, snake2])
        assert any(
            snake is snake1 and other_snake is snake2 and collision_type == "head"
            for snake, other_snake, collision_type in collisions
        )

    def test_no_body_collision(self):
        """Test no body collision."""
        snake1 = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake2 = Snake(1, (0, 255, 0), (200, 200), 10, 800, 600)

        assert GameLogic.check_body_collision(snake1, snake2) is False

    def test_find_empty_position(self):
        """Test finding an empty position."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        position = GameLogic.find_empty_position(800, 600, [snake])

        assert position is not None
        assert 0 <= position[0] < 800
        assert 0 <= position[1] < 600

    @pytest.mark.parametrize(
        "width,height",
        [
            (800, 600),
            (1000, 800),
            (1450, 830),
        ],
    )
    def test_find_empty_position_different_sizes(self, width, height):
        """Test finding empty position with different game sizes."""
        position = GameLogic.find_empty_position(width, height, [])

        assert position is not None
        assert 0 <= position[0] < width
        assert 0 <= position[1] < height

    def test_self_collision_short_snake(self):
        """Self-collision should not trigger for snakes with <= 3 segments."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.segments = [(100, 100), (90, 100), (80, 100)]
        assert GameLogic.check_self_collision(snake) is False

    def test_self_collision_long_snake(self):
        """Self-collision should trigger when head overlaps body segment."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        # Head at (100,100), body loops back to overlap the head
        snake.segments = [(100, 100), (110, 100), (110, 110), (100, 110), (100, 100)]
        # segments[4] == head → distance 0 < segment_size
        assert GameLogic.check_self_collision(snake) is True

    def test_no_self_collision(self):
        """Self-collision should not trigger for a straight snake."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.segments = [(100, 100), (90, 100), (80, 100), (70, 100), (60, 100)]
        assert GameLogic.check_self_collision(snake) is False

    def test_check_collisions_wall(self):
        """check_collisions should detect wall collision."""
        snake = Snake(0, (255, 0, 0), (-5, 100), 10, 800, 600)
        collisions = GameLogic.check_collisions([snake])
        assert len(collisions) == 1
        assert collisions[0][2] == "wall"

    def test_check_collisions_self(self):
        """check_collisions should detect self-collision."""
        snake = Snake(0, (255, 0, 0), (100, 100), 10, 800, 600)
        snake.segments = [(100, 100), (110, 100), (110, 110), (100, 110), (100, 100)]
        collisions = GameLogic.check_collisions([snake])
        assert len(collisions) == 1
        assert collisions[0][2] == "self"

    def test_check_collisions_wall_priority(self):
        """Wall collision takes priority — no other checks after wall hit."""
        snake = Snake(0, (255, 0, 0), (-5, 100), 10, 800, 600)
        snake.segments = [(-5, 100), (5, 100), (5, 110), (-5, 110), (-5, 100)]
        collisions = GameLogic.check_collisions([snake])
        # Should only report wall, not self-collision too
        assert len(collisions) == 1
        assert collisions[0][2] == "wall"
