"""Game logic and collision detection utilities."""

import math
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.core.game_config import GameConfig

if TYPE_CHECKING:
    from src.game.game_state import GameState
    from src.game.snake import Snake

# Relative action constants
TURN_LEFT = 0
TURN_STRAIGHT = 1
TURN_RIGHT = 2


class GameLogic:
    """Static utility class for game logic operations."""

    @staticmethod
    def _collision_positions(snake: "Snake") -> List[Tuple[int, int]]:
        """Return head positions traversed during the current frame."""
        move_positions = getattr(snake, "last_move_positions", None)
        return list(move_positions) if move_positions else [snake.head]

    @staticmethod
    def _head_path_positions(snake: "Snake") -> List[Tuple[int, int]]:
        """Return the previous head plus every traversed head position."""
        move_positions = getattr(snake, "last_move_positions", None)
        if not move_positions:
            return [snake.head]

        first_x, first_y = move_positions[0]
        dir_x, dir_y = snake.direction
        previous_head = (
            first_x - dir_x * snake.segment_size,
            first_y - dir_y * snake.segment_size,
        )
        return [previous_head] + list(move_positions)

    @staticmethod
    def _head_paths_crossed(snake1: "Snake", snake2: "Snake") -> bool:
        """Return whether two heads swapped positions between movement samples."""
        path1 = GameLogic._head_path_positions(snake1)
        path2 = GameLogic._head_path_positions(snake2)
        threshold = min(snake1.segment_size, snake2.segment_size)

        for start1, end1 in zip(path1, path1[1:]):
            for start2, end2 in zip(path2, path2[1:]):
                if (
                    GameLogic.distance(start1, end2) < threshold
                    and GameLogic.distance(start2, end1) < threshold
                ):
                    return True
        return False

    @staticmethod
    def get_circular_arena(
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Return circular arena center/radius for an effective board size."""
        width = GameConfig.WIDTH if width is None else width
        height = GameConfig.HEIGHT if height is None else height
        base_width = max(GameConfig.WIDTH, 1)
        base_height = max(GameConfig.HEIGHT, 1)
        scale_x = width / base_width
        scale_y = height / base_height
        scale = min(scale_x, scale_y)

        return (
            GameConfig.ARENA_CENTER_X * scale_x,
            GameConfig.ARENA_CENTER_Y * scale_y,
            GameConfig.ARENA_RADIUS * scale,
        )

    @staticmethod
    def get_random_circular_position(
        width: int,
        height: int,
        margin: int = 0,
    ) -> Tuple[int, int]:
        """Return a random point inside the effective circular arena."""
        cx, cy, radius = GameLogic.get_circular_arena(width, height)
        spawn_radius = max(0.0, radius - margin)
        angle = random.uniform(0, 2 * math.pi)
        r = spawn_radius * math.sqrt(random.uniform(0, 1))
        return (int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle)))

    @staticmethod
    def relative_to_absolute_direction(
        current_direction: Tuple[int, int],
        relative_action: int,
    ) -> Tuple[int, int]:
        """Convert relative action (left/straight/right) to absolute direction.

        Uses cardinal direction ordering: up(0,-1), right(1,0), down(0,1), left(-1,0).
        Left turn = counter-clockwise, Right turn = clockwise.

        Args:
            current_direction: Current absolute direction tuple (dx, dy)
            relative_action: 0=turn left, 1=go straight, 2=turn right

        Returns:
            New absolute direction tuple (dx, dy)
        """
        CARDINAL = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        try:
            idx = CARDINAL.index(current_direction)
        except ValueError:
            return current_direction  # Fallback: keep current direction

        if relative_action == TURN_LEFT:  # Turn left (counter-clockwise)
            new_idx = (idx - 1) % 4
        elif relative_action == TURN_RIGHT:  # Turn right (clockwise)
            new_idx = (idx + 1) % 4
        else:  # Go straight
            new_idx = idx

        return CARDINAL[new_idx]

    @staticmethod
    def check_collisions(snakes: List["Snake"]) -> List[Tuple["Snake", Optional["Snake"], str]]:
        """
        Check all collision types for all snakes.

        Single source of truth for collision detection. Checks:
        - Wall collisions (head out of bounds)
        - Self-collisions (head hitting own body, length > 3)
        - Head-on collisions between two snakes
        - Body collisions (head hitting another snake's body)

        Args:
            snakes: List of all snakes in the game

        Returns:
            List of tuples (colliding_snake, other_snake, collision_type)
            where collision_type is "wall", "self", "head", or "body"
        """
        collisions: List[Tuple["Snake", Optional["Snake"], str]] = []
        for i, snake in enumerate(snakes):
            if not snake.is_alive:
                continue
            # Check collision with walls
            if GameLogic.check_wall_collision(snake):
                collisions.append((snake, None, "wall"))
                continue  # Wall death takes priority, skip other checks
            # Check self-collision (head hitting own body)
            if GameLogic.check_self_collision(snake):
                collisions.append((snake, None, "self"))
                continue  # Self-collision death, skip other checks
            # Check collision with other snakes
            for j, other_snake in enumerate(snakes):
                if i != j and other_snake.is_alive:
                    if GameLogic.check_head_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "head"))
                    elif GameLogic.check_body_collision(snake, other_snake):
                        collisions.append((snake, other_snake, "body"))
        return collisions

    @staticmethod
    def check_wall_collision(snake: "Snake") -> bool:
        """
        Check if snake's head is outside the game boundaries.

        Supports both rectangular and circular arena types.

        Args:
            snake: The snake to check

        Returns:
            True if snake has collided with a wall
        """
        for head_x, head_y in GameLogic._collision_positions(snake):
            if GameConfig.ARENA_TYPE == "circular":
                cx, cy, radius = GameLogic.get_circular_arena(snake.game_width, snake.game_height)
                dx = head_x - cx
                dy = head_y - cy
                if (dx * dx + dy * dy) > radius**2:
                    return True
            elif (
                head_x < 0
                or head_x >= snake.game_width
                or head_y < 0
                or head_y >= snake.game_height
            ):
                return True
        return False

    @staticmethod
    def head_on_collision(snake1: "Snake", snake2: "Snake", game_state: "GameState") -> None:
        """
        Handle head-on collision between two snakes.

        Args:
            snake1: First snake in collision
            snake2: Second snake in collision
            game_state: The current game state (unused, kept for API compatibility)
        """
        snake1.die()
        snake2.die()
        # Note: alive_snakes is recalculated in GameState.update(), no manual decrement needed

    @staticmethod
    def body_collision(
        colliding_snake: "Snake", hit_snake: "Snake", game_state: "GameState"
    ) -> None:
        """
        Handle collision where one snake hits another's body.

        Args:
            colliding_snake: The snake that collided (dies)
            hit_snake: The snake that was hit (survives, unused)
            game_state: The current game state (unused, kept for API compatibility)
        """
        colliding_snake.die()
        # Note: alive_snakes is recalculated in GameState.update(), no manual decrement needed

    @staticmethod
    def check_self_collision(snake: "Snake") -> bool:
        """
        Check if snake's head collides with its own body.

        Only checks if snake is long enough to self-collide (length > 3).
        Skips first 3 segments (head + 2 adjacent) since they can't overlap.

        Args:
            snake: The snake to check

        Returns:
            True if snake head has collided with its own body
        """
        if len(snake.segments) <= 3:
            return False
        for head in GameLogic._collision_positions(snake):
            for segment in snake.segments[3:]:
                if GameLogic.distance(head, segment) < snake.segment_size:
                    return True
        return False

    @staticmethod
    def check_head_collision(snake1: "Snake", snake2: "Snake") -> bool:
        """
        Check if two snake heads are colliding.

        Args:
            snake1: First snake
            snake2: Second snake

        Returns:
            True if heads are within collision distance
        """
        if any(
            GameLogic.distance(head1, head2) < snake1.segment_size
            for head1 in GameLogic._collision_positions(snake1)
            for head2 in GameLogic._collision_positions(snake2)
        ):
            return True
        return GameLogic._head_paths_crossed(snake1, snake2)

    @staticmethod
    def check_body_collision(snake1: "Snake", snake2: "Snake") -> bool:
        """
        Check if snake1's head collides with snake2's body.

        Args:
            snake1: Snake whose head might be colliding
            snake2: Snake whose body might be hit

        Returns:
            True if snake1's head hits snake2's body
        """
        return any(
            GameLogic.distance(head, segment) < snake1.segment_size
            for head in GameLogic._collision_positions(snake1)
            for segment in snake2.segments[1:]
        )

    @staticmethod
    def distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point (x, y)
            point2: Second point (x, y)

        Returns:
            Euclidean distance between the points
        """
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    @staticmethod
    def find_empty_position(
        width: int, height: int, snakes: List["Snake"]
    ) -> Optional[Tuple[int, int]]:
        """
        Find an empty position inside walls (respecting wall thickness margin).

        Supports both rectangular and circular arena types.

        Args:
            width: Game width
            height: Game height
            snakes: List of all snakes to avoid

        Returns:
            A (x, y) position that doesn't overlap any snake, or None if not found
        """
        attempts = 0
        max_attempts = 100
        is_circular = GameConfig.ARENA_TYPE == "circular"
        margin = GameConfig.WALL_THICKNESS
        while attempts < max_attempts:
            if is_circular:
                x, y = GameLogic.get_random_circular_position(width, height, margin)
            else:
                x = random.randint(margin, width - margin - 1)
                y = random.randint(margin, height - margin - 1)
            if not GameLogic.position_overlaps_snakes((x, y), snakes):
                return (x, y)
            attempts += 1
        print("Warning: Could not find an empty position after max attempts")
        return None

    @staticmethod
    def position_overlaps_snakes(position: Tuple[int, int], snakes: List["Snake"]) -> bool:
        """Return whether a position overlaps any living snake segment."""
        return any(
            GameLogic.distance(position, segment) < snake.segment_size
            for snake in snakes
            if snake.is_alive
            for segment in snake.segments
        )
