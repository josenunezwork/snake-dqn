"""Food management for the snake game.

This module provides the FoodManager class which handles all food-related
logic including spawning, consumption, and maintaining food count.
"""

import random
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.core.game_config import GameConfig
from src.game.game_logic import GameLogic

if TYPE_CHECKING:
    from src.game.snake import Snake


class FoodManager:
    """Manages food spawning and consumption in the game.

    Responsibilities:
    - Spawn initial food at game start
    - Maintain food count at configured level
    - Handle food consumption detection
    - Provide food positions for game state

    Usage:
        fm = FoodManager(1450, 830, 300, 250, segment_size=10, wall_thickness=10)
        fm.maintain_count(snakes)  # Called each frame
        if fm.consume_at(snake.head, snake.segment_size):
            snake.grow()
    """

    def __init__(
        self,
        game_width: int,
        game_height: int,
        max_food: int,
        initial_food: int,
        segment_size: int = 10,
        wall_thickness: int = 10,
    ) -> None:
        """Initialize the food manager.

        Args:
            game_width: Width of the game area
            game_height: Height of the game area
            max_food: Maximum number of food items to maintain
            initial_food: Number of food items to spawn initially
            segment_size: Size of game segments (for collision detection)
            wall_thickness: Thickness of walls (for spawn boundaries)
        """
        self.game_width = game_width
        self.game_height = game_height
        self.max_food = max_food
        self.segment_size = segment_size
        self.wall_thickness = wall_thickness
        self.food: List[Tuple[int, int]] = []
        self._spawn_initial(initial_food)

    def _spawn_initial(self, count: int, snakes: Optional[List["Snake"]] = None) -> None:
        """Spawn initial food items at random positions.

        Args:
            count: Number of food items to spawn
            snakes: Optional list of snakes to avoid when spawning
        """
        self.food.clear()
        for _ in range(count):
            pos = self._find_spawn_position(snakes)
            if pos:
                self.add_food(pos)

    def _get_random_position(self) -> Tuple[int, int]:
        """Get a random position within game boundaries.

        Supports both rectangular and circular arena types.

        Returns:
            Random (x, y) position respecting wall margins
        """
        if GameConfig.ARENA_TYPE == "circular":
            return GameLogic.get_random_circular_position(
                self.game_width,
                self.game_height,
                self.wall_thickness,
            )
        return (
            random.randint(
                self.wall_thickness, self.game_width - self.wall_thickness - self.segment_size
            ),
            random.randint(
                self.wall_thickness, self.game_height - self.wall_thickness - self.segment_size
            ),
        )

    def _position_overlaps_food(self, position: Tuple[int, int]) -> bool:
        """Return whether a position overlaps existing food."""
        return any(
            GameLogic.distance(position, food_pos) < self.segment_size for food_pos in self.food
        )

    def add_food(self, position: Tuple[int, int]) -> bool:
        """Add food if it does not overlap existing food."""
        if self._position_overlaps_food(position):
            return False
        self.food.append(position)
        return True

    def _find_spawn_position(self, snakes: Optional[List["Snake"]] = None) -> Optional[Tuple[int, int]]:
        """Find a spawn position that avoids snakes and existing food."""
        for _ in range(100):
            if snakes:
                pos = GameLogic.find_empty_position(self.game_width, self.game_height, snakes)
                if pos is None:
                    return None
            else:
                pos = self._get_random_position()

            if not self._position_overlaps_food(pos):
                return pos
        return None

    def spawn(self, count: int, snakes: List["Snake"]) -> int:
        """Spawn food at empty positions avoiding snakes.

        Args:
            count: Number of food items to try to spawn
            snakes: List of snakes to avoid when spawning

        Returns:
            Number of food items actually spawned
        """
        spawned = 0
        for _ in range(count):
            pos = self._find_spawn_position(snakes)
            if pos and self.add_food(pos):
                spawned += 1
            else:
                # No empty position found, stop trying
                break
        return spawned

    def maintain_count(self, snakes: List["Snake"]) -> int:
        """Ensure food count stays at max_food level.

        This should be called each game frame to maintain food supply.

        Args:
            snakes: List of snakes to avoid when spawning

        Returns:
            Number of new food items spawned
        """
        deficit = self.max_food - len(self.food)
        if deficit > 0:
            return self.spawn(deficit, snakes)
        return 0

    def consume_at(self, position: Tuple[int, int], radius: int) -> bool:
        """Check and consume food at position.

        Args:
            position: (x, y) position to check for food
            radius: Collision radius (typically segment_size)

        Returns:
            True if food was consumed, False otherwise
        """
        eaten = [f for f in self.food if GameLogic.distance(f, position) < radius]
        if eaten:
            self.food = [f for f in self.food if f not in eaten]
            return True
        return False

    def reset(self, initial_count: int, snakes: Optional[List["Snake"]] = None) -> None:
        """Reset food for a new episode.

        Args:
            initial_count: Number of food items to spawn after reset
            snakes: Optional list of snakes to avoid when spawning
        """
        self._spawn_initial(initial_count, snakes)

    def clear(self) -> None:
        """Remove all food from the game."""
        self.food.clear()

    @property
    def count(self) -> int:
        """Get current food count."""
        return len(self.food)

    def __len__(self) -> int:
        """Get current food count."""
        return len(self.food)

    def __iter__(self):
        """Iterate over food positions."""
        return iter(self.food)

    def __repr__(self) -> str:
        return f"FoodManager(count={len(self.food)}, max={self.max_food})"
