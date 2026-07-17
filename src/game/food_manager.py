"""Food management for the snake game.

This module provides the FoodManager class which handles all food-related
logic including spawning, consumption, and maintaining food count.
"""

import random
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from src.core.game_config import GameConfig
from src.core.mechanics_constants import (
    CORPSE_EXEMPT_FROM_CAP_V2,
    corpse_food_cap,
    evict_oldest_corpse,
    same_cell,
    snap_to_cell,
)
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

    Food pools: pellets are either ambient (subject to the max_food cap that
    maintain_count() tops up to) or corpse-class (kill corpses + boost trail
    pellets, added with ``add_food(..., corpse=True)``). Corpse-class food is
    exempt from the ambient cap: maintain_count() ignores it when computing the
    spawn deficit, so a large corpse never suppresses baseline food spawns.
    Callers only pass ``corpse=True`` under mechanics v2; at v1 every pellet is
    ambient and behavior is unchanged.

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
        # Positions of corpse-class pellets (exempt from the ambient cap).
        # Membership is checked against the live food list when counting, so
        # external trims of self.food cannot inflate the corpse count.
        self._corpse_positions: Set[Tuple[int, int]] = set()
        self._spawn_initial(initial_food)

    def _spawn_initial(self, count: int, snakes: Optional[List["Snake"]] = None) -> None:
        """Spawn initial food items at random positions.

        Args:
            count: Number of food items to spawn
            snakes: Optional list of snakes to avoid when spawning
        """
        self.food.clear()
        self._corpse_positions.clear()
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
        # Snap to the segment lattice so cell-exact pickup matches the legacy
        # radius test (see mechanics_constants.snap_to_cell).
        return snap_to_cell(
            (
                random.randint(
                    self.wall_thickness, self.game_width - self.wall_thickness - self.segment_size
                ),
                random.randint(
                    self.wall_thickness, self.game_height - self.wall_thickness - self.segment_size
                ),
            ),
            self.segment_size,
        )

    def _position_overlaps_food(self, position: Tuple[int, int]) -> bool:
        """Return whether a position occupies the same cell as existing food.

        Cell-exact (matches the pickup/collision predicate). On the shared
        segment lattice this equals the legacy ``distance < segment_size`` test,
        but it never over-suppresses corpse/trail pellets that merely land near
        (but not on) an existing pellet.
        """
        return any(same_cell(position, food_pos, self.segment_size) for food_pos in self.food)

    def add_food(self, position: Tuple[int, int], corpse: bool = False) -> bool:
        """Add food if it does not overlap existing food.

        Args:
            position: (x, y) position for the new pellet
            corpse: Mark the pellet corpse-class (kill corpse / boost trail).
                Corpse-class food is exempt from the ambient max_food cap.
                Callers only pass True under mechanics v2.

        Returns:
            True if the pellet was added
        """
        if self._position_overlaps_food(position):
            return False
        self.food.append(position)
        if corpse and CORPSE_EXEMPT_FROM_CAP_V2:
            self._corpse_positions.add(position)
            # Bound corpse accumulation: once over the cap, evict the oldest
            # corpse pellet(s). Same rule + target as BatchSim._add_food, so the
            # ordered food list stays byte-identical (parity gate depends on it).
            cap = corpse_food_cap(self.max_food)
            while len(self._corpse_positions) > cap:
                if evict_oldest_corpse(self.food, self._corpse_positions) is None:
                    break
        return True

    def _find_spawn_position(
        self, snakes: Optional[List["Snake"]] = None
    ) -> Optional[Tuple[int, int]]:
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
        """Ensure the AMBIENT food count stays at max_food level.

        This should be called each game frame to maintain food supply.
        Corpse-class pellets are ignored when computing the deficit, so a
        large corpse never suppresses baseline spawns (only relevant under
        mechanics v2; at v1 all food is ambient and this is the legacy
        ``max_food - len(food)`` top-up).

        Args:
            snakes: List of snakes to avoid when spawning

        Returns:
            Number of new food items spawned
        """
        deficit = self.max_food - self.ambient_count
        if deficit > 0:
            return self.spawn(deficit, snakes)
        return 0

    def consume_at(self, position: Tuple[int, int], radius: int) -> bool:
        """Check and consume food at position.

        Cell-exact: a pellet is eaten iff it occupies the same integer cell
        (position // radius) as the given position. Equivalent to the legacy
        radius test on the segment lattice.

        Args:
            position: (x, y) position to check for food
            radius: Cell size for the pickup check (typically segment_size)

        Returns:
            True if food was consumed, False otherwise
        """
        eaten = [f for f in self.food if same_cell(f, position, radius)]
        if eaten:
            self.food = [f for f in self.food if f not in eaten]
            self._corpse_positions.difference_update(eaten)
            return True
        return False

    def trim_ambient(self, target: int) -> int:
        """Trim ambient (cap-subject) pellets down to ``target``, keeping corpse food.

        The canonical way for external callers (e.g. the web food-target control)
        to shrink the board: it removes the newest ambient pellets only, never
        corpse-class food (kill corpses / boost trail), so mechanics-v2 kill
        economics stay visible. Keeps ``_corpse_positions`` consistent with the
        live list, so it does not leak the way a raw ``del food[...]`` does.

        Args:
            target: Desired ambient pellet count.

        Returns:
            Number of ambient pellets removed.
        """
        overage = self.ambient_count - target
        if overage <= 0:
            return 0
        removed = 0
        result = list(self.food)
        i = len(result) - 1
        while i >= 0 and removed < overage:
            if result[i] not in self._corpse_positions:
                result.pop(i)
                removed += 1
            i -= 1
        self.food = result
        # Drop any corpse positions no longer on the board (defensive: keeps the
        # set from leaking across long sessions).
        self._corpse_positions.intersection_update(self.food)
        return removed

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
        self._corpse_positions.clear()

    @property
    def count(self) -> int:
        """Get current food count."""
        return len(self.food)

    @property
    def corpse_count(self) -> int:
        """Number of corpse-class pellets currently on the board.

        Counted against the live food list so external trims of ``food``
        (e.g. the web session's food-target control) stay consistent.
        """
        if not self._corpse_positions:
            return 0
        return sum(1 for f in self.food if f in self._corpse_positions)

    @property
    def ambient_count(self) -> int:
        """Number of ambient (cap-subject) pellets currently on the board."""
        return len(self.food) - self.corpse_count

    def __len__(self) -> int:
        """Get current food count."""
        return len(self.food)

    def __iter__(self):
        """Iterate over food positions."""
        return iter(self.food)

    def __repr__(self) -> str:
        return f"FoodManager(count={len(self.food)}, max={self.max_food})"
