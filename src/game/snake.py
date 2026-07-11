"""Base snake class: movement, lifecycle, and shared entity state.

State (observation) building and reward computation are factored into mixins —
see snake_state.py (SnakeStateMixin) and snake_reward.py (SnakeRewardMixin) —
which this class composes. Keeps each concern in its own module.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig
from src.core.mechanics_constants import BOOST_DROPS_TRAIL_V2
from src.game.snake_reward import SnakeRewardMixin
from src.game.snake_state import SnakeStateMixin


class Snake(SnakeStateMixin, SnakeRewardMixin):
    """
    Base class for all snake types (AI and Human).

    Provides shared behavior for movement, state representation,
    collision detection, and reward calculation.
    """

    # Color name mapping (shared by all snakes)
    COLOR_NAMES: Dict[Tuple[int, int, int], str] = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Cyan",
        (255, 165, 0): "Orange",
        (128, 0, 128): "Purple",
    }

    def __init__(
        self,
        id: int,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        segment_size: int,
        game_width: int,
        game_height: int,
        food_capacity: Optional[int] = None,
    ) -> None:
        """
        Initialize a snake.

        Args:
            id: Unique snake identifier
            color: RGB color tuple
            start_pos: Starting position (x, y)
            segment_size: Size of each body segment in pixels
            game_width: Game area width
            game_height: Game area height
            food_capacity: Effective max food count for this environment
        """
        self.id: int = id
        self.color: Tuple[int, int, int] = color
        self.segments: List[Tuple[int, int]] = [start_pos]
        self.direction: Tuple[int, int] = (1, 0)
        self.is_alive: bool = True
        self.length: int = 1
        self.segment_size: int = segment_size
        self.game_width: int = game_width
        self.game_height: int = game_height
        self.food_capacity: int = max(
            0,
            int(food_capacity if food_capacity is not None else GameConfig.MAX_FOOD),
        )
        self.respawn_timer: int = 0
        self.frames_since_food: int = 0  # Track frames since last food for starvation
        self._prev_nearest_enemy_dist: float = float("inf")  # For enemy distance trend
        self._prev_nearest_enemy_id: Optional[int] = None
        self.is_boosting: bool = False
        self.boost_frames: int = 0  # Counter for length cost
        self.last_move_positions: List[Tuple[int, int]] = []
        # Mechanics v2: tail cells vacated by boost segment burns this frame.
        # GameState.update() converts these into corpse-class food pellets.
        self.pending_trail_pellets: List[Tuple[int, int]] = []
        # Use centralized device manager for consistent device handling
        self.device: torch.device = DeviceManager.get_device()

    @property
    def head(self) -> Tuple[int, int]:
        """Get the head position of the snake."""
        return self.segments[0]

    @staticmethod
    def _angle_to_sector(dx: float, dy: float, num_sectors: int) -> int:
        """
        Convert delta (dx, dy) to sector index.

        Args:
            dx: X delta from reference point
            dy: Y delta from reference point
            num_sectors: Number of sectors (typically 16)

        Returns:
            Sector index in range [0, num_sectors)
        """
        angle = math.atan2(dy, dx)
        return int(((angle + math.pi) / (2 * math.pi)) * num_sectors) % num_sectors

    @classmethod
    def color_to_name(cls, color: Tuple[int, int, int]) -> str:
        """
        Convert RGB color tuple to human-readable name.

        Args:
            color: RGB tuple (r, g, b)

        Returns:
            Closest matching color name
        """
        r, g, b = color
        min_distance = float("inf")
        closest_color = "Unknown"
        for rgb, name in cls.COLOR_NAMES.items():
            distance = sum((a - b) ** 2 for a, b in zip((r, g, b), rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        return closest_color

    def _logical_length(self) -> int:
        """Return the rule-authoritative snake length, independent of body fill-in lag."""
        return max(1, int(self.length))

    def move(self) -> None:
        """Move snake one step in current direction, with optional boost.

        When boosting, moves two segments per frame and pays a length cost.
        Under mechanics v2 each boost segment burn records the vacated tail
        cell in ``pending_trail_pellets`` (mass transfer: GameState turns it
        into a corpse-class food pellet).
        Collision detection is centralized in GameState.handle_collisions().
        """
        self.last_move_positions = []
        self.pending_trail_pellets = []
        new_head = (
            self.head[0] + self.direction[0] * self.segment_size,
            self.head[1] + self.direction[1] * self.segment_size,
        )
        self.last_move_positions.append(new_head)

        self.segments.insert(0, new_head)
        if len(self.segments) > self.length:
            self.segments.pop()

        # If boosting, do a second step
        if self.is_boosting and self.length >= GameConfig.MIN_BOOST_LENGTH:
            new_head2 = (
                self.head[0] + self.direction[0] * self.segment_size,
                self.head[1] + self.direction[1] * self.segment_size,
            )
            self.last_move_positions.append(new_head2)
            self.segments.insert(0, new_head2)
            if len(self.segments) > self.length:
                self.segments.pop()

            # Length cost
            self.boost_frames += 1
            if self.boost_frames >= GameConfig.BOOST_LENGTH_COST_FRAMES:
                self.boost_frames = 0
                if self.length > 1:
                    self.length -= 1
                    if len(self.segments) > self.length:
                        burned_tail = self.segments.pop()
                        # v2: the burned segment becomes a trail pellet at the
                        # vacated tail cell instead of vanishing.
                        if BOOST_DROPS_TRAIL_V2 and GameConfig.MECHANICS_VERSION == 2:
                            self.pending_trail_pellets.append(burned_tail)

    def grow(self, amount: int = 1) -> None:
        """
        Increase snake length.

        Args:
            amount: Number of segments to add
        """
        self.length += amount

    def change_direction(self, new_direction: Tuple[int, int]) -> None:
        """
        Change snake direction if not a 180° turn.

        Args:
            new_direction: New direction tuple (dx, dy)
        """
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def die(self) -> None:
        """Mark snake as dead and set respawn timer."""
        self.is_alive = False
        self.respawn_timer = GameConfig.FRAME_RATE

    def respawn(self, new_pos: Tuple[int, int]) -> None:
        """
        Respawn snake at a new position.

        Args:
            new_pos: New starting position (x, y)
        """
        self.segments = [new_pos]
        self.direction = (1, 0)  # Reset direction to right
        self.is_alive = True
        self.length = 1
        # Reset boost-cost baseline so the first post-respawn boost isn't charged
        # for the (now-gone) pre-death length (mirrors soft_reset).
        self._reward_prev_length = 1
        self.respawn_timer = 0
        self.frames_since_food = 0  # Reset starvation counter
        self._prev_nearest_enemy_dist = float("inf")  # Reset enemy distance tracking
        self._prev_nearest_enemy_id = None
        self.is_boosting = False
        self.boost_frames = 0
        self.last_move_positions = []
        self.pending_trail_pellets = []
