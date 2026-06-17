"""Base snake class with shared behavior."""

import math
from typing import Dict, List, Optional, Tuple

import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig, StateIndices
from src.game.game_logic import GameLogic


class Snake:
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

    def get_state(
        self,
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
        update_enemy_memory: bool = True,
    ) -> torch.Tensor:
        """
        Get state representation for neural network input.

        Generates a state vector containing:
        - Direction (4): One-hot encoded
        - Length (1): Normalized
        - Food features (19): Relative position + density map
        - Danger map (16): Obstacle proximity per sector
        - Boundary distances (4): Distance to each wall
        - Enemy features (10): Nearest/2nd nearest enemy position, heading, trend, kill opportunity
        - Per-action danger (3): Danger score for left/straight/right actions

        Args:
            other_snakes: List of all snakes in the game
            food: List of food positions
            update_enemy_memory: Whether to update the nearest-enemy trend
                baseline after building this state. Replay next-state capture
                should leave this False so the next action state sees the same
                trend signal.

        Returns:
            State tensor on the snake's device (size = GameConfig.INPUT_SIZE)
        """
        head_x, head_y = self.head
        state: List[float] = []

        # Direction (one-hot encoded)
        direction_state = [0.0] * 4
        try:
            direction_idx = GameConfig.ACTIONS.index(self.direction)
            direction_state[direction_idx] = 1.0
        except ValueError:
            direction_state[0] = 1.0
        state.extend(direction_state)

        # Normalized logical length (clamped to 1.0). This uses the game-rule
        # length budget, not the currently filled body segments, so freshly
        # eaten growth and boost availability stay aligned in the observation.
        normalized_length = min(self._logical_length() / GameConfig.MAX_LENGTH, 1.0)
        state.append(normalized_length)

        # Food features
        food_features = self._get_enhanced_food_state(
            food, head_x, head_y, num_sectors=GameConfig.NUM_SECTORS
        )
        state.extend(food_features)

        # Danger map
        danger_map = self._get_danger_map(other_snakes, num_sectors=GameConfig.NUM_SECTORS)
        state.extend(danger_map)

        # Boundary distances (normalized) - 4 features at indices 40-43.
        # Rectangular and circular arenas both expose left/right/top/bottom
        # distance-to-boundary semantics so replay state contracts stay stable.
        if GameConfig.ARENA_TYPE == "circular":
            cx, cy, radius = GameLogic.get_circular_arena(self.game_width, self.game_height)
            radius = max(float(radius), 1.0)
            dx_c = head_x - cx
            dy_c = head_y - cy
            horizontal_extent = math.sqrt(max(radius**2 - dy_c**2, 0.0))
            vertical_extent = math.sqrt(max(radius**2 - dx_c**2, 0.0))
            left_edge = cx - horizontal_extent
            right_edge = cx + horizontal_extent
            top_edge = cy - vertical_extent
            bottom_edge = cy + vertical_extent
            diameter = radius * 2.0
            dist_left = max(0.0, min((head_x - left_edge) / diameter, 1.0))
            dist_right = max(0.0, min((right_edge - head_x) / diameter, 1.0))
            dist_top = max(0.0, min((head_y - top_edge) / diameter, 1.0))
            dist_bottom = max(0.0, min((bottom_edge - head_y) / diameter, 1.0))
            state.extend([dist_left, dist_right, dist_top, dist_bottom])
        else:
            dist_left = head_x / self.game_width
            dist_right = (self.game_width - head_x) / self.game_width
            dist_top = head_y / self.game_height
            dist_bottom = (self.game_height - head_y) / self.game_height
            state.extend([dist_left, dist_right, dist_top, dist_bottom])

        # Enemy features (10 values: nearest + heading + trend + 2nd nearest + kill opportunity)
        enemy_features = self._get_nearest_enemy_features(
            other_snakes,
            update_enemy_memory=update_enemy_memory,
        )
        state.extend(enemy_features)

        # Per-action danger signals (3 values: left, straight, right)
        per_action_danger = self._get_per_action_danger(other_snakes)
        state.extend(per_action_danger)

        # Boost availability (1 value)
        boost_state = self._get_boost_state()
        state.extend(boost_state)

        if len(state) != GameConfig.INPUT_SIZE:
            raise ValueError(
                f"State dimension mismatch. Expected {GameConfig.INPUT_SIZE}, got {len(state)}"
            )

        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _get_enhanced_food_state(
        self, food: List[Tuple[int, int]], head_x: int, head_y: int, num_sectors: int = 16
    ) -> List[float]:
        """
        Calculate food-related state features (colab-compatible format).

        Uses colab's normalization scheme:
        - Food position: normalized by max(game_width, game_height) → range [-1, 1]
        - Food distance: normalized by danger_distance * 2 (in grid units)
        - Food density: count-based (count / max_food) → range [0, ~0.1]

        Args:
            food: List of food positions
            head_x: Snake head x position
            head_y: Snake head y position
            num_sectors: Number of sectors for density map

        Returns:
            List of [rel_x, rel_y, norm_dist] + density_map
        """
        if not food:
            # No food: rel_x=0, rel_y=0, distance=1.0 (maximally far), density=0
            return [0.0, 0.0, 1.0] + [0.0] * num_sectors

        # Find nearest food
        nearest = min(food, key=lambda f: (f[0] - head_x) ** 2 + (f[1] - head_y) ** 2)
        dx, dy = nearest[0] - head_x, nearest[1] - head_y
        dist = math.sqrt(dx**2 + dy**2)

        # Colab-compatible normalization: use max dimension (matches colab's grid-based approach)
        max_dim = max(self.game_width, self.game_height)
        rel_x = (dx / max_dim) if max_dim else 0.0
        rel_y = (dy / max_dim) if max_dim else 0.0

        # Food distance: normalize by board diagonal so full range maps to [0, 1]
        board_diagonal = math.sqrt(self.game_width**2 + self.game_height**2)
        norm_dist = min(dist / board_diagonal, 1.0) if board_diagonal > 0 else 0.0

        # Food density per sector, normalized to [0, 1] range
        density_map: List[float] = [0.0] * num_sectors
        for fx, fy in food:
            ddx, ddy = fx - head_x, fy - head_y
            sector = self._angle_to_sector(ddx, ddy, num_sectors)
            density_map[sector] += 1.0

        # Normalize by expected food per sector for this environment. Actor and
        # curriculum worlds can scale max_food, so global MAX_FOOD would make
        # dense local food supplies look artificially sparse.
        expected_per_sector = max(self.food_capacity / num_sectors, 1.0)
        density_map = [min(d / expected_per_sector, 1.0) for d in density_map]

        return [rel_x, rel_y, norm_dist] + density_map

    def _get_danger_map(self, other_snakes: List["Snake"], num_sectors: int = 16) -> List[float]:
        """
        Calculate danger map for obstacles in each sector (colab-compatible format).

        Uses detailed wall sampling for better wall awareness.
        Output is clamped to [0, 1.0] to match colab's expected range.

        Danger hierarchy (preserved after clamping):
        - Wall: base_danger * 2.0, then clamped → effectively 1.0 when close
        - Other snakes: base_danger * 1.5, then clamped → max 1.0
        - Own body: base_danger * 1.5, then clamped → max 1.0

        Args:
            other_snakes: List of all snakes in the game
            num_sectors: Number of sectors for the danger map

        Returns:
            List of danger values per sector (0.0 to 1.0, colab-compatible)
        """
        head_x, head_y = self.head
        danger_map: List[float] = [0.0] * num_sectors
        danger_count: List[int] = [0] * num_sectors  # Track obstacle count per sector
        max_dist_pixels = GameConfig.DANGER_MAX_DISTANCE * self.segment_size

        def update_danger(
            seg_x: float,
            seg_y: float,
            is_self: bool = False,
            is_head: bool = False,
            is_wall: bool = False,
        ) -> None:
            dx, dy = seg_x - head_x, seg_y - head_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > max_dist_pixels:
                return
            sector = Snake._angle_to_sector(dx, dy, num_sectors)
            base_danger = max(0.0, 1.0 - dist / max_dist_pixels)

            # Different danger levels for different obstacles
            if is_wall:
                danger_val = base_danger * 2.0  # Walls = MAXIMUM danger (instant death!)
            elif is_self:
                danger_val = base_danger * 1.5  # Own body (same weight as other snakes)
            else:
                danger_val = base_danger * 1.5  # Other snakes = 1.5x danger

            if danger_val > 0:
                danger_count[sector] += 1
            danger_map[sector] = max(danger_map[sector], danger_val)

        # Own body danger. Skip the adjacent neck and a vacating tail so the
        # sector map agrees with immediate collision/mask semantics.
        self_obstacles = list(self.segments[2:])
        if len(self.segments) >= self.length and self_obstacles:
            self_obstacles = self_obstacles[:-1]
        for segment in self_obstacles:
            update_danger(segment[0], segment[1], is_self=True, is_head=False)

        # Other snakes danger
        for snake in other_snakes:
            if snake != self and snake.is_alive:
                update_danger(snake.head[0], snake.head[1], is_self=False, is_head=True)
                obstacle_segments = list(snake.segments[1:])
                if len(snake.segments) >= snake.length and obstacle_segments:
                    obstacle_segments = obstacle_segments[:-1]
                for segment in obstacle_segments:
                    update_danger(segment[0], segment[1], is_self=False, is_head=False)

        # Wall danger - sample multiple points along nearby walls
        if GameConfig.USE_BOUNDARY_AS_DANGER:
            if GameConfig.ARENA_TYPE == "circular":
                cx, cy, radius = GameLogic.get_circular_arena(self.game_width, self.game_height)
                dx_c = head_x - cx
                dy_c = head_y - cy
                dist_from_center = math.sqrt(dx_c**2 + dy_c**2)
                dist_to_edge = radius - dist_from_center

                if dist_to_edge < max_dist_pixels + self.segment_size:
                    # Sample points along the nearby arc of the circular boundary
                    num_samples = max(8, int(2 * math.pi * radius / self.segment_size / 4))
                    for i in range(num_samples):
                        angle = 2 * math.pi * i / num_samples
                        bx = cx + (radius + self.segment_size) * math.cos(angle)
                        by = cy + (radius + self.segment_size) * math.sin(angle)
                        d = math.sqrt((bx - head_x) ** 2 + (by - head_y) ** 2)
                        if d <= max_dist_pixels:
                            update_danger(bx, by, is_wall=True)
            else:
                wall_sample_step = self.segment_size
                wall_offset = self.segment_size  # How far "into" the wall to sample

                # Left wall
                if head_x < max_dist_pixels + wall_offset:
                    for offset_y in range(
                        -int(max_dist_pixels), int(max_dist_pixels) + 1, wall_sample_step
                    ):
                        sample_y = head_y + offset_y
                        if 0 <= sample_y < self.game_height:
                            update_danger(-wall_offset, sample_y, is_wall=True)

                # Right wall
                if self.game_width - head_x < max_dist_pixels + wall_offset:
                    for offset_y in range(
                        -int(max_dist_pixels), int(max_dist_pixels) + 1, wall_sample_step
                    ):
                        sample_y = head_y + offset_y
                        if 0 <= sample_y < self.game_height:
                            update_danger(self.game_width + wall_offset, sample_y, is_wall=True)

                # Top wall
                if head_y < max_dist_pixels + wall_offset:
                    for offset_x in range(
                        -int(max_dist_pixels), int(max_dist_pixels) + 1, wall_sample_step
                    ):
                        sample_x = head_x + offset_x
                        if 0 <= sample_x < self.game_width:
                            update_danger(sample_x, -wall_offset, is_wall=True)

                # Bottom wall
                if self.game_height - head_y < max_dist_pixels + wall_offset:
                    for offset_x in range(
                        -int(max_dist_pixels), int(max_dist_pixels) + 1, wall_sample_step
                    ):
                        sample_x = head_x + offset_x
                        if 0 <= sample_x < self.game_width:
                            update_danger(sample_x, self.game_height + wall_offset, is_wall=True)

        # Blend max-danger with density: more obstacles = slightly higher danger
        for i in range(num_sectors):
            if danger_count[i] > 1:
                # Add up to 0.2 for high-density sectors (5+ obstacles)
                density_bonus = min(danger_count[i] / 5.0, 1.0) * 0.2
                danger_map[i] = min(danger_map[i] + density_bonus, 2.0)  # Pre-clamp max

        # Clamp to [0, 1.0] for colab compatibility
        # This preserves threat hierarchy after clamping.
        danger_map = [min(d, 1.0) for d in danger_map]

        return danger_map

    def _get_nearest_enemy_features(
        self,
        other_snakes: List["Snake"],
        update_enemy_memory: bool = True,
    ) -> List[float]:
        """Get features about enemy snakes (10D).

        Args:
            other_snakes: List of all snakes in the game.
            update_enemy_memory: Whether to store the nearest-enemy distance/id
                as the baseline for the next state calculation.

        Returns:
            10 features:
            [0-2] nearest: rel_x, rel_y, rel_size
            [3-4] nearest heading: dx, dy
            [5] distance trend: +1 closing, -1 separating, 0 unchanged
            [6-8] 2nd nearest: rel_x, rel_y, rel_size
            [9] kill opportunity: 1.0 if adjacent to enemy's projected path
        """
        head_x, head_y = self.head
        max_dim = max(self.game_width, self.game_height)

        # Sort enemies by distance
        enemies = []
        for snake in other_snakes:
            if snake == self or not snake.is_alive:
                continue
            dx = snake.head[0] - head_x
            dy = snake.head[1] - head_y
            dist = math.sqrt(dx**2 + dy**2)
            enemies.append((snake, dx, dy, dist))
        enemies.sort(key=lambda e: e[3])

        features = [0.0] * 10  # Default: no enemies

        if enemies:
            snake, dx, dy, dist = enemies[0]
            # Basic position (existing)
            features[0] = dx / max_dim
            features[1] = dy / max_dim
            features[2] = (
                min(
                    snake._logical_length() / max(self._logical_length(), 1),
                    2.0,
                )
                / 2.0
            )
            # Heading (new)
            features[3] = float(snake.direction[0])  # Already -1/0/1
            features[4] = float(snake.direction[1])
            # Distance trend (new)
            prev_dist = self._prev_nearest_enemy_dist
            if math.isinf(prev_dist) or self._prev_nearest_enemy_id != snake.id:
                features[5] = 0.0
            else:
                features[5] = 1.0 if dist < prev_dist else (-1.0 if dist > prev_dist else 0.0)
            if update_enemy_memory:
                self._prev_nearest_enemy_dist = dist
                self._prev_nearest_enemy_id = snake.id
            # Kill opportunity (new)
            enemy_next_x = snake.head[0] + snake.direction[0] * snake.segment_size
            enemy_next_y = snake.head[1] + snake.direction[1] * snake.segment_size
            kill_dist = math.sqrt((head_x - enemy_next_x) ** 2 + (head_y - enemy_next_y) ** 2)
            features[9] = 1.0 if kill_dist < self.segment_size * 3 else 0.0
        else:
            if update_enemy_memory:
                self._prev_nearest_enemy_dist = float("inf")
                self._prev_nearest_enemy_id = None

        if len(enemies) >= 2:
            snake2, dx2, dy2, dist2 = enemies[1]
            features[6] = dx2 / max_dim
            features[7] = dy2 / max_dim
            features[8] = (
                min(
                    snake2._logical_length() / max(self._logical_length(), 1),
                    2.0,
                )
                / 2.0
            )

        return features

    def _get_per_action_danger(self, other_snakes: List["Snake"]) -> List[float]:
        """Compute immediate danger score for each relative action (left, straight, right).

        For each possible action, simulates one step in that direction and checks
        for wall collision, self-collision, and proximity to other snake bodies.

        Returns:
            [danger_left, danger_straight, danger_right] each in [0.0, 1.0]
        """
        head_x, head_y = self.head
        dangers = []
        # After a one-step move, old head and old first body segment become
        # adjacent to the new head and cannot be self-collision targets. If
        # the snake is not still growing, the old tail moves away before
        # collision checks and should not count as immediate danger either.
        self_collision_segments = list(self.segments[2:])
        if len(self.segments) >= self.length and self_collision_segments:
            self_collision_segments = self_collision_segments[:-1]

        for relative_action in range(3):  # 0=left, 1=straight, 2=right
            new_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
            new_x = head_x + new_dir[0] * self.segment_size
            new_y = head_y + new_dir[1] * self.segment_size

            danger = 0.0

            # Wall collision check
            if GameConfig.ARENA_TYPE == "circular":
                cx, cy, radius = GameLogic.get_circular_arena(self.game_width, self.game_height)
                ddx = new_x - cx
                ddy = new_y - cy
                wall_hit = (ddx * ddx + ddy * ddy) > radius**2
            else:
                wall_hit = (
                    new_x < 0 or new_x >= self.game_width or new_y < 0 or new_y >= self.game_height
                )
            if wall_hit:
                danger = 1.0
            else:
                # Self-collision check (would new head hit own body?)
                if self_collision_segments:
                    for seg in self_collision_segments:
                        dist = math.sqrt((new_x - seg[0]) ** 2 + (new_y - seg[1]) ** 2)
                        if dist < self.segment_size:
                            danger = 1.0
                            break

                # Other snake collision check
                if danger < 1.0:
                    for snake in other_snakes:
                        if snake == self or not snake.is_alive:
                            continue
                        obstacle_segments = list(snake.segments)
                        if len(snake.segments) > 1 and len(snake.segments) >= snake.length:
                            obstacle_segments = obstacle_segments[:-1]
                        for seg in obstacle_segments:
                            dist = math.sqrt((new_x - seg[0]) ** 2 + (new_y - seg[1]) ** 2)
                            if dist < self.segment_size:
                                danger = 1.0
                                break
                        if danger >= 1.0:
                            break

                # Proximity danger (softer signal for nearby obstacles)
                if danger < 1.0:
                    max_check_dist = self.segment_size * 3
                    min_obstacle_dist = max_check_dist

                    # Check walls
                    if GameConfig.ARENA_TYPE == "circular":
                        cx, cy, radius = GameLogic.get_circular_arena(
                            self.game_width,
                            self.game_height,
                        )
                        d_center = math.sqrt((new_x - cx) ** 2 + (new_y - cy) ** 2)
                        min_wall = max(0, radius - d_center)
                    else:
                        wall_dists = [
                            new_x,
                            self.game_width - new_x,
                            new_y,
                            self.game_height - new_y,
                        ]
                        min_wall = min(wall_dists)
                    min_obstacle_dist = min(min_obstacle_dist, min_wall)

                    # Check snake bodies
                    for snake in other_snakes:
                        if snake == self:
                            for seg in self_collision_segments:
                                d = math.sqrt((new_x - seg[0]) ** 2 + (new_y - seg[1]) ** 2)
                                min_obstacle_dist = min(min_obstacle_dist, d)
                        elif snake.is_alive:
                            obstacle_segments = list(snake.segments)
                            if len(snake.segments) > 1 and len(snake.segments) >= snake.length:
                                obstacle_segments = obstacle_segments[:-1]
                            for seg in obstacle_segments:
                                d = math.sqrt((new_x - seg[0]) ** 2 + (new_y - seg[1]) ** 2)
                                min_obstacle_dist = min(min_obstacle_dist, d)

                    # Soft danger: closer obstacles = higher danger
                    if min_obstacle_dist < max_check_dist:
                        # Reserve 1.0 for hard collisions so state-derived
                        # masks do not hide safe but wall-adjacent moves.
                        proximity_danger = 1.0 - min_obstacle_dist / max_check_dist
                        danger = max(danger, min(proximity_danger, 0.95))

            dangers.append(danger)

        return dangers

    def _get_boost_state(self) -> List[float]:
        """Get boost availability (1.0 if can boost, 0.0 if not)."""
        can_boost = 1.0 if self.length >= GameConfig.MIN_BOOST_LENGTH else 0.0
        return [can_boost]

    def move(self) -> None:
        """Move snake one step in current direction, with optional boost.

        When boosting, moves two segments per frame and pays a length cost.
        Collision detection is centralized in GameState.handle_collisions().
        """
        self.last_move_positions = []
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
                        self.segments.pop()

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
        self.respawn_timer = 0
        self.frames_since_food = 0  # Reset starvation counter
        self._prev_nearest_enemy_dist = float("inf")  # Reset enemy distance tracking
        self._prev_nearest_enemy_id = None
        self.is_boosting = False
        self.boost_frames = 0
        self.last_move_positions = []

    def calculate_reward(
        self,
        ate_food: bool,
        collided: bool,
        old_state: Optional[torch.Tensor],
        new_state: Optional[torch.Tensor],
        other_snakes: List["Snake"],
        food: List[Tuple[int, int]],
        frame_kills: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """
        Calculate reward with clean signals and proper incentives.

        Uses centralized reward constants from GameConfig for easy tuning.

        Fixed issues: wall oscillation exploit, food avoidance being positive,
        weak danger penalties, missing starvation penalty.

        Args:
            ate_food: Whether food was eaten this step
            collided: Whether snake died this step
            old_state: Previous state tensor (or None)
            new_state: Current state tensor (or None)
            other_snakes: List of all snakes
            food: List of food positions
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids] (or None)

        Returns:
            Reward value (clamped to [REWARD_MIN, REWARD_MAX])
        """
        # Terminal: death penalty (optionally scaled by mass, so dying while large
        # costs more and the agent learns to protect accumulated length).
        if collided:
            scale = GameConfig.REWARD_DEATH_LENGTH_SCALE
            if scale:
                max_len = max(int(getattr(GameConfig, "MAX_LENGTH", 150)), 1)
                norm_len = min(self.length / max_len, 2.0)
                return GameConfig.REWARD_DEATH * (1.0 + scale * norm_len)
            return GameConfig.REWARD_DEATH

        reward = 0.0
        if ate_food:
            self.frames_since_food = 0
            reward += GameConfig.REWARD_FOOD_BASE
        else:
            # Track frames since last food (starvation mechanic)
            self.frames_since_food += 1

            # Shaping rewards (when not eating or dying)
            if new_state is not None:
                # 1) Proportional food-distance shaping: reward scales with distance change
                if old_state is not None:
                    prev_food_dist = float(old_state[StateIndices.FOOD_DISTANCE])
                    curr_food_dist = float(new_state[StateIndices.FOOD_DISTANCE])
                    delta = prev_food_dist - curr_food_dist  # positive if closer
                    # Scale by TOWARD_FOOD/typical-step-delta, cap at ±0.1
                    alpha = GameConfig.REWARD_TOWARD_FOOD / 0.01
                    food_shaping = max(min(alpha * delta, 0.1), -0.1)
                    reward += food_shaping

                # 2) Wall proximity - penalty ONLY (no escape reward oscillation exploit)
                reward += self._calculate_wall_awareness_penalty(new_state)

                # 3) Immediate danger penalty. Use per-action danger instead of
                # the global sector map so the snake is not punished for harmless
                # obstacles behind it, such as its own adjacent neck after growth.
                reward += self._calculate_action_danger_penalty(new_state)

            # 4) Starvation penalty (encourages food hunting, prevents looping)
            if self.frames_since_food > GameConfig.STARVATION_START_FRAME:
                frames_starving = self.frames_since_food - GameConfig.STARVATION_START_FRAME
                starvation_factor = min(frames_starving / GameConfig.STARVATION_MAX_FRAMES, 1.0)
                reward -= GameConfig.STARVATION_MAX_PENALTY * starvation_factor

            # 6) Survival reward: small positive signal for staying alive
            reward += GameConfig.REWARD_SURVIVAL

        # 5) Inter-snake interaction rewards (multi-agent competitive/cooperative behavior)
        reward += self._calculate_interaction_reward(other_snakes, frame_kills=frame_kills)

        # 7) Boost cost: penalize burning body segments via boost so that eating
        # then immediately boosting the mass away is not reward-free. This fixes
        # the boost-abuse exploit (policies otherwise boost ~89% of the time).
        # Triggers only on the frames a boost action actually burns a segment.
        boost_seg_penalty = GameConfig.REWARD_BOOST_SEGMENT
        if boost_seg_penalty:
            action = getattr(self, "_pre_collision_action", None)
            prev_len = getattr(self, "_reward_prev_length", self.length)
            lost = prev_len - self.length
            if action is not None and int(action) >= 3 and lost > 0 and not ate_food:
                reward -= boost_seg_penalty * float(lost)
        self._reward_prev_length = self.length

        # Clamp to reasonable range (safety net)
        return max(min(reward, GameConfig.REWARD_MAX), GameConfig.REWARD_MIN)

    def _get_wall_awareness_distance(self, state: torch.Tensor) -> float:
        """Return the normalized distance signal used for wall-awareness reward."""
        boundary_slice = slice(StateIndices.BOUNDARY_LEFT, StateIndices.BOUNDARY_BOTTOM + 1)
        boundary_distances = state[boundary_slice]
        return float(min(boundary_distances))

    def _calculate_wall_awareness_penalty(self, state: torch.Tensor) -> float:
        """Return the smooth wall-proximity penalty for the current arena type."""
        min_boundary_dist = self._get_wall_awareness_distance(state)
        if min_boundary_dist >= GameConfig.WALL_AWARENESS_THRESHOLD:
            return 0.0

        # Linear interpolation: REWARD_WALL_DANGER at distance 0, 0 at threshold.
        return (
            GameConfig.REWARD_WALL_DANGER
            * (GameConfig.WALL_AWARENESS_THRESHOLD - min_boundary_dist)
            / GameConfig.WALL_AWARENESS_THRESHOLD
        )

    def _calculate_action_danger_penalty(self, state: torch.Tensor) -> float:
        """Return danger penalty from immediate action risk."""
        danger_start = StateIndices.PER_ACTION_DANGER_START
        danger_end = StateIndices.PER_ACTION_DANGER_END
        danger_slice = state[danger_start:danger_end]
        if len(danger_slice) == 0:
            return 0.0

        # Average pressure penalizes constrained states without treating a
        # single blocked direction as a critical state when escape moves exist.
        action_danger = float(danger_slice.float().mean())
        danger_thresholds = [
            (GameConfig.DANGER_CRITICAL_THRESHOLD, GameConfig.REWARD_DANGER_CRITICAL),
            (GameConfig.DANGER_HIGH_THRESHOLD, GameConfig.REWARD_DANGER_HIGH),
            (GameConfig.DANGER_MEDIUM_THRESHOLD, GameConfig.REWARD_DANGER_MEDIUM),
        ]
        for threshold, penalty in danger_thresholds:
            if action_danger > threshold:
                return penalty
        return 0.0

    def _calculate_interaction_reward(
        self,
        other_snakes: List["Snake"],
        frame_kills: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """
        Calculate inter-snake interaction reward.

        Uses collision-pair tracking (frame_kills) for accurate kill attribution
        when available. Falls back to proximity heuristic for backward compatibility.

        Args:
            other_snakes: List of all snakes in the game
            frame_kills: Dict mapping killer_snake_id → [victim_snake_ids] (or None)

        Returns:
            Interaction reward value
        """
        # No interaction rewards in single-snake mode
        if len(other_snakes) <= 1:
            return 0.0

        interaction_reward = 0.0

        # Use accurate kill tracking when available. An empty dict is an
        # authoritative "no kills this frame" signal from GameState.
        if frame_kills is not None:
            if self.id not in frame_kills:
                return 0.0

            victim_ids = frame_kills[self.id]
            for snake in other_snakes:
                if snake.id in victim_ids:
                    victim_length = snake._logical_length()
                    kill_reward = (
                        GameConfig.REWARD_KILL_BASE
                        + GameConfig.REWARD_KILL_LENGTH_SCALE * victim_length
                    )
                    interaction_reward += min(kill_reward, GameConfig.REWARD_KILL_MAX)
            return interaction_reward

        # Fallback: proximity-based heuristic (backward compatibility)
        for snake in other_snakes:
            if snake == self:
                continue

            # Reward only when another snake just died AND was close enough
            # that we likely caused it (within 2 segment sizes of our body)
            if not snake.is_alive and snake.respawn_timer == GameConfig.FRAME_RATE:
                dead_head = snake.head
                for segment in self.segments:
                    dist = math.sqrt(
                        (dead_head[0] - segment[0]) ** 2 + (dead_head[1] - segment[1]) ** 2
                    )
                    if dist < self.segment_size * 2:
                        interaction_reward += 1.0
                        break

        return interaction_reward
