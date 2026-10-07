"""State (observation) construction for Snake.

The 58-/61-D neural-network state vector and its feature helpers, factored out of
snake.py as a mixin so the game-entity, observation, and reward concerns live in
separate modules. Composed into Snake; `self` resolves at runtime via the MRO.
"""

import math
from typing import TYPE_CHECKING, List, Tuple

import torch

from src.core.game_config import GameConfig
from src.game.game_logic import GameLogic, exact_radius_sq
from src.game.snake_geometry import SnakeGeometry, geometry_for, near_obstacle_min_d2

if TYPE_CHECKING:  # avoid a runtime import cycle (Snake composes this mixin)
    from src.game.snake import Snake

# Free-space feature: bounded flood-fill caps. The BFS stops after FREE_SPACE_BFS_CAP
# reachable cells (a hard perf bound); the per-action value is normalized by an
# effective cap that scales with snake length, so the signal is "reachable space
# relative to my own body size" (value ~0.5 means a pocket about as big as the body).
FREE_SPACE_BFS_CAP = 160
FREE_SPACE_MIN_CAP = 32


class SnakeStateMixin:
    """Builds the neural-network observation vector for a Snake (see get_state)."""

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

        # Free-space ("don't trap yourself") features (3 values) — opt-in. Appended
        # last so 58-D contracts stay column-aligned with this 61-D layout.
        if GameConfig.USE_FREE_SPACE:
            state.extend(self._get_free_space_features(other_snakes))

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

        Normalization scheme:
        - Food position: normalized by max(game_width, game_height) → range [-1, 1]
        - Food distance: normalized by the board diagonal, clamped → range [0, 1]
        - Food density: count per sector normalized by the expected food per sector
          (food_capacity / num_sectors), clamped → range [0, 1]

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

        # Find nearest food (same key and first-minimum tie rule as
        # min(food, key=lambda f: (f[0] - head_x) ** 2 + (f[1] - head_y) ** 2)).
        nearest = None
        nearest_key = None
        for f in food:
            key = (f[0] - head_x) ** 2 + (f[1] - head_y) ** 2
            if nearest is None or key < nearest_key:
                nearest = f
                nearest_key = key
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
        # Snake._angle_to_sector inlined (same expression, same operation order).
        atan2 = math.atan2
        pi = math.pi
        two_pi = 2 * math.pi
        for fx, fy in food:
            ddx, ddy = fx - head_x, fy - head_y
            sector = int(((atan2(ddy, ddx) + pi) / two_pi) * num_sectors) % num_sectors
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
            is_wall: bool = False,
        ) -> None:
            dx, dy = seg_x - head_x, seg_y - head_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > max_dist_pixels:
                return
            sector = self._angle_to_sector(dx, dy, num_sectors)
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

        # Exact far-point skip (byte-identical): update_danger returns early when
        # sqrt(d2) > max_dist_pixels. On integer lattice offsets with an integer
        # max_dist_pixels whose square test is exact (exact_radius_sq), that is
        # d2 > max_dist_pixels**2, so far points are skipped without the call.
        far_sq = exact_radius_sq(max_dist_pixels)
        if far_sq is None:
            far_sq = -1  # disables the skip: every point takes the original path

        sqrt = math.sqrt
        atan2 = math.atan2
        pi = math.pi
        two_pi = 2 * math.pi

        def near_body(body: List[Tuple[int, int]], start: int, stop: int) -> None:
            """update_danger(x, y, is_self=...) for body[start:stop], inlined.

            Body points (self or other) both weigh 1.5, so is_self does not change
            the value. Points with d2 > far_sq would return early in update_danger.
            """
            for index in range(start, stop):
                sx, sy = body[index]
                dx = sx - head_x
                dy = sy - head_y
                d2 = dx**2 + dy**2
                if d2 > far_sq:
                    continue
                dist = sqrt(d2)
                if dist > max_dist_pixels:
                    continue
                sector = int(((atan2(dy, dx) + pi) / two_pi) * num_sectors) % num_sectors
                danger_val = max(0.0, 1.0 - dist / max_dist_pixels) * 1.5
                if danger_val > 0:
                    danger_count[sector] += 1
                danger_map[sector] = max(danger_map[sector], danger_val)

        # Own body danger. Skip the adjacent neck and a vacating tail so the
        # sector map agrees with immediate collision/mask semantics.
        body = self.segments
        stop = len(body)
        if stop >= self.length and stop > 2:
            stop -= 1
        if far_sq < 0:
            for index in range(2, stop):
                sx, sy = body[index]
                update_danger(sx, sy, is_self=True)
        else:
            near_body(body, 2, stop)

        # Other snakes danger
        for snake in other_snakes:
            if snake != self and snake.is_alive:
                if far_sq >= 0:
                    # Whole-snake skip: the danger points of another snake (head, then
                    # segments[1:] minus a vacating tail) are exactly its cached geometry's
                    # obstacle points; if their bounding box is more than max_dist_pixels
                    # away on an axis, every point has d2 > far_sq and would return early.
                    geometry = geometry_for(snake, self.segment_size)
                    min_x, max_x, min_y, max_y = geometry.bbox
                    if geometry.has_obstacles and (
                        head_x < min_x - max_dist_pixels
                        or head_x > max_x + max_dist_pixels
                        or head_y < min_y - max_dist_pixels
                        or head_y > max_y + max_dist_pixels
                    ):
                        continue
                update_danger(snake.head[0], snake.head[1], is_self=False)
                body = snake.segments
                stop = len(body)
                if stop >= snake.length and stop > 1:
                    stop -= 1
                if far_sq < 0:
                    for index in range(1, stop):
                        sx, sy = body[index]
                        update_danger(sx, sy, is_self=False)
                else:
                    near_body(body, 1, stop)

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

        circular = GameConfig.ARENA_TYPE == "circular"
        # Bucketed near-point queries (exact). Only obstacle distances below
        # max_check_dist = 3 * ss can change any output: the hard tests compare with
        # ss < 3 * ss, and the proximity value is min(max_check_dist, ...), whose value
        # a distance >= max_check_dist leaves unchanged. On integer coordinates,
        # d2 < (3 * ss)**2 implies |dx|, |dy| < 3 * ss, so such a point lies within 3
        # cells (cell = (x // ss, y // ss)) of the candidate head: the minimum over the
        # 7x7 cell neighbourhood equals the full minimum whenever the full minimum is
        # below (3 * ss)**2, and is otherwise discarded (both then have no effect).
        ss_int = self.segment_size
        near_sq = exact_radius_sq(ss_int * 3) if exact_radius_sq(ss_int) is not None else None
        # Obstacle points of every live other snake (head included, vacating tail
        # excluded when full): cached per-snake geometry on the fast path, else a flat
        # list in roster order; shared by the three candidates.
        other_points: List[Tuple[int, int]] = []
        other_geometries: List[SnakeGeometry] = []
        self_in_roster = False
        for snake in other_snakes:
            if snake == self:
                self_in_roster = True
                continue
            if not snake.is_alive:
                continue
            if near_sq is not None:
                geometry = geometry_for(snake, ss_int)
                if geometry.has_obstacles:
                    other_geometries.append(geometry)
                continue
            body = snake.segments
            if len(body) > 1 and len(body) >= snake.length:
                other_points.extend(body[:-1])
            else:
                other_points.extend(body)
        has_other_points = bool(other_geometries) or bool(other_points)

        self_cells: dict = {}
        if near_sq is not None:
            for px, py in self_collision_segments:
                key = (px // ss_int, py // ss_int)
                bucket = self_cells.get(key)
                if bucket is None:
                    self_cells[key] = [(px, py)]
                else:
                    bucket.append((px, py))

        def near_min(cells: dict, qx: int, qy: int) -> "int | None":
            best = None
            cx = qx // ss_int
            cy = qy // ss_int
            for kx in range(cx - 3, cx + 4):
                for ky in range(cy - 3, cy + 4):
                    bucket = cells.get((kx, ky))
                    if bucket is None:
                        continue
                    for px, py in bucket:
                        d2 = (qx - px) ** 2 + (qy - py) ** 2
                        if best is None or d2 < best:
                            best = d2
            if best is None or best >= near_sq:
                return None
            return best

        for relative_action in range(3):  # 0=left, 1=straight, 2=right
            new_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
            new_x = head_x + new_dir[0] * self.segment_size
            new_y = head_y + new_dir[1] * self.segment_size

            danger = 0.0

            # Wall collision check
            if circular:
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
                # Byte-identical restructuring of the original per-segment scans:
                # every test there is math.sqrt(d2) < ss or min(..., math.sqrt(d2)),
                # and math.sqrt is correctly rounded (monotone), so the min over
                # sqrt(d2) is sqrt(min d2) and "any sqrt(d2) < ss" is
                # "sqrt(min d2) < ss". d2 is the original expression.
                if near_sq is not None:
                    self_min = near_min(self_cells, new_x, new_y)
                else:
                    self_min = (
                        min(
                            (new_x - sx) ** 2 + (new_y - sy) ** 2
                            for sx, sy in self_collision_segments
                        )
                        if self_collision_segments
                        else None
                    )
                if self_min is not None and math.sqrt(self_min) < self.segment_size:
                    danger = 1.0
                other_min = None
                if danger < 1.0 and has_other_points:
                    if near_sq is not None:
                        other_min = near_obstacle_min_d2(other_geometries, new_x, new_y, ss_int, 3)
                        if other_min is not None and other_min >= near_sq:
                            other_min = None
                    else:
                        other_min = min(
                            (new_x - sx) ** 2 + (new_y - sy) ** 2 for sx, sy in other_points
                        )
                    if other_min is not None and math.sqrt(other_min) < self.segment_size:
                        danger = 1.0

                # Proximity danger (softer signal for nearby obstacles)
                if danger < 1.0:
                    max_check_dist = self.segment_size * 3
                    min_obstacle_dist = max_check_dist

                    # Check walls
                    if circular:
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

                    # Snake bodies: own body once per occurrence of self in the roster
                    # (as the original loop), then every live other snake's obstacles.
                    if self_in_roster and self_min is not None:
                        min_obstacle_dist = min(min_obstacle_dist, math.sqrt(self_min))
                    if other_min is not None:
                        min_obstacle_dist = min(min_obstacle_dist, math.sqrt(other_min))

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

    def _get_free_space_features(self, other_snakes: List["Snake"]) -> List[float]:
        """Reachable open area per relative action (left/straight/right).

        For each candidate next-head cell, flood-fills a segment-resolution grid
        counting reachable free cells, bounded by a cap that scales with the
        snake's length. A move that seals the snake into a small pocket scores
        low; an open move scores ~1.0. This gives the policy the spatial signal
        needed to avoid self-entrapment when long — which per-action danger
        (immediate only) and the coarse sector danger map cannot express.

        Returns:
            [free_left, free_straight, free_right], each in [0.0, 1.0].
        """
        ss = max(int(self.segment_size), 1)
        head_x, head_y = self.head
        # Half-open floor mapping: a pixel in [0, W) maps to cell [0, ceil(W/ss)).
        # (round-based mapping pushed a head in the rightmost/bottommost half-cell
        # to an out-of-bounds cell, collapsing all free-space to 0 along that edge.)
        grid_w = max(int(math.ceil(self.game_width / ss)), 1)
        grid_h = max(int(math.ceil(self.game_height / ss)), 1)

        def to_cell(x: float, y: float) -> Tuple[int, int]:
            return (int(x // ss), int(y // ss))

        # Circular arenas: cells whose top-left corner falls outside the radius
        # are treated as wall (corner test, matching to_cell's floor mapping).
        circular = GameConfig.ARENA_TYPE == "circular"
        cx = cy = radius_sq = 0.0
        if circular:
            cx, cy, radius = GameLogic.get_circular_arena(self.game_width, self.game_height)
            radius_sq = float(radius) ** 2

        def in_bounds(cell: Tuple[int, int]) -> bool:
            gx, gy = cell
            if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
                return False
            if circular:
                px, py = gx * ss, gy * ss
                if (px - cx) ** 2 + (py - cy) ** 2 > radius_sq:
                    return False
            return True

        # Blocked cells = full snake bodies (walls for reachability). Unlike the
        # immediate per-action danger, we block the snake's ENTIRE body — including
        # the head/neck cell it is leaving — so a pocket cannot "leak" back out
        # through the head and self-traps are detected. The tail clearing as the
        # snake advances is accounted for by the length-scaled cap below.
        # Cell sets per snake come from the content-validated geometry cache when the
        # cell size is the snake's own segment size (the same (x // ss, y // ss) cells).
        cache_key = None
        if ss == self.segment_size:
            own_geometry = geometry_for(self, ss)
            other_geometries = tuple(
                geometry_for(snake, ss)
                for snake in other_snakes
                if not (snake is self or not snake.is_alive)
            )
            # Result cache (exact): the features are a pure function of the cell size,
            # board, arena shape, own direction, own segments + length (own_geometry is
            # content-validated) and the other live snakes' segments (their geometry
            # objects, compared by identity: the key holds them, so an equal key means the
            # same cached content). The hero's veto recomputes exactly what its carried
            # state already did one call earlier, so this hit is common.
            cache_key = (
                ss,
                self.game_width,
                self.game_height,
                circular,
                cx,
                cy,
                radius_sq,
                tuple(self.direction),
                own_geometry,
                other_geometries,
            )
            cached = getattr(self, "_free_space_cache", None)
            if cached is not None and cached[0] == cache_key:
                return list(cached[1])
            blocked: set = set(own_geometry.body_cells)
            for geometry in other_geometries:
                blocked |= geometry.body_cells
        else:
            blocked = {(sx // ss, sy // ss) for sx, sy in self.segments}
            for snake in other_snakes:
                if snake is self or not snake.is_alive:
                    continue
                blocked.update([(sx // ss, sy // ss) for sx, sy in snake.segments])

        # Cap scales with body size ("room for my body?"), bounded for performance.
        cap = min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP, int(self.length) * 2))

        # Earlier searches of this call: (seen set, count). Every cell in a seen set is
        # free, in bounds and in the start's connected component C, and a search returns
        # min(cap, |C|) whatever its visit order (count < cap only after C is exhausted;
        # count == cap implies |C| >= cap). So a start already seen by an earlier search
        # has exactly that search's count: reuse it instead of flooding again.
        searches: List[Tuple[set, int]] = []

        def reachable(start: Tuple[int, int]) -> int:
            if start in blocked or not in_bounds(start):
                return 0
            for prior_seen, prior_count in searches:
                if start in prior_seen:
                    return prior_count
            count = _flood(start)
            return count

        def _flood(start: Tuple[int, int]) -> int:
            seen = {start}
            stack = [start]
            count = 0
            if circular:
                while stack and count < cap:
                    gx, gy = stack.pop()
                    count += 1
                    for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                        if nb not in seen and nb not in blocked and in_bounds(nb):
                            seen.add(nb)
                            stack.append(nb)
                searches.append((seen, count))
                return count
            # Rectangular arena: in_bounds inlined (same predicate, same visit order).
            seen_add = seen.add
            push = stack.append
            pop = stack.pop
            while stack and count < cap:
                gx, gy = pop()
                count += 1
                nb = (gx + 1, gy)
                if gx + 1 < grid_w and nb not in seen and nb not in blocked:
                    seen_add(nb)
                    push(nb)
                nb = (gx - 1, gy)
                if gx >= 1 and nb not in seen and nb not in blocked:
                    seen_add(nb)
                    push(nb)
                nb = (gx, gy + 1)
                if gy + 1 < grid_h and nb not in seen and nb not in blocked:
                    seen_add(nb)
                    push(nb)
                nb = (gx, gy - 1)
                if gy >= 1 and nb not in seen and nb not in blocked:
                    seen_add(nb)
                    push(nb)
            searches.append((seen, count))
            return count

        features: List[float] = []
        for relative_action in range(3):  # 0=left, 1=straight, 2=right
            new_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
            new_head_cell = to_cell(head_x + new_dir[0] * ss, head_y + new_dir[1] * ss)
            features.append(min(reachable(new_head_cell) / cap, 1.0))
        if cache_key is not None:
            self._free_space_cache = (cache_key, tuple(features))
        return features
