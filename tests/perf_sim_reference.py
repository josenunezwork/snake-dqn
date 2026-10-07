"""Verbatim pre-perf-sim (main db2ef7a) implementations of the optimized hot paths.

Reference oracles for tests/test_perf_sim_identity.py. Each function is the original method
body copied unchanged from main at db2ef7a (methods turned into module functions taking the
snake as ``self``); only names were prefixed with ``ref_`` and the two internal calls
re-pointed at their reference twins. Do not "fix" or modernize this file: it is evidence.
"""

# flake8: noqa
import math
from typing import List, Optional, Tuple

from src.core.game_config import GameConfig
from src.core.mechanics_constants import same_cell
from src.game.game_logic import GameLogic
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP


def ref_get_enhanced_food_state(
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


def ref_get_danger_map(self, other_snakes: List["Snake"], num_sectors: int = 16) -> List[float]:
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

    # Own body danger. Skip the adjacent neck and a vacating tail so the
    # sector map agrees with immediate collision/mask semantics.
    self_obstacles = list(self.segments[2:])
    if len(self.segments) >= self.length and self_obstacles:
        self_obstacles = self_obstacles[:-1]
    for segment in self_obstacles:
        update_danger(segment[0], segment[1], is_self=True)

    # Other snakes danger
    for snake in other_snakes:
        if snake != self and snake.is_alive:
            update_danger(snake.head[0], snake.head[1], is_self=False)
            obstacle_segments = list(snake.segments[1:])
            if len(snake.segments) >= snake.length and obstacle_segments:
                obstacle_segments = obstacle_segments[:-1]
            for segment in obstacle_segments:
                update_danger(segment[0], segment[1], is_self=False)

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


def ref_get_per_action_danger(self, other_snakes: List["Snake"]) -> List[float]:
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


def ref_get_free_space_features(self, other_snakes: List["Snake"]) -> List[float]:
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
    blocked: set = set()
    for sx, sy in self.segments:
        blocked.add(to_cell(sx, sy))
    for snake in other_snakes:
        if snake is self or not snake.is_alive:
            continue
        for sx, sy in snake.segments:
            blocked.add(to_cell(sx, sy))

    # Cap scales with body size ("room for my body?"), bounded for performance.
    cap = min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP, int(self.length) * 2))

    def reachable(start: Tuple[int, int]) -> int:
        if start in blocked or not in_bounds(start):
            return 0
        seen = {start}
        stack = [start]
        count = 0
        while stack and count < cap:
            gx, gy = stack.pop()
            count += 1
            for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if nb not in seen and nb not in blocked and in_bounds(nb):
                    seen.add(nb)
                    stack.append(nb)
        return count

    features: List[float] = []
    for relative_action in range(3):  # 0=left, 1=straight, 2=right
        new_dir = GameLogic.relative_to_absolute_direction(self.direction, relative_action)
        new_head_cell = to_cell(head_x + new_dir[0] * ss, head_y + new_dir[1] * ss)
        features.append(min(reachable(new_head_cell) / cap, 1.0))
    return features


def ref_segments_collide_after_move(
    self,
    segments: List[Tuple[int, int]],
    other_snakes: Optional[List["Snake"]] = None,
    traversed_heads: Optional[List[Tuple[int, int]]] = None,
) -> bool:
    """Return whether a simulated post-move body would collide."""
    collision_heads = traversed_heads or [segments[0]]
    for head in collision_heads:
        if not self._in_bounds_position(*head):
            return True

    if len(segments) > 3:
        for head in collision_heads:
            for segment in segments[3:]:
                if GameLogic.distance(head, segment) < self.segment_size:
                    return True

    for snake in other_snakes or []:
        if snake == self or not snake.is_alive:
            continue
        body_segments = list(snake.segments[1:])
        if len(snake.segments) >= snake.length and body_segments:
            body_segments = body_segments[:-1]
        for head in collision_heads:
            if GameLogic.distance(head, snake.head) < self.segment_size:
                return True
            for segment in body_segments:
                if GameLogic.distance(head, segment) < self.segment_size:
                    return True

    return False


def ref_check_self_collision(snake: "Snake") -> bool:
    """
    Check if snake's head collides with its own body.

    Cell-exact: head and segment collide iff they occupy the same integer
    cell (position // segment_size). Equivalent to the legacy radius test
    on the segment lattice.

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
            if same_cell(head, segment, snake.segment_size):
                return True
    return False


def ref_check_body_collision(snake1: "Snake", snake2: "Snake") -> bool:
    """
    Check if snake1's head collides with snake2's body.

    Cell-exact: head and body segment collide iff they occupy the same
    integer cell (position // segment_size).

    Args:
        snake1: Snake whose head might be colliding
        snake2: Snake whose body might be hit

    Returns:
        True if snake1's head hits snake2's body
    """
    return any(
        same_cell(head, segment, snake1.segment_size)
        for head in GameLogic._collision_positions(snake1)
        for segment in snake2.segments[1:]
    )


def ref_position_overlaps_food(self, position: Tuple[int, int]) -> bool:
    """Return whether a position occupies the same cell as existing food.

    Cell-exact (matches the pickup/collision predicate). On the shared
    segment lattice this equals the legacy ``distance < segment_size`` test,
    but it never over-suppresses corpse/trail pellets that merely land near
    (but not on) an existing pellet.
    """
    return any(same_cell(position, food_pos, self.segment_size) for food_pos in self.food)


def ref_consume_at(self, position: Tuple[int, int], radius: int) -> bool:
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


def ref_simulate_relative_action_fatality(
    snake: "Snake",
    other_snakes: Optional[List["Snake"]] = None,
) -> Tuple[List[bool], List[bool]]:
    """Simulate each candidate relative action ONCE and report per-action fatality.

    This is the single shared candidate-move simulation used for mask building
    (see :meth:`AISnake._get_safe_actions`). It is a module-level function (not a
    method) so classes that borrow ``_get_safe_actions`` from ``AISnake`` (e.g.
    ``ScriptedSnake``) resolve it without inheriting extra attributes.

    Per relative direction the normal 1-step move is simulated and scanned once.
    The boosted 2-step variant reuses the normal result for the shared first
    traversed head: the arena bounds and every other snake are static within a
    frame, so the first head's bounds/enemy scans are identical to the normal
    pass and are skipped. Only the snake's own post-boost body layout (which
    differs from the normal layout) is re-checked for the first head, then the
    second head gets the full scan. This eliminates the repeated segment scans
    the old per-action loops performed within a frame while producing exactly
    the same fatality verdicts.

    Args:
        snake: The snake to simulate. Any Snake exposing the AISnake simulation
            helpers works (AISnake, or ScriptedSnake which borrows them).
        other_snakes: Other snakes treated as obstacles (may include ``snake``).

    Returns:
        Tuple of ``(normal_fatal, boost_fatal)``, each a 3-element list indexed
        by relative action (0=left, 1=straight, 2=right). ``boost_fatal[i]`` is
        True when boosting is unavailable (length < MIN_BOOST_LENGTH), the
        normal move is fatal, or the 2-step boosted move collides.
    """
    normal_fatal = [True, True, True]
    boost_fatal = [True, True, True]
    can_boost = snake.length >= GameConfig.MIN_BOOST_LENGTH

    for relative_action in range(3):  # 0=left, 1=straight, 2=right
        abs_dir = GameLogic.relative_to_absolute_direction(snake.direction, relative_action)
        normal_segments, normal_heads = snake._simulate_move_after_action(abs_dir, steps=1)
        if ref_segments_collide_after_move(
            snake,
            normal_segments,
            other_snakes,
            traversed_heads=normal_heads,
        ):
            continue
        normal_fatal[relative_action] = False

        if not can_boost:
            continue
        boost_segments, boost_heads = snake._simulate_move_after_action(
            abs_dir,
            steps=2,
            is_boost=True,
        )
        # The first boosted head equals the normal head just proven safe against
        # the bounds and all other snakes (static within the frame); re-check it
        # only against the snake's own post-boost body layout.
        first_head = boost_heads[0]
        first_head_hits_own_body = False
        if len(boost_segments) > 3:
            for segment in boost_segments[3:]:
                if GameLogic.distance(first_head, segment) < snake.segment_size:
                    first_head_hits_own_body = True
                    break
        if first_head_hits_own_body:
            continue
        if ref_segments_collide_after_move(
            snake,
            boost_segments,
            other_snakes,
            traversed_heads=boost_heads[1:],
        ):
            continue
        boost_fatal[relative_action] = False

    return normal_fatal, boost_fatal
