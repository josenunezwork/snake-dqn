"""Tests for cell-exact collision canonicalization (blueprint §1.1).

Collision and food-pickup checks are integer cell-occupancy equality
(cell = position // segment_size) at ALL mechanics versions. For the shipped
geometry — 10px segments moving in fixed 10px cardinal steps — this is exactly
equivalent to the legacy ``distance < segment_size`` tests for any pair of
positions whose relative offset lies on the segment lattice, which these tests
enforce as a property sweep.
"""

import math

import pytest

from src.core.mechanics_constants import cell_index, same_cell, snap_to_cell
from src.game.food_manager import FoodManager
from src.game.game_logic import GameLogic
from src.game.snake import Snake

pytestmark = pytest.mark.usefixtures("setup_config")

SEGMENT_SIZE = 10


class TestSnapEquivalence:
    """After snapping to the lattice, cell decision == radius decision for ANY pair.

    The load-bearing invariant behind cell-exact canonicalization: spawns are
    drawn off-lattice, but ``snap_to_cell`` places every entity on one shared
    lattice where two distinct points are >= segment_size apart, so
    ``same_cell`` is exactly the legacy ``distance < segment_size`` test even
    across pre-snap offsets (the cross-offset case an on-lattice sweep misses).
    """

    def test_snapped_pairs_match_radius_decision(self):
        # Deterministic pseudo-random off-lattice pairs on different offsets.
        pairs = []
        v = 12345
        for _ in range(4000):
            coords = []
            for _ in range(4):
                v = (1103515245 * v + 12345) % 2147483648
                coords.append(v % 400)
            pairs.append(((coords[0], coords[1]), (coords[2], coords[3])))
        for a, b in pairs:
            sa = snap_to_cell(a, SEGMENT_SIZE)
            sb = snap_to_cell(b, SEGMENT_SIZE)
            assert (math.dist(sa, sb) < SEGMENT_SIZE) == same_cell(sa, sb, SEGMENT_SIZE)


class TestSpawnsAreLatticeAligned:
    """Spawn sources must emit lattice-aligned positions (the snap invariant)."""

    def test_food_spawns_on_lattice(self):
        fm = FoodManager(
            game_width=1450,
            game_height=830,
            max_food=50,
            initial_food=50,
            segment_size=SEGMENT_SIZE,
            wall_thickness=10,
        )
        assert fm.food, "expected initial food"
        for x, y in fm.food:
            assert x % SEGMENT_SIZE == 0 and y % SEGMENT_SIZE == 0, f"off-lattice food {(x, y)}"

    def test_find_empty_position_on_lattice(self):
        for _ in range(200):
            pos = GameLogic.find_empty_position(1450, 830, [])
            assert pos is not None
            x, y = pos
            assert x % SEGMENT_SIZE == 0 and y % SEGMENT_SIZE == 0, f"off-lattice spawn {(x, y)}"


class TestCellRadiusEquivalence:
    """Property sweep: cell decision == legacy radius decision on the lattice."""

    @pytest.mark.parametrize(
        "base",
        [
            (0, 0),  # origin (cell corner)
            (100, 100),  # on-lattice interior point
            (7, 3),  # off-lattice, near origin
            (104, 57),  # off-lattice interior point
            (1443, 829),  # off-lattice, near the full-board far corner
            (-14, 9),  # negative coordinate (out-of-bounds head)
        ],
    )
    def test_lattice_offsets_match_radius_decision(self, base):
        """For every relative offset on the segment lattice, the legacy radius
        decision (distance < segment_size) equals the cell decision."""
        bx, by = base
        span = 3 * SEGMENT_SIZE
        for dx in range(-span, span + 1, SEGMENT_SIZE):
            for dy in range(-span, span + 1, SEGMENT_SIZE):
                other = (bx + dx, by + dy)
                radius_decision = math.dist(base, other) < SEGMENT_SIZE
                cell_decision = same_cell(base, other, SEGMENT_SIZE)
                assert radius_decision == cell_decision, (
                    f"divergence at base={base}, offset=({dx},{dy}): "
                    f"radius={radius_decision}, cell={cell_decision}"
                )

    def test_cell_index_floor_division_for_negative_coordinates(self):
        """Out-of-bounds (negative) positions map to negative cells, not cell 0."""
        assert cell_index((-1, -10), SEGMENT_SIZE) == (-1, -1)
        assert cell_index((0, 9), SEGMENT_SIZE) == (0, 0)
        assert cell_index((10, 19), SEGMENT_SIZE) == (1, 1)


class TestCellExactCollisionChecks:
    """Functional checks: the canonicalized predicates behave on the lattice."""

    def _snake(self, sid, pos, segments=None):
        snake = Snake(sid, (255, 0, 0), pos, SEGMENT_SIZE, 800, 600)
        if segments is not None:
            snake.segments = list(segments)
            snake.length = len(segments)
        return snake

    def test_same_cell_heads_collide(self):
        snake1 = self._snake(0, (100, 100))
        snake2 = self._snake(1, (100, 100))
        assert GameLogic.check_head_collision(snake1, snake2) is True

    def test_adjacent_cell_heads_do_not_collide(self):
        snake1 = self._snake(0, (100, 100))
        snake2 = self._snake(1, (110, 100))
        assert GameLogic.check_head_collision(snake1, snake2) is False

    def test_body_collision_same_cell(self):
        snake1 = self._snake(0, (100, 100))
        snake2 = self._snake(1, (200, 200), segments=[(200, 200), (100, 100), (100, 110)])
        assert GameLogic.check_body_collision(snake1, snake2) is True

    def test_body_collision_adjacent_cell_is_clear(self):
        snake1 = self._snake(0, (100, 100))
        snake2 = self._snake(1, (200, 200), segments=[(200, 200), (110, 100), (110, 110)])
        assert GameLogic.check_body_collision(snake1, snake2) is False

    def test_self_collision_same_cell(self):
        snake = self._snake(
            0, (100, 100), segments=[(100, 100), (110, 100), (110, 110), (100, 110), (100, 100)]
        )
        assert GameLogic.check_self_collision(snake) is True

    def test_no_self_collision_straight_snake(self):
        snake = self._snake(
            0, (100, 100), segments=[(100, 100), (90, 100), (80, 100), (70, 100), (60, 100)]
        )
        assert GameLogic.check_self_collision(snake) is False


class TestCellExactFoodPickup:
    """Food pickup is cell-occupancy equality too."""

    def _food_manager(self):
        return FoodManager(
            game_width=800,
            game_height=600,
            max_food=10,
            initial_food=0,
            segment_size=SEGMENT_SIZE,
            wall_thickness=10,
        )

    def test_consume_at_same_cell(self):
        fm = self._food_manager()
        fm.add_food((105, 103))
        assert fm.consume_at((100, 100), SEGMENT_SIZE) is True
        assert fm.food == []

    def test_consume_at_adjacent_cell_leaves_food(self):
        fm = self._food_manager()
        fm.add_food((110, 100))
        assert fm.consume_at((100, 100), SEGMENT_SIZE) is False
        assert fm.food == [(110, 100)]
