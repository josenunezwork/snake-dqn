"""Per-environment RNG discipline for the vectorized batch simulator.

The live simulator draws spawn positions from the global CPython ``random``
module using rejection sampling, so the number of draws per frame is
world-dependent (see ``docs/simd_env_spec.md`` §5). To reproduce food and spawn
positions bit-for-bit when seeded identically, the batch simulator keeps one
real :class:`random.Random` per environment and consumes it in the exact
documented draw order.

This module centralizes that discipline in one place so the draw order (x then
y, ``randint`` semantics, snap-to-cell, and the nested snake/food rejection
loops) lives in a single, auditable spot rather than being scattered through
the step function.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from src.core.mechanics_constants import cell_index, snap_to_cell


class EnvRng:
    """A single environment's spawn RNG, matching the live game's draw order.

    Wraps a CPython :class:`random.Random` (Mersenne-Twister) so the batch
    simulator's spawn positions match the live game's when seeded identically.
    Reproducing the PRNG in NumPy is intentionally avoided: NumPy's stream
    differs from CPython's, and the rejection-sampling draw counts are
    world-dependent, so only a real ``random.Random`` stays in lockstep.

    Args:
        seed: Seed for the underlying ``random.Random``.
        game_width: Arena width in pixels.
        game_height: Arena height in pixels.
        segment_size: Cell edge length in pixels.
        wall_thickness: Spawn margin in pixels (== wall thickness).
    """

    def __init__(
        self,
        seed: int,
        game_width: int,
        game_height: int,
        segment_size: int,
        wall_thickness: int,
    ) -> None:
        self._rng = random.Random(seed)
        self.game_width = int(game_width)
        self.game_height = int(game_height)
        self.segment_size = int(segment_size)
        self.wall_thickness = int(wall_thickness)

    # ------------------------------------------------------------------
    # Low-level draws (exact spec draw order)
    # ------------------------------------------------------------------
    def get_random_position(self) -> Tuple[int, int]:
        """Draw one snapped candidate from ``GameState.get_random_position``.

        Draw order (spec §5.4): ``randint(wall_thickness, width - wall_thickness
        - segment_size)`` for x, then the same for y (note the ``-segment_size``
        upper bound, which differs from ``find_empty_position``'s ``-1``), then
        snap. Used for snake placement at reset, where the live game draws this
        FIRST and only falls back to ``find_empty_position`` on overlap.
        """
        wt = self.wall_thickness
        s = self.segment_size
        x = self._rng.randint(wt, self.game_width - wt - s)
        y = self._rng.randint(wt, self.game_height - wt - s)
        return snap_to_cell((x, y), s)

    def place_snake(self, placed_cells: np.ndarray) -> Tuple[int, int]:
        """Reproduce ``GameState._get_non_overlapping_snake_position``.

        Draws one ``get_random_position`` candidate; if it overlaps an
        already-placed snake cell (cell-exact on the shared lattice, equivalent
        to the legacy ``distance < segment_size`` test), falls back to
        ``find_empty_position`` (its own rejection loop); if that returns None,
        keeps the original candidate.

        Args:
            placed_cells: Integer array (N, 2) of already-placed snake cells.

        Returns:
            A snapped ``(x, y)`` pixel position (always non-None).
        """
        candidate = self.get_random_position()
        occupied = _cell_set(placed_cells)
        if not occupied or cell_index(candidate, self.segment_size) not in occupied:
            return candidate
        empty = self.find_empty_position(placed_cells)
        return empty if empty is not None else candidate

    def _draw_find_empty_candidate(self) -> Tuple[int, int]:
        """Draw one snapped candidate from ``GameLogic.find_empty_position``.

        Draw order (spec §5.2): ``randint(margin, width - margin - 1)`` for x,
        then ``randint(margin, height - margin - 1)`` for y, then snap.
        """
        margin = self.wall_thickness
        x = self._rng.randint(margin, self.game_width - margin - 1)
        y = self._rng.randint(margin, self.game_height - margin - 1)
        return snap_to_cell((x, y), self.segment_size)

    def find_empty_position(
        self,
        snake_cells: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Reproduce ``GameLogic.find_empty_position`` for a rectangular arena.

        Up to 100 attempts; each attempt draws (x, y), snaps, and rejects if the
        snapped cell overlaps any living snake segment (cell-exact on the shared
        lattice, equivalent to the legacy ``distance < segment_size`` test).

        Args:
            snake_cells: Integer array (N, 2) of occupied snake cell indices
                (already ``// segment_size``). Empty array means no snakes.

        Returns:
            A snapped ``(x, y)`` pixel position, or None after 100 failures.
        """
        occupied = _cell_set(snake_cells)
        for _ in range(100):
            pos = self._draw_find_empty_candidate()
            if cell_index(pos, self.segment_size) not in occupied:
                return pos
        return None

    def find_spawn_position_no_snakes(
        self,
        food_cells: set,
    ) -> Optional[Tuple[int, int]]:
        """Reproduce ``FoodManager._find_spawn_position`` with ``snakes=None``.

        This is the branch the FoodManager constructor's ``_spawn_initial`` takes
        (spec §5.1/§5.4): up to 100 attempts, each drawing a single
        ``_get_random_position`` candidate (``randint(wt, width-wt-s)`` x then y,
        the SAME upper bound as snake placement, NOT ``find_empty_position``'s
        ``-1``) and rejecting only on cell-exact food overlap. The constructor's
        food is discarded by the subsequent ``reset()``, but its RNG draws still
        advance the shared stream and MUST be reproduced.

        Args:
            food_cells: Set of ``(col, row)`` cells already holding food.

        Returns:
            A snapped ``(x, y)`` pixel position, or None after 100 failures.
        """
        s = self.segment_size
        for _ in range(100):
            pos = self.get_random_position()
            if cell_index(pos, s) not in food_cells:
                return pos
        return None

    def find_spawn_position(
        self,
        snake_cells: np.ndarray,
        food_cells: set,
    ) -> Optional[Tuple[int, int]]:
        """Reproduce ``FoodManager._find_spawn_position`` (with snakes).

        Nested rejection: up to 100 outer attempts, each calling
        ``find_empty_position`` (inner up-to-100 snake-overlap loop) and then
        re-checking cell-exact food overlap.

        Args:
            snake_cells: Integer array (N, 2) of occupied snake cell indices.
            food_cells: Set of ``(col, row)`` cells already holding food.

        Returns:
            A snapped ``(x, y)`` pixel position, or None if none found.
        """
        for _ in range(100):
            pos = self.find_empty_position(snake_cells)
            if pos is None:
                return None
            if cell_index(pos, self.segment_size) not in food_cells:
                return pos
        return None


def _cell_set(cells: np.ndarray) -> set:
    """Return a Python set of ``(col, row)`` tuples from an (N, 2) int array."""
    if cells.size == 0:
        return set()
    return {(int(c), int(r)) for c, r in cells}


def make_env_rngs(
    seeds: List[int],
    game_width: int,
    game_height: int,
    segment_size: int,
    wall_thickness: int,
) -> List[EnvRng]:
    """Construct one :class:`EnvRng` per environment.

    Args:
        seeds: Per-environment seeds (length E).
        game_width: Arena width in pixels.
        game_height: Arena height in pixels.
        segment_size: Cell edge length in pixels.
        wall_thickness: Spawn margin in pixels.

    Returns:
        A list of E :class:`EnvRng`.
    """
    return [
        EnvRng(int(seed), game_width, game_height, segment_size, wall_thickness) for seed in seeds
    ]
