"""Deterministic scenarios for GridBatchSim parity tests and throughput benches.

Two things the default reset cannot give quickly are provided here:

- :func:`inject_serpentines` places long boustrophedon ("serpentine") snakes into
  every env of one or more simulators with an identical layout, so parity and
  benchmarks can exercise the big-body regime (the champion's self-deaths happen
  at a median length of ~1068) without simulating thousands of growth frames.
- :class:`GreedySafePolicy` is a cheap, deterministic, mask-respecting policy
  that steers toward food and boosts sometimes, so snakes keep growing, burn
  boost segments, and eventually trap themselves or collide.

Neither touches a simulator's RNG streams, so lockstep comparisons stay exact.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from src.simd_env.batch_sim import CARDINAL, BatchSim

__all__ = ["serpentine_path", "inject_serpentines", "GreedySafePolicy"]


def serpentine_path(
    col0: int, row0: int, width: int, height: int, length: int
) -> List[Tuple[int, int]]:
    """Boustrophedon cells inside a ``width`` x ``height`` box, head first.

    Rows alternate direction and consecutive rows are joined vertically, so the
    path is 4-connected and self-avoiding. The returned list is ordered head ->
    tail; the head is the LAST cell of the sweep so it faces into open space.

    Args:
        col0: Left column of the box.
        row0: Top row of the box.
        width: Box width in cells (>= 2).
        height: Box height in cells; ``width * height`` must be >= ``length``.
        length: Number of cells.

    Returns:
        ``length`` distinct ``(col, row)`` cells, head first.
    """
    if width * height < length:
        raise ValueError("box too small for the requested serpentine length")
    cells: List[Tuple[int, int]] = []
    for r in range(height):
        cols = range(width) if r % 2 == 0 else range(width - 1, -1, -1)
        for c in cols:
            cells.append((col0 + c, row0 + r))
            if len(cells) == length:
                return cells[::-1]
    return cells[::-1]


def _heading(cells: Sequence[Tuple[int, int]]) -> int:
    """Cardinal index of the move ``cells[1] -> cells[0]`` (defaults to right)."""
    if len(cells) < 2:
        return 1
    d = (cells[0][0] - cells[1][0], cells[0][1] - cells[1][1])
    for i, v in enumerate(CARDINAL):
        if (int(v[0]), int(v[1])) == d:
            return i
    return 1


def inject_serpentines(
    sims: Sequence[BatchSim], lengths: Sequence[int], box_height: int = 6
) -> None:
    """Replace every env's snakes with identical stacked serpentines.

    Snake ``s`` occupies a horizontal band of ``box_height`` rows; bands are
    separated by two empty rows. Food lying on a new body cell is removed from
    each simulator's ordered food list (and corpse/membership indexes) in place,
    which consumes no RNG. ``GridBatchSim`` grids are rebuilt and masks refreshed.

    Args:
        sims: Simulators with identical ``E``, ``S`` and arena (mutated).
        lengths: Body length per snake slot (len == S). 0 kills that slot.
        box_height: Rows per serpentine band.
    """
    first = sims[0]
    S, grid_w, grid_h = first.S, first.grid_w, first.grid_h
    if len(lengths) != S:
        raise ValueError("need one length per snake slot")
    width = grid_w - 4
    layouts: List[List[Tuple[int, int]]] = []
    for sidx, length in enumerate(lengths):
        if length <= 0:
            layouts.append([])
            continue
        row0 = 2 + sidx * (box_height + 2)
        if row0 + box_height > grid_h - 2:
            raise ValueError("arena too small for this many serpentine bands")
        layouts.append(serpentine_path(2, row0, width, box_height, int(length)))
    for sim in sims:
        if sim.cap <= max(lengths) + 3:
            raise ValueError("ring capacity too small for the injected lengths")
        for e in range(sim.E):
            occupied = set()
            for sidx, cells in enumerate(layouts):
                sim.bodies[e, sidx] = 0
                if not cells:
                    sim.alive[e, sidx] = False
                    sim.seg_count[e, sidx] = 0
                    continue
                n = len(cells)
                # Head at ring slot n-1, tail at slot 0.
                sim.bodies[e, sidx, :n] = np.array(cells[::-1], dtype=np.int64)
                sim.head_ptr[e, sidx] = n - 1
                sim.seg_count[e, sidx] = n
                sim.length[e, sidx] = n
                sim.alive[e, sidx] = True
                sim.direction[e, sidx] = _heading(cells)
                sim.boost_frames[e, sidx] = 0
                sim.frames_since_food[e, sidx] = 0
                sim._reward_prev_length[e, sidx] = n
                occupied.update(cells)
            keep = [c for c in sim.food_cells[e] if c not in occupied]
            sim.food_cells[e][:] = keep
            sim.food_set[e].intersection_update(keep)
            sim.corpse_cells[e].intersection_update(keep)
            if hasattr(sim, "_rebuild_env_grids"):
                sim._rebuild_env_grids(e)
        all_envs = np.ones(sim.E, dtype=bool)
        sim._rebuild_traversed_from_heads(all_envs)
        sim._refresh_action_masks(all_envs)


class GreedySafePolicy:
    """Deterministic food-seeking policy restricted to safe actions.

    Per snake: among safe normal actions pick the one whose next head cell is
    closest (Manhattan, to a per-env food sample) to food, breaking ties with a
    fixed preference; boost the chosen direction with probability
    ``boost_prob`` when its boost bit is safe. With no safe action it goes
    straight (a deterministic death).

    Args:
        seed: Seed for the boost coin and tie-break preference.
        boost_prob: Probability of boosting when the boost variant is safe.
        food_sample: Max pellets per env considered (first N in list order).
    """

    def __init__(self, seed: int, boost_prob: float = 0.15, food_sample: int = 64) -> None:
        self._gen = np.random.default_rng(seed ^ 0x6EED)
        self.boost_prob = boost_prob
        self.food_sample = food_sample

    def actions(self, sim: BatchSim, mask: np.ndarray) -> np.ndarray:
        """Return ``(E, S)`` actions for the simulator's current state."""
        E, S = sim.E, sim.S
        heads = sim.heads()
        deltas = np.array([-1, 0, 1])
        dirs = (sim.direction[:, :, None] + deltas[None, None, :]) % 4
        nxt = heads[:, :, None, :] + CARDINAL[dirs]  # (E, S, 3, 2)
        fmax = self.food_sample
        food = np.full((E, fmax, 2), 10**6, dtype=np.int64)
        for e in range(E):
            fl = sim.food_cells[e][:fmax]
            if fl:
                food[e, : len(fl)] = np.asarray(fl, dtype=np.int64)
        dist = np.abs(nxt[:, :, :, None, :] - food[:, None, None, :, :]).sum(-1).min(-1)
        jitter = self._gen.random((E, S, 3)) * 0.5
        score = np.where(mask[:, :, :3], dist + jitter, np.inf)
        rel = score.argmin(axis=2)
        none_safe = ~mask[:, :, :3].any(axis=2)
        rel = np.where(none_safe, 1, rel)
        boost_ok = np.take_along_axis(mask[:, :, 3:], rel[..., None], axis=2)[..., 0]
        boost = boost_ok & (self._gen.random((E, S)) < self.boost_prob)
        return (rel + 3 * boost).astype(np.int64)
