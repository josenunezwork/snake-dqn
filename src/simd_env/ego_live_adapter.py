"""Live ``GameState`` -> :class:`~src.simd_env.ego_raster.EgoGridView` (redesign M1).

Serving builds the ``ego2s-draft`` observation from the live game through the SAME
featurizer the trainer runs on :class:`~src.simd_env.grid_sim.GridBatchSim`, by
rasterising the live state into the same padded occupancy grids:

- ``owner``: the slot index of the living snake on a cell (``-1`` empty), written
  snake by snake in slot order, each body tail first so its head wins a (never
  expected) self-overlap: the order of ``GridBatchSim._rebuild_env_grids``;
- ``slot``: chosen so the featurizer's ``(head_ptr - slot) % cap`` is the segment's
  offset ``k`` from its head (the sim's ring slots differ, but only ``k`` is
  observable): ``head_ptr = n - 1``, ``slot = n - 1 - k``, ``cap = max(n) + 1``;
- ``food``: 1 ambient pellet, 2 corpse/trail pellet (``FoodManager._corpse_positions``),
  ``FOOD_OUTSIDE`` on the border.

Hunger caveat: the live game advances ``frames_since_food`` only in reward bookkeeping
(``SnakeRewardMixin``), which a scripted snake never runs, so a scripted row's own-hunger
scalar is 0 here while the sim counts it. Only that row's own observation is affected, and
scripted anchors never read ego2s; a learned policy served through a non-reward wrapper
must keep the counter advancing before relying on this scalar.

Dead snakes own no cells (the sim drops them from its grids at death). Positions
go through :func:`~src.core.mechanics_constants.cell_index`, as in the raster31v2
adapter. Bytewise identity with the sim path is pinned by
``tests/test_ego_live_identity.py`` (live ``Snake`` + ``FoodManager`` lockstep with
``GridBatchSim``) and the gate-world check
``research/redesign_scope_20261007/ego_live_identity.py`` (real ``GameState``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.mechanics_constants import cell_index
from src.simd_env.ego_raster import EgoGridView, empty_view_like
from src.simd_env.grid_sim import FOOD_AMBIENT, FOOD_CORPSE, GRID_PAD
from src.simd_env.live_adapter import _heading_index

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.game_state import GameState

__all__ = ["game_state_to_ego_view"]


def _boost_cost_frames(default: int = 3) -> int:
    try:
        from src.core.game_config import GameConfig

        return int(GameConfig.BOOST_LENGTH_COST_FRAMES)
    except Exception:  # pragma: no cover - config not initialized
        return default


def game_state_to_ego_view(
    game_state: "GameState | Any", *, boost_length_cost_frames: int | None = None
) -> EgoGridView:
    """Snapshot a live game (``snakes``, ``food_manager``, arena size) as a 1-env view.

    Args:
        game_state: A live ``GameState`` (or the duck-typed ``PyRefGame``).
        boost_length_cost_frames: Boost burn cadence for the boost-phase scalar;
            defaults to ``GameConfig.BOOST_LENGTH_COST_FRAMES``.

    Returns:
        An :class:`EgoGridView` with ``E = 1`` and ``S = len(snakes)``.
    """
    snakes = list(game_state.snakes)
    S = len(snakes)
    fm = game_state.food_manager
    seg = int(getattr(snakes[0], "segment_size", 10)) if snakes else 10
    width = int(getattr(game_state, "_game_width", getattr(fm, "game_width", 0)))
    height = int(getattr(game_state, "_game_height", getattr(fm, "game_height", 0)))
    if width % seg or height % seg:
        raise ValueError("ego2s needs an arena whose size is a multiple of the segment size")
    grid_w, grid_h = width // seg, height // seg
    P = GRID_PAD
    grids = empty_view_like(1, S, grid_w, grid_h, pad=P)
    owner, slot, food = grids["owner_pad"][0], grids["slot_pad"][0], grids["food_pad"][0]

    head_ptr = np.zeros((1, S), dtype=np.int64)
    seg_count = np.zeros((1, S), dtype=np.int64)
    length = np.ones((1, S), dtype=np.int64)
    alive = np.zeros((1, S), dtype=bool)
    direction = np.ones((1, S), dtype=np.int64)
    heads = np.zeros((1, S, 2), dtype=np.int64)
    boost_frames = np.zeros((1, S), dtype=np.int64)
    frames_since_food = np.zeros((1, S), dtype=np.int64)
    bodies = []
    for s, snake in enumerate(snakes):
        cells = np.array([cell_index(p, seg) for p in snake.segments], dtype=np.int64)
        cells = cells.reshape(-1, 2)
        bodies.append(cells)
        is_alive = bool(getattr(snake, "is_alive", True))
        alive[0, s] = is_alive
        n = len(cells)
        seg_count[0, s] = n
        head_ptr[0, s] = max(n - 1, 0)
        length[0, s] = int(snake.length)
        direction[0, s] = _heading_index(getattr(snake, "direction", (1, 0)))
        boost_frames[0, s] = int(getattr(snake, "boost_frames", 0))
        frames_since_food[0, s] = int(getattr(snake, "frames_since_food", 0))
        if n:
            heads[0, s] = cells[0]
    cap = max([len(c) for c in bodies] + [0]) + 1
    for s, cells in enumerate(bodies):
        if not alive[0, s] or not len(cells):
            continue
        n = len(cells)
        k = np.arange(n - 1, -1, -1)  # tail first, so the head is written last
        c = cells[k]
        if (
            (c[:, 0] < 0).any()
            or (c[:, 0] >= grid_w).any()
            or (c[:, 1] < 0).any()
            or (c[:, 1] >= grid_h).any()
        ):
            raise ValueError(f"living snake {s} has a segment outside the arena")
        owner[c[:, 1] + P, c[:, 0] + P] = s
        slot[c[:, 1] + P, c[:, 0] + P] = n - 1 - k
    corpse = getattr(fm, "_corpse_positions", set())
    for pos in fm.food:
        col, row = cell_index(pos, seg)
        if not (0 <= col < grid_w and 0 <= row < grid_h):
            raise ValueError(f"food {pos} lies outside the arena")
        food[row + P, col + P] = FOOD_CORPSE if pos in corpse else FOOD_AMBIENT
    return EgoGridView(
        owner_pad=grids["owner_pad"],
        slot_pad=grids["slot_pad"],
        food_pad=grids["food_pad"],
        head_ptr=head_ptr,
        seg_count=seg_count,
        length=length,
        alive=alive,
        direction=direction,
        heads=heads,
        boost_frames=boost_frames,
        frames_since_food=frames_since_food,
        grid_w=grid_w,
        grid_h=grid_h,
        cap=cap,
        pad=P,
        boost_length_cost_frames=(
            _boost_cost_frames() if boost_length_cost_frames is None else boost_length_cost_frames
        ),
    )
