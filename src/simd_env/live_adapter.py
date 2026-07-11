"""Live ``GameState`` -> :class:`ObsInputs` bridge (blueprint §0.10c, §6).

The web backend steps a single live :class:`~src.game.game_state.GameState`
(``Snake`` objects + :class:`~src.game.food_manager.FoodManager`). This module
converts that state into the SAME :class:`~src.simd_env.featurizer.ObsInputs`
struct the batched trainer fills, so the one shared featurizer serves both paths
and serve-time observations are train-time observations by construction.

Everything is a single environment (``E = 1``). Positions are converted from
pixels to integer cell indices on the shared segment lattice via
:func:`~src.core.mechanics_constants.cell_index`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.mechanics_constants import cell_index
from src.simd_env.featurizer import ObsInputs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.game_state import GameState

# Live snake direction (dx, dy) -> cardinal heading index (0=up,1=right,2=down,
# 3=left), matching src.simd_env.batch_sim.CARDINAL and the featurizer.
_DIR_TO_HEADING = {
    (0, -1): 0,  # up
    (1, 0): 1,  # right
    (0, 1): 2,  # down
    (-1, 0): 3,  # left
}


def _heading_index(direction) -> int:
    """Map a live snake ``(dx, dy)`` unit direction to a cardinal heading index.

    Args:
        direction: ``(dx, dy)`` unit direction from a live ``Snake``.

    Returns:
        Cardinal heading index (0=up, 1=right, 2=down, 3=left); defaults to
        right (1) for any unexpected value.
    """
    dx = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
    dy = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)
    return _DIR_TO_HEADING.get((dx, dy), 1)


def game_state_to_obs_inputs(
    game_state: "GameState",
    max_frames: int = 5000,
    starvation_max: int = 500,
    max_length: int = 400,
) -> ObsInputs:
    """Build a single-env (``E = 1``) :class:`ObsInputs` from a live ``GameState``.

    Pulls heads/bodies/lengths/alive/direction from the ``Snake`` objects and
    food (with corpse flags) from the :class:`FoodManager`. Dead snakes keep
    zeroed geometry and ``alive = False`` so the featurizer skips them.

    Args:
        game_state: The live game to snapshot.
        max_frames: Episode-length cap for the episode-progress scalar.
        starvation_max: Starvation frame cap for the hunger scalar.
        max_length: Length cap for the length scalars.

    Returns:
        A filled :class:`ObsInputs` with ``E = 1``.
    """
    snakes = list(game_state.snakes)
    S = len(snakes)
    seg = game_state.food_manager.segment_size if hasattr(game_state, "food_manager") else 10
    # Fall back to the first snake's segment_size if the food manager lacks one.
    if snakes:
        seg = getattr(snakes[0], "segment_size", seg)

    # --- Snake arrays ---
    body_len = np.zeros((1, S), dtype=np.int64)
    lengths = np.zeros((1, S), dtype=np.int64)
    alive = np.zeros((1, S), dtype=bool)
    heading = np.zeros((1, S), dtype=np.int64)
    boost_frames = np.zeros((1, S), dtype=np.int64)
    frames_since_food = np.zeros((1, S), dtype=np.int64)
    boosting = np.zeros((1, S), dtype=bool)
    heads = np.zeros((1, S, 2), dtype=np.int64)

    body_cells = []  # per-snake list of (col, row)
    maxlen = 1
    for s, snake in enumerate(snakes):
        segs = [cell_index(p, seg) for p in snake.segments]
        body_cells.append(segs)
        maxlen = max(maxlen, len(segs))
        alive[0, s] = bool(getattr(snake, "is_alive", True))
        lengths[0, s] = max(1, int(getattr(snake, "length", 1)))
        body_len[0, s] = len(segs)
        heading[0, s] = _heading_index(getattr(snake, "direction", (1, 0)))
        boost_frames[0, s] = int(getattr(snake, "boost_frames", 0))
        frames_since_food[0, s] = int(getattr(snake, "frames_since_food", 0))
        boosting[0, s] = bool(getattr(snake, "is_boosting", False)) and alive[0, s]
        if segs:
            heads[0, s] = segs[0]

    bodies = np.zeros((1, S, maxlen, 2), dtype=np.int64)
    for s, segs in enumerate(body_cells):
        for k, cell in enumerate(segs):
            bodies[0, s, k] = cell

    # --- Food arrays ---
    fm = game_state.food_manager
    food_positions = list(fm.food)
    corpse_positions = set(getattr(fm, "_corpse_positions", set()))
    F = max(len(food_positions), 1)
    food_cells = np.zeros((1, F, 2), dtype=np.int64)
    food_mass = np.zeros((1, F), dtype=np.float64)
    food_is_corpse = np.zeros((1, F), dtype=bool)
    for i, pos in enumerate(food_positions):
        food_cells[0, i] = cell_index(pos, seg)
        food_mass[0, i] = 1.0
        food_is_corpse[0, i] = pos in corpse_positions

    grid_w = int(getattr(game_state, "_game_width", fm.game_width)) // seg
    grid_h = int(getattr(game_state, "_game_height", fm.game_height)) // seg
    arena_type = 1.0
    try:
        from src.core.game_config import GameConfig

        arena_type = 1.0 if GameConfig.ARENA_TYPE == "circular" else 0.0
    except Exception:  # pragma: no cover - config not initialized
        arena_type = 0.0

    return ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=alive,
        heading=heading,
        boost_frames=boost_frames,
        frames_since_food=frames_since_food,
        boosting=boosting,
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=grid_w,
        grid_h=grid_h,
        max_snakes=S,
        starvation_max=starvation_max,
        max_length=max_length,
        min_boost_length=_gc_int("MIN_BOOST_LENGTH", 5),
        boost_cost_frames=_gc_int("BOOST_LENGTH_COST_FRAMES", 3),
        frame=np.array([int(getattr(game_state, "frame", 0))], dtype=np.int64),
        max_frames=max_frames,
        arena_type_flag=arena_type,
    )


def _gc_int(name: str, default: int) -> int:
    """Read an int GameConfig knob, falling back to ``default`` if unavailable.

    GameConfig properties call ``get_config()`` and raise before the config
    singleton is initialized; the fallback keeps the adapter usable in bare
    unit tests that never call ``initialize_config``.
    """
    try:
        from src.core.game_config import GameConfig

        return int(getattr(GameConfig, name))
    except Exception:  # pragma: no cover - config not initialized
        return default
