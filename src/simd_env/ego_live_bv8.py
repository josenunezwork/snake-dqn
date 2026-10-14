"""Live game -> the v8 feature source of ``ego2s-bv8`` (M4-B live serving).

``ego2s-bv8`` appends the compiled v8 veto's 30 per-action diagnostics
(:func:`src.simd_env.ego_raster_bv8.v8_scalars`) to the ``ego2s-b`` scalars. On the SIMD grid
engine they are read from :class:`~src.simd_env.grid_sim.GridBatchSim`'s ring buffers; serving
reads the SAME compiled kernel (:func:`src.simd_env.v8_nb.v8_decide`) on
:class:`LiveV8Source`, a one-env duck-typed copy of exactly the sim fields the kernel reads:

- ``bodies`` ``(1, S, cap, 2)``: each live snake's cells (``cell_index``) laid out so the
  kernel's ``(head_ptr - k) % cap`` is segment ``k`` from the head (``head_ptr = n - 1``,
  ``bodies[0, s, n - 1 - k] = segments[k]``, ``cap = max(n) + 1``, the
  :func:`~src.simd_env.ego_live_adapter.game_state_to_ego_view` layout);
- ``head_ptr`` / ``seg_count`` / ``length`` / ``boost_frames`` / ``direction`` / ``alive`` and
  the arena grid, as the ego view records them;
- ``cfg``: ``min_boost_length`` / ``boost_length_cost_frames`` / ``mechanics_version`` /
  ``arena_type`` from the live ``GameConfig``;
- :meth:`LiveV8Source.get_advisory_action_mask`: the exact (no-fallback) safe-action mask from
  :func:`src.game.ai_snake.simulate_relative_action_fatality` against the pre-move roster,
  the function the live ``AISnake`` masks with and ``PyRefGame.action_masks`` checks the
  sim's advisory mask against.

Built at decision time (``Ego2sServingPolicy.prepare_frame``: after respawns, before any
snake moves), the features are bitwise the sim's (``tests/test_ego_live_bv8.py`` lockstep;
record-level P2 via ``research/redesign_m3c_20261008/p2_noveto.py``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np

from src.core.mechanics_constants import cell_index
from src.simd_env.live_adapter import _heading_index

__all__ = ["LiveV8Source", "game_state_to_v8_source"]


def _live_cfg() -> SimpleNamespace:
    from src.core.game_config import GameConfig

    return SimpleNamespace(
        min_boost_length=int(GameConfig.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(GameConfig.BOOST_LENGTH_COST_FRAMES),
        mechanics_version=int(GameConfig.MECHANICS_VERSION),
        arena_type=str(GameConfig.ARENA_TYPE),
    )


class LiveV8Source:
    """The sim surface :func:`~src.simd_env.v8_nb.v8_decide` reads, for one live game."""

    def __init__(
        self,
        snakes: list,
        grid_w: int,
        grid_h: int,
        seg: int,
        cfg: Optional[Any] = None,
        mask_slots: Optional[Iterable[int]] = None,
    ) -> None:
        S = len(snakes)
        self._snakes = snakes
        self._mask_slots = None if mask_slots is None else {int(s) for s in mask_slots}
        self._mask: Optional[np.ndarray] = None
        self.grid_w, self.grid_h = int(grid_w), int(grid_h)
        self.cfg = cfg if cfg is not None else _live_cfg()
        self.alive = np.zeros((1, S), dtype=bool)
        self.head_ptr = np.zeros((1, S), dtype=np.int64)
        self.seg_count = np.zeros((1, S), dtype=np.int64)
        self.length = np.ones((1, S), dtype=np.int64)
        self.boost_frames = np.zeros((1, S), dtype=np.int64)
        self.direction = np.ones((1, S), dtype=np.int64)
        cells = []
        for s, snake in enumerate(snakes):
            c = np.array([cell_index(p, seg) for p in snake.segments], dtype=np.int64)
            c = c.reshape(-1, 2)
            cells.append(c)
            alive = bool(getattr(snake, "is_alive", True)) and len(c) > 0
            self.alive[0, s] = alive
            self.seg_count[0, s] = len(c)
            self.head_ptr[0, s] = max(len(c) - 1, 0)
            self.length[0, s] = int(snake.length)
            self.boost_frames[0, s] = int(getattr(snake, "boost_frames", 0))
            self.direction[0, s] = _heading_index(getattr(snake, "direction", (1, 0)))
        cap = max([len(c) for c in cells] + [0]) + 1
        self.bodies = np.full((1, S, cap, 2), -1, dtype=np.int64)
        for s, c in enumerate(cells):
            if len(c):
                self.bodies[0, s, : len(c)] = c[::-1]  # bodies[n - 1 - k] = segment k

    def get_advisory_action_mask(self) -> np.ndarray:
        """``(1, S, 6)`` exact safe-action mask (no fallback; dead / unrequested rows False)."""
        if self._mask is None:
            from src.game.ai_snake import simulate_relative_action_fatality

            S = len(self._snakes)
            mask = np.zeros((1, S, 6), dtype=bool)
            for s, snake in enumerate(self._snakes):
                if not self.alive[0, s] or (
                    self._mask_slots is not None and s not in self._mask_slots
                ):
                    continue
                normal_fatal, boost_fatal = simulate_relative_action_fatality(snake, self._snakes)
                for rel in range(3):
                    mask[0, s, rel] = not normal_fatal[rel]
                    mask[0, s, rel + 3] = not boost_fatal[rel]
            self._mask = mask
        return self._mask


def game_state_to_v8_source(
    game_state: Any, *, mask_slots: Optional[Iterable[int]] = None
) -> LiveV8Source:
    """Snapshot a live game (``GameState`` or ``PyRefGame``) as a :class:`LiveV8Source`.

    Args:
        game_state: The live game, at decision time (before any snake moves this frame).
        mask_slots: Slots whose advisory mask is computed (default: every living snake);
            serving passes the snakes its policy drives (the only rows it featurizes).
    """
    snakes = list(game_state.snakes)
    fm = game_state.food_manager
    seg = int(getattr(snakes[0], "segment_size", 10)) if snakes else 10
    width = int(getattr(game_state, "_game_width", getattr(fm, "game_width", 0)))
    height = int(getattr(game_state, "_game_height", getattr(fm, "game_height", 0)))
    if width % seg or height % seg:
        raise ValueError("ego2s needs an arena whose size is a multiple of the segment size")
    return LiveV8Source(snakes, width // seg, height // seg, seg, mask_slots=mask_slots)
