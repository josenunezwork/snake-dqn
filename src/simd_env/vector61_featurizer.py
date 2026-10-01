"""Batched ``vector61`` (Apex) featurizer over :class:`~src.simd_env.batch_sim.BatchSim` state.

The live Apex champion consumes the 61-D (or 58-D) state vector built per snake by
:meth:`src.game.snake_state.SnakeStateMixin.get_state`. This module rebuilds the
same vector for many ``(env, slot)`` rows straight from the batch sim's
cell-index arrays, so a ``vector61`` checkpoint can be served by the batched
evaluation engine. It is featurization only: policy wiring into
:mod:`src.simd_env.eval_engine` is not done (see the plan doc below).

Parity contract
---------------
The target is **bit-exact float32 equality** with the live ``get_state`` for every
feature, not a tolerance. That is achievable because every live quantity is a
deterministic IEEE-754 double computation over lattice-aligned integer pixel
coordinates (``pixel == cell * segment_size``, see
:func:`src.core.mechanics_constants.snap_to_cell`):

- Integer deltas, squares and sums are exact in float64 (all magnitudes are far
  below 2**53), ``sqrt`` and ``/`` are correctly rounded in both Python and
  NumPy, and ``int / int`` true division in Python is correctly rounded too, so
  the NumPy float64 expressions below reproduce the live Python floats exactly
  as long as the *operation order* is identical. The final ``float64 -> float32``
  cast is the same round-to-nearest the live ``torch.tensor(..., float32)`` does.
- Sector indices go through ``math.atan2`` in the live code. NumPy's ``arctan2``
  is not guaranteed to be bit-identical to the platform libm, so sectors are read
  from a lookup table built once with ``math.atan2`` on the exact pixel deltas
  (all deltas are lattice multiples, so the table is exhaustive).
- Order-dependent reductions are reproduced exactly: nearest food uses the first
  minimum in live food-list order (``np.argmin``), enemies use a stable sort by
  distance in slot order, and sector danger uses order-independent ``max`` /
  ``count`` reductions.
- Integer shortcuts are exact: squared pixel distances are integers and a
  correctly rounded ``sqrt`` is monotone, so ``sqrt(d2) < t`` <=> ``d2 < t**2``
  for the integer thresholds used, and ``min(sqrt(d2)) == sqrt(min(d2))``.
- The capped flood fill (``_get_free_space_features``) returns
  ``min(component_size, cap)`` regardless of traversal order (every pop is a
  distinct cell of the start's 4-connected component). The batched version
  computes that same number with cap ``FREE_SPACE_BFS_CAP`` (then
  ``min(min(size, 160), cap_row) == min(size, cap_row)``) through two sound
  certificates (a fully free box, a closed local flood) and an exact memoized
  scalar fallback; see :func:`batched_capped_component_sizes`.

Stateful feature
----------------
Feature ``[5]`` of the enemy block (nearest-enemy distance trend) depends on the
previous ``get_state`` call of the same snake. :class:`Vector61Memory` holds that
per-``(env, slot)`` baseline. The caller owns the *call sequence*: the live
``AISnake`` advances the baseline once per frame (normally at its post-step
carry-forward capture) and a snake's baseline resets on respawn. Resetting a
row's memory when it dies is equivalent, because a dead snake is never
featurized before it respawns. See ``docs/research/simd_vector61_plan_2026-09-26.md``.

Scope: rectangular arenas with ``width`` and ``height`` multiples of
``segment_size`` (the only worlds :class:`BatchSim` supports). Circular arenas
raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.simd_env.batch_sim import CARDINAL, BatchSim, BatchSimConfig

__all__ = [
    "FREE_SPACE_BFS_CAP",
    "FREE_SPACE_MIN_CAP",
    "Vector61Params",
    "Vector61Memory",
    "Vector61Featurizer",
    "batched_capped_component_sizes",
    "capped_component_sizes",
]

# Mirrors src.game.snake_state (duplicated so this module does not import torch;
# tests/test_simd_vector61.py asserts the two stay equal).
FREE_SPACE_BFS_CAP = 160
FREE_SPACE_MIN_CAP = 32

# Column layout (SnakeStateMixin.get_state order): 0-3 direction one-hot, 4 length,
# 5-23 food, 24-39 sector danger, 40-43 boundary, 44-53 enemies, 54-56 per-action
# danger, 57 boost, 58-60 free space (61-D only).
_CARDINAL_TUPLES: Tuple[Tuple[int, int], ...] = tuple(
    (int(dx), int(dy)) for dx, dy in CARDINAL.tolist()
)


@dataclass(frozen=True)
class Vector61Params:
    """World and network knobs the vector61 state reads (live ``GameConfig`` values).

    Attributes:
        game_width: Arena width in pixels (``Snake.game_width``).
        game_height: Arena height in pixels.
        segment_size: Cell edge in pixels.
        max_length: ``GameConfig.MAX_LENGTH`` (length normalizer).
        num_sectors: ``GameConfig.NUM_SECTORS`` (16 for the vector contract).
        danger_max_distance: ``GameConfig.DANGER_MAX_DISTANCE`` in cells.
        use_boundary_as_danger: ``GameConfig.USE_BOUNDARY_AS_DANGER``.
        min_boost_length: ``GameConfig.MIN_BOOST_LENGTH``.
        food_capacity: The live snake's ``food_capacity`` (``GameState``'s
            effective max food; the density normalizer).
        use_free_space: Emit the 3 free-space columns (61-D) or not (58-D).
        actions: ``GameConfig.ACTIONS`` (direction one-hot order).
        arena_type: Must be ``"rectangular"``.
    """

    game_width: int
    game_height: int
    segment_size: int
    max_length: int
    num_sectors: int
    danger_max_distance: int
    use_boundary_as_danger: bool
    min_boost_length: int
    food_capacity: int
    use_free_space: bool = True
    actions: Tuple[Tuple[int, int], ...] = _CARDINAL_TUPLES
    arena_type: str = "rectangular"

    def __post_init__(self) -> None:
        if self.arena_type != "rectangular":
            raise NotImplementedError("vector61 batch featurizer supports rectangular arenas only")
        ss = int(self.segment_size)
        if ss <= 0:
            raise ValueError("segment_size must be positive")
        if self.game_width % ss or self.game_height % ss:
            raise ValueError(
                "vector61 batch featurizer requires width/height to be multiples of "
                f"segment_size (got {self.game_width}x{self.game_height}, ss={ss})"
            )
        if self.num_sectors <= 0:
            raise ValueError("num_sectors must be positive")
        if self.danger_max_distance < 0:
            raise ValueError("danger_max_distance must be non-negative")
        if set(self.actions) != set(_CARDINAL_TUPLES) or len(self.actions) != 4:
            raise ValueError("actions must be a permutation of the four cardinal directions")

    @property
    def input_size(self) -> int:
        """61 with free-space columns, else 58 (``expected_input_size``)."""
        return 61 if self.use_free_space else 58

    @classmethod
    def from_game_config(
        cls,
        sim_cfg: BatchSimConfig,
        food_capacity: Optional[int] = None,
    ) -> "Vector61Params":
        """Build params from the active ``GameConfig`` plus a batch-sim world config.

        World geometry and boost length come from ``sim_cfg`` (the world the batch
        sim actually runs). Network-side knobs (sectors, danger radius, free space,
        max length, action order) come from the active global ``GameConfig``, the
        same singleton the live ``get_state`` reads.

        Args:
            sim_cfg: The :class:`BatchSimConfig` of the batch being featurized.
            food_capacity: The live snakes' ``food_capacity``. Defaults to
                ``sim_cfg.max_food`` (``GameState`` with food multiplier 1.0).

        Raises:
            ValueError: If the active config's ``INPUT_SIZE`` disagrees with its
                ``USE_FREE_SPACE`` mode.
        """
        from src.core.game_config import GameConfig

        use_free_space = bool(GameConfig.USE_FREE_SPACE)
        expected = 61 if use_free_space else 58
        if int(GameConfig.INPUT_SIZE) != expected:
            raise ValueError(
                f"GameConfig.INPUT_SIZE={GameConfig.INPUT_SIZE} does not match "
                f"use_free_space={use_free_space} (expected {expected})"
            )
        return cls(
            game_width=int(sim_cfg.game_width),
            game_height=int(sim_cfg.game_height),
            segment_size=int(sim_cfg.segment_size),
            max_length=int(GameConfig.MAX_LENGTH),
            num_sectors=int(GameConfig.NUM_SECTORS),
            danger_max_distance=int(GameConfig.DANGER_MAX_DISTANCE),
            use_boundary_as_danger=bool(GameConfig.USE_BOUNDARY_AS_DANGER),
            min_boost_length=int(sim_cfg.min_boost_length),
            food_capacity=int(sim_cfg.max_food if food_capacity is None else food_capacity),
            use_free_space=use_free_space,
            actions=tuple((int(a[0]), int(a[1])) for a in GameConfig.ACTIONS),
            arena_type=str(sim_cfg.arena_type),
        )


class Vector61Memory:
    """Per-``(env, slot)`` nearest-enemy trend baseline (live ``_prev_nearest_enemy_*``).

    ``prev_dist`` is ``inf`` and ``prev_id`` is ``-1`` (live ``None``) until a
    featurization with ``update_memory=True`` stores a baseline.
    """

    def __init__(self, num_envs: int, num_snakes: int) -> None:
        self.prev_dist = np.full((num_envs, num_snakes), np.inf, dtype=np.float64)
        self.prev_id = np.full((num_envs, num_snakes), -1, dtype=np.int64)

    def reset(self) -> None:
        """Forget every baseline (episode reset)."""
        self.prev_dist.fill(np.inf)
        self.prev_id.fill(-1)

    def reset_where(self, mask: np.ndarray) -> None:
        """Forget the baselines of rows where ``mask`` (E, S) is True.

        Call with ``sim.get_done()`` after every step: the live snake resets its
        baseline on respawn, and a dead snake is never featurized in between, so
        resetting at death is equivalent.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.prev_dist.shape:
            raise ValueError(f"mask must have shape {self.prev_dist.shape}, got {mask.shape}")
        self.prev_dist[mask] = np.inf
        self.prev_id[mask] = -1


def capped_component_sizes(
    free: np.ndarray,
    starts: Sequence[Tuple[int, int]],
    cap: int = FREE_SPACE_BFS_CAP,
) -> List[int]:
    """Return ``min(component size, cap)`` of each start cell's 4-connected free region.

    Equivalent, per start, to the live capped flood fill's pop count (0 for a
    blocked or out-of-bounds start). Flood fills are memoized across starts: a
    fully explored component records its exact size and a capped exploration
    records ``cap`` on every cell it discovered, so later starts that touch an
    already-classified component finish immediately. Any start that reaches a
    memoized cell must share that cell's component; a component memoized with
    an exact size below ``cap`` was fully explored and would already contain the
    start, so a reached memo value is always ``cap``.

    Args:
        free: Bool grid indexed ``[x, y]`` (True == free and in bounds).
        starts: ``(x, y)`` cells.
        cap: Exploration cap (>= every row cap that will consume the result).

    Returns:
        One ``int`` per start.
    """
    gw, gh = free.shape
    free_flat = free.ravel().tolist()
    memo: Dict[int, int] = {}
    out: List[int] = []
    for x, y in starts:
        x, y = int(x), int(y)
        if x < 0 or x >= gw or y < 0 or y >= gh:
            out.append(0)
            continue
        start = x * gh + y
        if not free_flat[start]:
            out.append(0)
            continue
        known = memo.get(start)
        if known is not None:
            out.append(known)
            continue
        seen = {start}
        stack = [start]
        popped = 0
        result: Optional[int] = None
        while stack:
            if popped >= cap:
                result = cap
                break
            cell = stack.pop()
            popped += 1
            cx, cy = divmod(cell, gh)
            for nb, ok in (
                (cell + gh, cx + 1 < gw),
                (cell - gh, cx > 0),
                (cell + 1, cy + 1 < gh),
                (cell - 1, cy > 0),
            ):
                if not ok or nb in seen or not free_flat[nb]:
                    continue
                hit = memo.get(nb)
                if hit is not None:
                    result = hit
                    break
                seen.add(nb)
                stack.append(nb)
            if result is not None:
                break
        if result is None:
            result = min(popped, cap)
        for cell in seen:
            memo[cell] = result
        out.append(result)
    return out


def _sector_lut(gw: int, gh: int, ss: int, num_sectors: int) -> Tuple[np.ndarray, int, int]:
    """Sector index for every lattice pixel delta, via the live ``math.atan2`` formula.

    Returns ``(lut, rx, ry)`` with ``lut[dx_cell + rx, dy_cell + ry]`` equal to
    ``Snake._angle_to_sector(dx_cell * ss, dy_cell * ss, num_sectors)``. The range
    covers every head-to-{segment, food, wall-sample} delta on a ``gw x gh`` board.
    """
    rx, ry = gw + 2, gh + 2
    two_pi = 2 * math.pi
    lut = np.empty((2 * rx + 1, 2 * ry + 1), dtype=np.int64)
    for i, dxc in enumerate(range(-rx, rx + 1)):
        dxp = dxc * ss
        row = lut[i]
        for j, dyc in enumerate(range(-ry, ry + 1)):
            angle = math.atan2(dyc * ss, dxp)
            row[j] = int(((angle + math.pi) / two_pi) * num_sectors) % num_sectors
    return lut, rx, ry


_LUT_CACHE: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, int, int]] = {}


# Windowed pre-pass for the batched flood fill: Chebyshev radius of the local
# window and the maximum number of BFS layers before handing a start to the
# exact memoized fallback.
FLOOD_WINDOW_RADIUS = 10
FLOOD_MAX_LAYERS = 40


def batched_capped_component_sizes(
    free_grids: Sequence[np.ndarray],
    starts: np.ndarray,
    cap: int = FREE_SPACE_BFS_CAP,
    radius: int = FLOOD_WINDOW_RADIUS,
    max_layers: int = FLOOD_MAX_LAYERS,
) -> np.ndarray:
    """Vectorized :func:`capped_component_sizes` for many ``(grid, x, y)`` starts.

    Two certificates resolve most starts without a Python flood. Both only ever
    look at cells that are provably inside the start's true component:

    - a fully free, in-bounds ``ceil(sqrt(cap))``-square box with the start at a
      corner is 4-connected and holds ``>= cap`` cells, so the answer is ``cap``;
    - otherwise the start is flooded layer by layer inside its own ``(2r+1)^2``
      window (off-board cells count as blocked). Reaching ``cap`` cells proves
      the answer is ``cap``; converging without touching the window border
      proves the region is the whole component, so its count is exact.

    Starts resolved by neither rule (a region that leaves the window below
    ``cap`` cells, or needs more than ``max_layers`` layers) fall back to the
    exact memoized scalar flood. The result is therefore identical to the scalar
    function; the window only decides how fast it is computed.

    Args:
        free_grids: Same-shape bool grids indexed ``[x, y]`` (one per env).
        starts: ``(K, 3)`` int ``(grid_index, x, y)``.
        cap: Exploration cap.
        radius: Window radius ``r``.
        max_layers: Layer budget for the windowed pre-pass.

    Returns:
        ``(K,)`` int64 ``min(component size, cap)`` (0 for blocked/out-of-bounds).
    """
    starts = np.asarray(starts, dtype=np.int64).reshape(-1, 3)
    K = len(starts)
    out = np.zeros(K, dtype=np.int64)
    if K == 0:
        return out
    gw, gh = free_grids[0].shape
    g, x, y = starts[:, 0], starts[:, 1], starts[:, 2]
    inb = (x >= 0) & (x < gw) & (y >= 0) & (y < gh)
    ok = np.zeros(K, dtype=bool)
    stack = np.stack(free_grids)
    ok[inb] = stack[g[inb], x[inb], y[inb]]
    idx = np.flatnonzero(ok)
    if not len(idx):
        return out
    # Certificate 1 (one box scan per start): a fully free in-bounds ``side x side`` box
    # with the start at one of its corners is 4-connected and lies inside the
    # start's component, so ``side**2 >= cap`` proves the capped count is ``cap``.
    side = int(math.ceil(math.sqrt(cap)))
    if side <= min(gw, gh):
        boxes = np.lib.stride_tricks.sliding_window_view(~stack, (side, side), axis=(1, 2))
        gi, xi, yi = g[idx], x[idx], y[idx]
        open_box = np.zeros(len(idx), dtype=bool)
        for x0 in (xi, xi - side + 1):
            for y0 in (yi, yi - side + 1):
                inside = (x0 >= 0) & (x0 + side <= gw) & (y0 >= 0) & (y0 + side <= gh)
                x0c = np.clip(x0, 0, gw - side)
                y0c = np.clip(y0, 0, gh - side)
                todo = inside & ~open_box
                if np.any(todo):
                    hit = boxes[gi[todo], x0c[todo], y0c[todo]].any(axis=(1, 2))
                    open_box[np.flatnonzero(todo)[~hit]] = True
        out[idx[open_box]] = cap
        idx = idx[~open_box]
        if not len(idx):
            return out
    # Certificate 2: layer-by-layer flood inside a local window.
    r = int(radius)
    w = 2 * r + 1
    padded = np.zeros((len(free_grids), gw + 2 * r, gh + 2 * r), dtype=bool)
    padded[:, r : r + gw, r : r + gh] = stack
    views = np.lib.stride_tricks.sliding_window_view(padded, (w, w), axis=(1, 2))
    win = views[g[idx], x[idx], y[idx]].copy()  # (k, w, w), centre = start
    vis = np.zeros_like(win)
    vis[:, r, r] = True
    active = np.arange(len(idx))
    unresolved: List[int] = []
    for _ in range(int(max_layers)):
        if not len(active):
            break
        v = vis[active]
        nb = np.zeros_like(v)
        nb[:, 1:, :] |= v[:, :-1, :]
        nb[:, :-1, :] |= v[:, 1:, :]
        nb[:, :, 1:] |= v[:, :, :-1]
        nb[:, :, :-1] |= v[:, :, 1:]
        new = nb & win[active] & ~v
        grew = new.any(axis=(1, 2))
        v |= new
        vis[active] = v
        count = v.sum(axis=(1, 2))
        big = count >= cap
        border = v[:, 0, :].any(axis=1) | v[:, -1, :].any(axis=1)
        border |= v[:, :, 0].any(axis=1) | v[:, :, -1].any(axis=1)
        closed = ~grew & ~border & ~big
        out[idx[active[big]]] = cap
        out[idx[active[closed]]] = count[closed]
        leaked = ~grew & border & ~big
        unresolved.extend(active[leaked].tolist())
        active = active[grew & ~big]
    unresolved.extend(active.tolist())
    if unresolved:
        by_grid: Dict[int, List[int]] = {}
        for j in unresolved:
            by_grid.setdefault(int(g[idx[j]]), []).append(int(idx[j]))
        for grid_index, ks in by_grid.items():
            cells = [(int(x[k]), int(y[k])) for k in ks]
            sizes = capped_component_sizes(free_grids[grid_index], cells, cap)
            out[ks] = sizes
    return out


class _BatchView:
    """Padded pixel-space snapshot of every env touched by one featurize call.

    Body points are laid out per env as ``P = S * Kmax`` slots (snake-major,
    head-first, padded). For every alive snake ``t`` two masks select its points:
    ``other_ok`` (how others see ``t``: whole body, tail dropped when it will
    vacate) and ``self_ok`` (how ``t`` sees itself: ``segments[2:]``, tail
    dropped when it will vacate). Row ``(e, s)`` uses ``self_ok`` points owned by
    ``s`` and ``other_ok`` points owned by anyone else.
    """

    def __init__(self, sim: BatchSim, envs: np.ndarray, ss: int, gw: int, gh: int) -> None:
        S = sim.S
        u = len(envs)
        self.alive = sim.alive[envs]  # (u, S)
        self.length = sim.length[envs].astype(np.int64)
        self.direction = sim.direction[envs].astype(np.int64)
        n = sim.seg_count[envs].astype(np.int64)  # (u, S)
        kmax = max(int(n.max()) if n.size else 0, 1)
        k = np.arange(kmax, dtype=np.int64)
        ring = (sim.head_ptr[envs].astype(np.int64)[:, :, None] - k) % sim.cap
        cells = sim.bodies[envs[:, None, None], np.arange(S)[None, :, None], ring]
        in_body = k < n[:, :, None]  # (u, S, kmax)
        live_body = in_body & self.alive[:, :, None]
        length = self.length[:, :, None]
        nn = n[:, :, None]
        drop_other = (nn > 1) & (nn >= length)
        other_ok = live_body & (k < nn - drop_other)
        drop_self = (nn >= length) & (nn > 2)
        self_ok = live_body & (k >= 2) & (k < nn - drop_self)
        self.head_cell = cells[:, :, 0, :]  # (u, S, 2); meaningful where n > 0
        self.head_px = self.head_cell * ss
        self.body_px = (cells * ss).reshape(u, S * kmax, 2)
        self.owner = np.repeat(np.arange(S, dtype=np.int64), kmax)  # (P,)
        self.other_ok = other_ok.reshape(u, S * kmax)
        self.self_ok = self_ok.reshape(u, S * kmax)
        # Free-space blocked set: every alive snake's full body.
        free = np.ones((u, gw, gh), dtype=bool)
        bx, by = cells[..., 0], cells[..., 1]
        paint = live_body & (bx >= 0) & (bx < gw) & (by >= 0) & (by < gh)
        ui = np.broadcast_to(np.arange(u)[:, None, None], paint.shape)
        free[ui[paint], bx[paint], by[paint]] = False
        self.free = free
        # Food, padded to the longest list (order preserved).
        counts = [len(sim.food_cells[int(e)]) for e in envs]
        fmax = max(max(counts, default=0), 1)
        food = np.zeros((u, fmax, 2), dtype=np.int64)
        food_ok = np.zeros((u, fmax), dtype=bool)
        for j, e in enumerate(envs.tolist()):
            c = counts[j]
            if c:
                food[j, :c] = sim.food_cells[e]
                food_ok[j, :c] = True
        self.food_px = food * ss
        self.food_ok = food_ok
        self.food_count = np.asarray(counts, dtype=np.int64)


class Vector61Featurizer:
    """Build live-identical vector61/58 states for ``(env, slot)`` rows of a BatchSim.

    Every row of every env is computed in one set of NumPy broadcasts (no
    per-row Python loop); the free-space flood fill is batched across the call.

    Args:
        params: World/network knobs (see :meth:`Vector61Params.from_game_config`).
    """

    def __init__(self, params: Vector61Params) -> None:
        if params.danger_max_distance <= 0:
            raise ValueError("danger_max_distance must be positive")
        self.p = params
        ss = int(params.segment_size)
        self.ss = ss
        self.W = int(params.game_width)
        self.H = int(params.game_height)
        # Live free-space grid is ceil(W / ss); equal to W // ss on aligned worlds.
        self.gw = self.W // ss
        self.gh = self.H // ss
        self.max_dim = max(self.W, self.H)
        self.board_diagonal = math.sqrt(self.W**2 + self.H**2)
        self.maxd = int(params.danger_max_distance) * ss
        self.max_check = ss * 3
        self.wall_offsets = np.arange(-self.maxd, self.maxd + 1, ss, dtype=np.int64)
        key = (self.gw, self.gh, ss, int(params.num_sectors))
        if key not in _LUT_CACHE:
            _LUT_CACHE[key] = _sector_lut(*key)
        self.lut, self.rx, self.ry = _LUT_CACHE[key]
        self.onehot_index = np.array(
            [params.actions.index(t) for t in _CARDINAL_TUPLES], dtype=np.int64
        )
        self.expected_per_sector = max(params.food_capacity / params.num_sectors, 1.0)

    @property
    def input_size(self) -> int:
        """Width of each output row (61 or 58)."""
        return self.p.input_size

    def featurize(
        self,
        sim: BatchSim,
        rows: Optional[np.ndarray] = None,
        memory: Optional[Vector61Memory] = None,
        update_memory: bool = True,
    ) -> np.ndarray:
        """Return the ``(N, input_size)`` float32 live state for each row.

        Args:
            sim: The batch sim (read-only).
            rows: ``(N, 2)`` int ``(env, slot)`` rows. Defaults to every alive row in
                env-major, slot-minor order.
            memory: Nearest-enemy trend baselines. ``None`` behaves like a fresh
                live snake on every call (trend feature 0, nothing stored).
            update_memory: Store this call's nearest-enemy baseline (live
                ``get_state(update_enemy_memory=...)``). Rows must then be
                unique: the live game updates a snake's baseline once per call.

        Returns:
            ``float32`` array. Rows whose snake is dead are all-zero and leave
            ``memory`` untouched (the live game never featurizes a dead snake).
        """
        if sim.cfg.game_width != self.W or sim.cfg.game_height != self.H:
            raise ValueError("sim world size does not match the featurizer params")
        if sim.s != self.ss:
            raise ValueError("sim segment_size does not match the featurizer params")
        if rows is None:
            rows = np.argwhere(sim.alive)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        out = np.zeros((len(rows), self.input_size), dtype=np.float64)
        if not len(rows):
            return out.astype(np.float32)
        if memory is not None and update_memory:
            if len(np.unique(rows[:, 0] * sim.S + rows[:, 1])) != len(rows):
                raise ValueError("rows must be unique when updating trend memory")
        pos = np.flatnonzero(sim.alive[rows[:, 0], rows[:, 1]])
        if not len(pos):
            return out.astype(np.float32)
        envs, ui = np.unique(rows[pos, 0], return_inverse=True)
        view = _BatchView(sim, envs, self.ss, self.gw, self.gh)
        e_rows = rows[pos, 0]
        s_rows = rows[pos, 1]
        out[pos, :58] = self._block(view, ui, e_rows, s_rows, memory, update_memory)
        if self.p.use_free_space:
            out[pos, 58:61] = self._free_space(view, ui, s_rows)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def _block(
        self,
        v: _BatchView,
        ui: np.ndarray,
        e_rows: np.ndarray,
        s_rows: np.ndarray,
        memory: Optional[Vector61Memory],
        update_memory: bool,
    ) -> np.ndarray:
        """Columns 0..57 for alive rows, shape ``(N, 58)``."""
        p = self.p
        N = len(ui)
        block = np.zeros((N, 58), dtype=np.float64)
        hx = v.head_px[ui, s_rows, 0]
        hy = v.head_px[ui, s_rows, 1]
        length = v.length[ui, s_rows]
        logical = np.maximum(1, length)
        d = v.direction[ui, s_rows]
        block[np.arange(N), self.onehot_index[d]] = 1.0
        block[:, 4] = np.minimum(logical / p.max_length, 1.0)
        block[:, 5:24] = self._food(v, ui, hx, hy)
        self_mask = v.self_ok[ui] & (v.owner[None, :] == s_rows[:, None])
        other_mask = v.other_ok[ui] & (v.owner[None, :] != s_rows[:, None])
        bpx = v.body_px[ui]  # (N, P, 2)
        block[:, 24:40] = self._danger_map(bpx, self_mask | other_mask, hx, hy)
        W, H = self.W, self.H
        block[:, 40] = hx / W
        block[:, 41] = (W - hx) / W
        block[:, 42] = hy / H
        block[:, 43] = (H - hy) / H
        block[:, 44:54] = self._enemies(
            v, ui, e_rows, s_rows, hx, hy, logical, memory, update_memory
        )
        block[:, 54:57] = self._per_action(bpx, self_mask, other_mask, hx, hy, d)
        block[:, 57] = (length >= p.min_boost_length).astype(np.float64)
        return block

    def _food(self, v: _BatchView, ui: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
        n = self.p.num_sectors
        N = len(ui)
        out = np.zeros((N, 3 + n), dtype=np.float64)
        food = v.food_px[ui]  # (N, F, 2)
        ok = v.food_ok[ui]
        dxa = food[:, :, 0] - hx[:, None]
        dya = food[:, :, 1] - hy[:, None]
        d2 = np.where(ok, dxa * dxa + dya * dya, np.iinfo(np.int64).max)
        nearest = np.argmin(d2, axis=1)  # first minimum in food-list order
        ar = np.arange(N)
        dx = dxa[ar, nearest]
        dy = dya[ar, nearest]
        dist = np.sqrt(d2[ar, nearest].astype(np.float64))
        has = v.food_count[ui] > 0
        out[:, 0] = np.where(has, dx / self.max_dim, 0.0)
        out[:, 1] = np.where(has, dy / self.max_dim, 0.0)
        out[:, 2] = np.where(has, np.minimum(dist / self.board_diagonal, 1.0), 1.0)
        dxc = np.where(ok, dxa, 0) // self.ss + self.rx
        dyc = np.where(ok, dya, 0) // self.ss + self.ry
        flat = ar[:, None] * n + self.lut[dxc, dyc]
        counts = np.bincount(flat[ok], minlength=N * n).reshape(N, n)
        out[:, 3:] = np.minimum(counts.astype(np.float64) / self.expected_per_sector, 1.0)
        return out

    def _wall_points(
        self, hx: np.ndarray, hy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Live wall samples per row as padded ``(N, 4*O)`` x, y and validity."""
        ss, W, H, maxd = self.ss, self.W, self.H, self.maxd
        off = self.wall_offsets[None, :]
        ys = hy[:, None] + off
        xs = hx[:, None] + off
        ys_ok = (ys >= 0) & (ys < H)
        xs_ok = (xs >= 0) & (xs < W)
        left = (hx < maxd + ss)[:, None] & ys_ok
        right = ((W - hx) < maxd + ss)[:, None] & ys_ok
        top = (hy < maxd + ss)[:, None] & xs_ok
        bottom = ((H - hy) < maxd + ss)[:, None] & xs_ok
        wx = np.concatenate([np.full_like(ys, -ss), np.full_like(ys, W + ss), xs, xs], axis=1)
        wy = np.concatenate([ys, ys, np.full_like(xs, -ss), np.full_like(xs, H + ss)], axis=1)
        valid = np.concatenate([left, right, top, bottom], axis=1)
        return wx, wy, valid

    def _danger_map(
        self, bpx: np.ndarray, body_mask: np.ndarray, hx: np.ndarray, hy: np.ndarray
    ) -> np.ndarray:
        n = self.p.num_sectors
        N = len(hx)
        px, py, valid = bpx[:, :, 0], bpx[:, :, 1], body_mask
        fac = np.full(px.shape, 1.5)
        if self.p.use_boundary_as_danger:
            wx, wy, wvalid = self._wall_points(hx, hy)
            px = np.concatenate([px, wx], axis=1)
            py = np.concatenate([py, wy], axis=1)
            valid = np.concatenate([valid, wvalid], axis=1)
            fac = np.concatenate([fac, np.full(wx.shape, 2.0)], axis=1)
        dxp = px - hx[:, None]
        dyp = py - hy[:, None]
        # Integer squared distances are exact, and correctly rounded sqrt is
        # monotone, so ``sqrt(d2) > maxd`` <=> ``d2 > maxd**2`` on integers. Only
        # in-range points pay for the float path (identical per-element math).
        valid = valid & ((dxp * dxp + dyp * dyp) <= self.maxd * self.maxd)
        rr, cc = np.nonzero(valid)
        vx = dxp[rr, cc]
        vy = dyp[rr, cc]
        dist = np.sqrt((vx * vx + vy * vy).astype(np.float64))
        val = np.maximum(0.0, 1.0 - dist / float(self.maxd)) * fac[rr, cc]
        flat = rr * n + self.lut[vx // self.ss + self.rx, vy // self.ss + self.ry]
        danger = np.zeros(N * n, dtype=np.float64)
        np.maximum.at(danger, flat, val)
        count = np.bincount(flat[val > 0], minlength=N * n)
        bonus = np.minimum(count / 5.0, 1.0) * 0.2
        danger = np.where(count > 1, np.minimum(danger + bonus, 2.0), danger)
        return np.minimum(danger, 1.0).reshape(N, n)

    def _enemies(
        self,
        v: _BatchView,
        ui: np.ndarray,
        e_rows: np.ndarray,
        s_rows: np.ndarray,
        hx: np.ndarray,
        hy: np.ndarray,
        logical: np.ndarray,
        memory: Optional[Vector61Memory],
        update_memory: bool,
    ) -> np.ndarray:
        N = len(ui)
        S = v.alive.shape[1]
        ar = np.arange(N)
        f = np.zeros((N, 10), dtype=np.float64)
        heads = v.head_px[ui]  # (N, S, 2)
        dx = heads[:, :, 0] - hx[:, None]
        dy = heads[:, :, 1] - hy[:, None]
        dist = np.sqrt((dx * dx + dy * dy).astype(np.float64))
        ok = v.alive[ui] & (np.arange(S)[None, :] != s_rows[:, None])
        # Stable sort by distance: ties keep slot order, like the live list sort.
        order = np.argsort(np.where(ok, dist, np.inf), axis=1, kind="stable")
        n_ok = ok.sum(axis=1)
        first = order[:, 0]
        second = order[:, 1] if S > 1 else order[:, 0]
        has1 = n_ok >= 1
        has2 = n_ok >= 2
        lens = np.maximum(1, v.length[ui])  # (N, S) logical lengths
        own = np.maximum(logical, 1)
        dirs = v.direction[ui]
        t1 = first
        f[:, 0] = np.where(has1, dx[ar, t1] / self.max_dim, 0.0)
        f[:, 1] = np.where(has1, dy[ar, t1] / self.max_dim, 0.0)
        f[:, 2] = np.where(has1, np.minimum(lens[ar, t1] / own, 2.0) / 2.0, 0.0)
        hd = CARDINAL[dirs[ar, t1]]
        f[:, 3] = np.where(has1, hd[:, 0].astype(np.float64), 0.0)
        f[:, 4] = np.where(has1, hd[:, 1].astype(np.float64), 0.0)
        d1 = dist[ar, t1]
        if memory is not None:
            prev = memory.prev_dist[e_rows, s_rows]
            prev_id = memory.prev_id[e_rows, s_rows]
            same = ~np.isinf(prev) & (prev_id == t1)
            trend = np.where(d1 < prev, 1.0, np.where(d1 > prev, -1.0, 0.0))
            f[:, 5] = np.where(has1 & same, trend, 0.0)
            if update_memory:
                memory.prev_dist[e_rows, s_rows] = np.where(has1, d1, np.inf)
                memory.prev_id[e_rows, s_rows] = np.where(has1, t1, -1)
        ex = heads[ar, t1, 0] + hd[:, 0] * self.ss
        ey = heads[ar, t1, 1] + hd[:, 1] * self.ss
        kdx = hx - ex
        kdy = hy - ey
        kill_dist = np.sqrt((kdx * kdx + kdy * kdy).astype(np.float64))
        f[:, 9] = np.where(has1 & (kill_dist < self.ss * 3), 1.0, 0.0)
        t2 = second
        f[:, 6] = np.where(has2, dx[ar, t2] / self.max_dim, 0.0)
        f[:, 7] = np.where(has2, dy[ar, t2] / self.max_dim, 0.0)
        f[:, 8] = np.where(has2, np.minimum(lens[ar, t2] / own, 2.0) / 2.0, 0.0)
        return f

    def _per_action(
        self,
        bpx: np.ndarray,
        self_mask: np.ndarray,
        other_mask: np.ndarray,
        hx: np.ndarray,
        hy: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        ss, W, H, mc = self.ss, self.W, self.H, self.max_check
        dirs3 = (d[:, None] + np.array([-1, 0, 1])) % 4  # (N, 3): left, straight, right
        nx = hx[:, None] + CARDINAL[dirs3, 0] * ss
        ny = hy[:, None] + CARDINAL[dirs3, 1] * ss
        wall_hit = (nx < 0) | (nx >= W) | (ny < 0) | (ny >= H)
        min_wall = np.minimum(np.minimum(nx, W - nx), np.minimum(ny, H - ny))
        nearest = np.minimum(mc, min_wall).astype(np.float64)
        ddx = nx[:, :, None] - bpx[:, None, :, 0]
        ddy = ny[:, :, None] - bpx[:, None, :, 1]
        d2 = ddx * ddx + ddy * ddy  # (N, 3, P) exact integers
        # sqrt is correctly rounded and monotone: ``sqrt(d2) < ss`` <=> ``d2 < ss**2``
        # and ``min(sqrt(d2)) == sqrt(min(d2))`` bit for bit.
        close = d2 < ss * ss
        hit = wall_hit | (close & self_mask[:, None, :]).any(axis=2)
        hit |= (close & other_mask[:, None, :]).any(axis=2)
        either = (self_mask | other_mask)[:, None, :]
        if d2.shape[2]:
            big = np.iinfo(np.int64).max
            d2min = np.where(either, d2, big).min(axis=2)
            obstacle = np.where(d2min == big, np.inf, np.sqrt(d2min.astype(np.float64)))
            nearest = np.minimum(nearest, obstacle)
        soft = np.where(nearest < mc, np.minimum(1.0 - nearest / mc, 0.95), 0.0)
        return np.where(hit, 1.0, soft)

    def _free_space(self, v: _BatchView, ui: np.ndarray, s_rows: np.ndarray) -> np.ndarray:
        N = len(ui)
        d = v.direction[ui, s_rows]
        dirs3 = (d[:, None] + np.array([-1, 0, 1])) % 4
        cand = v.head_cell[ui, s_rows][:, None, :] + CARDINAL[dirs3]  # (N, 3, 2)
        starts = np.empty((N * 3, 3), dtype=np.int64)
        starts[:, 0] = np.repeat(ui, 3)
        starts[:, 1:] = cand.reshape(-1, 2)
        sizes = batched_capped_component_sizes(list(v.free), starts, FREE_SPACE_BFS_CAP)
        sizes = sizes.reshape(N, 3)
        length = v.length[ui, s_rows]
        cap = np.minimum(FREE_SPACE_BFS_CAP, np.maximum(FREE_SPACE_MIN_CAP, length * 2))
        count = np.minimum(sizes, cap[:, None])
        return np.minimum(count / cap[:, None], 1.0)
