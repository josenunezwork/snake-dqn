"""Vectorized, decision-identical scripted anchors for the grid sim (redesign M1).

:class:`FastProfileAnchorSimdPolicy` makes exactly the decisions of
``eval_engine._ProfileAnchorSimdPolicy`` (the gate's ``scripted-anchor/v1``
greedy_food / random_safe rules of :mod:`src.evaluation.anchors`) for the same
rows, at a few microseconds per row instead of building one ``AnchorContext``
(with a Python tuple of every pellet) per row:

- ``greedy_food``: a numba kernel. The nearest-pellet distance is a minimum over
  the env's pellet SET, so reading the pellets from ``GridBatchSim``'s food grid
  (whose cells equal ``food_cells`` as a set: a CI-pinned grid invariant) gives
  the same integer distances as the anchor's walk over the ordered list; ties
  break straight, left, right as in :func:`~src.evaluation.anchors.greedy_food_action`.
- ``random_safe``: the anchor's own SHA-256 key (world seed, slot, transition
  frame) per row; nothing to vectorize without changing the hash.

Pinned decision-identical by ``tests/test_fast_anchors.py``.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from numba import njit

from src.evaluation.anchors import (
    SCRIPTED_ANCHOR_KINDS,
    SCRIPTED_ANCHOR_VERSION,
    watch_anchor_frame,
)

__all__ = ["FastProfileAnchorSimdPolicy"]


@njit(cache=True, nogil=True)
def _nearest_food_d2(food_pad: np.ndarray, e: int, pad: int, H: int, W: int, x: int, y: int):
    """Exact min squared distance from ``(x, y)`` to a pellet of env ``e``; -1 if none.

    Rings of growing Chebyshev radius ``r`` around the cell; every cell on ring ``r`` is
    at squared distance >= ``r * r``, so once the best found is <= ``r * r`` no farther
    ring can beat it.
    """
    best = -1
    rmax = max(H, W) + abs(x) + abs(y)
    for r in range(rmax + 1):
        if best >= 0 and best <= r * r:
            break
        if r == 0:
            if 0 <= x < W and 0 <= y < H and food_pad[e, y + pad, x + pad] > 0:
                best = 0
            continue
        for yy in (y - r, y + r):  # top and bottom rows of the ring
            if 0 <= yy < H:
                for xx in range(max(x - r, 0), min(x + r, W - 1) + 1):
                    if food_pad[e, yy + pad, xx + pad] > 0:
                        v = (xx - x) * (xx - x) + (yy - y) * (yy - y)
                        if best < 0 or v < best:
                            best = v
        for xx in (x - r, x + r):  # left and right columns, corners excluded
            if 0 <= xx < W:
                for yy in range(max(y - r + 1, 0), min(y + r - 1, H - 1) + 1):
                    if food_pad[e, yy + pad, xx + pad] > 0:
                        v = (xx - x) * (xx - x) + (yy - y) * (yy - y)
                        if best < 0 or v < best:
                            best = v
        if best < 0 and x - r <= 0 and y - r <= 0 and x + r >= W - 1 and y + r >= H - 1:
            break  # the rings have covered the whole arena: no pellet at all
    return best


@njit(cache=True, nogil=True)
def _greedy_kernel(
    food_pad: np.ndarray,  # (E, Hp, Wp) int8 (>0 = pellet)
    pad: int,
    H: int,
    W: int,
    rows: np.ndarray,  # (N, 2) env, slot
    heads: np.ndarray,  # (E, S, 2) col, row
    headings: np.ndarray,  # (E, S, 2) dx, dy
    masks: np.ndarray,  # (N, 6) bool
    out: np.ndarray,  # (N,) int64
) -> None:
    for i in range(rows.shape[0]):
        e = rows[i, 0]
        s = rows[i, 1]
        best = -1
        best_d = 0
        best_p = 0
        for rel in range(3):
            if not masks[i, rel]:
                continue
            prio = 1 if rel == 0 else (0 if rel == 1 else 2)
            dx = headings[e, s, 0]
            dy = headings[e, s, 1]
            if rel == 0:
                dx, dy = dy, -dx
            elif rel == 2:
                dx, dy = -dy, dx
            d = _nearest_food_d2(food_pad, e, pad, H, W, heads[e, s, 0] + dx, heads[e, s, 1] + dy)
            if d < 0:
                d = 0  # no pellet in the env: the anchor ranks by priority alone
            if best < 0 or d < best_d or (d == best_d and prio < best_p):
                best = rel
                best_d = d
                best_p = prio
        out[i] = 1 if best < 0 else best


def _random_safe(world_seed: int, slot: int, frame: int, mask: np.ndarray) -> int:
    safe = [a for a in range(3) if mask[a]]
    if not safe:
        return 1
    key = f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{world_seed}|{slot}|{frame}"
    index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(safe)
    return safe[index]


class FastProfileAnchorSimdPolicy:
    """Drop-in for ``_ProfileAnchorSimdPolicy`` on a :class:`GridBatchSim` (same actions).

    Duck-types ``SimdPolicy.actions(masks, sim, slots)``.
    """

    def __init__(self, kind: str, world_seeds: Sequence[int]) -> None:
        if kind not in SCRIPTED_ANCHOR_KINDS:
            raise ValueError(f"unknown scripted anchor kind {kind!r}")
        self.kind = kind
        self._world_seeds = tuple(int(seed) for seed in world_seeds)

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        slots = np.ascontiguousarray(np.asarray(slots, dtype=np.int64).reshape(-1, 2))
        masks = np.ascontiguousarray(masks, dtype=np.bool_)
        out = np.ones(len(slots), dtype=np.int64)
        if not len(slots):
            return out
        if self.kind == "random_safe":
            for row, (env, slot) in enumerate(slots):
                out[row] = _random_safe(
                    self._world_seeds[int(env)],
                    int(slot),
                    watch_anchor_frame(int(sim.frame[int(env)])),
                    masks[row],
                )
            return out
        from src.simd_env.grid_sim import GRID_PAD

        _, _, food_pad = sim.get_padded_grids()
        _greedy_kernel(
            food_pad,
            GRID_PAD,
            int(sim.grid_h),
            int(sim.grid_w),
            slots,
            np.ascontiguousarray(sim.get_heads(), dtype=np.int64),
            np.ascontiguousarray(sim.get_direction_vectors(), dtype=np.int64),
            masks,
            out,
        )
        return out
