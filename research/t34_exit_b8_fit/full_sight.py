"""Full-arena planes for the optional ExIt-B8 S1a tower.

The representation deliberately starts from :class:`EgoGridView`, the common
surface already used by the SIMD and live ego-raster paths.  Consequently the
live builder and simulator builder execute identical painting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_raster import EgoGridView
from src.simd_env.grid_sim import FOOD_AMBIENT, FOOD_CORPSE

FULL_SIGHT_CHANNELS = 4
FULL_SIGHT_HEIGHT = 88
FULL_SIGHT_WIDTH = 152
FULL_SIGHT_SHAPE = (FULL_SIGHT_CHANNELS, FULL_SIGHT_HEIGHT, FULL_SIGHT_WIDTH)

__all__ = [
    "FULL_SIGHT_CHANNELS",
    "FULL_SIGHT_HEIGHT",
    "FULL_SIGHT_WIDTH",
    "FULL_SIGHT_SHAPE",
    "FullSight",
    "LiveFullSight",
    "full_sight_from_ego_view",
]


@dataclass(frozen=True)
class FullSight:
    """A uint8 ``(N, 4, 88, 152)`` plane batch with its source row identities."""

    planes: np.ndarray
    rows: np.ndarray


def _normalise_rows(
    view: EgoGridView, rows: np.ndarray | Iterable[tuple[int, int]] | None
) -> np.ndarray:
    if rows is None:
        env, slot = np.nonzero(view.alive)
        return np.stack((env, slot), axis=1).astype(np.int64, copy=False)
    result = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    if len(result) and (
        (result[:, 0] < 0).any()
        or (result[:, 0] >= view.E).any()
        or (result[:, 1] < 0).any()
        or (result[:, 1] >= view.S).any()
    ):
        raise ValueError("full-sight row is outside the source view")
    return result


def full_sight_from_ego_view(
    view: EgoGridView, rows: np.ndarray | Iterable[tuple[int, int]] | None = None
) -> FullSight:
    """Paint fixed-size own-TTV, other-TTV, food, and head planes.

    TTV is ``seg_count - ((head_ptr - slot) % cap)`` clipped to a byte.  Food
    uses 128 for ambient and 255 for corpse/trail food.  The head plane uses
    255 own, 128 enemy, and 64 for an enemy's next normal cell; stronger head
    values win if the marks overlap.  Smaller arenas are zero padded on the
    bottom/right, making the fixed 152x88 contract explicit.
    """
    if view.grid_w > FULL_SIGHT_WIDTH or view.grid_h > FULL_SIGHT_HEIGHT:
        raise ValueError(
            f"full-sight source {view.grid_w}x{view.grid_h} exceeds "
            f"{FULL_SIGHT_WIDTH}x{FULL_SIGHT_HEIGHT}"
        )
    picked = _normalise_rows(view, rows)
    out = np.zeros((len(picked), *FULL_SIGHT_SHAPE), dtype=np.uint8)
    if not len(picked):
        return FullSight(out, picked)

    P, H, W = view.pad, view.grid_h, view.grid_w
    owner_all = view.owner_pad[:, P : P + H, P : P + W].astype(np.int64, copy=False)
    slot_all = view.slot_pad[:, P : P + H, P : P + W]
    food_all = view.food_pad[:, P : P + H, P : P + W]
    for out_row, (env, hero) in enumerate(picked):
        owner = owner_all[env]
        occupied = owner >= 0
        safe_owner = np.where(occupied, owner, 0)
        head_ptr = view.head_ptr[env, safe_owner]
        seg_count = view.seg_count[env, safe_owner]
        offset = (head_ptr - slot_all) % view.cap
        ttl = np.clip(seg_count - offset, 0, 255).astype(np.uint8)
        own = occupied & (owner == hero)
        other = occupied & ~own
        out[out_row, 0, :H, :W] = np.where(own, ttl, 0)
        out[out_row, 1, :H, :W] = np.where(other, ttl, 0)
        food = out[out_row, 2, :H, :W]
        food[food_all[env] == FOOD_AMBIENT] = 128
        food[food_all[env] == FOOD_CORPSE] = 255

        heads = out[out_row, 3, :H, :W]
        for snake in range(view.S):
            if not view.alive[env, snake]:
                continue
            col, row = view.heads[env, snake]
            if 0 <= col < W and 0 <= row < H:
                heads[row, col] = max(heads[row, col], 255 if snake == hero else 128)
            if snake != hero:
                # heading order is up, right, down, left (the GridBatchSim contract).
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[int(view.direction[env, snake])]
                ncol, nrow = col + dx, row + dy
                if 0 <= ncol < W and 0 <= nrow < H:
                    heads[nrow, ncol] = max(heads[nrow, ncol], 64)
    return FullSight(out, picked)


class LiveFullSight:
    """Live counterpart of :func:`full_sight_from_ego_view`.

    This class contains no independent rasterization logic: it uses the same
    ``EgoGridView`` construction and same painter as the SIMD path, which is
    what the byte-identity test qualifies.
    """

    @staticmethod
    def build(game_state: Any, hero_slot: int) -> np.ndarray:
        view = game_state_to_ego_view(game_state)
        return full_sight_from_ego_view(view, np.array([[0, hero_slot]], dtype=np.int64)).planes[0]
