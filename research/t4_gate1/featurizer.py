"""RNG-stripped privileged Markovian arena + ego + scalar featurizer for Track-4 Gate 1.

Extracts compact lossless board samples (~4-6 KB/row, storing uint8 segment/food coordinates
and kinematics) from ``DeviceSim`` + ``DevicePolicy`` without ever reading ``rng_key`` or
``rng_counter``, and reconstructs 22 full-arena ``83x145`` planes, 23 heading-rotated ``31x31``
ego planes, and 64 privileged scalars on CPU or CUDA via pure vectorized tensor scatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.core.mechanics_constants import HEADON_SIZE_RATIO

ARENA_H = 83
ARENA_W = 145
ARENA_CHANNELS = 22
EGO_SIZE = 31
EGO_HALF = EGO_SIZE // 2
EGO_CHANNELS = ARENA_CHANNELS + 1  # +1 explicit wall_oob plane
SCALAR_DIM = 64

_CARDINAL_NP = np.asarray(((0, -1), (1, 0), (0, 1), (-1, 0)), dtype=np.int64)


@dataclass
class FeaturizedBatch:
    """Device-resident privileged critic inputs for a batch of B states."""

    arena: torch.Tensor  # (B, 22, 83, 145) float32
    ego: torch.Tensor  # (B, 23, 31, 31) float32
    scalars: torch.Tensor  # (B, 64) float32
    hero_xy_dir: torch.Tensor  # (B, 3) int64: (hx, hy, dir)
    legal_mask: torch.Tensor  # (B, 6) bool
    hero_length: torch.Tensor  # (B,) float32
    h_rem: torch.Tensor  # (B,) float32


@dataclass
class CompactBoardDataset:
    """Lossless RNG-stripped CSR-packed board dataset (~5 KB/row) with per-life return targets."""

    num_rows: int
    world_id: np.ndarray  # (N,) int32
    seed: np.ndarray  # (N,) int64
    life_id: np.ndarray  # (N,) int32
    tick: np.ndarray  # (N,) int32
    h_rem: np.ndarray  # (N,) int32
    b8_action: np.ndarray  # (N,) int64
    legal_mask: np.ndarray  # (N, 6) bool

    # Per-snake kinematics (slot 0 = hero, slots 1..5 = opponents)
    snake_alive: np.ndarray  # (N, 6) bool
    snake_length: np.ndarray  # (N, 6) int32
    snake_seg_count: np.ndarray  # (N, 6) int32
    snake_dir: np.ndarray  # (N, 6) int8
    snake_boost_frames: np.ndarray  # (N, 6) int8
    snake_hunger: np.ndarray  # (N, 6) int32
    snake_respawn: np.ndarray  # (N, 6) int8
    snake_boosted: np.ndarray  # (N, 6) bool
    head_xy: np.ndarray  # (N, 6, 2) uint8
    tail_xy: np.ndarray  # (N, 6, 2) uint8

    # CSR-packed segments across all rows
    seg_row_offsets: np.ndarray  # (N + 1,) int64
    seg_row_counts: np.ndarray  # (N,) int32
    seg_slot: np.ndarray  # (M_seg,) uint8
    seg_x: np.ndarray  # (M_seg,) uint8
    seg_y: np.ndarray  # (M_seg,) uint8
    seg_ttl: np.ndarray  # (M_seg,) int16

    # CSR-packed live food across all rows
    food_row_offsets: np.ndarray  # (N + 1,) int64
    food_row_counts: np.ndarray  # (N,) int32
    food_x: np.ndarray  # (M_food,) uint8
    food_y: np.ndarray  # (M_food,) uint8
    food_is_corpse: np.ndarray  # (M_food,) bool

    # Supervised targets populated once trajectory ends (zeros on standalone root batches)
    tau_alive: np.ndarray  # (N,) int32
    died_in_session: np.ndarray  # (N,) bool
    return_g: np.ndarray  # (N,) float32
    forage_short: np.ndarray  # (N,) float32
    forage_short_weight: np.ndarray  # (N,) float32
    forage_long: np.ndarray  # (N,) float32
    forage_long_valid: np.ndarray  # (N,) bool


class CompactSampleBuilder:
    """Incremental collector for ``CompactBoardDataset`` chunks during rollout."""

    def __init__(self) -> None:
        self._num_rows = 0
        self._world_id: list[np.ndarray] = []
        self._seed: list[np.ndarray] = []
        self._life_id: list[np.ndarray] = []
        self._tick: list[np.ndarray] = []
        self._h_rem: list[np.ndarray] = []
        self._b8_action: list[np.ndarray] = []
        self._legal_mask: list[np.ndarray] = []
        self._snake_alive: list[np.ndarray] = []
        self._snake_length: list[np.ndarray] = []
        self._snake_seg_count: list[np.ndarray] = []
        self._snake_dir: list[np.ndarray] = []
        self._snake_boost_frames: list[np.ndarray] = []
        self._snake_hunger: list[np.ndarray] = []
        self._snake_respawn: list[np.ndarray] = []
        self._snake_boosted: list[np.ndarray] = []
        self._head_xy: list[np.ndarray] = []
        self._tail_xy: list[np.ndarray] = []
        self._seg_counts: list[np.ndarray] = []
        self._seg_slot: list[np.ndarray] = []
        self._seg_x: list[np.ndarray] = []
        self._seg_y: list[np.ndarray] = []
        self._seg_ttl: list[np.ndarray] = []
        self._food_counts: list[np.ndarray] = []
        self._food_x: list[np.ndarray] = []
        self._food_y: list[np.ndarray] = []
        self._food_corpse: list[np.ndarray] = []

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @torch.inference_mode()
    def append_from_sim(
        self,
        sim: Any,
        *,
        env_indices: torch.Tensor,
        world_ids: np.ndarray,
        seeds: np.ndarray,
        life_ids: np.ndarray,
        ticks: np.ndarray,
        h_rems: np.ndarray,
        b8_actions: np.ndarray,
        legal_masks: np.ndarray,
    ) -> np.ndarray:
        """Append selected live rows from ``sim`` without ever reading RNG fields.

        Returns the integer dataset row indices ``[start .. start + M - 1]`` assigned.
        """
        idx_t = env_indices.to(device=sim.device, dtype=torch.long).reshape(-1)
        m = int(idx_t.numel())
        if m == 0:
            return np.zeros(0, dtype=np.int64)
        if int(sim.grid_h) != ARENA_H or int(sim.grid_w) != ARENA_W or int(sim.S) != 6:
            raise ValueError(
                f"Expected Watch6 83x145 6-snake sim, got {sim.grid_h}x{sim.grid_w} S={sim.S}"
            )

        start_row = self._num_rows
        self._num_rows += m

        self._world_id.append(np.asarray(world_ids, dtype=np.int32).reshape(m))
        self._seed.append(np.asarray(seeds, dtype=np.int64).reshape(m))
        self._life_id.append(np.asarray(life_ids, dtype=np.int32).reshape(m))
        self._tick.append(np.asarray(ticks, dtype=np.int32).reshape(m))
        self._h_rem.append(np.asarray(h_rems, dtype=np.int32).reshape(m))
        self._b8_action.append(np.asarray(b8_actions, dtype=np.int64).reshape(m))

        leg = np.asarray(legal_masks, dtype=bool).reshape(m, 6).copy()
        no_leg = ~leg.any(axis=1)
        if np.any(no_leg):
            leg[no_leg, :3] = True
        self._legal_mask.append(leg)

        al_t = sim.alive.index_select(0, idx_t)
        ln_t = sim.length.index_select(0, idx_t)
        sc_t = sim.seg_count.index_select(0, idx_t)
        hp_t = sim.head_ptr.index_select(0, idx_t)
        bodies_t = sim.bodies.index_select(0, idx_t)

        self._snake_alive.append(al_t.detach().cpu().numpy().astype(bool))
        self._snake_length.append(ln_t.detach().cpu().numpy().astype(np.int32))
        self._snake_seg_count.append(sc_t.detach().cpu().numpy().astype(np.int32))
        self._snake_dir.append(
            sim.direction.index_select(0, idx_t).detach().cpu().numpy().astype(np.int8)
        )
        self._snake_boost_frames.append(
            sim.boost_frames.index_select(0, idx_t).detach().cpu().numpy().astype(np.int8)
        )
        self._snake_hunger.append(
            sim.frames_since_food.index_select(0, idx_t).detach().cpu().numpy().astype(np.int32)
        )
        self._snake_respawn.append(
            sim.respawn_timer.index_select(0, idx_t).detach().cpu().numpy().astype(np.int8)
        )
        self._snake_boosted.append(
            sim._boosted_this_step.index_select(0, idx_t).detach().cpu().numpy().astype(bool)
        )

        # Head and tail coordinates per snake (clamped to valid uint8 arena bounds).
        e_grid = torch.arange(m, device=sim.device)[:, None]
        s_grid = torch.arange(6, device=sim.device)[None, :]
        head_cells = bodies_t[e_grid, s_grid, hp_t % sim.cap]
        tail_ring = (hp_t - (sc_t - 1).clamp_min(0)) % sim.cap
        tail_cells = bodies_t[e_grid, s_grid, tail_ring]
        head_clamped = torch.stack(
            (head_cells[..., 0].clamp(0, ARENA_W - 1), head_cells[..., 1].clamp(0, ARENA_H - 1)),
            dim=-1,
        )
        tail_clamped = torch.stack(
            (tail_cells[..., 0].clamp(0, ARENA_W - 1), tail_cells[..., 1].clamp(0, ARENA_H - 1)),
            dim=-1,
        )
        self._head_xy.append(head_clamped.detach().cpu().numpy().astype(np.uint8))
        self._tail_xy.append(tail_clamped.detach().cpu().numpy().astype(np.uint8))

        # Vectorized segment extraction across all 6 snakes.
        max_sc = int(sc_t.max().item()) if m > 0 else 0
        if max_sc > 0:
            k_t = torch.arange(max_sc, device=sim.device)[None, None, :]
            valid_seg = al_t[..., None] & (k_t < sc_t[..., None])
            rings = (hp_t[..., None] - k_t) % sim.cap
            seg_cells = bodies_t.gather(2, rings[..., None].expand(-1, -1, -1, 2))
            sx, sy = seg_cells[..., 0], seg_cells[..., 1]
            valid_seg &= (sx >= 0) & (sx < ARENA_W) & (sy >= 0) & (sy < ARENA_H)
            ttl_t = (ln_t[..., None] - k_t).clamp(1, 32767)
            slot_t = torch.arange(6, device=sim.device)[None, :, None].expand_as(sx)

            row_counts = valid_seg.reshape(m, -1).sum(dim=1)
            flat_mask = valid_seg.reshape(-1)
            self._seg_counts.append(row_counts.detach().cpu().numpy().astype(np.int32))
            self._seg_slot.append(
                slot_t.reshape(-1)[flat_mask].detach().cpu().numpy().astype(np.uint8)
            )
            self._seg_x.append(sx.reshape(-1)[flat_mask].detach().cpu().numpy().astype(np.uint8))
            self._seg_y.append(sy.reshape(-1)[flat_mask].detach().cpu().numpy().astype(np.uint8))
            self._seg_ttl.append(
                ttl_t.reshape(-1)[flat_mask].detach().cpu().numpy().astype(np.int16)
            )
        else:
            self._seg_counts.append(np.zeros(m, dtype=np.int32))

        # Vectorized live food extraction.
        fc_t = sim.food_count.index_select(0, idx_t)
        max_fc = int(fc_t.max().item()) if m > 0 else 0
        if max_fc > 0:
            fpres = sim.food_present.index_select(0, idx_t)[:, :max_fc]
            fcells = sim.food_cells.index_select(0, idx_t)[:, :max_fc]
            fcorpse = sim.corpse_food.index_select(0, idx_t)[:, :max_fc]
            fx, fy = fcells[..., 0], fcells[..., 1]
            fvalid = fpres & (fx >= 0) & (fx < ARENA_W) & (fy >= 0) & (fy < ARENA_H)
            f_row_counts = fvalid.sum(dim=1)
            f_flat = fvalid.reshape(-1)
            self._food_counts.append(f_row_counts.detach().cpu().numpy().astype(np.int32))
            self._food_x.append(fx.reshape(-1)[f_flat].detach().cpu().numpy().astype(np.uint8))
            self._food_y.append(fy.reshape(-1)[f_flat].detach().cpu().numpy().astype(np.uint8))
            self._food_corpse.append(
                fcorpse.reshape(-1)[f_flat].detach().cpu().numpy().astype(bool)
            )
        else:
            self._food_counts.append(np.zeros(m, dtype=np.int32))

        return np.arange(start_row, start_row + m, dtype=np.int64)

    def build(
        self,
        *,
        tau_alive: np.ndarray | None = None,
        died_in_session: np.ndarray | None = None,
        return_g: np.ndarray | None = None,
        forage_short: np.ndarray | None = None,
        forage_short_weight: np.ndarray | None = None,
        forage_long: np.ndarray | None = None,
        forage_long_valid: np.ndarray | None = None,
    ) -> CompactBoardDataset:
        n = self._num_rows
        if n == 0:
            empty_i32 = np.zeros(0, dtype=np.int32)
            empty_i64 = np.zeros(0, dtype=np.int64)
            empty_f32 = np.zeros(0, dtype=np.float32)
            empty_b = np.zeros(0, dtype=bool)
            empty_u8 = np.zeros(0, dtype=np.uint8)
            return CompactBoardDataset(
                num_rows=0,
                world_id=empty_i32,
                seed=empty_i64,
                life_id=empty_i32,
                tick=empty_i32,
                h_rem=empty_i32,
                b8_action=empty_i64,
                legal_mask=np.zeros((0, 6), dtype=bool),
                snake_alive=np.zeros((0, 6), dtype=bool),
                snake_length=np.zeros((0, 6), dtype=np.int32),
                snake_seg_count=np.zeros((0, 6), dtype=np.int32),
                snake_dir=np.zeros((0, 6), dtype=np.int8),
                snake_boost_frames=np.zeros((0, 6), dtype=np.int8),
                snake_hunger=np.zeros((0, 6), dtype=np.int32),
                snake_respawn=np.zeros((0, 6), dtype=np.int8),
                snake_boosted=np.zeros((0, 6), dtype=bool),
                head_xy=np.zeros((0, 6, 2), dtype=np.uint8),
                tail_xy=np.zeros((0, 6, 2), dtype=np.uint8),
                seg_row_offsets=np.zeros(1, dtype=np.int64),
                seg_row_counts=empty_i32,
                seg_slot=empty_u8,
                seg_x=empty_u8,
                seg_y=empty_u8,
                seg_ttl=np.zeros(0, dtype=np.int16),
                food_row_offsets=np.zeros(1, dtype=np.int64),
                food_row_counts=empty_i32,
                food_x=empty_u8,
                food_y=empty_u8,
                food_is_corpse=empty_b,
                tau_alive=empty_i32,
                died_in_session=empty_b,
                return_g=empty_f32,
                forage_short=empty_f32,
                forage_short_weight=empty_f32,
                forage_long=empty_f32,
                forage_long_valid=empty_b,
            )

        seg_counts = np.concatenate(self._seg_counts, axis=0).astype(np.int32)
        seg_offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(seg_counts, out=seg_offsets[1:])

        food_counts = np.concatenate(self._food_counts, axis=0).astype(np.int32)
        food_offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(food_counts, out=food_offsets[1:])

        return CompactBoardDataset(
            num_rows=n,
            world_id=np.concatenate(self._world_id, axis=0),
            seed=np.concatenate(self._seed, axis=0),
            life_id=np.concatenate(self._life_id, axis=0),
            tick=np.concatenate(self._tick, axis=0),
            h_rem=np.concatenate(self._h_rem, axis=0),
            b8_action=np.concatenate(self._b8_action, axis=0),
            legal_mask=np.concatenate(self._legal_mask, axis=0),
            snake_alive=np.concatenate(self._snake_alive, axis=0),
            snake_length=np.concatenate(self._snake_length, axis=0),
            snake_seg_count=np.concatenate(self._snake_seg_count, axis=0),
            snake_dir=np.concatenate(self._snake_dir, axis=0),
            snake_boost_frames=np.concatenate(self._snake_boost_frames, axis=0),
            snake_hunger=np.concatenate(self._snake_hunger, axis=0),
            snake_respawn=np.concatenate(self._snake_respawn, axis=0),
            snake_boosted=np.concatenate(self._snake_boosted, axis=0),
            head_xy=np.concatenate(self._head_xy, axis=0),
            tail_xy=np.concatenate(self._tail_xy, axis=0),
            seg_row_offsets=seg_offsets,
            seg_row_counts=seg_counts,
            seg_slot=(
                np.concatenate(self._seg_slot, axis=0)
                if self._seg_slot
                else np.zeros(0, dtype=np.uint8)
            ),
            seg_x=(
                np.concatenate(self._seg_x, axis=0) if self._seg_x else np.zeros(0, dtype=np.uint8)
            ),
            seg_y=(
                np.concatenate(self._seg_y, axis=0) if self._seg_y else np.zeros(0, dtype=np.uint8)
            ),
            seg_ttl=(
                np.concatenate(self._seg_ttl, axis=0)
                if self._seg_ttl
                else np.zeros(0, dtype=np.int16)
            ),
            food_row_offsets=food_offsets,
            food_row_counts=food_counts,
            food_x=(
                np.concatenate(self._food_x, axis=0)
                if self._food_x
                else np.zeros(0, dtype=np.uint8)
            ),
            food_y=(
                np.concatenate(self._food_y, axis=0)
                if self._food_y
                else np.zeros(0, dtype=np.uint8)
            ),
            food_is_corpse=(
                np.concatenate(self._food_corpse, axis=0)
                if self._food_corpse
                else np.zeros(0, dtype=bool)
            ),
            tau_alive=(
                np.asarray(tau_alive, dtype=np.int32).reshape(n)
                if tau_alive is not None
                else np.zeros(n, dtype=np.int32)
            ),
            died_in_session=(
                np.asarray(died_in_session, dtype=bool).reshape(n)
                if died_in_session is not None
                else np.zeros(n, dtype=bool)
            ),
            return_g=(
                np.asarray(return_g, dtype=np.float32).reshape(n)
                if return_g is not None
                else np.zeros(n, dtype=np.float32)
            ),
            forage_short=(
                np.asarray(forage_short, dtype=np.float32).reshape(n)
                if forage_short is not None
                else np.zeros(n, dtype=np.float32)
            ),
            forage_short_weight=(
                np.asarray(forage_short_weight, dtype=np.float32).reshape(n)
                if forage_short_weight is not None
                else np.zeros(n, dtype=np.float32)
            ),
            forage_long=(
                np.asarray(forage_long, dtype=np.float32).reshape(n)
                if forage_long is not None
                else np.zeros(n, dtype=np.float32)
            ),
            forage_long_valid=(
                np.asarray(forage_long_valid, dtype=bool).reshape(n)
                if forage_long_valid is not None
                else np.zeros(n, dtype=bool)
            ),
        )


def _csr_gather_indices(
    row_offsets: np.ndarray,
    row_counts: np.ndarray,
    batch_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather flat CSR indices and batch-row IDs ``(0..B-1)`` without Python loops."""
    counts = row_counts[batch_indices].astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = row_offsets[batch_indices].astype(np.int64)
    cum = np.cumsum(counts) - counts
    shifts = np.repeat(starts - cum, counts)
    flat_idx = np.arange(total, dtype=np.int64) + shifts
    batch_row_id = np.repeat(np.arange(len(batch_indices), dtype=np.int64), counts)
    return flat_idx, batch_row_id


def featurize_dataset_batch(
    dataset: CompactBoardDataset,
    batch_indices: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    h_rem_override: np.ndarray | torch.Tensor | int | None = None,
) -> FeaturizedBatch:
    """Reconstruct privileged 22-ch arena, 23-ch ego crop, and 64 scalars for ``batch_indices``."""
    target = torch.device(device)
    if isinstance(batch_indices, torch.Tensor):
        idx_np = batch_indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        idx_np = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
    b = int(idx_np.size)
    hw = ARENA_H * ARENA_W

    arena = torch.zeros((b, ARENA_CHANNELS, ARENA_H, ARENA_W), dtype=torch.float32, device=target)
    if b == 0:
        return FeaturizedBatch(
            arena=arena,
            ego=torch.zeros(
                (0, EGO_CHANNELS, EGO_SIZE, EGO_SIZE), dtype=torch.float32, device=target
            ),
            scalars=torch.zeros((0, SCALAR_DIM), dtype=torch.float32, device=target),
            hero_xy_dir=torch.zeros((0, 3), dtype=torch.long, device=target),
            legal_mask=torch.zeros((0, 6), dtype=torch.bool, device=target),
            hero_length=torch.zeros(0, dtype=torch.float32, device=target),
            h_rem=torch.zeros(0, dtype=torch.float32, device=target),
        )

    alive_np = dataset.snake_alive[idx_np]  # (B, 6)
    len_np = dataset.snake_length[idx_np].astype(np.float32)  # (B, 6)
    seg_cnt_np = dataset.snake_seg_count[idx_np].astype(np.float32)  # (B, 6)
    dir_np = dataset.snake_dir[idx_np].astype(np.int64) % 4  # (B, 6)
    boost_f_np = dataset.snake_boost_frames[idx_np].astype(np.float32)  # (B, 6)
    hunger_np = dataset.snake_hunger[idx_np].astype(np.float32)  # (B, 6)
    respawn_np = dataset.snake_respawn[idx_np].astype(np.float32)  # (B, 6)
    boosted_np = dataset.snake_boosted[idx_np].astype(np.float32)  # (B, 6)
    head_np = dataset.head_xy[idx_np].astype(np.int64)  # (B, 6, 2)
    tail_np = dataset.tail_xy[idx_np].astype(np.int64)  # (B, 6, 2)
    legal_np = dataset.legal_mask[idx_np]  # (B, 6)

    if h_rem_override is None:
        h_rem_np = dataset.h_rem[idx_np].astype(np.float32)
    elif isinstance(h_rem_override, (int, float)):
        h_rem_np = np.full(b, float(h_rem_override), dtype=np.float32)
    elif isinstance(h_rem_override, torch.Tensor):
        h_rem_np = h_rem_override.detach().cpu().numpy().astype(np.float32).reshape(b)
    else:
        h_rem_np = np.asarray(h_rem_override, dtype=np.float32).reshape(b)

    # Distance-sort opponents 1..5 relative to hero head (Manhattan, deterministic slot tie-break).
    hx = head_np[:, 0, 0]
    hy = head_np[:, 0, 1]
    ox = head_np[:, 1:, 0]
    oy = head_np[:, 1:, 1]
    opp_alive = alive_np[:, 1:]  # (B, 5)
    man_dist = np.abs(ox - hx[:, None]) + np.abs(oy - hy[:, None])
    slot_tie = np.arange(5, dtype=np.float64)[None, :] * 1e-3
    sort_key = np.where(opp_alive, man_dist.astype(np.float64) + slot_tie, 10000.0 + slot_tie)
    opp_order = np.argsort(sort_key, axis=1)  # (B, 5): original opp idx 0..4 at rank r
    opp_rank_of_slot = np.argsort(opp_order, axis=1)  # (B, 5): rank 0..4 of opp idx 0..4

    # Scatter body segments onto arena ch 0..3 (own), 6..9 (enemy union), 14..18 (sorted opps).
    seg_flat_idx, seg_b_np = _csr_gather_indices(
        dataset.seg_row_offsets, dataset.seg_row_counts, idx_np
    )
    arena_flat = arena.view(-1)
    if seg_flat_idx.size > 0:
        seg_b = torch.as_tensor(seg_b_np, dtype=torch.long, device=target)
        seg_s = torch.as_tensor(dataset.seg_slot[seg_flat_idx], dtype=torch.long, device=target)
        seg_x = torch.as_tensor(dataset.seg_x[seg_flat_idx], dtype=torch.long, device=target)
        seg_y = torch.as_tensor(dataset.seg_y[seg_flat_idx], dtype=torch.long, device=target)
        seg_ttl = torch.as_tensor(dataset.seg_ttl[seg_flat_idx], dtype=torch.float32, device=target)

        cell_off = seg_y * ARENA_W + seg_x
        base_row = seg_b * (ARENA_CHANNELS * hw)

        own_mask = seg_s == 0
        if bool(own_mask.any()):
            ob = base_row[own_mask]
            oc = cell_off[own_mask]
            ottl = seg_ttl[own_mask]
            ones_o = torch.ones_like(ottl)
            arena_flat.scatter_reduce_(
                0, ob + 0 * hw + oc, ones_o, reduce="amax", include_self=True
            )
            arena_flat.scatter_reduce_(
                0, ob + 1 * hw + oc, ottl.clamp(0.0, 64.0) / 64.0, reduce="amax", include_self=True
            )
            arena_flat.scatter_reduce_(
                0,
                ob + 2 * hw + oc,
                ottl.clamp(0.0, 300.0) / 300.0,
                reduce="amax",
                include_self=True,
            )
            arena_flat.scatter_reduce_(
                0,
                ob + 3 * hw + oc,
                ottl.clamp(0.0, 1200.0) / 1200.0,
                reduce="amax",
                include_self=True,
            )

        ene_mask = seg_s > 0
        if bool(ene_mask.any()):
            eb = base_row[ene_mask]
            ec = cell_off[ene_mask]
            ettl = seg_ttl[ene_mask]
            eslot0 = seg_s[ene_mask] - 1  # 0..4
            ones_e = torch.ones_like(ettl)
            t300_e = ettl.clamp(0.0, 300.0) / 300.0
            arena_flat.scatter_reduce_(
                0, eb + 6 * hw + ec, ones_e, reduce="amax", include_self=True
            )
            arena_flat.scatter_reduce_(
                0, eb + 7 * hw + ec, ettl.clamp(0.0, 64.0) / 64.0, reduce="amax", include_self=True
            )
            arena_flat.scatter_reduce_(
                0, eb + 8 * hw + ec, t300_e, reduce="amax", include_self=True
            )
            arena_flat.scatter_reduce_(
                0,
                eb + 9 * hw + ec,
                ettl.clamp(0.0, 1200.0) / 1200.0,
                reduce="amax",
                include_self=True,
            )
            rank_t = torch.as_tensor(opp_rank_of_slot, dtype=torch.long, device=target)
            erank = rank_t[seg_b[ene_mask], eslot0]  # 0..4
            arena_flat.scatter_reduce_(
                0, eb + (14 + erank) * hw + ec, t300_e, reduce="amax", include_self=True
            )

    # Scatter hero head (4), tail (5), enemy heads (10), and 1-2 step v2 1.15x threats (11..13).
    b_arange = torch.arange(b, dtype=torch.long, device=target)
    head_t = torch.as_tensor(head_np, dtype=torch.long, device=target)
    tail_t = torch.as_tensor(tail_np, dtype=torch.long, device=target)
    alive_t = torch.as_tensor(alive_np, dtype=torch.bool, device=target)
    len_t = torch.as_tensor(len_np, dtype=torch.float32, device=target)
    dir_t = torch.as_tensor(dir_np, dtype=torch.long, device=target)

    hero_alive_t = alive_t[:, 0]
    arena[b_arange, 4, head_t[:, 0, 1], head_t[:, 0, 0]] = hero_alive_t.float()
    arena[b_arange, 5, tail_t[:, 0, 1], tail_t[:, 0, 0]] = hero_alive_t.float()

    # Enemy heads & 1-2 step reachable threat cones split by mechanics-v2 1.15x head-on outcome.
    card_t = torch.as_tensor(_CARDINAL_NP, dtype=torch.long, device=target)  # (4, 2)
    turns_t = torch.tensor((-1, 0, 1), dtype=torch.long, device=target)
    hero_len = len_t[:, 0:1]  # (B, 1)
    opp_len = len_t[:, 1:]  # (B, 5)
    opp_alive_t = alive_t[:, 1:]  # (B, 5)

    hero_wins = opp_alive_t & (hero_len >= HEADON_SIZE_RATIO * opp_len)
    hero_loses = opp_alive_t & (opp_len >= HEADON_SIZE_RATIO * hero_len)
    mutual_die = opp_alive_t & (~hero_wins) & (~hero_loses)

    opp_heads = head_t[:, 1:]  # (B, 5, 2)
    opp_dirs = dir_t[:, 1:]  # (B, 5)
    # Scatter alive opponent heads onto ch 10.
    oh_b = b_arange[:, None].expand(b, 5)
    oh_pos = (
        oh_b * (ARENA_CHANNELS * hw) + 10 * hw + opp_heads[..., 1] * ARENA_W + opp_heads[..., 0]
    )
    arena_flat.scatter_reduce_(
        0, oh_pos.reshape(-1), opp_alive_t.float().reshape(-1), reduce="amax", include_self=True
    )

    cand_dirs = (opp_dirs[..., None] + turns_t[None, None, :]) % 4  # (B, 5, 3)
    deltas = card_t[cand_dirs]  # (B, 5, 3, 2)
    step1 = opp_heads[:, :, None, :] + deltas  # (B, 5, 3, 2)
    step2 = opp_heads[:, :, None, :] + 2 * deltas  # (B, 5, 3, 2)
    can_opp_boost = opp_alive_t & (opp_len >= 5.0)

    for step_cells, weight, valid_base in (
        (step1, 1.0, opp_alive_t),
        (step2, 0.5, can_opp_boost),
    ):
        tx = step_cells[..., 0]
        ty = step_cells[..., 1]
        inb = valid_base[..., None] & (tx >= 0) & (tx < ARENA_W) & (ty >= 0) & (ty < ARENA_H)
        t_cell = ty.clamp(0, ARENA_H - 1) * ARENA_W + tx.clamp(0, ARENA_W - 1)
        tb = b_arange[:, None, None].expand_as(t_cell) * (ARENA_CHANNELS * hw)
        for ch_idx, rel_mask in ((11, hero_loses), (12, mutual_die), (13, hero_wins)):
            vals = torch.where(
                inb & rel_mask[..., None],
                torch.full_like(tx, weight, dtype=torch.float32),
                torch.zeros_like(tx, dtype=torch.float32),
            )
            arena_flat.scatter_reduce_(
                0,
                (tb + ch_idx * hw + t_cell).reshape(-1),
                vals.reshape(-1),
                reduce="amax",
                include_self=True,
            )

    # Scatter food onto ch 19 (ambient) and ch 20 (corpse).
    food_flat_idx, food_b_np = _csr_gather_indices(
        dataset.food_row_offsets, dataset.food_row_counts, idx_np
    )
    amb_counts = np.zeros(b, dtype=np.float32)
    cor_counts = np.zeros(b, dtype=np.float32)
    if food_flat_idx.size > 0:
        is_cor = dataset.food_is_corpse[food_flat_idx]
        np.add.at(cor_counts, food_b_np, is_cor.astype(np.float32))
        np.add.at(amb_counts, food_b_np, (~is_cor).astype(np.float32))

        fb = torch.as_tensor(food_b_np, dtype=torch.long, device=target)
        fx = torch.as_tensor(dataset.food_x[food_flat_idx], dtype=torch.long, device=target)
        fy = torch.as_tensor(dataset.food_y[food_flat_idx], dtype=torch.long, device=target)
        fcor = torch.as_tensor(is_cor, dtype=torch.long, device=target)
        fch = 19 + fcor  # 19 = ambient, 20 = corpse
        fpos = fb * (ARENA_CHANNELS * hw) + fch * hw + fy * ARENA_W + fx
        arena_flat.scatter_reduce_(
            0,
            fpos,
            torch.ones(food_flat_idx.size, dtype=torch.float32, device=target),
            reduce="amax",
            include_self=True,
        )

    # Channel 21: local 4-neighbor free degree (in-bounds & unoccupied by any snake body).
    free_cell = ((arena[:, 0] == 0.0) & (arena[:, 6] == 0.0)).float()  # (B, 83, 145)
    deg = torch.zeros_like(free_cell)
    deg[:, 1:, :] += free_cell[:, :-1, :]
    deg[:, :-1, :] += free_cell[:, 1:, :]
    deg[:, :, 1:] += free_cell[:, :, :-1]
    deg[:, :, :-1] += free_cell[:, :, 1:]
    arena[:, 21] = deg * 0.25

    # Build 23-channel 31x31 heading-rotated 1:1 ego crop centered at hero head (forward = -u).
    hero_dir = dir_t[:, 0]  # (B,)
    fwd = card_t[hero_dir]  # (B, 2)
    rgt = card_t[(hero_dir + 1) % 4]  # (B, 2)
    offs = torch.arange(-EGO_HALF, EGO_HALF + 1, dtype=torch.long, device=target)
    du = offs[:, None]  # (31, 1): row offset (-15 = forward)
    dv = offs[None, :]  # (1, 31): col offset (+15 = right)
    hx_t = head_t[:, 0, 0][:, None, None]
    hy_t = head_t[:, 0, 1][:, None, None]
    wx = hx_t + (-du)[None, :, :] * fwd[:, 0, None, None] + dv[None, :, :] * rgt[:, 0, None, None]
    wy = hy_t + (-du)[None, :, :] * fwd[:, 1, None, None] + dv[None, :, :] * rgt[:, 1, None, None]
    inb_ego = (wx >= 0) & (wx < ARENA_W) & (wy >= 0) & (wy < ARENA_H)
    ego_cell = wy.clamp(0, ARENA_H - 1) * ARENA_W + wx.clamp(0, ARENA_W - 1)  # (B, 31, 31)
    arena_by_cell = arena.view(b, ARENA_CHANNELS, hw)
    gathered_ego = arena_by_cell.gather(
        2, ego_cell.view(b, 1, EGO_SIZE * EGO_SIZE).expand(-1, ARENA_CHANNELS, -1)
    ).view(b, ARENA_CHANNELS, EGO_SIZE, EGO_SIZE)
    gathered_ego = gathered_ego * inb_ego[:, None, :, :].float()
    ego = torch.cat((gathered_ego, (~inb_ego)[:, None, :, :].float()), dim=1)

    # Build 64-D privileged scalar vector (strictly excluding b8_action and all RNG state).
    sc = np.zeros((b, SCALAR_DIM), dtype=np.float32)
    l0 = len_np[:, 0]
    h_rem_clamped = np.maximum(h_rem_np, 0.0)
    sc[:, 0] = l0 / 1000.0
    sc[:, 1] = np.log1p(np.maximum(l0, 0.0)) / 8.0
    sc[:, 2] = np.sqrt(np.maximum(l0, 0.0) / 1000.0)
    sc[:, 3] = seg_cnt_np[:, 0] / np.maximum(l0, 1.0)
    sc[:, 4] = h_rem_clamped / 6000.0
    sc[:, 5] = np.minimum(h_rem_clamped, 300.0) / 300.0
    sc[:, 6] = np.minimum(h_rem_clamped, 1200.0) / 1200.0
    sc[:, 7] = (l0 / 1000.0) * (h_rem_clamped / 6000.0)
    hdir_np = dir_np[:, 0]
    sc[np.arange(b), 8 + hdir_np] = 1.0
    sc[:, 12] = boost_f_np[:, 0] / 3.0
    sc[:, 13] = (l0 >= 5.0).astype(np.float32)
    sc[:, 14] = boosted_np[:, 0]
    sc[:, 15] = np.minimum(hunger_np[:, 0], 300.0) / 300.0

    # Heading-aligned distances to arena walls (forward, left, right) in cells / 145.
    dist_n = (hy + 1).astype(np.float32)
    dist_e = (ARENA_W - hx).astype(np.float32)
    dist_s = (ARENA_H - hy).astype(np.float32)
    dist_w = (hx + 1).astype(np.float32)
    wall_by_dir = np.stack((dist_n, dist_e, dist_s, dist_w), axis=1)  # (B, 4)
    row_i = np.arange(b)
    sc[:, 16] = wall_by_dir[row_i, hdir_np] / 145.0
    sc[:, 17] = wall_by_dir[row_i, (hdir_np - 1) % 4] / 145.0
    sc[:, 18] = wall_by_dir[row_i, (hdir_np + 1) % 4] / 145.0

    # Canonical legal action mask (19..24).
    sc[:, 19:25] = legal_np.astype(np.float32)

    # Board economy (25..28).
    sc[:, 25] = amb_counts / 300.0
    sc[:, 26] = cor_counts / 150.0
    sc[:, 27] = opp_alive.sum(axis=1).astype(np.float32) / 5.0
    sc[:, 28] = (len_np[:, 1:] * opp_alive.astype(np.float32)).sum(axis=1) / 4000.0

    # 5 distance-sorted opponents x 7 features (29..63).
    fwd_np = _CARDINAL_NP[hdir_np].astype(np.float32)  # (B, 2)
    rgt_np = _CARDINAL_NP[(hdir_np + 1) % 4].astype(np.float32)  # (B, 2)
    for rank in range(5):
        orig = opp_order[:, rank]  # 0..4
        slot = orig + 1  # 1..5
        al_j = alive_np[row_i, slot].astype(np.float32)
        ln_j = len_np[row_i, slot] * al_j
        win_j = ((l0 >= HEADON_SIZE_RATIO * len_np[row_i, slot]) & (al_j > 0.5)).astype(np.float32)
        lose_j = ((len_np[row_i, slot] >= HEADON_SIZE_RATIO * l0) & (al_j > 0.5)).astype(np.float32)
        dx_j = (head_np[row_i, slot, 0] - hx).astype(np.float32)
        dy_j = (head_np[row_i, slot, 1] - hy).astype(np.float32)
        rel_fwd = ((dx_j * fwd_np[:, 0] + dy_j * fwd_np[:, 1]) / 145.0) * al_j
        rel_rgt = ((dx_j * rgt_np[:, 0] + dy_j * rgt_np[:, 1]) / 145.0) * al_j

        col = 29 + rank * 7
        sc[:, col + 0] = al_j
        sc[:, col + 1] = ln_j / 1000.0
        sc[:, col + 2] = ((l0 - ln_j) / 1000.0) * al_j
        sc[:, col + 3] = win_j
        sc[:, col + 4] = lose_j
        sc[:, col + 5] = rel_fwd
        sc[:, col + 6] = np.where(al_j > 0.5, rel_rgt, -respawn_np[row_i, slot] / 5.0)

    hero_xy_dir = torch.stack((head_t[:, 0, 0], head_t[:, 0, 1], hero_dir), dim=1)
    return FeaturizedBatch(
        arena=arena,
        ego=ego,
        scalars=torch.as_tensor(sc, dtype=torch.float32, device=target),
        hero_xy_dir=hero_xy_dir,
        legal_mask=torch.as_tensor(legal_np, dtype=torch.bool, device=target),
        hero_length=torch.as_tensor(l0, dtype=torch.float32, device=target),
        h_rem=torch.as_tensor(h_rem_clamped, dtype=torch.float32, device=target),
    )


@torch.inference_mode()
def featurize_live_sim(
    sim: Any,
    *,
    h_rem: int | np.ndarray,
    env_indices: torch.Tensor | None = None,
    legal_masks: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> FeaturizedBatch:
    """Featurize live ``DeviceSim`` rows through ``CompactSampleBuilder`` for 100% parity."""
    target = sim.device if device is None else torch.device(device)
    if env_indices is None:
        idx_t = torch.arange(sim.E, dtype=torch.long, device=sim.device)
    else:
        idx_t = env_indices.to(device=sim.device, dtype=torch.long).reshape(-1)
    m = int(idx_t.numel())
    if legal_masks is None:
        leg_np = sim.advisory_mask().index_select(0, idx_t)[:, 0].detach().cpu().numpy()
    elif isinstance(legal_masks, torch.Tensor):
        leg_np = legal_masks.detach().cpu().numpy()
    else:
        leg_np = np.asarray(legal_masks, dtype=bool)

    if isinstance(h_rem, (int, float)):
        h_rems = np.full(m, int(h_rem), dtype=np.int32)
    else:
        h_rems = np.asarray(h_rem, dtype=np.int32).reshape(m)

    zeros_i = np.zeros(m, dtype=np.int32)
    builder = CompactSampleBuilder()
    builder.append_from_sim(
        sim,
        env_indices=idx_t,
        world_ids=zeros_i,
        seeds=np.zeros(m, dtype=np.int64),
        life_ids=zeros_i,
        ticks=zeros_i,
        h_rems=h_rems,
        b8_actions=np.zeros(m, dtype=np.int64),
        legal_masks=leg_np,
    )
    ds = builder.build()
    return featurize_dataset_batch(ds, np.arange(m, dtype=np.int64), device=target)
