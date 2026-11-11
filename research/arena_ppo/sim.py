"""GPU-resident exact mechanics-v2 simulator wrapper (`TrainSim`) for Stage 1 PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch

from research.arena_ppo.runtime import normalize_device
from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_sim import DeviceSim
from src.core.mechanics_constants import corpse_food_cap
from src.simd_env.batch_sim import BatchSimConfig

OpponentKind = Literal["self", "snapshot", "scripted"]
OPPONENT_KIND_SELF = 0
OPPONENT_KIND_SNAPSHOT = 1
OPPONENT_KIND_SCRIPTED = 2

_OPPONENT_NAME_TO_ID: dict[str, int] = {
    "self": OPPONENT_KIND_SELF,
    "snapshot": OPPONENT_KIND_SNAPSHOT,
    "scripted": OPPONENT_KIND_SCRIPTED,
}
_CARDINAL_TUPLE = ((0, -1), (1, 0), (0, 1), (-1, 0))
_SIM_STATE_TENSORS: tuple[str, ...] = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
    "_boosted_this_step",
    "terminal_slots",
)


def arena_v2_config(num_envs: int, *, max_capacity: int = 1600) -> BatchSimConfig:
    """Return the exact rectangular mechanics-v2 configuration (`83 x 145` cells).

    Matches ``configs/mechanics_v2.yaml`` and Watch6 continuous-respawn v2 arena
    geometry (`1450 x 830` px at `10` px segment size -> `83` rows x `145` cols).

    Args:
        num_envs: Number of parallel worlds ``E``.
        max_capacity: Ring buffer capacity per snake (`<= 2048`).

    Returns:
        Frozen ``BatchSimConfig`` instance.
    """
    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    return BatchSimConfig(
        num_envs=int(num_envs),
        num_snakes=6,
        game_width=1450,
        game_height=830,
        segment_size=10,
        wall_thickness=10,
        initial_food=250,
        max_food=300,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        gamma=1.0,
        max_capacity=int(max_capacity),
        arena_type="rectangular",
        frame_rate=1,
    )


def derive_arena_world_seeds(base_seed: int, num_envs: int) -> list[int]:
    """Derive deterministic disjoint positive 63-bit world seeds.

    Args:
        base_seed: Non-negative campaign/run seed.
        num_envs: Number of parallel worlds ``E``.

    Returns:
        List of ``num_envs`` positive 63-bit integer seeds.
    """
    u64 = 0xFFFF_FFFF_FFFF_FFFF
    root = ((int(base_seed) & u64) ^ 0x4152_454E_415F_5632) & u64
    seeds: list[int] = []
    for idx in range(int(num_envs)):
        z = (root + ((idx + 1) * 0x9E3779B97F4A7C15)) & u64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & u64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & u64
        val = int((z ^ (z >> 31)) & 0x7FFF_FFFF_FFFF_FFFF)
        seeds.append(val if val != 0 else 1)
    return seeds


class _MPSCompatDeviceSim(DeviceSim):
    """DeviceSim adapter that keeps CPU/CUDA untouched and uses int32 scatter_reduce on MPS.

    PyTorch's Metal (`mps`) backend raises ``RuntimeError: not supported for torch.int64``
    on ``Tensor.scatter_reduce_`` with ``int64`` payloads. All reduced values in
    ``food_grid``, ``occupancy``, ``_append_many``, and ``advisory_mask`` are small bounded
    integers (`<= 20,000`), so performing the reduction in ``int32`` on ``mps`` ONLY and
    casting back to ``int64`` yields bit-identical state transitions to CPU ``DeviceSim``.
    """

    def food_grid(self) -> torch.Tensor:
        """Food grid with 0 empty, 1 ambient, 2 corpse."""
        if self.device.type != "mps":
            return super().food_grid()
        flat = torch.zeros(
            (self.E * self.grid_h * self.grid_w,), dtype=torch.int32, device=self.device
        )
        e = torch.arange(self.E, device=self.device)[:, None]
        x, y = self.food_cells[..., 0], self.food_cells[..., 1]
        valid = self.food_present & (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        pos = (
            e * (self.grid_h * self.grid_w)
            + y.clamp(0, self.grid_h - 1) * self.grid_w
            + x.clamp(0, self.grid_w - 1)
        )
        kind = torch.where(
            self.corpse_food,
            torch.full(x.shape, 2, dtype=torch.int32, device=self.device),
            torch.ones(x.shape, dtype=torch.int32, device=self.device),
        )
        values = torch.where(valid, kind, torch.zeros_like(kind))
        flat.scatter_reduce_(
            0, pos.reshape(-1), values.reshape(-1), reduce="amax", include_self=True
        )
        return flat.to(torch.long).reshape(self.E, self.grid_h, self.grid_w)

    def occupancy(self) -> torch.Tensor:
        """Owner grid for policy/spawn queries; -1 is empty."""
        if self.device.type != "mps":
            return super().occupancy()
        flat = torch.full(
            (self.E * self.grid_h * self.grid_w,), -1, dtype=torch.int32, device=self.device
        )
        k = torch.arange(self.cap, device=self.device)[None, None, :]
        valid = self.alive[..., None] & (k < self.seg_count[..., None])
        ring = (self.head_ptr[..., None] - k) % self.cap
        e = torch.arange(self.E, device=self.device)[:, None, None]
        s = torch.arange(self.S, device=self.device)[None, :, None]
        cells = self.bodies[e, s, ring]
        x, y = cells[..., 0], cells[..., 1]
        valid &= (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        pos = (
            e.expand_as(x) * (self.grid_h * self.grid_w)
            + y.clamp(0, self.grid_h - 1) * self.grid_w
            + x.clamp(0, self.grid_w - 1)
        )
        s_i32 = s.to(torch.int32).expand_as(x)
        values = torch.where(valid, s_i32, torch.full_like(s_i32, -1))
        flat.scatter_reduce_(
            0, pos.reshape(-1), values.reshape(-1), reduce="amax", include_self=True
        )
        return flat.to(torch.long).reshape(self.E, self.grid_h, self.grid_w)

    def _append_many(self, cells: torch.Tensor, mask: torch.Tensor, corpse: bool) -> None:
        """Append batch of food/corpse cells using int32 scatter_reduce on MPS only."""
        if self.device.type != "mps":
            super()._append_many(cells, mask, corpse)
            return
        indices = mask.nonzero(as_tuple=False)
        if indices.shape[0] == 0:
            return
        counts = mask.sum(dim=1)
        width = int(counts.max().item())
        packed = torch.zeros((self.E, width, 2), dtype=cells.dtype, device=self.device)
        packed_mask = torch.zeros((self.E, width), dtype=torch.bool, device=self.device)
        ranks = mask.long().cumsum(dim=1) - 1
        er, col = indices.unbind(dim=1)
        dest = ranks[er, col]
        packed[er, dest] = cells[er, col]
        packed_mask[er, dest] = True
        cells, mask = packed, packed_mask

        n = int(cells.shape[1])
        x, y = cells[..., 0], cells[..., 1]
        inb = mask & (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        old_f, board_n = int(self.food_capacity), int(self.grid_h) * int(self.grid_w)
        rows = torch.arange(self.E, device=self.device)
        ox, oy = self.food_cells[..., 0], self.food_cells[..., 1]
        old_live = (
            self.food_present & (ox >= 0) & (ox < self.grid_w) & (oy >= 0) & (oy < self.grid_h)
        )
        old_id = oy.clamp(0, self.grid_h - 1) * self.grid_w + ox.clamp(0, self.grid_w - 1)
        tag_i32 = torch.full((self.E, board_n), -2, dtype=torch.int32, device=self.device)
        old_stamp = torch.where(
            self.corpse_food,
            torch.arange(old_f, device=self.device)[None] + 1,
            torch.full_like(old_id, -1),
        )
        old_stamp_i32 = old_stamp.to(torch.int32)
        tag_i32.scatter_reduce_(
            1,
            old_id,
            torch.where(old_live, old_stamp_i32, torch.full_like(old_stamp_i32, -2)),
            reduce="amax",
            include_self=True,
        )
        tag = tag_i32.to(torch.long)
        tag = torch.where(tag == -2, torch.zeros_like(tag), tag)
        cap = corpse_food_cap(int(self.cfg.max_food)) if corpse else 0
        queue_i32 = torch.full((self.E, max(cap, 1)), -1, dtype=torch.int32, device=self.device)
        if corpse:
            rank = torch.cumsum(self.corpse_food.long(), dim=1) - 1
            old_id_p1_i32 = (old_id + 1).to(torch.int32)
            queue_i32.scatter_reduce_(
                1,
                rank.clamp(0, cap - 1),
                torch.where(self.corpse_food, old_id_p1_i32, torch.full_like(old_id_p1_i32, -1)),
                reduce="amax",
                include_self=True,
            )
        queue = queue_i32.to(torch.long)
        head = torch.zeros((self.E,), dtype=torch.long, device=self.device)
        count = (
            self.corpse_food.sum(dim=1).clamp_max(cap)
            if corpse
            else torch.zeros((self.E,), dtype=torch.long, device=self.device)
        )
        cell_id = y.clamp(0, self.grid_h - 1) * self.grid_w + x.clamp(0, self.grid_w - 1)
        accepted = torch.zeros((self.E, n), dtype=torch.bool, device=self.device)
        for c_idx in range(n):
            cid = cell_id[:, c_idx]
            accept = inb[:, c_idx] & (tag[rows, cid] == 0)
            accepted[:, c_idx] = accept
            if corpse:
                full = accept & (count >= cap)
                evicted = queue[rows, head].clamp_min(1) - 1
                tag[rows, evicted] = torch.where(
                    full, torch.zeros_like(evicted), tag[rows, evicted]
                )
                slot_q = torch.where(full, head, (head + count).remainder(cap))
                queue[rows, slot_q] = torch.where(accept, cid + 1, queue[rows, slot_q])
                head = torch.where(full, (head + 1).remainder(cap), head)
                count = torch.where(accept & ~full, count + 1, count)
                tag[rows, cid] = torch.where(
                    accept, torch.full_like(cid, old_f + c_idx + 1), tag[rows, cid]
                )
            else:
                tag[rows, cid] = torch.where(accept, torch.full_like(cid, -1), tag[rows, cid])
        old_keep = old_live & torch.where(
            self.corpse_food, tag.gather(1, old_id) == old_stamp, tag.gather(1, old_id) == -1
        )
        incoming_keep = (
            accepted
            & (tag.gather(1, cell_id) == (old_f + torch.arange(n, device=self.device)[None] + 1))
            if corpse
            else accepted
        )
        combined_cells = torch.cat((self.food_cells, cells), dim=1)
        combined_present = torch.cat((old_keep, incoming_keep), dim=1)
        combined_corpse = torch.cat(
            (self.corpse_food & old_keep, torch.full_like(incoming_keep, corpse) & incoming_keep),
            dim=1,
        )
        total = old_f + n
        source = torch.arange(total, device=self.device)[None]
        order = torch.where(combined_present, source, source + total).argsort(dim=1, stable=True)[
            :, :old_f
        ]
        self.food_cells = combined_cells.gather(1, order[..., None].expand(-1, -1, 2))
        self.food_present = combined_present.gather(1, order)
        self.corpse_food = combined_corpse.gather(1, order)
        self.food_count = self.food_present.sum(dim=1)

    def advisory_mask(self) -> torch.Tensor:
        """Return six-action advisory masks (used only for optional scripted greedy_food)."""
        if self.device.type != "mps":
            return super().advisory_mask()
        ecount, slots, cap = self.E, self.S, self.cap
        device, width, height = self.device, self.grid_w, self.grid_h
        env = torch.arange(ecount, device=device)[:, None, None]
        snake = torch.arange(slots, device=device)[None, :, None]
        offset = torch.arange(cap, device=device)[None, None, :]
        ring = (self.head_ptr[..., None] - offset) % cap
        body = self.bodies[env, snake, ring]
        bx, by = body[..., 0], body[..., 1]
        active = self.alive[..., None] & (offset < self.seg_count[..., None])
        valid = active & (bx >= 0) & (bx < width) & (by >= 0) & (by < height)
        rank = ((snake + 1) * (cap + 1) + cap - offset).to(torch.int32)
        values = torch.where(valid, rank.expand_as(bx), torch.zeros_like(rank.expand_as(bx)))
        grid_i32 = torch.zeros((ecount, height * width), dtype=torch.int32, device=device)
        index = by.clamp(0, height - 1) * width + bx.clamp(0, width - 1)
        grid_i32.scatter_reduce_(
            1, index.reshape(ecount, -1), values.reshape(ecount, -1), "amax", include_self=True
        )
        grid = grid_i32.to(torch.long)
        turns = torch.tensor((-1, 0, 1), dtype=torch.long, device=device)
        cardinal = torch.tensor(_CARDINAL_TUPLE, dtype=torch.long, device=device)
        direction = (self.direction[..., None] + turns) % 4
        delta = cardinal[direction]
        first = self.heads()[:, :, None] + delta
        cells = torch.stack((first, first + delta), dim=3)
        x, y = cells[..., 0], cells[..., 1]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        positions = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
        gathered = grid.gather(1, positions.reshape(ecount, -1)).reshape_as(x)
        owner = torch.div(gathered, cap + 1, rounding_mode="floor") - 1
        k = cap - gathered.remainder(cap + 1)
        safe_owner = owner.clamp_min(0)
        e4 = torch.arange(ecount, device=device)[:, None, None, None]
        owner_n, owner_l = self.seg_count[e4, safe_owner], self.length[e4, safe_owner]
        owner_alive = self.alive[e4, safe_owner]
        own = owner == torch.arange(slots, device=device)[None, :, None, None]
        other = (owner >= 0) & ~own & owner_alive
        drop_tail = (owner_n >= 2) & (owner_n >= owner_l)
        other_last = torch.where(drop_tail, owner_n - 2, owner_n - 1)
        other_hit = other & (k <= other_last)
        length, n_seg = self.length[..., None], self.seg_count[..., None]
        normal_last = torch.minimum(n_seg + 1, length) - 2
        own_normal = own[..., 0] & (k[..., 0] >= 2) & (k[..., 0] <= normal_last)
        fatal_normal = ~inside[..., 0] | own_normal | other_hit[..., 0]
        l1 = torch.minimum(n_seg + 1, length)
        l2 = torch.minimum(l1 + 1, length)
        burn = self.boost_frames[..., None] + 1 >= self.cfg.boost_length_cost_frames
        final = torch.where(burn, torch.minimum(l2, (length - 1).clamp_min(1)), l2)
        own_boost = own & (k >= 1) & (k <= (final - 3)[..., None])
        fatal_boost = (~inside | own_boost | other_hit).any(dim=3)
        normal = ~fatal_normal & self.alive[..., None]
        boost = normal & (length >= self.cfg.min_boost_length) & ~fatal_boost
        return torch.cat((normal, boost), dim=2)


@dataclass(frozen=True)
class StepInfo:
    """Per-tick transition metadata emitted by ``TrainSim.step``."""

    acted: torch.Tensor
    died: torch.Tensor
    survived: torch.Tensor
    post_length: torch.Tensor
    reward: torch.Tensor
    boosted: torch.Tensor
    learner_mask: torch.Tensor


class TrainSim:
    """Exact mechanics-v2 GPU-resident simulator wrapper for multi-seat PPO self-play.

    Composes ``DeviceSim`` (`train_mode=False, allow_respawn=True, hero_respawn=True,
    exact_mechanics=True`) without modifying or forking simulator mechanics. All 6 seats
    respawn continuously and share batched policy evaluation; optional per-world opponent
    routing supports ``"self"``, ``"snapshot"``, and ``"scripted"`` (`greedy_food`).
    """

    def __init__(
        self,
        sim: DeviceSim,
        *,
        opponent_kinds: Sequence[OpponentKind] | torch.Tensor | None = None,
    ) -> None:
        """Wrap an initialized ``DeviceSim`` instance on its canonical device.

        Args:
            sim: Underlying exact-v2 ``DeviceSim`` instance.
            opponent_kinds: Optional per-world opponent assignment (`length == E`)
                drawn from ``{"self", "snapshot", "scripted"}``. Defaults to all ``"self"``.
        """
        self.sim: DeviceSim = sim
        self.device: torch.device = sim.device
        self.E: int = int(sim.E)
        self.S: int = int(sim.S)
        self.grid_h: int = int(sim.grid_h)
        self.grid_w: int = int(sim.grid_w)
        self.cap: int = int(sim.cap)
        self.min_boost_length: int = int(sim.cfg.min_boost_length)

        self.prev_heads: torch.Tensor = torch.zeros(
            (self.E, self.S, 2), dtype=torch.long, device=self.device
        )
        self.prev_alive: torch.Tensor = torch.zeros(
            (self.E, self.S), dtype=torch.bool, device=self.device
        )
        self._turns = torch.tensor((-1, 0, 1), dtype=torch.long, device=self.device)
        self._cardinal = torch.tensor(_CARDINAL_TUPLE, dtype=torch.long, device=self.device)
        self._greedy_priority = torch.tensor((1, 0, 2), dtype=torch.long, device=self.device)

        self.world_opponent_kind: torch.Tensor = self._parse_opponent_kinds(opponent_kinds)
        self._learner_seat_mask: torch.Tensor = self._build_learner_seat_mask(
            self.world_opponent_kind
        )

    def _parse_opponent_kinds(
        self, opponent_kinds: Sequence[OpponentKind] | torch.Tensor | None
    ) -> torch.Tensor:
        if opponent_kinds is None:
            return torch.full((self.E,), OPPONENT_KIND_SELF, dtype=torch.long, device=self.device)
        if isinstance(opponent_kinds, torch.Tensor):
            kinds = opponent_kinds.to(device=self.device, dtype=torch.long)
            if kinds.shape != (self.E,):
                raise ValueError(f"opponent_kinds tensor must have shape {(self.E,)}")
            return kinds.clone()
        if len(opponent_kinds) != self.E:
            raise ValueError(f"opponent_kinds length {len(opponent_kinds)} != E={self.E}")
        ids = [_OPPONENT_NAME_TO_ID[str(name)] for name in opponent_kinds]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _build_learner_seat_mask(self, world_kinds: torch.Tensor) -> torch.Tensor:
        is_self_world = (world_kinds == OPPONENT_KIND_SELF)[:, None]
        is_hero_seat = (torch.arange(self.S, device=self.device)[None, :] == 0).expand(self.E, -1)
        return is_self_world | is_hero_seat

    def set_opponent_kinds(
        self, opponent_kinds: Sequence[OpponentKind] | torch.Tensor | None
    ) -> None:
        """Update per-world opponent assignments and refresh learner seat mask.

        Args:
            opponent_kinds: Per-world opponent assignments (`"self"`, `"snapshot"`, `"scripted"`).
        """
        self.world_opponent_kind = self._parse_opponent_kinds(opponent_kinds)
        self._learner_seat_mask = self._build_learner_seat_mask(self.world_opponent_kind)

    @classmethod
    def from_reference(
        cls,
        ref_sim: Any,
        *,
        device: str | torch.device = "cpu",
        opponent_kinds: Sequence[OpponentKind] | torch.Tensor | None = None,
    ) -> "TrainSim":
        """Create a ``TrainSim`` on ``device`` from a ``GridBatchSim`` or ``DeviceSim`` snapshot.

        Args:
            ref_sim: Reference ``GridBatchSim`` (after ``reset()``) or ``DeviceSim``.
            device: Target torch device (`cpu`, `mps`, or `cuda`).
            opponent_kinds: Optional per-world opponent kinds.

        Returns:
            Initialized ``TrainSim`` with exact mechanics-v2 dynamics.
        """
        resolved = normalize_device(device)
        backend = "indexed" if resolved.type == "cuda" else "reference"
        sim_cls: type[DeviceSim] = _MPSCompatDeviceSim if resolved.type == "mps" else DeviceSim

        if isinstance(ref_sim, DeviceSim):
            out = sim_cls(
                ref_sim.cfg,
                device=resolved,
                train_mode=False,
                allow_respawn=True,
                cap=ref_sim.cap,
                move_backend=backend,
                hero_respawn=True,
                exact_mechanics=True,
            )
            for name in _SIM_STATE_TENSORS:
                src_tensor = getattr(ref_sim, name)
                if src_tensor.dtype == torch.float64:
                    src_tensor = src_tensor.to(torch.float32)
                setattr(out, name, src_tensor.to(device=resolved).clone())
            out._prepared = bool(ref_sim._prepared)
            r2_cache.bump(out)
            return cls(out, opponent_kinds=opponent_kinds)

        out = sim_cls(
            ref_sim.cfg,
            device=resolved,
            train_mode=False,
            allow_respawn=True,
            cap=ref_sim.cap,
            move_backend=backend,
            hero_respawn=True,
            exact_mechanics=True,
        )
        names = (
            "bodies",
            "head_ptr",
            "seg_count",
            "length",
            "alive",
            "direction",
            "boost_frames",
            "frames_since_food",
            "respawn_timer",
            "_reward_prev_length",
            "frame",
            "_boosted_this_step",
            "_last_reward",
            "_last_done",
            "_last_death_cause",
            "_last_kills",
            "_last_food_ate",
            "_last_ambient_food_ate",
            "_last_corpse_food_ate",
            "_last_transition_valid",
        )
        for name in names:
            arr = np.asarray(getattr(ref_sim, name))
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            setattr(out, name, torch.as_tensor(arr, device=resolved).clone())

        max_live = max((len(x) for x in ref_sim.food_cells), default=0)
        if max_live > out.food_capacity:
            raise ValueError("reference food list exceeds device fixed capacity")
        for dst in range(out.E):
            cells = ref_sim.food_cells[dst]
            n_cells = len(cells)
            if n_cells:
                out.food_cells[dst, :n_cells] = torch.as_tensor(
                    cells, dtype=torch.long, device=resolved
                )
                out.food_present[dst, :n_cells] = True
                flags = [cell in ref_sim.corpse_cells[dst] for cell in cells]
                out.corpse_food[dst, :n_cells] = torch.as_tensor(
                    flags, dtype=torch.bool, device=resolved
                )
            out.food_count[dst] = n_cells

        out._prepared = bool(
            getattr(ref_sim, "_policy_phase_prepared", False)
            or getattr(ref_sim, "_policy_step_in_progress", False)
        )
        raw_seeds = getattr(ref_sim, "_seeds", list(range(ref_sim.E)))
        seeds = [int(s) if int(s) < (1 << 63) else int(s) - (1 << 64) for s in raw_seeds]
        out.rng_key = (
            torch.as_tensor(seeds, dtype=torch.long, device=resolved) & 0x7FFF_FFFF_FFFF_FFFF
        )
        out.rng_key = torch.where(out.rng_key == 0, torch.ones_like(out.rng_key), out.rng_key)
        out.rng_counter = out.frame.clone() * 65536
        r2_cache.bump(out)
        return cls(out, opponent_kinds=opponent_kinds)

    @classmethod
    def create(
        cls,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
        world_seeds: Sequence[int] | None = None,
        max_capacity: int = 1600,
        opponent_kinds: Sequence[OpponentKind] | torch.Tensor | None = None,
    ) -> "TrainSim":
        """Allocate and spawn ``num_envs`` exact mechanics-v2 worlds directly on ``device``.

        Uses ``DeviceSim``'s own exact margined SplitMix64 spawn and ambient-food refill
        paths so even ``E = 8,192`` initializes on GPU in milliseconds without slow Python loops.

        Args:
            num_envs: Number of parallel worlds ``E``.
            device: Target torch device (`cpu`, `mps`, or `cuda`).
            seed: Base seed used when ``world_seeds`` is omitted.
            world_seeds: Optional explicit per-world integer seeds (`len == num_envs`).
            max_capacity: Body ring capacity per snake.
            opponent_kinds: Optional per-world opponent assignments.

        Returns:
            Spawned ``TrainSim`` ready for rollout collection.
        """
        resolved = normalize_device(device)
        cfg = arena_v2_config(num_envs, max_capacity=max_capacity)
        backend = "indexed" if resolved.type == "cuda" else "reference"
        sim_cls: type[DeviceSim] = _MPSCompatDeviceSim if resolved.type == "mps" else DeviceSim
        sim = sim_cls(
            cfg,
            device=resolved,
            train_mode=False,
            allow_respawn=True,
            cap=cfg.max_capacity,
            move_backend=backend,
            hero_respawn=True,
            exact_mechanics=True,
        )
        seeds = (
            [int(s) for s in world_seeds]
            if world_seeds is not None
            else derive_arena_world_seeds(seed, num_envs)
        )
        if len(seeds) != int(num_envs):
            raise ValueError(f"world_seeds length {len(seeds)} != num_envs={num_envs}")
        norm_seeds = [s if s < (1 << 63) else s - (1 << 64) for s in seeds]
        sim.rng_key = (
            torch.tensor(norm_seeds, dtype=torch.long, device=resolved) & 0x7FFF_FFFF_FFFF_FFFF
        )
        sim.rng_key = torch.where(sim.rng_key == 0, torch.ones_like(sim.rng_key), sim.rng_key)
        sim.rng_counter.zero_()

        # Spawn all 6 snakes and initial food on device via exact v2 placement.
        sim.alive.fill_(False)
        sim.respawn_timer.zero_()
        sim.seg_count.zero_()
        sim.length.fill_(1)
        sim.food_present.fill_(False)
        sim.corpse_food.fill_(False)
        sim.food_count.zero_()
        sim._respawn_exact()
        all_active = torch.ones((sim.E,), dtype=torch.bool, device=resolved)
        sim._refill_food_rows(all_active)
        sim.frame.zero_()
        sim._prepared = False
        r2_cache.bump(sim)
        return cls(sim, opponent_kinds=opponent_kinds)

    def prepare(self) -> None:
        """Advance pre-decision phase (`frame`, food maintenance, respawns) idempotently."""
        self.sim.prepare()

    def learner_seat_mask(self) -> torch.Tensor:
        """Return boolean mask ``(E, S)`` of seats belonging to the learning policy."""
        return self._learner_seat_mask

    def legal_action_mask(self) -> torch.Tensor:
        """Compute legal 6-action masks ``(E, S, 6)`` on device with zero host syncs.

        Normal turns (`0, 1, 2`) are always legal so dead/masked rows retain valid
        categorical distributions; boost turns (`3, 4, 5`) are legal if and only if
        ``alive & (length >= min_boost_length)``.

        Returns:
            Boolean tensor of shape ``(E, S, 6)`` on ``self.device``.
        """
        normal = torch.ones((self.E, self.S, 3), dtype=torch.bool, device=self.device)
        can_boost = self.sim.alive & (self.sim.length >= self.min_boost_length)
        boost = can_boost[..., None].expand(-1, -1, 3)
        return torch.cat((normal, boost), dim=2)

    def scripted_greedy_food_actions(self, *, chunk_envs: int = 512) -> torch.Tensor:
        """Compute exact ``greedy_food`` actions ``(E, S)`` on device with zero host syncs.

        Vectorizes ``research.gpu_sim_spike.anchors_device.choose_anchor(..., "greedy_food")``
        across all ``S = 6`` slots simultaneously and chunks along ``E`` so temporary
        distance tensors stay bounded even at ``E = 8,192``.

        Args:
            chunk_envs: Maximum worlds per vectorized distance chunk.

        Returns:
            Long action tensor of shape ``(E, S)`` with values in ``0..2``.
        """
        safe_all = self.sim.advisory_mask()[:, :, :3]
        safe_all = torch.where(
            safe_all.any(dim=2, keepdim=True), safe_all, torch.ones_like(safe_all)
        )
        heads = self.sim.heads()
        out = torch.ones((self.E, self.S), dtype=torch.long, device=self.device)
        max_i64 = torch.iinfo(torch.long).max
        has_any_food = self.sim.food_present.any(dim=1)

        step = max(1, int(chunk_envs))
        for start in range(0, self.E, step):
            end = min(self.E, start + step)
            dirs = (self.sim.direction[start:end, :, None] + self._turns[None, None, :]) % 4
            cand = heads[start:end, :, None, :] + self._cardinal[dirs]
            food_xy = self.sim.food_cells[start:end, None, None, :, :]
            food_ok = self.sim.food_present[start:end, None, None, :]
            dist = (cand[:, :, :, None, :] - food_xy).square().sum(dim=-1)
            dist = dist.masked_fill(~food_ok, max_i64)
            nearest = dist.amin(dim=3)
            nearest = torch.where(
                has_any_food[start:end, None, None], nearest, torch.zeros_like(nearest)
            )
            score = nearest.clamp_max(max_i64 // 8) * 4 + self._greedy_priority[None, None, :]
            chosen = score.masked_fill(~safe_all[start:end], max_i64).argmin(dim=2)
            out[start:end] = torch.where(self.sim.alive[start:end], chosen, torch.ones_like(chosen))
        return out

    def compose_world_actions(
        self,
        self_actions: torch.Tensor,
        *,
        snapshot_actions: torch.Tensor | None = None,
        scripted_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route per-seat actions ``(E, S)`` from ``self``, ``snapshot``, and ``scripted``.

        Seat ``0`` always executes ``self_actions[:, 0]``; seats ``1..5`` execute
        ``self_actions``, ``snapshot_actions``, or ``scripted_actions`` according to
        each world's ``world_opponent_kind`` without host synchronization.

        Args:
            self_actions: Actions ``(E, S)`` sampled from the learner network.
            snapshot_actions: Optional actions ``(E, S)`` from an older snapshot network.
            scripted_actions: Optional actions ``(E, S)`` from ``scripted_greedy_food_actions``.

        Returns:
            Combined integer action tensor ``(E, S)`` on ``self.device``.
        """
        if self_actions.shape != (self.E, self.S):
            raise ValueError(f"self_actions must have shape {(self.E, self.S)}")
        combined = self_actions
        if snapshot_actions is not None:
            use_snap = (self.world_opponent_kind == OPPONENT_KIND_SNAPSHOT)[:, None] & (
                ~self._learner_seat_mask
            )
            combined = torch.where(use_snap, snapshot_actions, combined)
        if scripted_actions is not None:
            use_scr = (self.world_opponent_kind == OPPONENT_KIND_SCRIPTED)[:, None] & (
                ~self._learner_seat_mask
            )
            combined = torch.where(use_scr, scripted_actions, combined)
        return combined

    def step(self, actions: torch.Tensor) -> StepInfo:
        """Advance all ``E`` worlds by one tick and update 2-frame head history.

        Dynamics are byte-identical to ``DeviceSim.step(actions)``. Reward returned in
        ``StepInfo.reward`` is ``L_t / 1000.0`` per post-step alive tick (`0.0` on death).

        Args:
            actions: Integer actions ``(E, S)`` on ``self.device`` in ``0..5``.

        Returns:
            ``StepInfo`` with per-seat transition tensors on ``self.device``.
        """
        if actions.shape != (self.E, self.S):
            raise ValueError(f"actions must have shape {(self.E, self.S)}, got {actions.shape}")
        if actions.device != self.device:
            raise ValueError(
                f"actions device {actions.device} does not match TrainSim device {self.device}"
            )
        self.sim.prepare()
        decision_heads = self.sim.heads().clone()
        self.sim.step(actions)

        acted = self.sim._last_transition_valid
        died = self.sim._last_done
        survived = self.sim.alive
        post_length = self.sim.length
        boosted = self.sim._boosted_this_step

        self.prev_heads = torch.where(
            survived[..., None], decision_heads, torch.zeros_like(decision_heads)
        )
        self.prev_alive = survived.clone()

        reward = torch.where(
            survived,
            post_length.to(torch.float32) * (1.0 / 1000.0),
            torch.zeros((self.E, self.S), dtype=torch.float32, device=self.device),
        )
        return StepInfo(
            acted=acted,
            died=died,
            survived=survived,
            post_length=post_length,
            reward=reward,
            boosted=boosted,
            learner_mask=self._learner_seat_mask,
        )

    def state_dict(self) -> dict[str, Any]:
        """Export full deterministic simulator state on CPU for checkpointing."""
        sim_tensors = {
            name: getattr(self.sim, name).detach().cpu().clone() for name in _SIM_STATE_TENSORS
        }
        return {
            "E": self.E,
            "S": self.S,
            "cap": self.cap,
            "prepared": bool(self.sim._prepared),
            "sim_tensors": sim_tensors,
            "prev_heads": self.prev_heads.detach().cpu().clone(),
            "prev_alive": self.prev_alive.detach().cpu().clone(),
            "world_opponent_kind": self.world_opponent_kind.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore full deterministic simulator state onto ``self.device``.

        Args:
            state: Dictionary previously produced by :meth:`state_dict`.
        """
        if int(state["E"]) != self.E or int(state["S"]) != self.S or int(state["cap"]) != self.cap:
            raise ValueError(
                f"TrainSim shape mismatch on load: checkpoint={(state['E'], state['S'], state['cap'])} "
                f"vs current={(self.E, self.S, self.cap)}"
            )
        sim_tensors = state["sim_tensors"]
        for name in _SIM_STATE_TENSORS:
            target = getattr(self.sim, name)
            loaded = sim_tensors[name].to(device=self.device, dtype=target.dtype)
            target.copy_(loaded)
        self.sim._prepared = bool(state["prepared"])
        self.prev_heads.copy_(state["prev_heads"].to(device=self.device, dtype=torch.long))
        self.prev_alive.copy_(state["prev_alive"].to(device=self.device, dtype=torch.bool))
        self.set_opponent_kinds(state["world_opponent_kind"])
        r2_cache.bump(self.sim)
