"""Torch implementation of the rollout-state portion of the GPU sim spike.

``DeviceSim`` keeps all mutable rollout state in fixed-shape tensors.  In
particular, event dispatch compacts actual death candidates on device. Event
reduction/compaction introduces explicit scalar synchronization; detailed stage
synchronization and profiling are enabled only by the instrumentation mode.
It intentionally uses a counter RNG, so it is expected to
diverge from GridBatchSim's MT stream.  The parity harness measures that
divergence; this module must never be used as a promotion or serving simulator.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from research.gpu_sim_spike.stage_timing import StageTiming
from src.core.mechanics_constants import HEADON_SIZE_RATIO, corpse_food_cap
from src.core.reward_events import PHI_LENGTH_DIVISOR
from src.simd_env.batch_sim import DEATH_BODY, DEATH_HEAD, DEATH_SELF, DEATH_WALL

_CARDINAL = ((0, -1), (1, 0), (0, 1), (-1, 0))


class DeviceSim:
    """Fixed-shape device world cloned by the dense-search branch loop.

    The public arrays use cell coordinates and follow ``GridBatchSim`` names
    where possible.  ``food_cells`` is append ordered; holes are compacted
    after consumption, so its leading ``food_count[e]`` cells are the live list.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        device: str | torch.device = "cpu",
        train_mode: bool = True,
        allow_respawn: bool = True,
        cap: int | None = None,
        move_backend: str = "reference",
    ) -> None:
        if move_backend not in ("reference", "indexed"):
            raise ValueError("move_backend must be reference or indexed")
        self.move_backend = move_backend
        self.cfg = copy.copy(cfg)
        requested_device = torch.device(device)
        self.device = (
            torch.device("cuda:0")
            if requested_device.type == "cuda" and requested_device.index is None
            else requested_device
        )
        self.train_mode = bool(train_mode)
        self.allow_respawn = bool(allow_respawn)
        self.E, self.S = int(cfg.num_envs), int(cfg.num_snakes)
        self.cap = int(cap if cap is not None else (cfg.body_storage_capacity or cfg.max_capacity))
        self.grid_w = int(cfg.game_width // cfg.segment_size)
        self.grid_h = int(cfg.game_height // cfg.segment_size)
        self.v2 = int(cfg.mechanics_version) == 2
        # Ambient plus the bounded v2 corpse economy is an absolute fixed bound.
        self.food_capacity = int(
            cfg.max_food + (corpse_food_cap(cfg.max_food) if self.v2 else cfg.max_capacity)
        )
        es = (self.E, self.S)
        self.bodies = torch.zeros(
            (self.E, self.S, self.cap, 2), dtype=torch.long, device=self.device
        )
        self.head_ptr = torch.zeros(es, dtype=torch.long, device=self.device)
        self.seg_count = torch.zeros(es, dtype=torch.long, device=self.device)
        self.length = torch.ones(es, dtype=torch.long, device=self.device)
        self.alive = torch.ones(es, dtype=torch.bool, device=self.device)
        self.direction = torch.ones(es, dtype=torch.long, device=self.device)
        self.boost_frames = torch.zeros(es, dtype=torch.long, device=self.device)
        self.frames_since_food = torch.zeros(es, dtype=torch.long, device=self.device)
        self.respawn_timer = torch.zeros(es, dtype=torch.long, device=self.device)
        self._reward_prev_length = torch.ones(es, dtype=torch.long, device=self.device)
        self.frame = torch.zeros((self.E,), dtype=torch.long, device=self.device)
        self.rng_counter = torch.zeros((self.E,), dtype=torch.long, device=self.device)
        self.rng_key = torch.arange(self.E, dtype=torch.long, device=self.device) + 1
        self.food_cells = torch.zeros(
            (self.E, self.food_capacity, 2), dtype=torch.long, device=self.device
        )
        self.food_present = torch.zeros(
            (self.E, self.food_capacity), dtype=torch.bool, device=self.device
        )
        self.corpse_food = torch.zeros(
            (self.E, self.food_capacity), dtype=torch.bool, device=self.device
        )
        self.food_count = torch.zeros((self.E,), dtype=torch.long, device=self.device)
        self._last_reward = torch.zeros(es, dtype=torch.float32, device=self.device)
        self._last_done = torch.zeros(es, dtype=torch.bool, device=self.device)
        self._last_death_cause = torch.zeros(es, dtype=torch.long, device=self.device)
        self._last_kills = torch.zeros(es, dtype=torch.long, device=self.device)
        self._last_food_ate = torch.zeros(es, dtype=torch.bool, device=self.device)
        self._last_ambient_food_ate = torch.zeros(es, dtype=torch.bool, device=self.device)
        self._last_corpse_food_ate = torch.zeros(es, dtype=torch.bool, device=self.device)
        self._last_transition_valid = torch.zeros(es, dtype=torch.bool, device=self.device)
        self._boosted_this_step = torch.zeros(es, dtype=torch.bool, device=self.device)
        # Dense-search heroes terminate at death.  Opponents can retain the
        # reference respawn behaviour when the reference world enables it.
        self.terminal_slots = (
            (torch.arange(self.S, device=self.device)[None, :] == 0).expand(self.E, -1).clone()
        )
        self.port_limitations = (
            "Counter RNG is intentionally not CPython MT; bounded device rejection "
            "sampling can exhaust on crowded boards; circular arena uses "
            "cell-centre geometry.  Results are an upper-bound device result "
            "until the parity harness classifies every disagreement."
        )
        self._prepared = False
        self._timing = StageTiming(self.device)

    @classmethod
    def from_reference(
        cls, sim: Any, device: str | torch.device = "cpu", repeat: int = 1
    ) -> "DeviceSim":
        """Copy a CPU GridBatchSim snapshot once; host access ends on return."""
        if repeat < 1:
            raise ValueError("repeat must be positive")
        out = cls(
            sim.cfg,
            device=device,
            train_mode=sim.train_mode,
            allow_respawn=sim.allow_respawn,
            cap=sim.cap,
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
            value = np.asarray(getattr(sim, name))
            setattr(out, name, torch.as_tensor(value, device=out.device).clone())
        # Allocate a fixed buffer once from the reference's actual ordered lists.
        max_live = max((len(x) for x in sim.food_cells), default=0)
        if max_live > out.food_capacity:
            raise ValueError("reference food list exceeds device fixed capacity")
        for dst in range(out.E):
            src = dst
            cells = sim.food_cells[src]
            n = len(cells)
            if n:
                out.food_cells[dst, :n] = torch.as_tensor(
                    cells, dtype=torch.long, device=out.device
                )
                out.food_present[dst, :n] = True
                flags = [cell in sim.corpse_cells[src] for cell in cells]
                out.corpse_food[dst, :n] = torch.as_tensor(
                    flags, dtype=torch.bool, device=out.device
                )
            out.food_count[dst] = n
        out._prepared = bool(getattr(sim, "_policy_phase_prepared", False))
        # Counter RNG deliberately differs from CPython MT, but its key is
        # bound to the source world seed and phase so branch worlds are neither
        # accidentally identical across fixtures nor reset at offset seams.
        seeds = getattr(sim, "_seeds", list(range(sim.E)))
        out.rng_key = (
            torch.as_tensor(seeds, dtype=torch.long, device=out.device) & 0x7FFF_FFFF_FFFF_FFFF
        )
        out.rng_key = torch.where(out.rng_key == 0, torch.ones_like(out.rng_key), out.rng_key)
        out.rng_counter = out.frame.clone() * 65536
        out.rng_metadata = {
            "family": "splitmix-lcg-counter",
            "key_source": "reference_world_seed",
            "counter": "frame*65536",
        }
        return out.repeat_interleave(repeat)

    def clone(self) -> "DeviceSim":
        """Deep-copy every tensor without a device-to-host round trip."""
        out = object.__new__(DeviceSim)
        for key, value in self.__dict__.items():
            setattr(
                out, key, value.clone() if isinstance(value, torch.Tensor) else copy.copy(value)
            )
        return out

    def copy_env_to_reference(self, reference: Any, env: int = 0) -> None:
        """Explicit, measured host bridge for real B8/v8 decisions.

        This is deliberately outside :meth:`step`: callers time it as host work
        and label an end-to-end run an upper bound.  ``reference`` is a one-env
        GridBatchSim/BatchSim copy used only for the policy decision seam.
        """
        if reference.E != 1:
            raise ValueError("copy_env_to_reference requires a one-environment reference")
        if env < 0 or env >= self.E:
            raise IndexError("environment outside DeviceSim batch")
        fields = (
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
        for name in fields:
            source = getattr(self, name)[env : env + 1].detach().cpu().numpy()
            getattr(reference, name)[...] = source
        count = int(self.food_count[env].detach().cpu())
        cells = self.food_cells[env, :count].detach().cpu().tolist()
        corpse = self.corpse_food[env, :count].detach().cpu().tolist()
        reference.food_cells[0] = [tuple(map(int, cell)) for cell in cells]
        reference.food_set[0] = set(reference.food_cells[0])
        reference.corpse_cells[0] = {
            cell for cell, is_corpse in zip(reference.food_cells[0], corpse) if is_corpse
        }
        # GridBatchSim maintains derived occupancy/food mirrors for its true
        # featurizer and v8 path; BatchSim simply has no such method.
        rebuild = getattr(reference, "_rebuild_env_grids", None)
        if rebuild is not None:
            rebuild(0, food=True)
            reference._rebuild_traversed_from_heads(np.ones(1, dtype=bool))
            reference._refresh_action_masks(np.ones(1, dtype=bool))
        reference._policy_phase_prepared = self._prepared

    def sync_to_reference(self, reference: Any) -> None:
        """Copy matching batch worlds to a CPU reference at a policy seam."""
        if reference.E != self.E:
            raise ValueError("reference batch count must match DeviceSim")
        for env in range(self.E):
            # Reuse the single-world bridge semantics without mutating a
            # different reference row through Python aliases.
            for name in (
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
            ):
                getattr(reference, name)[env] = getattr(self, name)[env].detach().cpu().numpy()
            count = int(self.food_count[env].detach().cpu())
            cells = self.food_cells[env, :count].detach().cpu().tolist()
            corpse = self.corpse_food[env, :count].detach().cpu().tolist()
            reference.food_cells[env] = [tuple(map(int, cell)) for cell in cells]
            reference.food_set[env] = set(reference.food_cells[env])
            reference.corpse_cells[env] = {
                cell for cell, is_corpse in zip(reference.food_cells[env], corpse) if is_corpse
            }
            rebuild = getattr(reference, "_rebuild_env_grids", None)
            if rebuild is not None:
                rebuild(env, food=True)
                reference._rebuild_traversed_from_heads(np.arange(reference.E) == env)
                reference._refresh_action_masks(np.arange(reference.E) == env)
        reference._policy_phase_prepared = self._prepared

    def repeat_interleave(self, repeats: int) -> "DeviceSim":
        """Return branch worlds repeated contiguously, entirely on this device."""
        if repeats < 1:
            raise ValueError("repeats must be positive")
        out = self.clone()
        old_e = self.E
        for key, value in tuple(out.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == old_e:
                setattr(out, key, value.repeat_interleave(repeats, dim=0))
        out.E = old_e * repeats
        return out

    def heads(self) -> torch.Tensor:
        e = torch.arange(self.E, device=self.device)[:, None]
        s = torch.arange(self.S, device=self.device)[None, :]
        return self.bodies[e, s, self.head_ptr]

    def food_grid(self) -> torch.Tensor:
        e = torch.arange(self.E, device=self.device)[:, None].expand_as(self.food_present)
        live = self.food_present
        x, y = self.food_cells[..., 0], self.food_cells[..., 1]
        inb = live & (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        flat = torch.zeros(
            (self.E * self.grid_h * self.grid_w,), dtype=torch.int8, device=self.device
        )
        pos = (
            e * (self.grid_h * self.grid_w)
            + y.clamp(0, self.grid_h - 1) * self.grid_w
            + x.clamp(0, self.grid_w - 1)
        )
        values = torch.where(
            inb,
            torch.where(self.corpse_food, torch.full_like(x, 2), torch.ones_like(x)),
            torch.zeros_like(x),
        ).to(torch.int8)
        flat.scatter_reduce_(
            0, pos.reshape(-1), values.reshape(-1), reduce="amax", include_self=True
        )
        return flat.reshape(self.E, self.grid_h, self.grid_w)

    def occupancy(self) -> torch.Tensor:
        """Owner grid for policy raster extraction; -1 is empty."""
        grid = torch.full(
            (self.E, self.grid_h, self.grid_w), -1, dtype=torch.long, device=self.device
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
        values = torch.where(valid, s.expand_as(x), torch.full_like(x, -1))
        flat = grid.reshape(-1)
        flat.scatter_reduce_(
            0, pos.reshape(-1), values.reshape(-1), reduce="amax", include_self=True
        )
        return flat.reshape_as(grid)

    def advisory_mask(self) -> torch.Tensor:
        """Return exact-shape six-action advisory masks from the device state."""
        from research.gpu_sim_spike.masks_device import advisory_mask

        return advisory_mask(self)

    def _random_cells(self) -> torch.Tensor:
        # Counter-based integer mixer: reproducible across branch cloning.
        self.rng_counter += 1
        z = (
            self.rng_key * 6364136223846793005 + self.rng_counter * 1442695040888963407
        ) & 0x7FFF_FFFF_FFFF_FFFF
        x = z.remainder(self.grid_w)
        y = (z // self.grid_w).remainder(self.grid_h)
        return torch.stack((x, y), dim=-1)

    def _append_food(
        self, cells: torch.Tensor, mask: torch.Tensor, corpse: torch.Tensor | bool
    ) -> None:
        slots = self.food_count.clamp_max(self.food_capacity - 1)
        room = self.food_count < self.food_capacity
        use = mask & room
        e = torch.arange(self.E, device=self.device)
        old_grid = self.food_grid()
        duplicate = (
            old_grid[
                e, cells[:, 1].clamp(0, self.grid_h - 1), cells[:, 0].clamp(0, self.grid_w - 1)
            ]
            != 0
        )
        use &= ~duplicate
        self.food_cells[e, slots] = torch.where(use[:, None], cells, self.food_cells[e, slots])
        self.food_present[e, slots] = self.food_present[e, slots] | use
        corpse_value = torch.full_like(use, bool(corpse)) if isinstance(corpse, bool) else corpse
        self.corpse_food[e, slots] = torch.where(use, corpse_value, self.corpse_food[e, slots])
        self.food_count += use.long()
        if isinstance(corpse, bool) and corpse:
            self._trim_corpse_food()

    def _append_many(self, cells: torch.Tensor, mask: torch.Tensor, corpse: bool) -> None:
        """Append candidates with an exact, fixed-shape tagged FIFO corpse path."""
        # Meta checks retain the static tensor path. Real execution rejects an
        # empty event and packs only actual candidates, in original stream order.
        if self.device.type != "meta":
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
        n = cells.shape[1]
        x, y = cells[..., 0], cells[..., 1]
        inb = mask & (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        old_f, board_n = self.food_capacity, self.grid_h * self.grid_w
        rows = torch.arange(self.E, device=self.device)
        ox, oy = self.food_cells[..., 0], self.food_cells[..., 1]
        old_live = (
            self.food_present & (ox >= 0) & (ox < self.grid_w) & (oy >= 0) & (oy < self.grid_h)
        )
        old_id = oy.clamp(0, self.grid_h - 1) * self.grid_w + ox.clamp(0, self.grid_w - 1)
        # Tag values: 0 empty, -1 ambient, positive unique corpse timestamp.
        tag = torch.full((self.E, board_n), -2, dtype=torch.long, device=self.device)
        old_stamp = torch.where(
            self.corpse_food,
            torch.arange(old_f, device=self.device)[None] + 1,
            torch.full_like(old_id, -1),
        )
        tag.scatter_reduce_(
            1,
            old_id,
            torch.where(old_live, old_stamp, torch.full_like(old_stamp, -2)),
            reduce="amax",
            include_self=True,
        )
        tag = torch.where(tag == -2, torch.zeros_like(tag), tag)
        cap = corpse_food_cap(int(self.cfg.max_food)) if corpse else 0
        queue = torch.full((self.E, max(cap, 1)), -1, dtype=torch.long, device=self.device)
        if corpse:
            rank = torch.cumsum(self.corpse_food.long(), dim=1) - 1
            queue.scatter_reduce_(
                1,
                rank.clamp(0, cap - 1),
                torch.where(self.corpse_food, old_id + 1, torch.full_like(old_id, -1)),
                reduce="amax",
                include_self=True,
            )
        head = torch.zeros((self.E,), dtype=torch.long, device=self.device)
        count = (
            self.corpse_food.sum(dim=1).clamp_max(cap)
            if corpse
            else torch.zeros((self.E,), dtype=torch.long, device=self.device)
        )
        cell_id = y.clamp(0, self.grid_h - 1) * self.grid_w + x.clamp(0, self.grid_w - 1)
        accepted = torch.zeros((self.E, n), dtype=torch.bool, device=self.device)
        if self.device.type == "cuda":
            from research.gpu_sim_spike.event_cuda import append_fifo

            append_fifo(tag, queue, count, cell_id, inb, accepted, old_f, cap, corpse)
        else:
            for col in range(n):
                cid = cell_id[:, col]
                accept = inb[:, col] & (tag[rows, cid] == 0)
                accepted[:, col] = accept
                if corpse:
                    full = accept & (count >= cap)
                    evicted = queue[rows, head].clamp_min(1) - 1
                    tag[rows, evicted] = torch.where(
                        full, torch.zeros_like(evicted), tag[rows, evicted]
                    )
                    slot = torch.where(full, head, (head + count).remainder(cap))
                    queue[rows, slot] = torch.where(accept, cid + 1, queue[rows, slot])
                    head = torch.where(full, (head + 1).remainder(cap), head)
                    count = torch.where(accept & ~full, count + 1, count)
                    tag[rows, cid] = torch.where(
                        accept, torch.full_like(cid, old_f + col + 1), tag[rows, cid]
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

    def _trim_corpse_food(self) -> None:
        """Evict oldest corpse entries exactly by append-list order."""
        cap = corpse_food_cap(int(self.cfg.max_food))
        count = self.corpse_food.sum(dim=1)
        self._evict_oldest_corpse((count - cap).clamp_min(0))

    def _evict_oldest_corpse(self, evict: torch.Tensor) -> None:
        """Remove a per-world number of oldest corpse entries, then compact."""
        order = torch.cumsum(self.corpse_food.long(), dim=1)
        remove = self.corpse_food & (order <= evict[:, None])
        self.food_present &= ~remove
        self.corpse_food &= ~remove
        self._compact_food()

    def _maintain_food(self) -> None:
        with self.stage("food"):
            self._maintain_food_impl()

    def _maintain_food_impl(self) -> None:
        """Refill needy worlds while consuming exactly the old fixed RNG budget."""
        trials = int(self.cfg.max_food)
        if self.device.type == "meta":
            # Static shape-only oracle; no data-dependent scalar extraction.
            for _ in range(trials):
                want = (self.food_present & ~self.corpse_food).sum(1) < trials
                candidate = self._random_cells()
                occ = self.occupancy()
                e = torch.arange(self.E, device=self.device)
                self._append_food(
                    candidate, want & (occ[e, candidate[:, 1], candidate[:, 0]] < 0), False
                )
            return
        ambient = (self.food_present & ~self.corpse_food).sum(dim=1)
        active = (ambient < trials).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            self.rng_counter += trials
            return
        # Select only worlds needing refill. All fields have independent storage;
        # bodies/occupancy do not change during maintenance, so build it once.
        sub = object.__new__(DeviceSim)
        sub.__dict__ = self.__dict__.copy()
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.E:
                setattr(sub, name, value.index_select(0, active))
        sub.E = active.numel()
        occ = sub.occupancy()
        e = torch.arange(sub.E, device=self.device)
        ambient = ambient.index_select(0, active)
        used = 0
        for used in range(1, trials + 1):
            candidate = sub._random_cells()
            want = ambient < trials
            before = sub.food_count.clone()
            free = occ[e, candidate[:, 1], candidate[:, 0]] < 0
            sub._append_food(candidate, want & free, False)
            ambient += sub.food_count - before
            if bool((ambient >= trials).all().item()):
                break
        # The reference draws max_food candidates for EVERY world on every call,
        # including full worlds and trials after the final successful insertion.
        self.rng_counter += trials
        for name in ("food_cells", "food_present", "corpse_food", "food_count"):
            getattr(self, name)[active] = getattr(sub, name)

    def set_instrumentation(self, enabled: bool) -> None:
        self._timing = StageTiming(self.device, enabled=enabled)

    def stage(self, name: str) -> Any:
        return self._timing.stage(name)

    def consume_timing(self) -> dict[str, Any]:
        return self._timing.consume()

    def _compact_food(self) -> None:
        # Stable sort avoids dynamic-shape boolean indexing, including FakeTensor.
        slots = torch.arange(self.food_capacity, device=self.device)[None, :]
        key = torch.where(self.food_present, slots, slots + self.food_capacity)
        order = key.argsort(dim=1, stable=True)
        self.food_cells = self.food_cells.gather(1, order[..., None].expand(-1, -1, 2))
        self.corpse_food = self.corpse_food.gather(1, order)
        self.food_present = self.food_present.gather(1, order)
        self.food_count = self.food_present.sum(dim=1)

    def _push(self, cells: torch.Tensor, move: torch.Tensor) -> None:
        # Capacity is fixed at construction and validated by the spike harness.
        # Avoid a host-side scalar assertion here: CUDA branch steps must remain
        # graph/shape testable.  A harness-visible flag makes overflow explicit.
        self.capacity_overflow = getattr(self, "capacity_overflow", torch.zeros_like(move)) | (
            move & (self.seg_count >= self.cap)
        )
        e = torch.arange(self.E, device=self.device)[:, None]
        s = torch.arange(self.S, device=self.device)[None, :]
        new_ptr = (self.head_ptr + 1) % self.cap
        target = torch.where(move, new_ptr, self.head_ptr)
        self.bodies[e, s, target] = torch.where(move[..., None], cells, self.bodies[e, s, target])
        self.head_ptr = target
        self.seg_count += move.long()
        pop = (self.seg_count > self.length) & move
        self.seg_count -= pop.long()

    def _in_arena(self, cells: torch.Tensor) -> torch.Tensor:
        x, y = cells[..., 0], cells[..., 1]
        rect = (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)
        if getattr(self.cfg, "arena_type", "rectangular") != "circular":
            return rect
        # configs store pixel values; use cell centres for the device lattice.
        radius = (
            float(
                getattr(
                    self.cfg, "arena_radius", min(self.cfg.game_width, self.cfg.game_height) // 2
                )
            )
            / self.cfg.segment_size
        )
        cx = (
            float(getattr(self.cfg, "arena_center_x", self.cfg.game_width // 2))
            / self.cfg.segment_size
        )
        cy = (
            float(getattr(self.cfg, "arena_center_y", self.cfg.game_height // 2))
            / self.cfg.segment_size
        )
        return rect & ((x.float() - cx).square() + (y.float() - cy).square() < radius * radius)

    def _respawn(self) -> None:
        if self.move_backend == "indexed" and self.device.type != "meta":
            from research.gpu_sim_spike.move_cuda import respawn

            respawn(self)
            return
        self._respawn_reference()

    def _respawn_reference(self) -> None:
        dead = ~self.alive
        self.respawn_timer = torch.where(
            dead & (self.respawn_timer > 0), self.respawn_timer - 1, self.respawn_timer
        )
        ready = dead & (self.respawn_timer <= 0) & ~self.terminal_slots
        # Bounded device-only search; a row without a free candidate remains dead.
        placed = torch.zeros_like(ready)
        for _ in range(32):
            base = self._random_cells()
            slots = torch.arange(self.S, device=self.device)[None, :]
            candidate = torch.stack(
                (
                    (base[:, None, 0] + slots).remainder(self.grid_w),
                    (base[:, None, 1] + slots // self.grid_w).remainder(self.grid_h),
                ),
                dim=-1,
            )
            occ = self.occupancy()
            e = torch.arange(self.E, device=self.device)[:, None]
            free = occ[e, candidate[..., 1], candidate[..., 0]] < 0
            # Lowest slot owns duplicate candidate cells in the same respawn pass.
            equal = (candidate[:, :, None, :] == candidate[:, None, :, :]).all(dim=-1)
            earlier = torch.tril(
                torch.ones((self.S, self.S), dtype=torch.bool, device=self.device), diagonal=-1
            )[None]
            unique = ~(equal & earlier).any(dim=2)
            choose = ready & ~placed & free & unique
            self.bodies[:, :, 0] = torch.where(choose[..., None], candidate, self.bodies[:, :, 0])
            # Make the newly claimed cell visible to occupancy on the next
            # trial even when this dead snake had a nonzero old ring pointer.
            self.head_ptr = torch.where(choose, torch.zeros_like(self.head_ptr), self.head_ptr)
            self.seg_count = torch.where(choose, torch.ones_like(self.seg_count), self.seg_count)
            self.length = torch.where(choose, torch.ones_like(self.length), self.length)
            placed |= choose
            # Claim successful positions before the next deterministic trial.
            self.alive |= choose
        self.head_ptr = torch.where(placed, torch.zeros_like(self.head_ptr), self.head_ptr)
        self.seg_count = torch.where(placed, torch.ones_like(self.seg_count), self.seg_count)
        self.length = torch.where(placed, torch.ones_like(self.length), self.length)
        self.alive |= placed
        self.direction = torch.where(placed, torch.ones_like(self.direction), self.direction)
        self.boost_frames = torch.where(
            placed, torch.zeros_like(self.boost_frames), self.boost_frames
        )
        self.frames_since_food = torch.where(
            placed, torch.zeros_like(self.frames_since_food), self.frames_since_food
        )
        self.respawn_timer = torch.where(
            placed, torch.zeros_like(self.respawn_timer), self.respawn_timer
        )
        self._reward_prev_length = torch.where(
            placed, torch.ones_like(self._reward_prev_length), self._reward_prev_length
        )

    def prepare(self) -> None:
        """Run the reference policy-phase lifecycle once before a decision.

        ``GridBatchSim.step_with_policy`` advances frame, maintains food, and
        respawns before it calls B8/v8.  Branching from that seam must not run
        those operations a second time, hence the explicit prepared flag.
        """
        if self._prepared:
            return
        self.frame += 1
        self._maintain_food()
        if self.allow_respawn:
            with self.stage("movement"):
                self._respawn()
        self._prepared = True

    def _collision_flags_reference(
        self, head0: torch.Tensor, traversed: torch.Tensor, traversed_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retain the original dense comparison as the default regression arm."""
        k = torch.arange(self.cap, device=self.device)[None, None, :]
        rings = (self.head_ptr[..., None] - k) % self.cap
        body = self.bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
        valid_body = self.alive[..., None] & (k < self.seg_count[..., None])
        same = (traversed[:, :, :, None, None, :] == body[:, None, None, :, :, :]).all(dim=-1)
        same &= traversed_valid[:, :, :, None, None]
        own_offsets = k >= 3
        own = torch.eye(self.S, dtype=torch.bool, device=self.device)[None, :, None, :, None]
        self_hit = (
            same & own & valid_body[:, None, None, :, :] & own_offsets[:, None, None, :, :]
        ).any(dim=(2, 3, 4))
        head_equal = (traversed[:, :, :, None, None, :] == traversed[:, None, None, :, :, :]).all(
            dim=-1
        )
        head_equal &= traversed_valid[:, :, :, None, None] & traversed_valid[:, None, None, :, :]
        head_equal = head_equal.any(dim=(2, 4))
        # CPU also treats crossing paths as head-on, including normal-vs-boost.
        swap_a = (head0[:, :, None, None, None, :] == traversed[:, None, None, :, :, :]).all(dim=-1)
        swap_b = (traversed[:, :, :, None, None, :] == head0[:, None, None, :, None, :]).all(dim=-1)
        head_equal |= (swap_a & swap_b).any(dim=(2, 4))
        eye = torch.eye(self.S, dtype=torch.bool, device=self.device)[None]
        head_equal &= ~eye
        head_pair = head_equal
        body_other = same & valid_body[:, None, None, :, :]
        body_other &= ~eye[:, :, None, :, None]
        body_pair = body_other.any(dim=(2, 4))
        return body, self_hit, head_pair, body_pair

    def _resolve_collisions_reference(
        self,
        in_arena: torch.Tensor,
        self_hit: torch.Tensor,
        head_pair: torch.Tensor,
        body_pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retain the original ordered tensor resolver as the regression arm."""
        # Resolve exactly in the CPU collision-list order.  The loops are fixed
        # by S (no device scalar extraction) and carry a per-env dead set.
        death = torch.zeros_like(self.length)
        death_rank = torch.full_like(self.length, self.S * self.S + self.S)
        dead = torch.zeros_like(self.alive)
        kills = torch.zeros_like(self.length)
        kill_mass = torch.zeros_like(self.length)
        rank = 0
        for i in range(self.S):
            wall = self.alive[:, i] & ~in_arena[:, i] & ~dead[:, i]
            death[:, i] = torch.where(wall, torch.full_like(death[:, i], DEATH_WALL), death[:, i])
            death_rank[:, i] = torch.where(
                wall, torch.full_like(death_rank[:, i], rank), death_rank[:, i]
            )
            dead |= (
                torch.nn.functional.one_hot(
                    torch.full((self.E,), i, device=self.device), self.S
                ).bool()
                & wall[:, None]
            )
            rank += 1
            selfdeath = self.alive[:, i] & self_hit[:, i] & ~dead[:, i]
            death[:, i] = torch.where(
                selfdeath, torch.full_like(death[:, i], DEATH_SELF), death[:, i]
            )
            death_rank[:, i] = torch.where(
                selfdeath, torch.full_like(death_rank[:, i], rank), death_rank[:, i]
            )
            dead |= (
                torch.nn.functional.one_hot(
                    torch.full((self.E,), i, device=self.device), self.S
                ).bool()
                & selfdeath[:, None]
            )
            rank += 1
            for j in range(self.S):
                if i == j:
                    continue
                pair_i = self.alive[:, i] & ~dead[:, i]
                head = pair_i & self.alive[:, j] & ~dead[:, j] & head_pair[:, i, j]
                body_target_live = (
                    torch.ones_like(pair_i) if self.v2 else (self.alive[:, j] & ~dead[:, j])
                )
                bodyhit = pair_i & body_target_live & ~head_pair[:, i, j] & body_pair[:, i, j]
                if self.v2:
                    iwins = head & (
                        self.length[:, i].float() >= HEADON_SIZE_RATIO * self.length[:, j].float()
                    )
                    jwins = head & (
                        self.length[:, j].float() >= HEADON_SIZE_RATIO * self.length[:, i].float()
                    )
                else:
                    iwins = torch.zeros_like(head)
                    jwins = torch.zeros_like(head)
                mutual = head & ~iwins & ~jwins
                die_i = bodyhit | mutual | jwins
                die_j = mutual | iwins
                cause_i = torch.where(bodyhit, DEATH_BODY, DEATH_HEAD)
                death[:, i] = torch.where(die_i, cause_i, death[:, i])
                death[:, j] = torch.where(
                    die_j, torch.full_like(death[:, j], DEATH_HEAD), death[:, j]
                )
                death_rank[:, i] = torch.where(
                    die_i, torch.full_like(death_rank[:, i], rank), death_rank[:, i]
                )
                death_rank[:, j] = torch.where(
                    die_j, torch.full_like(death_rank[:, j], rank), death_rank[:, j]
                )
                kills[:, i] += iwins.long()
                kills[:, j] += (jwins | bodyhit).long()
                kill_mass[:, i] += torch.where(
                    iwins, self.length[:, j], torch.zeros_like(self.length[:, j])
                )
                kill_mass[:, j] += torch.where(
                    jwins | bodyhit, self.length[:, i], torch.zeros_like(self.length[:, i])
                )
                dead |= (
                    torch.nn.functional.one_hot(
                        torch.full((self.E,), i, device=self.device), self.S
                    ).bool()
                    & die_i[:, None]
                )
                dead |= (
                    torch.nn.functional.one_hot(
                        torch.full((self.E,), j, device=self.device), self.S
                    ).bool()
                    & die_j[:, None]
                )
                rank += 1
        return death, death_rank, dead, kills, kill_mass

    def step(self, actions: torch.Tensor) -> None:
        """Advance all environments, compacting event candidates before dispatch."""
        if actions.shape != (self.E, self.S):
            raise ValueError(f"actions must have shape {(self.E, self.S)}")
        if actions.device != self.device:
            raise ValueError("actions must be on the DeviceSim device")
        actions = actions.long().clamp(0, 5)
        self.prepare()
        movement_stage = self.stage("movement")
        movement_stage.__enter__()
        acted, prev_length = self.alive.clone(), self._reward_prev_length.clone()
        new_dir = (
            self.direction
            + torch.where(
                actions.remainder(3) == 0, -1, torch.where(actions.remainder(3) == 2, 1, 0)
            )
        ) % 4
        self.direction = torch.where(self.alive, new_dir, self.direction)
        cardinal = torch.tensor(_CARDINAL, dtype=torch.long, device=self.device)
        delta = cardinal[self.direction]
        head0 = self.heads()
        move1 = self.alive
        head1 = head0 + delta
        self._push(head1, move1)
        boost = self.alive & (actions >= 3) & (self.length >= int(self.cfg.min_boost_length))
        head2 = head1 + delta
        self._push(head2, boost)
        self._boosted_this_step = boost
        self.boost_frames = torch.where(boost, self.boost_frames + 1, self.boost_frames)
        burn = (
            boost
            & (self.boost_frames >= int(self.cfg.boost_length_cost_frames))
            & (self.length > 1)
        )
        self.boost_frames = torch.where(
            burn, torch.zeros_like(self.boost_frames), self.boost_frames
        )
        # Capture a paid tail before the additional pop; v2 transfers it to food.
        tail_ring = (self.head_ptr - (self.seg_count - 1).clamp_min(0)) % self.cap
        e = torch.arange(self.E, device=self.device)[:, None]
        s = torch.arange(self.S, device=self.device)[None, :]
        paid_tail = self.bodies[e, s, tail_ring]
        self.length -= burn.long()
        extra = burn & (self.seg_count > self.length)
        self.seg_count -= extra.long()
        if self.v2:
            for slot in range(self.S):
                self._append_food(
                    paid_tail[:, slot], extra[:, slot] & self._in_arena(paid_tail[:, slot]), True
                )
        traversed = torch.stack((head1, head2), dim=2)
        traversed_valid = torch.stack((move1, boost), dim=2)
        # Food is consumed at the first traversed cell.  Matching is vectorized,
        # then stable-compacted once, retaining append order of survivors.
        live_food = self.food_present
        eq = (traversed[:, :, :, None, :] == self.food_cells[:, None, None, :, :]).all(dim=-1)
        hit = eq & live_food[:, None, None, :] & traversed_valid[..., None]
        # CPU resolves food in snake-slot order and rechecks the remaining food
        # for each snake.  The fixed-S loop preserves that contested-boost case
        # while all state and selection work remain device tensors.
        picked = torch.zeros_like(self.food_present)
        ate = torch.zeros_like(self.alive)
        ate_corpse = torch.zeros_like(self.alive)
        food_slots = torch.arange(self.food_capacity, device=self.device)[None, :]
        for slot in range(self.S):
            available = hit[:, slot] & ~picked[:, None, :]
            by_t = available.any(dim=2)
            first_t = by_t.long().argmax(dim=1)
            tmask = torch.arange(2, device=self.device)[None, :] == first_t[:, None]
            candidates = (available & tmask[:, :, None]).any(dim=1)
            has = candidates.any(dim=1) & self.alive[:, slot]
            chosen = candidates.long().argmax(dim=1)
            select = has[:, None] & (food_slots == chosen[:, None])
            ate[:, slot] = has
            ate_corpse[:, slot] = (select & self.corpse_food).any(dim=1)
            picked |= select
        self.length += ate.long()
        self.food_present &= ~picked
        self.corpse_food &= self.food_present
        self._compact_food()
        movement_stage.__exit__(None, None, None)
        if self.train_mode:
            self._maintain_food()
        death_stage = self.stage("deaths_corpse")
        death_stage.__enter__()
        # Collisions use the post-move ring.  This is all tensor work, including
        # boost traversals and head swaps.
        in_arena = ((~traversed_valid) | self._in_arena(traversed)).all(dim=2)
        if self.move_backend == "indexed" and self.device.type != "meta":
            from research.gpu_sim_spike.move_cuda import collisions

            death, death_rank, dead, kills, kill_mass = collisions(
                self, head0, traversed, traversed_valid, in_arena
            )
            body = None
        else:
            body, self_hit, head_pair, body_pair = self._collision_flags_reference(
                head0, traversed, traversed_valid
            )
            death, death_rank, dead, kills, kill_mass = self._resolve_collisions_reference(
                in_arena, self_hit, head_pair, body_pair
            )
        died = dead
        # A single reduction rejects empty death events before body reordering,
        # candidate compaction, or native corpse dispatch. Meta retains shapes.
        if self.device.type == "meta" or bool(died.any().item()):
            if body is None:
                offsets = torch.arange(self.cap, device=self.device)[None, None, :]
                rings = (self.head_ptr[..., None] - offsets) % self.cap
                body = self.bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
            # Reorder corpse candidates by recorded resolution order, not snake id.
            snake_order = death_rank.argsort(dim=1, stable=True)
            body = body.gather(1, snake_order[:, :, None, None].expand(-1, -1, self.cap, 2))
            died_ordered = died.gather(1, snake_order)
            seg_ordered = self.seg_count.gather(1, snake_order)
            offsets = torch.arange(self.cap, device=self.device)[None, None, :]
            corpse_mask = died_ordered[..., None] & (offsets < seg_ordered[..., None])
            if not self.v2:
                corpse_mask &= offsets.remainder(2) == 0
            self._append_many(
                body.reshape(self.E, self.S * self.cap, 2),
                corpse_mask.reshape(self.E, self.S * self.cap),
                self.v2,
            )
        death_stage.__exit__(None, None, None)
        reward_stage = self.stage("movement")
        reward_stage.__enter__()
        phi_prev = prev_length.float() / PHI_LENGTH_DIVISOR
        phi_new = torch.where(
            died, torch.zeros_like(phi_prev), self.length.float() / PHI_LENGTH_DIVISOR
        )
        self._last_reward = (
            float(self.cfg.gamma) * phi_new
            - phi_prev
            + torch.where(
                died,
                torch.full_like(phi_new, float(self.cfg.death_value)),
                torch.zeros_like(phi_new),
            )
            + kill_mass.float() * float(self.cfg.kill_scale)
        )
        self.frames_since_food = torch.where(
            died,
            self.frames_since_food,
            torch.where(
                ate,
                torch.zeros_like(self.frames_since_food),
                self.frames_since_food + self.alive.long(),
            ),
        )
        self._reward_prev_length = torch.where(died, self._reward_prev_length, self.length)
        self.alive &= ~died
        self.respawn_timer = torch.where(
            died, torch.full_like(self.respawn_timer, int(self.cfg.frame_rate)), self.respawn_timer
        )
        self._last_done, self._last_death_cause, self._last_kills, self._last_food_ate = (
            died,
            death,
            kills,
            ate,
        )
        self._last_ambient_food_ate = ate & ~ate_corpse
        self._last_corpse_food_ate = ate_corpse
        self._last_transition_valid = acted
        self._prepared = False
        reward_stage.__exit__(None, None, None)
