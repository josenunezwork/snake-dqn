"""Sandbox-only natural reset sampler for resident training worlds.

The certified simulator keeps its original respawn path.  This adapter is used
only to create fresh task worlds: it independently draws each slot, rejects
occupied candidates, and excludes the outer wall ring.  It is distributional
CPU-comparable support, not a common-RNG-tape parity claim.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.device_sim import ResidentDeviceSim


class TrainingDeviceSim(ResidentDeviceSim):
    """Adds a non-adjacent per-slot reset sampler without editing DeviceSim."""

    @classmethod
    def from_reference(
        cls, sim: Any, device: str | torch.device = "cpu", repeat: int = 1
    ) -> "TrainingDeviceSim":
        """Preserve uint64 seed bits before the certified signed-tensor boundary."""
        view = copy.copy(sim)
        view._seeds = [int(seed) if int(seed) < 2**63 else int(seed) - 2**64 for seed in sim._seeds]
        return super().from_reference(view, device=device, repeat=repeat)

    @torch.inference_mode()
    def _random_cells(self) -> torch.Tensor:
        """Use independent mixed x/y bits for sandbox food/rejection streams."""
        self.rng_counter += 1
        z = self.rng_key ^ (self.rng_counter * 0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        z = z ^ (z >> 31)
        return torch.stack((z.remainder(self.grid_w), (z >> 32).remainder(self.grid_h)), dim=-1)

    def _maintain_food_impl(self) -> None:
        """Refill in FIFO order with the sandbox stream on every compact row.

        Every world consumes the same fixed per-frame counter budget regardless
        of occupancy in any other world. The mutable compact view retains this
        class, so it cannot accidentally call the certified assay's LCG mixer.
        """
        trials = int(self.cfg.max_food)
        ambient = (self.food_present & ~self.corpse_food).sum(1)
        rows = (ambient < trials).nonzero(as_tuple=False).flatten()
        if not rows.numel():
            self.rng_counter += trials
            return
        sub = object.__new__(type(self))
        sub.__dict__ = self.__dict__.copy()
        # Explicit field ownership also covers the E == S case.
        from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS

        for name in _ENV_TENSOR_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(sub, name, value.index_select(0, rows))
        sub.E = int(rows.numel())
        r2_cache.inherit(sub, self)
        occupied = sub.occupancy()
        env = torch.arange(sub.E, device=self.device)
        count = ambient.index_select(0, rows)
        for _ in range(trials):
            candidate = sub._random_cells()
            free = occupied[env, candidate[:, 1], candidate[:, 0]] < 0
            previous = sub.food_count.clone()
            sub._append_food(candidate, (count < trials) & free, False)
            count += sub.food_count - previous
            if bool((count >= trials).all()):
                break
        self.rng_counter += trials
        for name in ("food_cells", "food_present", "corpse_food", "food_count"):
            getattr(self, name).index_copy_(0, rows, getattr(sub, name))

    @torch.inference_mode()
    def _respawn(self) -> None:
        """Apply normal timers then independently sample every ready opponent."""
        dead = ~self.alive
        self.respawn_timer = torch.where(
            dead & (self.respawn_timer > 0), self.respawn_timer - 1, self.respawn_timer
        )
        # Reuse the reset sampler for ready lanes only. Terminal hero remains
        # excluded, so its task-boundary death cannot silently respawn.
        self.reset_fresh_worlds()

    @torch.inference_mode()
    def reset_fresh_worlds(self) -> None:
        """Place every dead, nonterminal slot from independent counter lanes."""
        pending = ~self.alive & ~self.terminal_slots & (self.respawn_timer <= 0)
        # Ordinary fully-live frames must not pay 64 occupancy paints.  This is
        # a sandbox sampler branch, so its explicit scalar check is recorded as
        # a simulator-D throughput tradeoff rather than hidden in the engine.
        if not bool(pending.any()):
            return
        placed = torch.zeros_like(pending)
        slots = torch.arange(self.S, device=self.device)[None, :]
        for trial in range(64):
            # Slot decorrelation is explicit; do not derive five adjacent cells
            # from one base candidate as the A-3 timing assay did.
            counter = self.rng_counter[:, None] + trial * self.S + slots + 1
            z = self.rng_key[:, None] ^ (counter * 0x9E3779B97F4A7C15)
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB
            z = z ^ (z >> 31)
            x = 1 + z.remainder(max(1, self.grid_w - 2))
            y = 1 + (z // max(1, self.grid_w - 2)).remainder(max(1, self.grid_h - 2))
            candidate = torch.stack((x, y), dim=-1)
            occupied = self.occupancy()
            env = torch.arange(self.E, device=self.device)[:, None]
            free = occupied[env, y, x] < 0
            equal = (candidate[:, :, None] == candidate[:, None, :]).all(-1)
            prior = torch.tril(
                torch.ones((self.S, self.S), dtype=torch.bool, device=self.device), -1
            )
            unique = ~(equal & prior[None]).any(2)
            take = pending & ~placed & free & unique
            self.bodies[:, :, 0] = torch.where(take[..., None], candidate, self.bodies[:, :, 0])
            self.head_ptr = torch.where(take, torch.zeros_like(self.head_ptr), self.head_ptr)
            self.seg_count = torch.where(take, torch.ones_like(self.seg_count), self.seg_count)
            self.length = torch.where(take, torch.ones_like(self.length), self.length)
            self.alive |= take
            placed |= take
            if bool((pending & ~placed).any()) is False:
                break
        # A respawn allocation belongs only to ready lanes.  Advancing every
        # world when one unrelated opponent dies would couple later food draws.
        self.rng_counter += pending.any(dim=1).long() * (64 * self.S)
        if not bool((placed | ~pending).all()):
            raise RuntimeError("sandbox reset spawn rejection exhausted")
        self.direction = torch.where(placed, torch.ones_like(self.direction), self.direction)
        self.respawn_timer = torch.where(
            placed, torch.zeros_like(self.respawn_timer), self.respawn_timer
        )
        self.boost_frames = torch.where(
            placed, torch.zeros_like(self.boost_frames), self.boost_frames
        )
        self.frames_since_food = torch.where(
            placed, torch.zeros_like(self.frames_since_food), self.frames_since_food
        )
        self._reward_prev_length = torch.where(
            placed, torch.ones_like(self.length), self._reward_prev_length
        )
