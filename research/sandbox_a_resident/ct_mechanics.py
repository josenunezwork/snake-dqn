"""Sandbox-local mechanics adapter, RNG inventory, and forced-state determinism suite for C/T.

Implements Ruling §4 ("C/T — corrected GPU premise test, before TRAIN-02"):
1. Explicit task identities separating the terminal-hero mechanics-v1 skill task
   with repaired non-train in-step ambient food refill from the mechanics-v2
   continuous-respawn Watch6 product task (zero v2 transfer claim).
2. Leaves certified ``gpu_sim_spike.DeviceSim`` and ``TRAIN-01``'s
   ``TrainingDeviceSim`` / ``ResidentTrainingTask`` completely untouched.
3. Repairs ``train_mode=False`` in-step ambient food refill ordering to execute
   sequentially per snake slot (0..S-1) immediately upon food consumption before
   collision resolution and corpse-food drop, matching ``GridBatchSim._consume_food``,
   with ``wall_thickness`` spawn margins, ``direction=1`` spawn/respawn heading,
   and strictly row-independent RNG counter advancement.
4. Inventories and rekeys every active post-prepare RNG consumer with
   domain-separated action-independent keys (Common Random Numbers across legal
   actions within each suffix).
5. Provides deterministic forced-state tests against CPU ``GridBatchSim`` with
   injected identical candidate draws across eat/death, simultaneous/contested
   food, boost, corpse, respawn, prepared-state/carry, and root restoration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import PHI_LENGTH_DIVISOR
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.sandbox_a_resident.training_rosters import (
    TrainingRoster,
    vector61_params,
)
from research.sandbox_a_resident.training_sim import TrainingDeviceSim
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim

CT_STUDY_SLUG = "t34-cf-1028"
CT_DOMAIN_PREFIX = "a100-dev/t34-cf-1028"
CT_SKILL_TASK_ID = "t34-cf-v1-terminal-hero-skill-instep-refill"
SEPARATED_WATCH_V2_TASK_ID = "watch6-v2-continuous-respawn-product"

_CARDINAL = ((0, -1), (1, 0), (0, 1), (-1, 0))

RNG_CONSUMER_INVENTORY: tuple[dict[str, Any], ...] = (
    {
        "id": "sim_instep_food_refill",
        "owner": "CTDeviceSim._refill_food_rows",
        "active_post_prepare": True,
        "state_fields": ("sim.rng_key", "sim.rng_counter"),
        "rekey_domain": f"{CT_DOMAIN_PREFIX}/branch-sim-rng/v1",
        "includes_action_id": False,
        "description": "In-step per-slot ambient food refill during step() when train_mode=False.",
    },
    {
        "id": "sim_prepare_food_refill",
        "owner": "CTDeviceSim._maintain_food_impl",
        "active_post_prepare": True,
        "state_fields": ("sim.rng_key", "sim.rng_counter"),
        "rekey_domain": f"{CT_DOMAIN_PREFIX}/branch-sim-rng/v1",
        "includes_action_id": False,
        "description": "Pre-decision ambient food maintenance on branch steps h >= 1.",
    },
    {
        "id": "sim_prepare_opponent_respawn",
        "owner": "CTDeviceSim.reset_fresh_worlds",
        "active_post_prepare": True,
        "state_fields": ("sim.rng_key", "sim.rng_counter"),
        "rekey_domain": f"{CT_DOMAIN_PREFIX}/branch-sim-rng/v1",
        "includes_action_id": False,
        "description": "Non-terminal dead opponent respawn placement on branch steps h >= 1.",
    },
    {
        "id": "opponent_random_safe",
        "owner": "CTOpponentRuntime.actions",
        "active_post_prepare": True,
        "state_fields": ("opponents.rng_key", "opponents.rng_step"),
        "rekey_domain": f"{CT_DOMAIN_PREFIX}/branch-opp-rng/v1",
        "includes_action_id": False,
        "description": "Mixed-roster random_safe opponent tie-breaking on branch steps h >= 0.",
    },
)


def task_separation_report() -> dict[str, Any]:
    """Explicit task-identity declaration separating v1 C/T skill from v2 Watch6."""
    return {
        "active_task_id": CT_SKILL_TASK_ID,
        "mechanics_version": 1,
        "train_mode": False,
        "hero_respawn": False,
        "opponent_respawn": True,
        "in_step_refill_ordering": "per_slot_sequential_before_collision_and_corpse",
        "separated_product_task_id": SEPARATED_WATCH_V2_TASK_ID,
        "separated_product_differences": [
            (
                "Watch6 uses mechanics_version=2 (size-ratio head-on wins, "
                "boost paid-tail food, capped corpse class)."
            ),
            (
                "Watch6 continuously respawns hero over 6,000-frame product sessions "
                "rather than terminal hero skill episodes."
            ),
            (
                "Watch6 uses five FRP3-s12+v8 opponents rather than balanced "
                "scripted/frozen/mixed skill rosters."
            ),
        ],
        "v2_transfer_claim": False,
        "train01_packet15_modified": False,
    }


def derive_63bit_key(domain: str, *parts: str | int) -> int:
    """Derive a nonzero positive 63-bit deterministic key via SHA-256."""
    joined = "|".join([domain, *(str(p) for p in parts)])
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    if val == 0:
        val = (int.from_bytes(digest[8:16], "big") & 0x7FFF_FFFF_FFFF_FFFF) or 1
    return val


@dataclass(frozen=True)
class BranchKeys:
    """Action-independent domain-separated keys for one (root_uid, suffix_k) pair."""

    root_uid: str
    suffix_index: int
    phase: str
    sim_rng_key: int
    opp_rng_key: int


def derive_branch_keys(root_uid: str, suffix_index: int) -> BranchKeys:
    """Derive CRN branch keys for suffix_index in 0..7 (0..3 selection, 4..7 scoring).

    Action ID is deliberately excluded so all legal candidate actions evaluated
    under (root_uid, suffix_index) share Common Random Numbers across every
    active post-prepare RNG consumer.
    """
    if not isinstance(root_uid, str) or not root_uid:
        raise ValueError("root_uid must be a nonempty string")
    if type(suffix_index) is not int or not 0 <= suffix_index < 8:
        raise ValueError("suffix_index must be an integer in 0..7")
    phase = "selection" if suffix_index < 4 else "scoring"
    sim_key = derive_63bit_key(
        f"{CT_DOMAIN_PREFIX}/branch-sim-rng/v1", phase, root_uid, suffix_index
    )
    opp_key = derive_63bit_key(
        f"{CT_DOMAIN_PREFIX}/branch-opp-rng/v1", phase, root_uid, suffix_index
    )
    return BranchKeys(
        root_uid=root_uid,
        suffix_index=suffix_index,
        phase=phase,
        sim_rng_key=sim_key,
        opp_rng_key=opp_key,
    )


def _splitmix_positive(key: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    """Device-portable 63-bit positive mixer using unsigned-equivalent wraparound."""
    # Constants wrapped to signed int64 so PyTorch CPU and CUDA wrap identically.
    c1 = -7046029254386353131  # 0x9E3779B97F4A7C15 - 2**64
    c2 = -4658895280553007687  # 0xBF58476D1CE4E5B9 - 2**64
    c3 = -7723592293110705685  # 0x94D049BB133111EB - 2**64
    z = key ^ (counter * c1)
    z = (z ^ ((z >> 30) & 0x3_FFFF_FFFF)) * c2
    z = (z ^ ((z >> 27) & 0x1F_FFFF_FFFF)) * c3
    z = z ^ ((z >> 31) & 0x1_FFFF_FFFF)
    return z & 0x7FFF_FFFF_FFFF_FFFF


class _InjectedCpuRngProxy:
    """Thin EnvRng wrapper feeding identical snapped cell draws to CPU GridBatchSim."""

    def __init__(
        self,
        queue: list[tuple[int, int]],
        *,
        segment_size: int,
        game_width: int,
        game_height: int,
        wall_thickness: int,
    ) -> None:
        self._queue = queue
        self.segment_size = int(segment_size)
        self.game_width = int(game_width)
        self.game_height = int(game_height)
        self.wall_thickness = int(wall_thickness)

    def _draw_find_empty_candidate(self) -> tuple[int, int]:
        if not self._queue:
            raise RuntimeError("exhausted injected CPU EnvRng draw queue")
        cx, cy = self._queue.pop(0)
        return (int(cx) * self.segment_size, int(cy) * self.segment_size)


class CTDeviceSim(TrainingDeviceSim):
    """Sandbox-local C/T simulator matching GridBatchSim refill and respawn rejection.

    Certified ``DeviceSim`` and ``TrainingDeviceSim`` remain untouched.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "capacity_overflow"):
            self.capacity_overflow = torch.zeros(
                (self.E, self.S), dtype=torch.bool, device=self.device
            )

    @classmethod
    def from_reference(
        cls, sim: Any, device: str | torch.device = "cpu", repeat: int = 1
    ) -> "CTDeviceSim":
        out = super().from_reference(sim, device=device, repeat=repeat)
        out.__class__ = cls
        if not hasattr(out, "capacity_overflow"):
            out.capacity_overflow = torch.zeros((out.E, out.S), dtype=torch.bool, device=out.device)
        return out  # type: ignore[return-value]

    def _spawn_bounds(self) -> tuple[int, int, int, int]:
        margin = max(0, int(self.cfg.wall_thickness) // int(self.cfg.segment_size))
        span_w = max(1, int(self.grid_w) - 2 * margin)
        span_h = max(1, int(self.grid_h) - 2 * margin)
        return margin, span_w, margin, span_h

    @torch.inference_mode()
    def _random_cells(self) -> torch.Tensor:
        """Draw candidate cells within wall_thickness interior margin."""
        self.rng_counter += 1
        mx, span_w, my, span_h = self._spawn_bounds()
        z = _splitmix_positive(self.rng_key, self.rng_counter)
        x = mx + z.remainder(span_w)
        y = my + (z >> 32).remainder(span_h)
        return torch.stack((x, y), dim=-1)

    @torch.inference_mode()
    def _draw_for_rows(self, active_rows: torch.Tensor) -> torch.Tensor:
        """Draw one candidate cell per active row, advancing only active rng_counters."""
        if "_random_cells" in self.__dict__:
            before = self.rng_counter.clone()
            full_cand = self._random_cells()
            after_active = torch.where(
                self.rng_counter.index_select(0, active_rows)
                == before.index_select(0, active_rows),
                before.index_select(0, active_rows) + 1,
                self.rng_counter.index_select(0, active_rows),
            )
            self.rng_counter.copy_(before)
            self.rng_counter.index_copy_(0, active_rows, after_active)
            return full_cand.index_select(0, active_rows)
        if int(active_rows.numel()) == self.E:
            return self._random_cells()
        self.rng_counter.index_add_(0, active_rows, torch.ones_like(active_rows))
        mx, span_w, my, span_h = self._spawn_bounds()
        z = _splitmix_positive(
            self.rng_key.index_select(0, active_rows),
            self.rng_counter.index_select(0, active_rows),
        )
        x = mx + z.remainder(span_w)
        y = my + (z >> 32).remainder(span_h)
        return torch.stack((x, y), dim=-1)

    def _refill_food_rows(self, active_mask: torch.Tensor) -> None:
        """Refill ambient deficit up to max_food matching GridBatchSim._spawn nested rejection."""
        max_food = int(self.cfg.max_food)
        ambient = (self.food_present & ~self.corpse_food).sum(dim=1)
        need = active_mask & (ambient < max_food)
        rows = need.nonzero(as_tuple=False).flatten()
        if not rows.numel():
            return

        sub = object.__new__(type(self))
        sub.__dict__ = self.__dict__.copy()
        for name in _ENV_TENSOR_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(sub, name, value.index_select(0, rows))
        sub.E = int(rows.numel())
        r2_cache.inherit(sub, self)
        occupied = sub.occupancy()

        remaining = (max_food - ambient.index_select(0, rows)).clamp_min(0)
        outer_tries = torch.zeros(sub.E, dtype=torch.long, device=self.device)
        inner_tries = torch.zeros(sub.E, dtype=torch.long, device=self.device)
        exhausted = torch.zeros(sub.E, dtype=torch.bool, device=self.device)

        while True:
            active_local = (remaining > 0) & (~exhausted)
            if not bool(active_local.any()):
                break
            local_idx = active_local.nonzero(as_tuple=False).flatten()
            global_idx = rows.index_select(0, local_idx)
            cand = self._draw_for_rows(global_idx)
            cx = cand[:, 0]
            cy = cand[:, 1]
            in_bounds = (cx >= 0) & (cx < self.grid_w) & (cy >= 0) & (cy < self.grid_h)
            safe_x = cx.clamp(0, self.grid_w - 1)
            safe_y = cy.clamp(0, self.grid_h - 1)
            occ_free = in_bounds & (occupied[local_idx, safe_y, safe_x] < 0)

            sub_cells = sub.food_cells.index_select(0, local_idx)
            sub_pres = sub.food_present.index_select(0, local_idx)
            dup = (
                sub_pres & (sub_cells[:, :, 0] == cx[:, None]) & (sub_cells[:, :, 1] == cy[:, None])
            ).any(dim=1)

            placed = occ_free & (~dup)
            if bool(placed.any()):
                placed_local = local_idx[placed]
                placed_cand = cand[placed]
                insert_mask = torch.zeros(sub.E, dtype=torch.bool, device=self.device)
                insert_full = torch.zeros((sub.E, 2), dtype=torch.long, device=self.device)
                insert_mask[placed_local] = True
                insert_full[placed_local] = placed_cand
                prev_fc = sub.food_count.clone()
                sub._append_food(insert_full, insert_mask, False)
                added = (sub.food_count - prev_fc) > 0
                remaining = torch.where(added, remaining - 1, remaining)
                exhausted |= insert_mask & (~added)
                inner_tries = torch.where(insert_mask, torch.zeros_like(inner_tries), inner_tries)
                outer_tries = torch.where(insert_mask, torch.zeros_like(outer_tries), outer_tries)

            dup_only = occ_free & dup
            if bool(dup_only.any()):
                dup_local = local_idx[dup_only]
                inner_tries[dup_local] = 0
                outer_tries[dup_local] += 1
                exhausted[dup_local] |= outer_tries[dup_local] >= 100

            occ_blocked = ~occ_free
            if bool(occ_blocked.any()):
                occ_local = local_idx[occ_blocked]
                inner_tries[occ_local] += 1
                exhausted[occ_local] |= inner_tries[occ_local] >= 100

        for name in ("food_cells", "food_present", "corpse_food", "food_count"):
            getattr(self, name).index_copy_(0, rows, getattr(sub, name))

    def _maintain_food_impl(self) -> None:
        """Pre-decision maintenance phase called once per frame for all worlds."""
        all_rows = torch.ones(self.E, dtype=torch.bool, device=self.device)
        self._refill_food_rows(all_rows)

    @torch.inference_mode()
    def reset_fresh_worlds(self) -> None:
        """Place dead nonterminal slots sequentially matching GridBatchSim._respawn_dead."""
        pending = ~self.alive & ~self.terminal_slots & (self.respawn_timer <= 0)
        if not bool(pending.any()):
            return
        any_placed = False
        for sidx in range(self.S):
            slot_need = pending[:, sidx]
            if not bool(slot_need.any()):
                continue
            rows = slot_need.nonzero(as_tuple=False).flatten()
            r2_cache.bump(self)
            sub = object.__new__(type(self))
            sub.__dict__ = self.__dict__.copy()
            for name in _ENV_TENSOR_FIELDS:
                value = getattr(self, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(sub, name, value.index_select(0, rows))
            sub.E = int(rows.numel())
            r2_cache.inherit(sub, self)
            occupied = sub.occupancy()

            unplaced = torch.ones(sub.E, dtype=torch.bool, device=self.device)
            for _ in range(100):
                if not bool(unplaced.any()):
                    break
                local_idx = unplaced.nonzero(as_tuple=False).flatten()
                global_idx = rows.index_select(0, local_idx)
                cand = self._draw_for_rows(global_idx)
                cx = cand[:, 0]
                cy = cand[:, 1]
                in_bounds = (cx >= 0) & (cx < self.grid_w) & (cy >= 0) & (cy < self.grid_h)
                safe_x = cx.clamp(0, self.grid_w - 1)
                safe_y = cy.clamp(0, self.grid_h - 1)
                free = in_bounds & (occupied[local_idx, safe_y, safe_x] < 0)
                if bool(free.any()):
                    win_global = global_idx[free]
                    win_local = local_idx[free]
                    win_cand = cand[free]
                    self.bodies[win_global, sidx, 0] = win_cand.to(self.bodies.dtype)
                    self.head_ptr[win_global, sidx] = 0
                    self.seg_count[win_global, sidx] = 1
                    self.length[win_global, sidx] = 1
                    self.direction[win_global, sidx] = 1
                    self.alive[win_global, sidx] = True
                    self.respawn_timer[win_global, sidx] = 0
                    self.boost_frames[win_global, sidx] = 0
                    self.frames_since_food[win_global, sidx] = 0
                    self._reward_prev_length[win_global, sidx] = 1
                    unplaced[win_local] = False
                    any_placed = True
        if any_placed:
            r2_cache.bump(self)

    @torch.inference_mode()
    def step(self, actions: torch.Tensor) -> None:
        """Advance one frame with per-slot sequential non-train food refill before deaths."""
        if self.train_mode:
            return super().step(actions)
        if actions.shape != (self.E, self.S):
            raise ValueError(f"actions must have shape {(self.E, self.S)}")
        if actions.device != self.device:
            raise ValueError("actions must be on the CTDeviceSim device")
        actions = actions.long().clamp(0, 5)
        r2_cache.bump(self)
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
        cardinal = (
            r2_cache.const(_CARDINAL, torch.long, self.device)
            if r2_cache.enabled(self)
            else torch.tensor(_CARDINAL, dtype=torch.long, device=self.device)
        )
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

        # Ordered per-slot food consumption and immediate per-eat ambient refill
        # matching CPU GridBatchSim._consume_food (train_mode=False).
        ate = torch.zeros_like(self.alive)
        ate_corpse = torch.zeros_like(self.alive)
        eq_any = (traversed[:, :, :, None, :] == self.food_cells[:, None, None, :, :]).all(dim=-1)
        any_candidate = bool(
            (eq_any & self.food_present[:, None, None, :] & traversed_valid[..., None]).any()
        )
        if any_candidate:
            food_slots = torch.arange(self.food_capacity, device=self.device)[None, :]
            t_idx = torch.arange(2, device=self.device)[None, :]
            for slot in range(self.S):
                eq_s = (traversed[:, slot, :, None, :] == self.food_cells[:, None, :, :]).all(
                    dim=-1
                )
                available = eq_s & self.food_present[:, None, :] & traversed_valid[:, slot, :, None]
                by_t = available.any(dim=2)
                first_t = by_t.long().argmax(dim=1)
                tmask = t_idx == first_t[:, None]
                candidates = (available & tmask[:, :, None]).any(dim=1)
                has = candidates.any(dim=1) & self.alive[:, slot]
                if not bool(has.any()):
                    continue
                chosen = candidates.long().argmax(dim=1)
                select = has[:, None] & (food_slots == chosen[:, None])
                is_corpse = (select & self.corpse_food).any(dim=1)
                ate[:, slot] = has
                ate_corpse[:, slot] = is_corpse
                self.length[:, slot] += has.long()
                self.food_present &= ~select
                self.corpse_food &= self.food_present
                self._compact_food()
                r2_cache.bump(self)
                with self.stage("food"):
                    self._refill_food_rows(has)
                r2_cache.bump(self)

        movement_stage.__exit__(None, None, None)
        death_stage = self.stage("deaths_corpse")
        death_stage.__enter__()
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
        if self.device.type == "meta" or bool(died.any().item()):
            if body is None:
                offsets = torch.arange(self.cap, device=self.device)[None, None, :]
                rings = (self.head_ptr[..., None] - offsets) % self.cap
                body = self.bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
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
        r2_cache.bump(self)


def _sim_view(sim: Any, rows: torch.Tensor) -> Any:
    out = object.__new__(type(sim))
    out.__dict__ = sim.__dict__.copy()
    out.E = int(rows.numel())
    out.cfg = replace(sim.cfg, num_envs=out.E)
    for name in _ENV_TENSOR_FIELDS:
        setattr(out, name, getattr(sim, name).index_select(0, rows))
    r2_cache.inherit(out, sim)
    return out


_VECTOR_RUNTIME_ENV_FIELDS: tuple[str, ...] = (
    "controlled",
    "prev_dist",
    "prev_id",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
)


class CTOpponentRuntime:
    """Device-resident opponent runtime with explicit rekeyable random_safe counter RNG.

    Replaces the 5,000-frame clamped ``random_table`` with per-world
    ``(rng_key, rng_step)`` SplitMix64 state so:
    1. Collection, 5-tick ring replay, and arbitrary-horizon rollouts never clamp;
    2. Per-world carry and RNG state can be losslessly snapshotted and restored;
    3. Counterfactual H300 branches rekey ``(rng_key, rng_step=0)`` per suffix
       independently of hero action ID (exact CRN across actions).
    """

    def __init__(
        self,
        sim: Any,
        rosters: Sequence[TrainingRoster],
        *,
        b8_network: nn.Module,
        checkpoint_networks: Mapping[str, nn.Module],
        opp_rng_keys: Sequence[int] | None = None,
    ) -> None:
        if len(rosters) != int(sim.E):
            raise ValueError("every world needs one immutable roster")
        self.device = torch.device(sim.device)
        self.rosters = tuple(rosters)
        self.E = int(sim.E)
        if opp_rng_keys is None:
            keys = [
                derive_63bit_key(f"{CT_DOMAIN_PREFIX}/opp-init/v1", idx) for idx in range(self.E)
            ]
        else:
            if len(opp_rng_keys) != self.E:
                raise ValueError("opp_rng_keys length must match sim.E")
            keys = [int(k) & 0x7FFF_FFFF_FFFF_FFFF or 1 for k in opp_rng_keys]
        self.rng_key = torch.tensor(keys, dtype=torch.long, device=self.device)
        self.rng_step = torch.zeros(self.E, dtype=torch.long, device=self.device)

        self._groups: list[tuple[torch.Tensor, DevicePolicy, tuple[int, ...]]] = []
        self._row_to_group: list[tuple[int, int]] = [(-1, -1)] * self.E
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for index, roster in enumerate(self.rosters):
            grouped.setdefault(roster.identity, []).append(index)
        for group_idx, (_identity, indexes) in enumerate(grouped.items()):
            rows = torch.tensor(indexes, dtype=torch.long, device=self.device)
            for local_idx, global_idx in enumerate(indexes):
                self._row_to_group[global_idx] = (group_idx, local_idx)
            roster = self.rosters[indexes[0]]
            nets: list[nn.Module | None] = []
            kinds: list[str | None] = []
            tables: list[torch.Tensor | None] = []
            random_slots: list[int] = []
            for slot, (kind, reference) in enumerate(roster.slots, start=1):
                if kind == "checkpoint":
                    if reference not in checkpoint_networks:
                        raise KeyError(f"missing verified frozen checkpoint {reference}")
                    nets.append(checkpoint_networks[reference].to(self.device).eval())
                    kinds.append(None)
                    tables.append(None)
                else:
                    nets.append(None)
                    kinds.append("greedy_food" if reference == "random_safe" else reference)
                    tables.append(None)
                    if reference == "random_safe":
                        random_slots.append(slot)
            runtime = DeviceVectorRuntime(vector61_params(sim.cfg))
            controlled = torch.tensor(
                [[False] + [kind == "checkpoint" for kind, _ in roster.slots]],
                device=self.device,
                dtype=torch.bool,
            ).expand(len(indexes), -1)
            runtime.bind(_sim_view(sim, rows), controlled)
            policy = DevicePolicy(
                self.device,
                b8_network,
                tuple(nets),
                tuple(kinds),
                tuple(tables),
                runtime,
                vector_prepared=False,
                v8_lambda=roster.veto_lambda,
            )
            policy._training_pending_prepare = torch.ones(
                len(indexes), dtype=torch.bool, device=self.device
            )
            self._groups.append((rows, policy, tuple(random_slots)))

    @torch.inference_mode()
    def actions(self, sim: Any, hero_actions: torch.Tensor) -> torch.Tensor:
        if hero_actions.shape != (sim.E,):
            raise ValueError("hero actions must have shape (E,)")
        out = torch.empty((sim.E, sim.S), dtype=torch.long, device=self.device)
        for rows, policy, random_slots in self._groups:
            sub = _sim_view(sim, rows)
            pending = getattr(policy, "_training_pending_prepare", None)
            was_prepared = policy.vector_prepared
            if was_prepared and pending is not None and bool(pending.any()):
                policy.vector_runtime.prepare(sub, pending)
                pending.zero_()
            action = policy.actions(sub, hero_actions.index_select(0, rows))
            if pending is not None and not was_prepared:
                pending.zero_()
            if random_slots:
                g_key = self.rng_key.index_select(0, rows)
                g_step = self.rng_step.index_select(0, rows)
                adv = sub.advisory_mask()
                for slot in random_slots:
                    safe = adv[:, slot, :3]
                    safe = torch.where(safe.any(1, keepdim=True), safe, torch.ones_like(safe))
                    count = safe.sum(1)
                    counter = g_step * 8 + int(slot) + 1
                    mixed = _splitmix_positive(g_key, counter)
                    rank = mixed.remainder(count)
                    chosen = ((safe.cumsum(1) - 1 == rank[:, None]) & safe).long().argmax(1)
                    action[:, slot] = torch.where(
                        sub.alive[:, slot], chosen, torch.ones_like(chosen)
                    )
            out.index_copy_(0, rows, action)
        self.rng_step += 1
        return out

    @torch.inference_mode()
    def after_step(self, sim: Any) -> None:
        for rows, policy, _random_slots in self._groups:
            policy.after_step(_sim_view(sim, rows))

    @torch.inference_mode()
    def reset(
        self, mask: torch.Tensor, opp_keys: Mapping[int, int] | Sequence[int] | None = None
    ) -> None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        reset_rows = mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if opp_keys is not None:
            for r in reset_rows:
                val = int(opp_keys[r]) & 0x7FFF_FFFF_FFFF_FFFF or 1
                self.rng_key[r] = val
                self.rng_step[r] = 0
        else:
            self.rng_step[mask] = 0
        for rows, policy, _random_slots in self._groups:
            local = mask.index_select(0, rows)
            runtime = policy.vector_runtime
            if not bool(local.any()) or runtime is None:
                continue
            for name in ("_carry_valid", "sel_ready", "sel_fresh", "_alive_post"):
                value = getattr(runtime, name, None)
                if value is not None:
                    value[local] = False
            for name in ("prev_dist",):
                value = getattr(runtime, name, None)
                if value is not None:
                    value[local] = torch.inf
            for name in ("prev_id",):
                value = getattr(runtime, name, None)
                if value is not None:
                    value[local] = -1
            pending = getattr(policy, "_training_pending_prepare", None)
            if pending is None:
                pending = torch.zeros_like(local)
                policy._training_pending_prepare = pending
            pending |= local

    @torch.inference_mode()
    def rekey_branches(self, opp_keys: torch.Tensor) -> None:
        """Rekey random_safe post-prepare streams with action-independent suffix keys."""
        if opp_keys.shape != (self.E,):
            raise ValueError("opp_keys must have shape (E,)")
        self.rng_key.copy_(opp_keys.to(device=self.device, dtype=torch.long))
        self.rng_step.zero_()

    def snapshot_row(self, row: int) -> dict[str, Any]:
        """Extract lossless CPU carry and RNG state for a single environment row."""
        group_idx, local_idx = self._row_to_group[int(row)]
        _rows, policy, _random_slots = self._groups[group_idx]
        runtime = policy.vector_runtime
        carry: dict[str, torch.Tensor] = {}
        for key in _VECTOR_RUNTIME_ENV_FIELDS:
            value = getattr(runtime, key, None)
            if isinstance(value, torch.Tensor):
                carry[key] = value[local_idx].detach().to("cpu").clone()
        pending = getattr(policy, "_training_pending_prepare", None)
        pending_flag = bool(pending[local_idx].item()) if pending is not None else False
        return {
            "roster_identity": self.rosters[int(row)].identity,
            "vector_prepared": bool(policy.vector_prepared),
            "pending_prepare": pending_flag,
            "carry": carry,
            "rng_key": int(self.rng_key[int(row)].item()),
            "rng_step": int(self.rng_step[int(row)].item()),
        }

    @torch.inference_mode()
    def load_row_snapshots(self, snapshots: Sequence[Mapping[str, Any]]) -> None:
        """Restore lossless per-row carry and RNG snapshots across all E worlds."""
        if len(snapshots) != self.E:
            raise ValueError("snapshot count must equal runtime world count")
        for row_idx, snap in enumerate(snapshots):
            if snap["roster_identity"] != self.rosters[row_idx].identity:
                raise ValueError("opponent snapshot roster identity mismatch")
            self.rng_key[row_idx] = int(snap["rng_key"])
            self.rng_step[row_idx] = int(snap["rng_step"])
            group_idx, local_idx = self._row_to_group[row_idx]
            _rows, policy, _random_slots = self._groups[group_idx]
            runtime = policy.vector_runtime
            for key, tensor_val in snap["carry"].items():
                target = getattr(runtime, key)
                target[local_idx].copy_(tensor_val.to(device=self.device, dtype=target.dtype))
            pending = getattr(policy, "_training_pending_prepare", None)
            if pending is None:
                pending = torch.zeros(_rows.numel(), dtype=torch.bool, device=self.device)
                policy._training_pending_prepare = pending
            # If the saved policy group was not yet vector_prepared, mark this
            # row pending so actions() prepares its selector cleanly even when
            # mixed into a group with already-prepared rows.
            need_prepare = bool(snap["pending_prepare"]) or (not bool(snap["vector_prepared"]))
            pending[local_idx] = need_prepare
        for _rows, policy, _random_slots in self._groups:
            # With per-row pending_prepare set for any unprepared row, setting
            # vector_prepared=True ensures already-prepared rows preserve their
            # exact selector state while unprepared rows refresh via pending.
            policy.vector_prepared = True

    @torch.inference_mode()
    def compact_rows(self, keep: torch.Tensor) -> None:
        """Compact opponent runtime in-place to surviving world indices ``keep``."""
        keep = keep.to(device=self.device, dtype=torch.long).flatten()
        keep_cpu = [int(i) for i in keep.detach().cpu().tolist()]
        new_e = len(keep_cpu)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_cpu)}
        self.rosters = tuple(self.rosters[i] for i in keep_cpu)
        self.rng_key = self.rng_key.index_select(0, keep)
        self.rng_step = self.rng_step.index_select(0, keep)
        self.E = new_e
        new_groups: list[tuple[torch.Tensor, DevicePolicy, tuple[int, ...]]] = []
        self._row_to_group = [(-1, -1)] * new_e
        for rows, policy, random_slots in self._groups:
            old_group_rows = [int(r) for r in rows.detach().cpu().tolist()]
            local_keep_list: list[int] = []
            mapped_global_list: list[int] = []
            for local_pos, old_global in enumerate(old_group_rows):
                new_global = old_to_new.get(old_global)
                if new_global is not None:
                    local_keep_list.append(local_pos)
                    mapped_global_list.append(new_global)
            if not local_keep_list:
                continue
            local_keep_t = torch.tensor(local_keep_list, dtype=torch.long, device=self.device)
            new_rows_t = torch.tensor(mapped_global_list, dtype=torch.long, device=self.device)
            runtime = policy.vector_runtime
            for key in _VECTOR_RUNTIME_ENV_FIELDS:
                val = getattr(runtime, key, None)
                if isinstance(val, torch.Tensor):
                    setattr(runtime, key, val.index_select(0, local_keep_t))
            pending = getattr(policy, "_training_pending_prepare", None)
            if isinstance(pending, torch.Tensor):
                policy._training_pending_prepare = pending.index_select(0, local_keep_t)
            new_group_idx = len(new_groups)
            for new_local, new_global in enumerate(mapped_global_list):
                self._row_to_group[new_global] = (new_group_idx, new_local)
            new_groups.append((new_rows_t, policy, random_slots))
        self._groups = new_groups


def _sync_forced_state_to_cpu_and_gpu(
    cfg: BatchSimConfig,
    *,
    bodies: list[list[tuple[int, int]]],
    lengths: list[int],
    alive: list[bool],
    directions: list[int],
    food: list[tuple[int, int, bool]],
    boost_frames: list[int] | None = None,
    respawn_timer: list[int] | None = None,
    draws: Sequence[tuple[int, int]] = (),
    device: str | torch.device = "cpu",
) -> tuple[GridBatchSim, CTDeviceSim]:
    """Build matched 1-world CPU GridBatchSim and CTDeviceSim on an exact forced state."""
    assert cfg.num_envs == 1
    S = cfg.num_snakes
    cpu = GridBatchSim(cfg, seeds=[42], train_mode=False, allow_respawn=True)
    cpu.bodies.fill(0)
    cpu.head_ptr.fill(0)
    cpu.seg_count.fill(0)
    cpu.length[0] = np.asarray(lengths, dtype=np.int64)
    cpu._reward_prev_length[0] = np.asarray(lengths, dtype=np.int64)
    cpu.alive[0] = np.asarray(alive, dtype=bool)
    cpu.direction[0] = np.asarray(directions, dtype=np.int64)
    cpu.boost_frames[0] = np.asarray(boost_frames or [0] * S, dtype=np.int64)
    cpu.frames_since_food.fill(0)
    cpu.respawn_timer[0] = np.asarray(respawn_timer or [0] * S, dtype=np.int64)
    cpu._boosted_this_step.fill(False)
    cpu.frame[0] = 10

    for sidx in range(S):
        segs = bodies[sidx]
        if not alive[sidx] or not segs:
            continue
        n = len(segs)
        # segs[0] is head, segs[-1] is tail -> place head at head_ptr = n - 1
        cpu.head_ptr[0, sidx] = n - 1
        cpu.seg_count[0, sidx] = n
        for k, (cx, cy) in enumerate(segs):
            cpu.bodies[0, sidx, n - 1 - k] = (cx, cy)

    cpu.food_cells[0] = [(cx, cy) for cx, cy, _c in food]
    cpu.food_set[0] = set(cpu.food_cells[0])
    cpu.corpse_cells[0] = {(cx, cy) for cx, cy, is_c in food if is_c}
    cpu._rebuild_env_grids(0, food=True)
    cpu._rebuild_traversed_from_heads(np.ones(1, dtype=bool))
    cpu._refresh_action_masks(np.ones(1, dtype=bool))

    gpu = CTDeviceSim.from_reference(cpu, device=device)
    if gpu.device.type == "cuda":
        gpu.move_backend = "indexed"
    gpu.capacity_overflow = torch.zeros((gpu.E, gpu.S), dtype=torch.bool, device=gpu.device)
    r2_cache.enable(gpu)

    cpu_queue = list(draws)
    cpu._rngs[0] = _InjectedCpuRngProxy(  # type: ignore[assignment]
        cpu_queue,
        segment_size=cfg.segment_size,
        game_width=cfg.game_width,
        game_height=cfg.game_height,
        wall_thickness=cfg.wall_thickness,
    )
    gpu_queue = [(int(x), int(y)) for x, y in draws]

    def _injected_random_cells() -> torch.Tensor:
        if not gpu_queue:
            raise RuntimeError("exhausted injected CTDeviceSim _random_cells draw queue")
        cx, cy = gpu_queue.pop(0)
        gpu.rng_counter += 1
        return torch.tensor([[cx, cy]], dtype=torch.long, device=gpu.device)

    gpu._random_cells = _injected_random_cells  # type: ignore[method-assign]
    return cpu, gpu


def _assert_discrete_state_exact(cpu: GridBatchSim, gpu: CTDeviceSim, label: str) -> None:
    """Assert 100% exact equality of discrete mechanics, food order, and advisory masks."""
    assert int(cpu.frame[0]) == int(gpu.frame[0].item()), f"{label}: frame mismatch"
    np.testing.assert_array_equal(
        cpu.alive[0], gpu.alive[0].cpu().numpy(), err_msg=f"{label}: alive"
    )
    np.testing.assert_array_equal(
        cpu.length[0], gpu.length[0].cpu().numpy(), err_msg=f"{label}: length"
    )
    np.testing.assert_array_equal(
        cpu.seg_count[0], gpu.seg_count[0].cpu().numpy(), err_msg=f"{label}: seg_count"
    )
    np.testing.assert_array_equal(
        cpu.direction[0], gpu.direction[0].cpu().numpy(), err_msg=f"{label}: direction"
    )
    np.testing.assert_array_equal(
        cpu.boost_frames[0], gpu.boost_frames[0].cpu().numpy(), err_msg=f"{label}: boost_frames"
    )
    np.testing.assert_array_equal(
        cpu.respawn_timer[0], gpu.respawn_timer[0].cpu().numpy(), err_msg=f"{label}: respawn_timer"
    )
    np.testing.assert_array_equal(
        cpu._last_done[0], gpu._last_done[0].cpu().numpy(), err_msg=f"{label}: _last_done"
    )
    np.testing.assert_array_equal(
        cpu._last_death_cause[0],
        gpu._last_death_cause[0].cpu().numpy(),
        err_msg=f"{label}: _last_death_cause",
    )
    np.testing.assert_array_equal(
        cpu._last_kills[0], gpu._last_kills[0].cpu().numpy(), err_msg=f"{label}: _last_kills"
    )
    np.testing.assert_array_equal(
        cpu._last_food_ate[0],
        gpu._last_food_ate[0].cpu().numpy(),
        err_msg=f"{label}: _last_food_ate",
    )
    np.testing.assert_array_equal(
        cpu._last_ambient_food_ate[0],
        gpu._last_ambient_food_ate[0].cpu().numpy(),
        err_msg=f"{label}: _last_ambient_food_ate",
    )
    np.testing.assert_array_equal(
        cpu._last_corpse_food_ate[0],
        gpu._last_corpse_food_ate[0].cpu().numpy(),
        err_msg=f"{label}: _last_corpse_food_ate",
    )
    for sidx in range(cpu.S):
        if not cpu.alive[0, sidx]:
            continue
        n = int(cpu.seg_count[0, sidx])
        cpu_hp = int(cpu.head_ptr[0, sidx])
        gpu_hp = int(gpu.head_ptr[0, sidx].item())
        cpu_segs = [tuple(map(int, cpu.bodies[0, sidx, (cpu_hp - k) % cpu.cap])) for k in range(n)]
        gpu_segs = [
            tuple(map(int, gpu.bodies[0, sidx, (gpu_hp - k) % gpu.cap].cpu().tolist()))
            for k in range(n)
        ]
        assert cpu_segs == gpu_segs, f"{label}: snake {sidx} segments differ"

    gpu_present = gpu.food_present[0].cpu().numpy()
    gpu_cells = [tuple(map(int, rc)) for rc in gpu.food_cells[0].cpu().numpy()[gpu_present]]
    gpu_corpse = {
        tuple(map(int, rc))
        for rc, ic in zip(
            gpu.food_cells[0].cpu().numpy()[gpu_present],
            gpu.corpse_food[0].cpu().numpy()[gpu_present],
        )
        if ic
    }
    assert (
        list(cpu.food_cells[0]) == gpu_cells
    ), f"{label}: food_cells order mismatch: cpu={cpu.food_cells[0]} vs gpu={gpu_cells}"
    assert set(cpu.corpse_cells[0]) == gpu_corpse, f"{label}: corpse_cells mismatch"

    cpu_mask = cpu.get_advisory_action_mask()[0]
    gpu_mask = gpu.advisory_mask()[0].cpu().numpy()
    np.testing.assert_array_equal(cpu_mask, gpu_mask, err_msg=f"{label}: advisory_mask")


@torch.inference_mode()
def forced_state_determinism_suite(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Run all 7 required deterministic forced-state tests before scoring.

    Covers:
    1. Simultaneous eat + death (verifying v1 in-step refill before corpse drop)
    2. Simultaneous / contested food + boost 2-cell traversal + multi-eater refill
    3. Boost burn and min_boost_length gating
    4. Ordered corpse drop across multiple simultaneous deaths
    5. Opponent respawn timer countdown, rejection placement, and terminal hero lock
    6. Prepared-state observation / v8 / vector61 carry numerical & mask parity
    7. Mid-episode root snapshot restoration and continuation determinism
    """
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.vector61_policy import Vector61Runtime

    target_device = torch.device(device)
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=400,
        game_height=400,
        segment_size=20,
        wall_thickness=20,
        initial_food=4,
        max_food=4,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    draws_pool = [(x, y) for y in range(2, 18) for x in range(2, 18)]
    results: dict[str, str] = {}

    # 1. Eat + Death on the same frame (FROZEN14 v1 critical seam: snake 0 eats
    # ambient food at (6,5) so ambient drops 4->3 and refills IMMEDIATELY to 4
    # BEFORE snake 1 hits wall at (19,10)->(20,10) and drops v1 ambient-counted corpse pellets!).
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(5, 5), (4, 5), (3, 5), (2, 5)],
            [(19, 10), (18, 10), (17, 10), (16, 10), (15, 10), (14, 10)],
            [(5, 15), (4, 15)],
            [],
            [],
            [],
        ],
        lengths=[4, 6, 2, 1, 1, 1],
        alive=[True, True, True, False, False, False],
        directions=[1, 1, 1, 1, 1, 1],
        food=[(6, 5, False), (10, 10, False), (12, 12, False), (14, 14, False)],
        respawn_timer=[0, 0, 0, 5, 5, 5],
        draws=draws_pool,
        device=target_device,
    )
    act = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64)
    cpu.step(act)
    gpu.step(torch.tensor(act, dtype=torch.long, device=target_device))
    _assert_discrete_state_exact(cpu, gpu, "case1_eat_plus_death_v1_refill")
    # Confirm snake 0 grew to 5, snake 1 died at wall (k=0 OOB at x=20, k=2,4 -> 2 corpse pellets),
    # and total food is 4 (3 surviving + 1 in-step refill) + 2 corpse = 6 (vs 5 without refill)!
    assert len(cpu.food_cells[0]) == 6
    results["eat_and_death"] = "PASS"

    # 2. Simultaneous / contested food + boost 2-cell fallback + sequential 2-eater refill:
    # Snake 0 (normal) and Snake 1 (boost) both contest food at (6, 6); Snake 0 wins (6, 6),
    # so Snake 1's t=0 misses (6, 6) and its boost t=1 eats (6, 7)! Both trigger sequential refills!
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(5, 6), (4, 6), (3, 6), (2, 6), (2, 7)],
            [(6, 5), (6, 4), (6, 3), (6, 2), (7, 2), (8, 2)],
            [(10, 15), (9, 15)],
            [],
            [],
            [],
        ],
        lengths=[5, 6, 2, 1, 1, 1],
        alive=[True, True, True, False, False, False],
        directions=[1, 2, 1, 1, 1, 1],  # snake 0 right -> (6,6); snake 1 down+boost -> (6,6),(6,7)
        food=[(6, 6, False), (6, 7, False), (12, 12, False), (14, 14, False)],
        respawn_timer=[0, 0, 0, 5, 5, 5],
        draws=[(6, 6), (6, 7), (12, 12), (9, 9), (11, 11)] + draws_pool,
        device=target_device,
    )
    act = np.array([[1, 4, 1, 1, 1, 1]], dtype=np.int64)
    cpu.step(act)
    gpu.step(torch.tensor(act, dtype=torch.long, device=target_device))
    _assert_discrete_state_exact(cpu, gpu, "case2_contested_boost_food")
    assert bool(cpu._last_food_ate[0, 0]) and bool(cpu._last_food_ate[0, 1])
    results["simultaneous_contested_food"] = "PASS"

    # 3. Boost burn and min_boost_length refusal:
    # Snake 0 has length=5 and boost_frames = boost_length_cost_frames - 1 -> boosts & burns 1 seg!
    # Snake 1 has length=4 (< min_boost_length=5) -> requests boost (4) -> refused, moves 1 cell!
    burn_threshold = int(cfg.boost_length_cost_frames)
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(5, 5), (4, 5), (3, 5), (2, 5), (2, 6)],
            [(5, 10), (4, 10), (3, 10), (2, 10)],
            [],
            [],
            [],
            [],
        ],
        lengths=[5, 4, 1, 1, 1, 1],
        alive=[True, True, False, False, False, False],
        directions=[1, 1, 1, 1, 1, 1],
        food=[(15, 15, False), (16, 15, False), (17, 15, False), (18, 15, False)],
        boost_frames=[burn_threshold - 1, 0, 0, 0, 0, 0],
        respawn_timer=[0, 0, 5, 5, 5, 5],
        draws=draws_pool,
        device=target_device,
    )
    act = np.array([[4, 4, 1, 1, 1, 1]], dtype=np.int64)
    cpu.step(act)
    gpu.step(torch.tensor(act, dtype=torch.long, device=target_device))
    _assert_discrete_state_exact(cpu, gpu, "case3_boost_burn_and_min_len")
    assert int(cpu.length[0, 0]) == 4 and int(cpu.boost_frames[0, 0]) == 0
    assert tuple(map(int, cpu.heads()[0, 0])) == (7, 5)
    assert tuple(map(int, cpu.heads()[0, 1])) == (6, 10)
    results["boost_mechanics"] = "PASS"

    # 4. Ordered corpse drop across multiple deaths (body collision + wall death):
    # Snake 2 hits wall vs snake 1->0 body collision; verify exact corpse drop ordering parity.
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(8, 8), (8, 9), (8, 10), (8, 11)],
            [(7, 9), (6, 9), (5, 9), (4, 9)],  # moves right into snake 0's body at (8,9)
            [(19, 4), (18, 4), (17, 4), (16, 4)],  # moves right into wall at (20,4)
            [],
            [],
            [],
        ],
        lengths=[4, 4, 4, 1, 1, 1],
        alive=[True, True, True, False, False, False],
        directions=[0, 1, 1, 1, 1, 1],
        food=[(2, 2, False), (3, 2, False), (4, 2, False), (5, 2, False)],
        respawn_timer=[0, 0, 0, 5, 5, 5],
        draws=draws_pool,
        device=target_device,
    )
    act = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64)
    cpu.step(act)
    gpu.step(torch.tensor(act, dtype=torch.long, device=target_device))
    _assert_discrete_state_exact(cpu, gpu, "case4_corpse_drop_order")
    results["corpse_ordering"] = "PASS"

    # 5. Respawn timer countdown, multi-slot cross-collision rejection, and terminal hero lock:
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(5, 5), (4, 5), (3, 5)],
            [],  # dead timer=1 -> decrements to 0, rejects (5,5), spawns at (9,9)
            [],  # dead timer=1 -> decrements to 0, rejects newly occupied (9,9), spawns at (11,11)
            [],
            [],  # dead timer=2 -> decrements to 1, stays dead!
            [],
        ],
        lengths=[3, 1, 1, 1, 1, 1],
        alive=[True, False, False, False, False, False],
        directions=[1, 3, 2, 1, 2, 1],
        food=[(12, 12, False), (13, 12, False), (14, 12, False), (15, 12, False)],
        respawn_timer=[0, 1, 1, 5, 2, 5],
        draws=[(5, 5), (9, 9), (9, 9), (11, 11)] + draws_pool,
        device=target_device,
    )
    gpu.terminal_slots[0, 3] = True
    gpu.respawn_timer[0, 3] = 1
    cpu.respawn_timer[0, 3] = 5  # keep CPU slot 3 unready to match terminal lock on slot 3
    act = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64)
    cpu.step(act)
    gpu.step(torch.tensor(act, dtype=torch.long, device=target_device))
    assert bool(gpu.alive[0, 1].item()) and bool(gpu.alive[0, 2].item())
    assert not bool(gpu.alive[0, 4].item())
    assert not bool(gpu.alive[0, 3].item()), "terminal slot must never respawn"
    # Slot 1 spawned at (9,9) dir=1 -> (10,9); slot 2 spawned at (11,11) -> (12,11)
    assert tuple(map(int, gpu.heads()[0, 1].cpu().tolist())) == (10, 9)
    assert tuple(map(int, cpu.heads()[0, 1].tolist())) == (10, 9)
    assert tuple(map(int, gpu.heads()[0, 2].cpu().tolist())) == (12, 11)
    assert tuple(map(int, cpu.heads()[0, 2].tolist())) == (12, 11)
    assert int(gpu.direction[0, 1].item()) == 1 and int(gpu.direction[0, 2].item()) == 1

    # Verify empty-board duplicate + snake-occupied refill rejection directly on production path:
    cfg_dup = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=400,
        game_height=400,
        segment_size=20,
        wall_thickness=20,
        initial_food=2,
        max_food=2,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    cpu_dup, gpu_dup = _sync_forced_state_to_cpu_and_gpu(
        cfg_dup,
        bodies=[[(5, 5), (4, 5), (3, 5)], [], [], [], [], []],
        lengths=[3, 1, 1, 1, 1, 1],
        alive=[True, False, False, False, False, False],
        directions=[1, 1, 1, 1, 1, 1],
        food=[],
        respawn_timer=[0, 5, 5, 5, 5, 5],
        # (2,2) placed -> (2,2) rejected as duplicate -> (5,5) rejected as occupied -> (3,3) placed!
        draws=[(2, 2), (2, 2), (5, 5), (3, 3)],
        device=target_device,
    )
    cpu_dup.step(act)
    gpu_dup.step(torch.tensor(act, dtype=torch.long, device=target_device))
    _assert_discrete_state_exact(cpu_dup, gpu_dup, "case5b_duplicate_and_occupied_refill_rejection")
    assert list(cpu_dup.food_cells[0]) == [(2, 2), (3, 3)]
    results["respawn_and_terminal_lock"] = "PASS"

    # 6. Prepared-state ego2s-bv8 observation + vector61 carry numerical parity:
    cpu, gpu = _sync_forced_state_to_cpu_and_gpu(
        cfg,
        bodies=[
            [(8, 8), (7, 8), (6, 8), (5, 8), (5, 9), (5, 10)],
            [(11, 8), (12, 8), (13, 8), (14, 8), (14, 9)],
            [(8, 12), (9, 12), (10, 12)],
            [],
            [],
            [],
        ],
        lengths=[6, 5, 3, 1, 1, 1],
        alive=[True, True, True, False, False, False],
        directions=[1, 3, 0, 1, 1, 1],
        food=[(9, 8, False), (10, 10, False), (15, 15, False), (3, 3, False)],
        respawn_timer=[0, 0, 0, 5, 5, 5],
        draws=draws_pool,
        device=target_device,
    )
    # Prepare both at the decision seam and compare ego2s-bv8 + vector61 carry across 2 steps.
    vparams = vector61_params(cfg)
    cpu_vrun = Vector61Runtime(vparams)
    cpu_vrun.bind(cpu, np.array([[False, True, True, False, False, False]], dtype=bool))
    gpu_vrun = DeviceVectorRuntime(vparams)
    gpu_vrun.bind(
        gpu, torch.tensor([[False, True, True, False, False, False]], device=target_device)
    )

    for step_idx in range(2):
        # Advance to prepared seam identically:
        cpu.frame += 1
        cpu._maintain_food()
        cpu._respawn_dead()
        cpu._refresh_action_masks(np.ones(1, dtype=bool))
        gpu.prepare()

        cpu_obs = build_ego_raster(
            cpu, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        rows_hero = torch.tensor([[0, 0]], dtype=torch.long, device=target_device)
        gpu_legal = gpu.advisory_mask()[:, 0]
        gpu_v8 = (
            v8_decide(
                gpu,
                rows_hero,
                torch.zeros((1, 6), device=target_device),
                gpu_legal,
                8.0,
                features=True,
            )
            .feature_planes()
            .reshape(1, 30)
        )
        g_loc, g_glob, g_scal = build_observation(gpu, rows_hero, gpu_v8)
        np.testing.assert_allclose(
            cpu_obs["local"], g_loc.cpu().numpy(), atol=1, err_msg=f"step {step_idx} local"
        )
        np.testing.assert_allclose(
            cpu_obs["global"], g_glob.cpu().numpy(), atol=1, err_msg=f"step {step_idx} global"
        )
        np.testing.assert_allclose(
            cpu_obs["scalars"],
            g_scal.cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"step {step_idx} scalars",
        )

        cpu_vrun.prepare(cpu)
        gpu_vrun.prepare(gpu)
        cpu_vec, cpu_vmask = cpu_vrun.selection(np.array([[0, 1], [0, 2]], dtype=np.int64))
        gpu_vec, gpu_vmask = gpu_vrun.selection(
            torch.tensor([[0, 1], [0, 2]], dtype=torch.long, device=target_device)
        )
        np.testing.assert_array_equal(
            cpu_vmask, gpu_vmask.cpu().numpy(), err_msg=f"step {step_idx} vector61 mask"
        )
        np.testing.assert_allclose(
            cpu_vec,
            gpu_vec.cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"step {step_idx} vector61 carry",
        )

        if step_idx == 0:
            # Execute one synchronous step without double-preparing CPU:
            # CPU already ran frame+=1, maintain_food, respawn_dead above;
            # undo frame+=1 since food was full (4) and timers (5) > 1 so cpu.step matches gpu.step!
            cpu.frame -= 1
            cpu.respawn_timer[0, 3:] += 1
            step_act = np.array([[0, 2, 1, 1, 1, 1]], dtype=np.int64)
            cpu.step(step_act)
            cpu_vrun.observe_step(cpu)
            gpu.step(torch.tensor(step_act, dtype=torch.long, device=target_device))
            gpu_vrun.observe_step(gpu)
    results["prepared_state_and_carry"] = "PASS"

    # 7. Root restoration determinism (prepared state + carry + RNG continuation):
    gpu.__dict__.pop("_random_cells", None)
    gpu.rng_key.fill_(derive_63bit_key("test-restore-sim", 0))
    gpu.rng_counter.fill_(17)
    gpu.prepare()
    saved_sim = {name: getattr(gpu, name).detach().clone() for name in _ENV_TENSOR_FIELDS}
    saved_prepared = bool(gpu._prepared)
    saved_vcarry = {
        k: v.detach().clone() for k, v in gpu_vrun.__dict__.items() if isinstance(v, torch.Tensor)
    }

    # Roll forward 5 steps from the prepared root
    actions_seq = [
        torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long, device=target_device),
        torch.tensor([[0, 2, 0, 1, 1, 1]], dtype=torch.long, device=target_device),
        torch.tensor([[2, 0, 2, 1, 1, 1]], dtype=torch.long, device=target_device),
    ]
    expected_heads = []
    for a_step in actions_seq:
        gpu.prepare()
        gpu_vrun.prepare(gpu)
        v_before, _ = gpu_vrun.selection(
            torch.tensor([[0, 1]], dtype=torch.long, device=target_device)
        )
        gpu.step(a_step)
        gpu_vrun.observe_step(gpu)
        expected_heads.append((gpu.heads().clone(), v_before.clone(), gpu.food_cells.clone()))

    # Mutate simulator arbitrarily, then restore saved root and replay
    gpu.step(torch.tensor([[2, 2, 2, 1, 1, 1]], dtype=torch.long, device=target_device))
    for name, val in saved_sim.items():
        getattr(gpu, name).copy_(val)
    gpu._prepared = saved_prepared
    r2_cache.bump(gpu)
    for k, val in saved_vcarry.items():
        getattr(gpu_vrun, k).copy_(val)

    for idx_step, a_step in enumerate(actions_seq):
        gpu.prepare()
        gpu_vrun.prepare(gpu)
        v_replay, _ = gpu_vrun.selection(
            torch.tensor([[0, 1]], dtype=torch.long, device=target_device)
        )
        gpu.step(a_step)
        gpu_vrun.observe_step(gpu)
        exp_h, exp_v, exp_f = expected_heads[idx_step]
        assert torch.equal(gpu.heads(), exp_h), f"restore step {idx_step} heads"
        assert torch.equal(v_replay, exp_v), f"restore step {idx_step} vector61"
        assert torch.equal(gpu.food_cells, exp_f), f"restore step {idx_step} food"
    results["root_restoration"] = "PASS"

    return {
        "schema": "t34-cf-forced-state-determinism/v1",
        "status": "PASS",
        "device": str(target_device),
        "cases": results,
    }
