"""Pilot-only CT sync hoisting, derived from frozen a2d489a9 CT mechanics.

No RNG consumer, slot ordering, rejection limit, physics or reward changes.
The shared CT implementation remains the independent oracle.
"""

from __future__ import annotations

import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import (
    _CARDINAL,
    _ENV_TENSOR_FIELDS,
    PHI_LENGTH_DIVISOR,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim as ReferenceCTDeviceSim


class PilotCTDeviceSim(ReferenceCTDeviceSim):
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
            local_idx = active_local.nonzero(as_tuple=False).flatten()
            if not local_idx.numel():
                break
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
            dup_only = occ_free & dup
            occ_blocked = ~occ_free
            event_flags = (
                torch.stack((placed.any(), dup_only.any(), occ_blocked.any())).cpu().tolist()
            )
            if event_flags[0]:
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
            if event_flags[1]:
                dup_local = local_idx[dup_only]
                inner_tries[dup_local] = 0
                outer_tries[dup_local] += 1
                exhausted[dup_local] |= outer_tries[dup_local] >= 100

            occ_blocked = ~occ_free
            if event_flags[2]:
                occ_local = local_idx[occ_blocked]
                inner_tries[occ_local] += 1
                exhausted[occ_local] |= inner_tries[occ_local] >= 100

        for name in ("food_cells", "food_present", "corpse_food", "food_count"):
            getattr(self, name).index_copy_(0, rows, getattr(sub, name))

    @torch.inference_mode()
    def reset_fresh_worlds(self) -> None:
        """Place dead nonterminal slots sequentially matching GridBatchSim._respawn_dead."""
        pending = ~self.alive & ~self.terminal_slots & (self.respawn_timer <= 0)
        slot_flags = pending.any(0).cpu().tolist()
        if not any(slot_flags):
            return
        any_placed = False
        for sidx in range(self.S):
            slot_need = pending[:, sidx]
            if not slot_flags[sidx]:
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
                local_idx = unplaced.nonzero(as_tuple=False).flatten()
                if not local_idx.numel():
                    break
                global_idx = rows.index_select(0, local_idx)
                cand = self._draw_for_rows(global_idx)
                cx = cand[:, 0]
                cy = cand[:, 1]
                in_bounds = (cx >= 0) & (cx < self.grid_w) & (cy >= 0) & (cy < self.grid_h)
                safe_x = cx.clamp(0, self.grid_w - 1)
                safe_y = cy.clamp(0, self.grid_h - 1)
                free = in_bounds & (occupied[local_idx, safe_y, safe_x] < 0)
                winners = free.nonzero(as_tuple=False).flatten()
                if winners.numel():
                    win_global = global_idx[winners]
                    win_local = local_idx[winners]
                    win_cand = cand[winners]
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
        # One event readback also bounds corpse gather to the exact live span.
        if self.device.type == "meta":
            has_death, span = True, self.cap
        else:
            has_death, span = torch.stack((died.any().long(), self.seg_count.max())).cpu().tolist()
            span = max(1, span)
        if has_death:
            if body is None:
                offsets = torch.arange(span, device=self.device)[None, None, :]
                rings = (self.head_ptr[..., None] - offsets) % self.cap
                body = self.bodies.gather(2, rings[..., None].expand(-1, -1, -1, 2))
            body = body[:, :, :span]
            snake_order = death_rank.argsort(dim=1, stable=True)
            body = body.gather(1, snake_order[:, :, None, None].expand(-1, -1, span, 2))
            died_ordered = died.gather(1, snake_order)
            seg_ordered = self.seg_count.gather(1, snake_order)
            offsets = torch.arange(span, device=self.device)[None, None, :]
            corpse_mask = died_ordered[..., None] & (offsets < seg_ordered[..., None])
            if not self.v2:
                corpse_mask &= offsets.remainder(2) == 0
            self._append_many(
                body.reshape(self.E, self.S * span, 2),
                corpse_mask.reshape(self.E, self.S * span),
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
