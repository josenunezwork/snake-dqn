"""Outcome-independent resident batching; frozen CT paths remain the parity oracle."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike.anchors_device import choose_anchor
from research.gpu_sim_spike.device_policy import _masked_argmax
from research.gpu_sim_spike.ego_device import build_observation as ego_observation
from research.gpu_sim_spike.v8_device import v8_decide
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from research.sandbox_a_resident.ct_mechanics import (
    _VECTOR_RUNTIME_ENV_FIELDS,
    CTOpponentRuntime,
    _splitmix_positive,
    derive_63bit_key,
)
from research.sandbox_a_resident.ct_snapshots import (
    _SMALL_ENV_TENSOR_FIELDS,
    CompactWorldSnapshot,
    SnapshotOverflowError,
)
from research.sandbox_a_resident.training_model import canonical_legal_mask
from research.sandbox_a_resident.training_rosters import vector61_params
from research.spatial_student_pilot.observation import compact_from_sim

# Resolve close argmax decisions with the same single-row fp32 forward in every
# slab. Never round Q, add epsilon to scores, or change the lowest-index tie rule.
NEAR_TIE = 1e-5


def stable_q(
    network: nn.Module,
    inputs: Sequence[torch.Tensor],
    q: torch.Tensor,
    legal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Re-evaluate potentially tied rows independently of the outer slab size."""
    if legal is None:
        legal = torch.ones_like(q, dtype=torch.bool)
    legal = torch.where(
        legal.any(1, keepdim=True), legal, torch.arange(6, device=q.device)[None] < 3
    )
    ordered = q.masked_fill(~legal, -torch.inf).sort(dim=1, descending=True).values
    # Check all pairs, since v8 can restrict the eligible set after its base argmax.
    near = (ordered[:, :-1] - ordered[:, 1:]).abs().le(NEAR_TIE).any(1)
    ids = near.nonzero(as_tuple=False).flatten().tolist()
    if ids:
        q = q.clone()
        for row in ids:
            value = network(*(x[row : row + 1] for x in inputs))
            if isinstance(value, tuple):
                value = value[0]
            q[row] = value[0].float()
    return q


@torch.inference_mode()
def observe_hero_bv8(sim: Any, b8: nn.Module) -> tuple[torch.Tensor, ...]:
    """Canonical hero decision, preserving all raster/mask/feature math."""
    if not sim._prepared:
        sim.prepare()
    world = torch.arange(sim.E, device=sim.device)
    rows = torch.stack((world, torch.zeros_like(world)), dim=1)
    raw = sim.advisory_mask()[:, 0]
    features = (
        v8_decide(sim, rows, torch.zeros((sim.E, 6), device=sim.device), raw, 8.0, features=True)
        .feature_planes()
        .reshape(sim.E, 30)
    )
    local, glob, scalar = ego_observation(sim, rows, features)
    legal = canonical_legal_mask(raw)
    q = grouped_q(b8, (local, glob, scalar), legal)
    return local, glob, scalar, legal, q, _masked_argmax(q, legal)


def grouped_q(
    network: nn.Module, inputs: Sequence[torch.Tensor], legal: torch.Tensor
) -> torch.Tensor:
    """Keep E128 inference groups independent of the resident physics width."""
    parts = []
    for start in range(0, len(legal), 128):
        args = tuple(x[start : start + 128] for x in inputs)
        value = network(*args)
        if isinstance(value, tuple):
            value = value[0]
        parts.append(stable_q(network, args, value.float(), legal[start : start + 128]))
    return torch.cat(parts)


def legacy_checkpoint_q(
    network: nn.Module, state: torch.Tensor, groups: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Forward each E128 roster/slot group exactly as DevicePolicy does.

    A checkpoint may control several slots in one roster. Combining those rows
    changes GEMM shapes and can change float32 bytes on CUDA. Keep observation
    and physics batched, but gather each legacy group in ascending world order.
    Unlike hero inference, the legacy opponent path has no near-tie re-forward.
    """
    q = torch.empty((len(state), 6), dtype=torch.float32, device=state.device)
    for ids in groups:
        q.index_copy_(0, ids, network(state.index_select(0, ids)).float())
    return q


class ControlledVectorRuntime(DeviceVectorRuntime):
    """Static controlled identities only; keep lifecycle/carry updates full shape."""

    def features(
        self, sim: Any, rows: Any = None, *, update_memory: bool = True, memory_mask: Any = None
    ) -> torch.Tensor:
        if rows is not None:
            return super().features(sim, rows, update_memory=update_memory, memory_mask=memory_mask)
        selected = self.controlled_rows
        er, sr = selected.unbind(1)
        subset_mask = None if memory_mask is None else memory_mask.reshape(sim.E, sim.S)[er, sr]
        values = super().features(
            sim, selected, update_memory=update_memory, memory_mask=subset_mask
        )
        out = torch.zeros((sim.E, sim.S, self.params.input_size), device=sim.device)
        out[er, sr] = values
        return out.reshape(sim.E * sim.S, -1)


class BatchedOpponents(CTOpponentRuntime):
    """One vector61 prepare/observe pass; legacy E128 roster/slot forwards.

    World/slot indices address the original sim directly. No _sim_view or full
    body index-select copies, and R2's live-span/memo generation is shared by
    hero, vector61, scripted and v8 consumers of this decision seam.
    """

    def __init__(
        self,
        sim: Any,
        rosters: Sequence[Any],
        *,
        b8_network: nn.Module,
        checkpoint_networks: Mapping[str, nn.Module],
        opp_rng_keys: Sequence[int] | None = None,
    ) -> None:
        if len(rosters) != sim.E:
            raise ValueError("every world needs one immutable roster")
        self.device, self.E, self.rosters = sim.device, sim.E, tuple(rosters)
        keys = (
            opp_rng_keys
            if opp_rng_keys is not None
            else [derive_63bit_key("t34-spatial/opponent-unused-init", i) for i in range(sim.E)]
        )
        if len(keys) != sim.E:
            raise ValueError("opp_rng_keys length must match sim.E")
        self.rng_key = torch.tensor(keys, device=sim.device, dtype=torch.long)
        self.rng_step = torch.zeros(sim.E, device=sim.device, dtype=torch.long)
        self.runtime = ControlledVectorRuntime(vector61_params(sim.cfg))
        controlled = torch.tensor(
            [[False] + [kind == "checkpoint" for kind, _ in r.slots] for r in rosters],
            dtype=torch.bool,
            device=sim.device,
        )
        self.runtime.bind(sim, controlled)
        self.pending = torch.ones(sim.E, dtype=torch.bool, device=sim.device)
        self.prepared = False
        self._needs_prepare = True
        self.networks = checkpoint_networks
        self._dispatch()

    def _dispatch(self) -> None:
        groups: dict[str, list[tuple[int, int]]] = {}
        anchors: dict[str, list[tuple[int, int]]] = {}
        for row, roster in enumerate(self.rosters):
            for slot, (kind, ref) in enumerate(roster.slots, 1):
                (groups if kind == "checkpoint" else anchors).setdefault(ref, []).append(
                    (row, slot)
                )
        self.reference_q_groups = {}
        for key, entries in groups.items():
            reference = {}
            for i, (row, slot) in enumerate(entries):
                reference.setdefault((row // 128, self.rosters[row].identity, slot), []).append(i)
            self.reference_q_groups[key] = [
                torch.tensor(ids, device=self.device) for ids in reference.values()
            ]
        self.groups = {
            key: torch.tensor(rows, device=self.device, dtype=torch.long)
            for key, rows in groups.items()
        }
        if any(ref not in self.networks for ref in self.groups):
            raise KeyError("missing verified frozen checkpoint")
        self.runtime.controlled_rows = torch.tensor(
            [row for rows in groups.values() for row in rows], device=self.device, dtype=torch.long
        ).reshape(-1, 2)
        self.anchors = {
            key: torch.tensor(rows, device=self.device, dtype=torch.long)
            for key, rows in anchors.items()
        }

    @torch.inference_mode()
    def actions(self, sim: Any, hero_actions: torch.Tensor) -> torch.Tensor:
        if hero_actions.shape != (sim.E,):
            raise ValueError("hero actions must have shape (E,)")
        if not self.prepared or self._needs_prepare:
            self.runtime.prepare(sim, self.pending if self.prepared else None)
            self.pending.zero_()
            self.prepared = True
            self._needs_prepare = False
        out = torch.ones((sim.E, sim.S), dtype=torch.long, device=self.device)
        out[:, 0] = hero_actions
        # Preserve legacy GEMM composition while sharing observation and v8 work.
        all_rows, all_q, all_masks = [], [], []
        for ref, rows in self.groups.items():
            state, mask = self.runtime.selection(rows)
            q = legacy_checkpoint_q(self.networks[ref], state, self.reference_q_groups[ref])
            if getattr(self, "verify_q_bytes", False):
                for ids in self.reference_q_groups[ref]:
                    inputs = state.index_select(0, ids)
                    legacy = self.networks[ref](inputs).float()
                    if not torch.equal(q.index_select(0, ids), legacy):
                        raise ValueError(
                            "checkpoint Q bytes differ from E128 legacy inference groups"
                        )
            all_rows.append(rows)
            all_q.append(q)
            all_masks.append(mask)
        if all_rows:
            rows = torch.cat(all_rows)
            out[rows[:, 0], rows[:, 1]] = v8_decide(
                sim, rows, torch.cat(all_q), torch.cat(all_masks), 8.0
            ).action
        for ref, rows in self.anchors.items():
            if ref == "random_safe":
                safe = sim.advisory_mask()[rows[:, 0], rows[:, 1], :3]
                safe = torch.where(safe.any(1, keepdim=True), safe, torch.ones_like(safe))
                mixed = _splitmix_positive(
                    self.rng_key[rows[:, 0]], self.rng_step[rows[:, 0]] * 8 + rows[:, 1] + 1
                )
                rank = mixed.remainder(safe.sum(1))
                action = ((safe.cumsum(1) - 1 == rank[:, None]) & safe).long().argmax(1)
                out[rows[:, 0], rows[:, 1]] = torch.where(
                    sim.alive[rows[:, 0], rows[:, 1]], action, 1
                )
            else:
                # choose_anchor is slot-specific; full-sim readers reuse R2 memo.
                for slot in range(1, sim.S):
                    selected = rows[rows[:, 1] == slot, 0]
                    if selected.numel():
                        out[selected, slot] = choose_anchor(sim, slot, ref, None)[selected]
        self.rng_step += 1
        return out

    @torch.inference_mode()
    def after_step(self, sim: Any) -> None:
        self.runtime.observe_step(sim)
        self.prepared = False

    def snapshot_rows(self, rows: Sequence[int]) -> list[dict[str, Any]]:
        ids = torch.tensor(rows, device=self.device, dtype=torch.long)
        carry = {
            key: getattr(self.runtime, key).index_select(0, ids).detach().cpu().clone()
            for key in _VECTOR_RUNTIME_ENV_FIELDS
        }
        pending = self.pending[ids].cpu().tolist()
        keys, steps = self.rng_key[ids].cpu().tolist(), self.rng_step[ids].cpu().tolist()
        return [
            dict(
                roster_identity=self.rosters[row].identity,
                vector_prepared=self.prepared,
                pending_prepare=bool(pending[i]),
                carry={key: value[i].clone() for key, value in carry.items()},
                rng_key=int(keys[i]),
                rng_step=int(steps[i]),
            )
            for i, row in enumerate(rows)
        ]

    def snapshot_row(self, row: int) -> dict[str, Any]:
        return self.snapshot_rows([row])[0]

    @torch.inference_mode()
    def load_row_snapshots(self, snapshots: Sequence[Mapping[str, Any]]) -> None:
        if len(snapshots) != self.E:
            raise ValueError("snapshot count must equal runtime world count")
        for row, snap in enumerate(snapshots):
            if snap["roster_identity"] != self.rosters[row].identity:
                raise ValueError("opponent snapshot roster identity mismatch")
        for key in _VECTOR_RUNTIME_ENV_FIELDS:
            target = getattr(self.runtime, key)
            target.copy_(torch.stack([s["carry"][key] for s in snapshots]).to(target))
        self.rng_key.copy_(torch.tensor([s["rng_key"] for s in snapshots], device=self.device))
        self.rng_step.copy_(torch.tensor([s["rng_step"] for s in snapshots], device=self.device))
        flags = {bool(s["vector_prepared"]) for s in snapshots}
        self.prepared = flags.pop() if len(flags) == 1 else True
        self._needs_prepare = any(
            s["pending_prepare"] or (self.prepared and not s["vector_prepared"]) for s in snapshots
        )
        self.pending.copy_(
            torch.tensor(
                [
                    s["pending_prepare"] or (self.prepared and not s["vector_prepared"])
                    for s in snapshots
                ],
                device=self.device,
            )
        )

    @torch.inference_mode()
    def compact_rows(self, keep: torch.Tensor) -> None:
        indexes = keep.cpu().tolist()
        self.rosters = tuple(self.rosters[i] for i in indexes)
        self.E = len(indexes)
        for key in _VECTOR_RUNTIME_ENV_FIELDS:
            setattr(self.runtime, key, getattr(self.runtime, key).index_select(0, keep))
        for key in ("rng_key", "rng_step", "pending"):
            setattr(self, key, getattr(self, key).index_select(0, keep))
        self._dispatch()


def split_compact(compact: dict[str, Any]) -> list[dict[str, Any]]:
    """Trim each accepted row to exactly the previous scalar extraction's bytes."""
    result = []
    for row in range(compact["body_count"].shape[0]):
        body_span = max(1, int(compact["body_count"][row].max()))
        used = (compact["food_kind"][row] > 0).nonzero(as_tuple=False).flatten()
        food_span = int(used[-1]) + 1 if used.numel() else 1
        item = {}
        for key, value in compact.items():
            if not isinstance(value, torch.Tensor):
                item[key] = value
                continue
            value = value[row : row + 1]
            if key == "body_cells":
                value = value[:, :, :body_span]
            elif key in ("food_cells", "food_kind"):
                value = value[:, :food_span]
            item[key] = value.clone()
        result.append(item)
    return result


@torch.inference_mode()
def capture_rows(
    sim: Any,
    opponents: Any,
    rows: Sequence[int],
    *,
    world_ids: Sequence[int],
    tick: int,
    max_body_capacity: int,
) -> tuple[list[dict[str, Any]], list[CompactWorldSnapshot]]:
    """Gather accepted live spans once, transfer per field, then split on host."""
    ids = torch.tensor(rows, device=sim.device, dtype=torch.long)
    if sim.cap > max_body_capacity or bool(sim.capacity_overflow[ids].any()):
        raise SnapshotOverflowError("body capacity overflow")
    compact = {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in compact_from_sim(sim, rows).items()
    }
    compacts = split_compact(compact)
    small = {
        name: getattr(sim, name).index_select(0, ids).detach().cpu().clone()
        for name in _SMALL_ENV_TENSOR_FIELDS
    }
    # Food slot identities and absent-buffer bytes are deliberately kept separate
    # from the compact observation (which is an observation, not resume state).
    food = sim.food_cells[ids].cpu().numpy()
    present = sim.food_present[ids].cpu().numpy()
    corpse = sim.corpse_food[ids].cpu().numpy()
    states = (
        opponents.snapshot_rows(rows)
        if hasattr(opponents, "snapshot_rows")
        else [opponents.snapshot_row(row) for row in rows]
    )
    snapshots = []
    for i, world in enumerate(world_ids):
        counts = small["seg_count"][i].numpy().astype(np.int64)
        if (
            (counts < 0).any()
            or (counts > sim.cap).any()
            or bool((small["length"][i] > max_body_capacity).any())
        ):
            raise SnapshotOverflowError("body span overflow")
        offsets = np.concatenate(([0], counts.cumsum())).astype(np.int32)
        segments = np.concatenate(
            [
                compact["body_cells"][i, slot, : int(count)].numpy()
                for slot, count in enumerate(counts)
            ]
        ).astype(np.int16)
        active = np.flatnonzero(present[i]).astype(np.int16)
        snapshots.append(
            CompactWorldSnapshot(
                world_id=int(world),
                life_id=0,
                tick=tick,
                prepared=bool(sim._prepared),
                num_snakes=sim.S,
                body_cap=sim.cap,
                food_cap=sim.food_capacity,
                body_offsets=offsets,
                body_segments_xy=segments,
                food_slot_indices=active,
                food_xy=food[i, active].astype(np.int16),
                food_corpse=corpse[i, active].astype(np.uint8),
                small_tensors={key: value[i].clone() for key, value in small.items()},
                opponent_state=states[i],
            )
        )
    return compacts, snapshots


def restore_fanout(snapshots: Sequence[CompactWorldSnapshot], sim: Any, opponents: Any) -> None:
    """Restore each distinct root once per slab, then fan out device tensors."""
    from types import SimpleNamespace

    from research.sandbox_a_resident.ct_snapshots import restore_world_snapshots_into

    unique, inverse, index = [], [], {}
    for snapshot in snapshots:
        # Caller interns snapshots by root identity; distinct roots never alias.
        key = id(snapshot)
        if key not in index:
            index[key] = len(unique)
            unique.append(snapshot)
        inverse.append(index[key])
    if len(unique) == len(snapshots):
        restore_world_snapshots_into(snapshots, sim, opponents)
        return
    fields = (*_SMALL_ENV_TENSOR_FIELDS, "bodies", "food_cells", "food_present", "corpse_food")
    proxy = SimpleNamespace(**sim.__dict__)
    proxy.E = len(unique)
    for name in fields:
        value = getattr(sim, name)
        setattr(
            proxy,
            name,
            torch.empty((len(unique), *value.shape[1:]), device=value.device, dtype=value.dtype),
        )
    sink = SimpleNamespace(E=len(unique), load_row_snapshots=lambda states: None)
    restore_world_snapshots_into(unique, proxy, sink)
    gather = torch.tensor(inverse, device=sim.device, dtype=torch.long)
    for name in fields:
        getattr(sim, name).copy_(getattr(proxy, name).index_select(0, gather))
    sim._prepared = proxy._prepared
    from research.gpu_sim_spike import r2_cache

    r2_cache.bump(sim)
    opponents.load_row_snapshots([s.opponent_state for s in snapshots])
