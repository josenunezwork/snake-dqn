"""Lossless bounded root snapshot store and 5-tick lookback ring for C/T.

Implements Ruling §4 ("Build a lossless bounded root snapshot store with explicit
overflow refusal for full legal body spans up to 10,400, food, timers, prepared
state, and opponent memory/RNG; never truncate a long body" and "5-tick checkpoint
ring + exact action-tape replay").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_policy import _masked_argmax
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_topology import (
    IncrementalSparseTopologyBuilder,
    SparseTopologyBatch,
    compute_topology_features,
)
from research.sandbox_a_resident.training_model import canonical_legal_mask
from research.sandbox_a_resident.training_rosters import TrainingRoster


@torch.inference_mode()
def observe_hero_bv8(
    sim: CTDeviceSim,
    b8_network: nn.Module,
    *,
    v8_lambda: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact prepared hero (local_u8, glob_u8, scalars_f32, legal_mask, q_b8, a_b8)."""
    if not sim._prepared:
        sim.prepare()
    world_ids = torch.arange(sim.E, device=sim.device, dtype=torch.long)
    rows = torch.stack((world_ids, torch.zeros_like(world_ids)), dim=1)
    raw_legal = sim.advisory_mask()[:, 0]
    v8_feat = (
        v8_decide(
            sim,
            rows,
            torch.zeros((sim.E, 6), device=sim.device),
            raw_legal,
            v8_lambda,
            features=True,
        )
        .feature_planes()
        .reshape(sim.E, 30)
    )
    local_obs, glob_obs, scalars_obs = build_observation(sim, rows, v8_feat)
    legal = canonical_legal_mask(raw_legal)
    q_b8 = b8_network(local_obs, glob_obs, scalars_obs).float()
    a_b8 = _masked_argmax(q_b8, legal)
    return local_obs, glob_obs, scalars_obs, legal, q_b8, a_b8


MAX_LEGAL_BODY_CAPACITY = 10_400

_SMALL_ENV_TENSOR_FIELDS: tuple[str, ...] = (
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
    "capacity_overflow",
)


class SnapshotOverflowError(RuntimeError):
    """Raised when a root snapshot exceeds bounded store capacity or body span."""


@dataclass
class CompactWorldSnapshot:
    """Lossless variable-length CPU snapshot of one prepared world and its opponents.

    Preserves full legal body spans up to ``max_body_capacity`` (10,400) without
    ever truncating a body, exact ring ``head_ptr`` indices, active food slots,
    all simulator timers/counters/flags, ``_prepared`` state, and lossless
    ``CTOpponentRuntime`` carry + counter-RNG state.
    """

    world_id: int
    life_id: int
    tick: int
    prepared: bool
    num_snakes: int
    body_cap: int
    food_cap: int
    body_offsets: np.ndarray  # (S + 1,) int32
    body_segments_xy: np.ndarray  # (total_segs, 2) int16 ordered k=0..seg_count-1 from head_ptr
    food_slot_indices: np.ndarray  # (num_food,) int16
    food_xy: np.ndarray  # (num_food, 2) int16
    food_corpse: np.ndarray  # (num_food,) uint8
    small_tensors: dict[str, torch.Tensor]
    opponent_state: dict[str, Any]

    @property
    def nbytes(self) -> int:
        total = int(
            self.body_offsets.nbytes
            + self.body_segments_xy.nbytes
            + self.food_slot_indices.nbytes
            + self.food_xy.nbytes
            + self.food_corpse.nbytes
        )
        for tensor in self.small_tensors.values():
            total += int(tensor.numel() * tensor.element_size())
        for tensor in self.opponent_state.get("carry", {}).values():
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel() * tensor.element_size())
        return total + 256

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "world_id": int(self.world_id),
            "life_id": int(self.life_id),
            "tick": int(self.tick),
            "prepared": bool(self.prepared),
            "num_snakes": int(self.num_snakes),
            "body_cap": int(self.body_cap),
            "food_cap": int(self.food_cap),
            "body_offsets": torch.from_numpy(self.body_offsets.copy()),
            "body_segments_xy": torch.from_numpy(self.body_segments_xy.copy()),
            "food_slot_indices": torch.from_numpy(self.food_slot_indices.copy()),
            "food_xy": torch.from_numpy(self.food_xy.copy()),
            "food_corpse": torch.from_numpy(self.food_corpse.copy()),
            "small_tensors": {k: v.clone() for k, v in self.small_tensors.items()},
            "opponent_state": {
                "roster_identity": self.opponent_state["roster_identity"],
                "vector_prepared": bool(self.opponent_state["vector_prepared"]),
                "pending_prepare": bool(self.opponent_state["pending_prepare"]),
                "carry": {k: v.clone() for k, v in self.opponent_state["carry"].items()},
                "rng_key": int(self.opponent_state["rng_key"]),
                "rng_step": int(self.opponent_state["rng_step"]),
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CompactWorldSnapshot":
        return cls(
            world_id=int(state["world_id"]),
            life_id=int(state["life_id"]),
            tick=int(state["tick"]),
            prepared=bool(state["prepared"]),
            num_snakes=int(state["num_snakes"]),
            body_cap=int(state["body_cap"]),
            food_cap=int(state["food_cap"]),
            body_offsets=state["body_offsets"].detach().cpu().numpy().astype(np.int32),
            body_segments_xy=state["body_segments_xy"].detach().cpu().numpy().astype(np.int16),
            food_slot_indices=state["food_slot_indices"].detach().cpu().numpy().astype(np.int16),
            food_xy=state["food_xy"].detach().cpu().numpy().astype(np.int16),
            food_corpse=state["food_corpse"].detach().cpu().numpy().astype(np.uint8),
            small_tensors={k: v.detach().cpu().clone() for k, v in state["small_tensors"].items()},
            opponent_state={
                "roster_identity": state["opponent_state"]["roster_identity"],
                "vector_prepared": bool(state["opponent_state"]["vector_prepared"]),
                "pending_prepare": bool(state["opponent_state"]["pending_prepare"]),
                "carry": {
                    k: v.detach().cpu().clone() for k, v in state["opponent_state"]["carry"].items()
                },
                "rng_key": int(state["opponent_state"]["rng_key"]),
                "rng_step": int(state["opponent_state"]["rng_step"]),
            },
        )


def capture_world_snapshot(
    sim: CTDeviceSim,
    opponents: CTOpponentRuntime,
    row: int,
    *,
    world_id: int,
    life_id: int,
    tick: int,
    max_body_capacity: int = MAX_LEGAL_BODY_CAPACITY,
) -> CompactWorldSnapshot:
    """Capture one lossless CPU snapshot of ``row``, refusing any body overflow."""
    r = int(row)
    if bool(sim.capacity_overflow[r].any().item()):
        raise SnapshotOverflowError(
            f"simulator capacity_overflow set on world={world_id} row={r} tick={tick}"
        )
    S = int(sim.S)
    cap = int(sim.cap)
    if cap > max_body_capacity:
        raise SnapshotOverflowError(f"sim.cap={cap} exceeds max_body_capacity={max_body_capacity}")

    seg_counts = sim.seg_count[r].detach().cpu().numpy().astype(np.int64)
    lengths = sim.length[r].detach().cpu().numpy().astype(np.int64)
    head_ptrs = sim.head_ptr[r].detach().cpu().numpy().astype(np.int64)
    bodies_row = sim.bodies[r].detach().cpu().numpy()

    for s_idx in range(S):
        sc = int(seg_counts[s_idx])
        ln = int(lengths[s_idx])
        if sc < 0 or sc > max_body_capacity or sc > cap or ln > max_body_capacity:
            raise SnapshotOverflowError(
                f"body span overflow on world={world_id} slot={s_idx} "
                f"seg_count={sc} length={ln} max={max_body_capacity}"
            )

    offsets = np.zeros(S + 1, dtype=np.int32)
    total_segs = int(seg_counts.sum())
    seg_xy = np.empty((total_segs, 2), dtype=np.int16)
    cursor = 0
    for s_idx in range(S):
        sc = int(seg_counts[s_idx])
        hp = int(head_ptrs[s_idx])
        if sc > 0:
            ring_indices = (hp - np.arange(sc, dtype=np.int64)) % cap
            seg_xy[cursor : cursor + sc] = bodies_row[s_idx, ring_indices].astype(np.int16)
            cursor += sc
        offsets[s_idx + 1] = cursor

    present_row = sim.food_present[r].detach().cpu().numpy()
    active_food = np.flatnonzero(present_row).astype(np.int16)
    if active_food.size > int(sim.food_capacity):
        raise SnapshotOverflowError(
            f"food overflow on world={world_id} active={active_food.size} cap={sim.food_capacity}"
        )
    if active_food.size > 0:
        f_xy = sim.food_cells[r].detach().cpu().numpy()[active_food].astype(np.int16)
        f_corpse = sim.corpse_food[r].detach().cpu().numpy()[active_food].astype(np.uint8)
    else:
        f_xy = np.empty((0, 2), dtype=np.int16)
        f_corpse = np.empty((0,), dtype=np.uint8)

    small: dict[str, torch.Tensor] = {}
    for field in _SMALL_ENV_TENSOR_FIELDS:
        small[field] = getattr(sim, field)[r].detach().to("cpu").clone()

    return CompactWorldSnapshot(
        world_id=int(world_id),
        life_id=int(life_id),
        tick=int(tick),
        prepared=bool(sim._prepared),
        num_snakes=S,
        body_cap=cap,
        food_cap=int(sim.food_capacity),
        body_offsets=offsets,
        body_segments_xy=seg_xy,
        food_slot_indices=active_food,
        food_xy=f_xy,
        food_corpse=f_corpse,
        small_tensors=small,
        opponent_state=opponents.snapshot_row(r),
    )


@torch.inference_mode()
def restore_world_snapshots_into(
    snapshots: Sequence[CompactWorldSnapshot],
    sim: CTDeviceSim,
    opponents: CTOpponentRuntime,
) -> None:
    """Losslessly restore ``len(snapshots) == sim.E`` prepared world snapshots."""
    B = len(snapshots)
    if B != int(sim.E) or B != int(opponents.E):
        raise ValueError(
            f"snapshot count {B} must match sim.E={sim.E} and opponents.E={opponents.E}"
        )
    if B == 0:
        return
    prepared_flags = {bool(s.prepared) for s in snapshots}
    if len(prepared_flags) != 1:
        raise ValueError("all snapshots restored into one batch must share prepared state")

    S = int(sim.S)
    cap = int(sim.cap)
    fcap = int(sim.food_capacity)
    bodies_cpu = np.zeros((B, S, cap, 2), dtype=np.int64)
    food_cells_cpu = np.zeros((B, fcap, 2), dtype=np.int64)
    food_present_cpu = np.zeros((B, fcap), dtype=bool)
    corpse_food_cpu = np.zeros((B, fcap), dtype=bool)

    for b, snap in enumerate(snapshots):
        if snap.num_snakes != S or snap.body_cap > cap or snap.food_cap > fcap:
            raise SnapshotOverflowError("target simulator capacity smaller than snapshot")
        head_ptrs = snap.small_tensors["head_ptr"].numpy().astype(np.int64)
        seg_counts = snap.small_tensors["seg_count"].numpy().astype(np.int64)
        for s_idx in range(S):
            start = int(snap.body_offsets[s_idx])
            end = int(snap.body_offsets[s_idx + 1])
            sc = end - start
            if sc != int(seg_counts[s_idx]):
                raise SnapshotOverflowError("corrupt snapshot body_offsets vs seg_count")
            if sc > 0:
                hp = int(head_ptrs[s_idx])
                ring_indices = (hp - np.arange(sc, dtype=np.int64)) % cap
                bodies_cpu[b, s_idx, ring_indices] = snap.body_segments_xy[start:end].astype(
                    np.int64
                )
        if snap.food_slot_indices.size > 0:
            f_idx = snap.food_slot_indices.astype(np.int64)
            food_cells_cpu[b, f_idx] = snap.food_xy.astype(np.int64)
            food_present_cpu[b, f_idx] = True
            corpse_food_cpu[b, f_idx] = snap.food_corpse.astype(bool)

    sim.bodies.copy_(torch.from_numpy(bodies_cpu).to(device=sim.device, dtype=sim.bodies.dtype))
    sim.food_cells.copy_(
        torch.from_numpy(food_cells_cpu).to(device=sim.device, dtype=sim.food_cells.dtype)
    )
    sim.food_present.copy_(
        torch.from_numpy(food_present_cpu).to(device=sim.device, dtype=sim.food_present.dtype)
    )
    sim.corpse_food.copy_(
        torch.from_numpy(corpse_food_cpu).to(device=sim.device, dtype=sim.corpse_food.dtype)
    )

    for field in _SMALL_ENV_TENSOR_FIELDS:
        stacked = torch.stack([s.small_tensors[field] for s in snapshots], dim=0)
        target = getattr(sim, field)
        target.copy_(stacked.to(device=sim.device, dtype=target.dtype))

    sim._prepared = bool(snapshots[0].prepared)
    r2_cache.bump(sim)
    opponents.load_row_snapshots([s.opponent_state for s in snapshots])


@dataclass(frozen=True)
class RootMetadata:
    """Immutable metadata for one retained counterfactual root."""

    root_index: int
    root_uid: str
    kind: str  # "death_lookback" or "ordinary"
    partition: str  # "train" or "assessment"
    world_id: int
    mix: str
    roster_index: int
    life_id: int
    tick: int
    hero_length: int
    death_event_id: int | None
    lookback: int | None
    death_tick: int | None
    death_length: int | None
    b8_action: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_index": int(self.root_index),
            "root_uid": str(self.root_uid),
            "kind": str(self.kind),
            "partition": str(self.partition),
            "world_id": int(self.world_id),
            "mix": str(self.mix),
            "roster_index": int(self.roster_index),
            "life_id": int(self.life_id),
            "tick": int(self.tick),
            "hero_length": int(self.hero_length),
            "death_event_id": None if self.death_event_id is None else int(self.death_event_id),
            "lookback": None if self.lookback is None else int(self.lookback),
            "death_tick": None if self.death_tick is None else int(self.death_tick),
            "death_length": None if self.death_length is None else int(self.death_length),
            "b8_action": int(self.b8_action),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RootMetadata":
        return cls(
            root_index=int(row["root_index"]),
            root_uid=str(row["root_uid"]),
            kind=str(row["kind"]),
            partition=str(row["partition"]),
            world_id=int(row["world_id"]),
            mix=str(row["mix"]),
            roster_index=int(row["roster_index"]),
            life_id=int(row["life_id"]),
            tick=int(row["tick"]),
            hero_length=int(row["hero_length"]),
            death_event_id=(None if row["death_event_id"] is None else int(row["death_event_id"])),
            lookback=None if row["lookback"] is None else int(row["lookback"]),
            death_tick=None if row["death_tick"] is None else int(row["death_tick"]),
            death_length=None if row["death_length"] is None else int(row["death_length"]),
            b8_action=int(row["b8_action"]),
        )


class LosslessBoundedSnapshotStore:
    """Bounded store of lossless world snapshots + exact prepared root observations.

    Refuses overflow explicitly (`SnapshotOverflowError`) if root count, total byte
    budget, or any snake's body span exceeds its frozen limit; never truncates.
    """

    def __init__(
        self,
        *,
        grid_h: int,
        grid_w: int,
        max_roots: int = 2_560,
        max_bytes: int = 512 * 1024 * 1024,
        max_body_capacity: int = MAX_LEGAL_BODY_CAPACITY,
    ) -> None:
        if max_roots < 1 or max_bytes < 1 or max_body_capacity < 1:
            raise ValueError("positive store limits required")
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.max_roots = int(max_roots)
        self.max_bytes = int(max_bytes)
        self.max_body_capacity = int(max_body_capacity)

        self.metadata: list[RootMetadata] = []
        self._uid_set: set[str] = set()
        self.snapshots: list[CompactWorldSnapshot] = []
        self._locals: list[torch.Tensor] = []
        self._globs: list[torch.Tensor] = []
        self._scalars: list[torch.Tensor] = []
        self._legal_masks: list[torch.Tensor] = []
        self._q_b8: list[torch.Tensor] = []
        self._topo_builder = IncrementalSparseTopologyBuilder(self.grid_h, self.grid_w)
        self._finalized_topology: SparseTopologyBatch | None = None
        self._bytes_used = 0

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def bytes_used(self) -> int:
        topo_bytes = self._finalized_topology.nbytes if self._finalized_topology is not None else 0
        return int(self._bytes_used + topo_bytes)

    def add_root(
        self,
        *,
        meta: OmitIndexRootMeta | RootMetadata,
        snapshot: CompactWorldSnapshot,
        local_obs: torch.Tensor,
        glob_obs: torch.Tensor,
        scalars_obs: torch.Tensor,
        legal_mask: torch.Tensor,
        q_b8: torch.Tensor,
        topo_planes_u16: np.ndarray,
        topo_summaries_f32: np.ndarray,
    ) -> RootMetadata:
        """Append one verified prepared root or raise ``SnapshotOverflowError``."""
        if self._finalized_topology is not None:
            raise RuntimeError("cannot add roots after topology batch is finalized")
        if len(self.snapshots) >= self.max_roots:
            raise SnapshotOverflowError(
                f"snapshot store root limit reached: {len(self.snapshots)} >= {self.max_roots}"
            )
        if not snapshot.prepared:
            raise ValueError("counterfactual root snapshot must be captured in prepared state")
        for s_idx in range(snapshot.num_snakes):
            sc = int(snapshot.body_offsets[s_idx + 1] - snapshot.body_offsets[s_idx])
            ln = int(snapshot.small_tensors["length"][s_idx].item())
            if sc > self.max_body_capacity or ln > self.max_body_capacity:
                raise SnapshotOverflowError(
                    f"root body span {max(sc, ln)} exceeds "
                    f"max_body_capacity={self.max_body_capacity}"
                )
        if int(legal_mask.sum().item()) < 1:
            raise ValueError(f"root {meta.root_uid} has zero legal actions")

        loc_cpu = local_obs.detach().to("cpu", dtype=torch.uint8).clone()
        glb_cpu = glob_obs.detach().to("cpu", dtype=torch.uint8).clone()
        scl_cpu = scalars_obs.detach().to("cpu", dtype=torch.float32).clone()
        msk_cpu = legal_mask.detach().to("cpu", dtype=torch.bool).clone()
        qb8_cpu = q_b8.detach().to("cpu", dtype=torch.float32).clone()

        if topo_planes_u16.shape != (4, self.grid_h, self.grid_w):
            raise ValueError(f"expected topo_planes_u16 shape {(4, self.grid_h, self.grid_w)}")
        if topo_summaries_f32.shape != (6,):
            raise ValueError("expected topo_summaries_f32 shape (6,)")
        nnz = int(np.count_nonzero(topo_planes_u16))
        added_bytes = (
            snapshot.nbytes
            + int(loc_cpu.numel() + glb_cpu.numel() + scl_cpu.numel() * 4)
            + int(msk_cpu.numel() + qb8_cpu.numel() * 4)
            + nnz * 4
            + 32
        )
        if self._bytes_used + added_bytes > self.max_bytes:
            raise SnapshotOverflowError(
                f"snapshot store byte limit exceeded: "
                f"{self._bytes_used + added_bytes} > {self.max_bytes}"
            )
        if meta.root_uid in self._uid_set:
            raise ValueError(f"duplicate root_uid in store: {meta.root_uid}")

        idx = len(self.snapshots)
        bound_meta = RootMetadata(
            root_index=idx,
            root_uid=meta.root_uid,
            kind=meta.kind,
            partition=meta.partition,
            world_id=meta.world_id,
            mix=meta.mix,
            roster_index=meta.roster_index,
            life_id=meta.life_id,
            tick=meta.tick,
            hero_length=meta.hero_length,
            death_event_id=meta.death_event_id,
            lookback=meta.lookback,
            death_tick=meta.death_tick,
            death_length=meta.death_length,
            b8_action=meta.b8_action,
        )
        self._uid_set.add(bound_meta.root_uid)
        self.metadata.append(bound_meta)
        self.snapshots.append(snapshot)
        self._locals.append(loc_cpu)
        self._globs.append(glb_cpu)
        self._scalars.append(scl_cpu)
        self._legal_masks.append(msk_cpu)
        self._q_b8.append(qb8_cpu)
        self._topo_builder.append_batch(topo_planes_u16[None, ...], topo_summaries_f32[None, ...])
        self._bytes_used += added_bytes
        return bound_meta

    def finalize_topology(self) -> SparseTopologyBatch:
        if self._finalized_topology is None:
            self._finalized_topology = self._topo_builder.build()
        return self._finalized_topology

    def stacked_observations(self) -> dict[str, torch.Tensor]:
        """Return exact prepared root observations/masks/Q stacked along root_index."""
        if not self.snapshots:
            return {
                "local": torch.empty((0, 7, 31, 31), dtype=torch.uint8),
                "glob": torch.empty((0, 4, 37, 37), dtype=torch.uint8),
                "scalars": torch.empty((0, 60), dtype=torch.float32),
                "legal_mask": torch.empty((0, 6), dtype=torch.bool),
                "q_b8": torch.empty((0, 6), dtype=torch.float32),
            }
        return {
            "local": torch.stack(self._locals, dim=0),
            "glob": torch.stack(self._globs, dim=0),
            "scalars": torch.stack(self._scalars, dim=0),
            "legal_mask": torch.stack(self._legal_masks, dim=0),
            "q_b8": torch.stack(self._q_b8, dim=0),
        }

    def restore_batch_into(
        self,
        root_indices: Sequence[int],
        sim: CTDeviceSim,
        opponents: CTOpponentRuntime,
    ) -> None:
        """Restore an arbitrary batch of root indices (allowing repeats across actions)."""
        selected = [self.snapshots[int(i)] for i in root_indices]
        restore_world_snapshots_into(selected, sim, opponents)

    def to_state_dict(self) -> dict[str, Any]:
        topo = self.finalize_topology()
        obs = self.stacked_observations()
        return {
            "schema": "ct-lossless-snapshot-store/v1",
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "max_roots": self.max_roots,
            "max_bytes": self.max_bytes,
            "max_body_capacity": self.max_body_capacity,
            "bytes_used": self._bytes_used,
            "metadata": [m.to_dict() for m in self.metadata],
            "snapshots": [s.to_state_dict() for s in self.snapshots],
            "observations": obs,
            "topology": topo.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LosslessBoundedSnapshotStore":
        if state.get("schema") != "ct-lossless-snapshot-store/v1":
            raise ValueError("foreign LosslessBoundedSnapshotStore schema")
        store = cls(
            grid_h=int(state["grid_h"]),
            grid_w=int(state["grid_w"]),
            max_roots=int(state["max_roots"]),
            max_bytes=int(state["max_bytes"]),
            max_body_capacity=int(state["max_body_capacity"]),
        )
        store.metadata = [RootMetadata.from_dict(row) for row in state["metadata"]]
        store._uid_set = {m.root_uid for m in store.metadata}
        store.snapshots = [CompactWorldSnapshot.from_state_dict(s) for s in state["snapshots"]]
        obs = state["observations"]
        n = len(store.snapshots)
        store._locals = [obs["local"][i].detach().cpu().clone() for i in range(n)]
        store._globs = [obs["glob"][i].detach().cpu().clone() for i in range(n)]
        store._scalars = [obs["scalars"][i].detach().cpu().clone() for i in range(n)]
        store._legal_masks = [obs["legal_mask"][i].detach().cpu().clone() for i in range(n)]
        store._q_b8 = [obs["q_b8"][i].detach().cpu().clone() for i in range(n)]
        store._finalized_topology = SparseTopologyBatch.from_state_dict(state["topology"])
        store._bytes_used = int(state["bytes_used"])
        return store


@dataclass(frozen=True)
class OmitIndexRootMeta:
    """Unindexed root metadata passed to ``LosslessBoundedSnapshotStore.add_root``."""

    root_uid: str
    kind: str
    partition: str
    world_id: int
    mix: str
    roster_index: int
    life_id: int
    tick: int
    hero_length: int
    death_event_id: int | None
    lookback: int | None
    death_tick: int | None
    death_length: int | None
    b8_action: int


class LookbackRing:
    """Rolling 5-tick prepared checkpoint ring + action tape for exact lookback recovery.

    Stores prepared world snapshots at ticks ``t % stride == 0`` across a rolling
    window covering ``max_lookback + stride`` ticks (default 59 slots * 5 = 295 ticks
    >= 285 + 5), plus an exact per-tick hero action and life-ID tape so any
    non-multiple-of-5 lookback ``t_death - L`` is reconstructed losslessly in
    ``0..4`` single-world replay steps.
    """

    def __init__(
        self,
        *,
        num_worlds: int,
        stride: int = 5,
        max_lookback: int = 285,
        max_body_capacity: int = MAX_LEGAL_BODY_CAPACITY,
    ) -> None:
        if num_worlds < 1 or stride < 1 or max_lookback < 1:
            raise ValueError("positive ring parameters required")
        self.num_worlds = int(num_worlds)
        self.stride = int(stride)
        self.max_lookback = int(max_lookback)
        self.max_body_capacity = int(max_body_capacity)
        self.num_slots = (self.max_lookback // self.stride) + 3
        self.tape_len = self.num_slots * self.stride

        self._slots: list[list[CompactWorldSnapshot | None]] = [
            [None] * self.num_slots for _ in range(self.num_worlds)
        ]
        self._action_tape = np.full((self.num_worlds, self.tape_len), -1, dtype=np.int16)
        self._tape_tick = np.full((self.num_worlds, self.tape_len), -1, dtype=np.int64)
        self._tape_life = np.full((self.num_worlds, self.tape_len), -1, dtype=np.int64)

    def maybe_capture_checkpoint(
        self,
        sim: CTDeviceSim,
        opponents: CTOpponentRuntime,
        *,
        tick: int,
        life_ids: Sequence[int],
    ) -> None:
        """Capture prepared world snapshots when ``tick % stride == 0``."""
        if int(tick) % self.stride != 0:
            return
        if not sim._prepared:
            raise RuntimeError("LookbackRing checkpoints must be captured after sim.prepare()")
        slot = (int(tick) // self.stride) % self.num_slots
        for w in range(self.num_worlds):
            self._slots[w][slot] = capture_world_snapshot(
                sim,
                opponents,
                w,
                world_id=w,
                life_id=int(life_ids[w]),
                tick=int(tick),
                max_body_capacity=self.max_body_capacity,
            )

    def record_tick_actions(
        self,
        *,
        tick: int,
        hero_actions: np.ndarray | Sequence[int],
        life_ids: Sequence[int],
    ) -> None:
        """Record exact hero greedy actions and life IDs for prepared tick ``tick``."""
        pos = int(tick) % self.tape_len
        acts = np.asarray(hero_actions, dtype=np.int16)
        if acts.shape != (self.num_worlds,):
            raise ValueError("hero_actions shape mismatch")
        self._action_tape[:, pos] = acts
        self._tape_tick[:, pos] = int(tick)
        self._tape_life[:, pos] = np.asarray(life_ids, dtype=np.int64)

    def has_same_life_lookbacks(
        self, world_id: int, *, death_tick: int, life_id: int, lookbacks: Sequence[int]
    ) -> bool:
        """Verify all requested lookbacks exist in-ring within the same hero life."""
        w = int(world_id)
        for lb in lookbacks:
            target_tick = int(death_tick) - int(lb)
            if target_tick < 0:
                return False
            base_tick = (target_tick // self.stride) * self.stride
            slot = (base_tick // self.stride) % self.num_slots
            snap = self._slots[w][slot]
            if snap is None or snap.tick != base_tick or snap.life_id != int(life_id):
                return False
            for tau in range(base_tick, target_tick + 1):
                pos = tau % self.tape_len
                if int(self._tape_tick[w, pos]) != tau or int(self._tape_life[w, pos]) != int(
                    life_id
                ):
                    return False
        return True

    @torch.inference_mode()
    def reconstruct_lookback_root(
        self,
        *,
        world_id: int,
        target_tick: int,
        expected_life_id: int,
        roster: TrainingRoster,
        cfg: Any,
        b8_network: nn.Module,
        checkpoint_networks: Mapping[str, nn.Module],
        device: torch.device | str,
        replay_cache: dict[tuple[Any, ...], tuple[CTDeviceSim, CTOpponentRuntime]] | None = None,
    ) -> tuple[
        CompactWorldSnapshot,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        np.ndarray,
        np.ndarray,
    ]:
        """Deterministically replay ``0..stride-1`` ticks to recover prepared ``target_tick``."""
        w = int(world_id)
        t_target = int(target_tick)
        t_base = (t_target // self.stride) * self.stride
        delta = t_target - t_base
        if not 0 <= delta < self.stride:
            raise RuntimeError(f"invalid replay delta={delta}")
        slot = (t_base // self.stride) % self.num_slots
        base_snap = self._slots[w][slot]
        if base_snap is None or base_snap.tick != t_base:
            raise RuntimeError(
                f"missing base checkpoint for world={w} target={t_target} base={t_base}"
            )
        if base_snap.life_id != int(expected_life_id):
            raise RuntimeError(
                f"life boundary crossed in lookback replay world={w} "
                f"base_life={base_snap.life_id} expected={expected_life_id}"
            )

        dev = torch.device(device)
        cache_key = (roster.identity, str(dev))
        if replay_cache is not None and cache_key in replay_cache:
            r_sim, r_opp = replay_cache[cache_key]
        else:
            from dataclasses import replace as dc_replace

            cfg1 = dc_replace(cfg, num_envs=1)
            r_sim = CTDeviceSim(cfg1, device=dev, train_mode=False)
            r_sim.terminal_slots[:, 0] = True
            r_opp = CTOpponentRuntime(
                r_sim,
                [roster],
                b8_network=b8_network,
                checkpoint_networks=checkpoint_networks,
            )
            if replay_cache is not None:
                replay_cache[cache_key] = (r_sim, r_opp)

        restore_world_snapshots_into([base_snap], r_sim, r_opp)

        for step_offset in range(delta):
            tau = t_base + step_offset
            pos = tau % self.tape_len
            if int(self._tape_tick[w, pos]) != tau or int(self._tape_life[w, pos]) != int(
                expected_life_id
            ):
                raise RuntimeError(f"missing action tape for world={w} tick={tau}")
            r_sim.prepare()
            recorded_act = int(self._action_tape[w, pos])
            hero_act_t = torch.tensor([recorded_act], dtype=torch.long, device=dev)
            full_actions = r_opp.actions(r_sim, hero_act_t)
            r_sim.step(full_actions)
            r_opp.after_step(r_sim)
            if bool(r_sim._last_done[0, 0].item()):
                raise RuntimeError(
                    f"unexpected hero death during lookback replay at tau={tau} < {t_target}"
                )

        r_sim.prepare()
        exact_snap = capture_world_snapshot(
            r_sim,
            r_opp,
            0,
            world_id=w,
            life_id=int(expected_life_id),
            tick=t_target,
            max_body_capacity=self.max_body_capacity,
        )
        local_obs, glob_obs, scalars_obs, legal_mask, q_b8, a_b8_t = observe_hero_bv8(
            r_sim, b8_network
        )
        b8_act = int(a_b8_t[0].item())

        pos_target = t_target % self.tape_len
        if int(self._tape_tick[w, pos_target]) == t_target:
            recorded_target_act = int(self._action_tape[w, pos_target])
            if b8_act != recorded_target_act:
                raise RuntimeError(
                    f"replayed B8 action {b8_act} != recorded {recorded_target_act} "
                    f"at world={w} tick={t_target}"
                )

        planes_u16, sums_f32 = compute_topology_features(r_sim, [0])
        return (
            exact_snap,
            local_obs[0],
            glob_obs[0],
            scalars_obs[0],
            legal_mask[0],
            q_b8[0],
            b8_act,
            planes_u16[0],
            sums_f32[0],
        )
