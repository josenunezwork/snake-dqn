"""C/T frozen control, whole-world 80/20 split, seed domains, and E1024x5000 successor collector.

Implements Ruling §4:
- New ``a100-dev/t34-ct-h5000-1029/`` namespace with distinct ``training``, ``validation``,
  and ``assessment`` seed domains excluding ``TRAIN-01`` ``monitor``/``nomination``
  and every protected bank.
- Whole-world 80/20 stratified split frozen *before* collection and fitting so all
  lives, lookbacks, actions, suffixes, ordinary roots, and anchor rows of a world
  belong to exactly one partition.
- Bare B8 s3 collection for ``E1024 x 5,000`` ticks (5,120,000 task rows), extracting:
  * up to 160 eligible ``L >= 600`` death events with 8 same-life lookbacks
    ``{15, 35, 65, 100, 140, 185, 235, 285}`` (``<= 1,280`` death-lookback roots),
  * up to 1,280 ``(mix, length_bin)``-matched ``L >= 600`` ordinary roots without
    conditioning on future survival,
  * a fixed 50,000-row unsearched ordinary anchor/flip archive.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_h5000.replay import LookbackRing
from research.sandbox_a_resident.ct_h5000.sampling import quota_sample, reservoir_admissions
from research.sandbox_a_resident.ct_h5000.units import StageExportOverflowError
from research.sandbox_a_resident.ct_mechanics import (
    CTDeviceSim,
    CTOpponentRuntime,
    derive_63bit_key,
    task_separation_report,
)
from research.sandbox_a_resident.ct_snapshots import (
    MAX_LEGAL_BODY_CAPACITY,
    CompactWorldSnapshot,
    LosslessBoundedSnapshotStore,
    OmitIndexRootMeta,
    capture_world_snapshot,
    observe_hero_bv8,
)
from research.sandbox_a_resident.ct_topology import (
    TOPOLOGY_SPEC,
    IncrementalSparseTopologyBuilder,
    SparseTopologyBatch,
    compress_array_to_u8_tensor,
    compute_topology_features,
    decompress_u8_tensor_to_array,
)
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.sandbox_a_resident.training_rosters import MIXES, TrainingRoster, training_rosters
from src.simd_env.batch_sim import BatchSimConfig

CT_STUDY_SLUG = "t34-ct-h5000-1029"
CT_DOMAIN_PREFIX = f"a100-dev/{CT_STUDY_SLUG}"

TRAIN01_NAMESPACE = "a100-dev/sandbox-a-train01"
TRAIN01_MONITOR_OFFSET = 400_000_000
TRAIN01_MONITOR_COUNT = 3 * 96
TRAIN01_NOMINATION_OFFSET = 500_000_000
TRAIN01_NOMINATION_COUNT = 3 * 256

ORDINARY_LENGTH_BIN_EDGES: tuple[int, ...] = (1, 600, 800, 1000, 1300, 1700)
ORDINARY_LENGTH_BIN_LABELS: tuple[str, ...] = (
    "[1,600)",
    "[600,800)",
    "[800,1000)",
    "[1000,1300)",
    "[1300,1700)",
    "[1700,inf)",
)


class CollectionCapReached(RuntimeError):
    """Collection exceeded its frozen wall cap; no partial bank is accepted."""

    def __init__(self, completed_ticks: int, worlds: int, elapsed_sec: float) -> None:
        super().__init__("natural collection wall cap reached")
        self.telemetry = {
            "completed_ticks": completed_ticks,
            "charged_collection_rows": completed_ticks * worlds,
            "collection_wall_sec": elapsed_sec,
            "collection_complete": False,
        }


@dataclass(frozen=True)
class CTControl:
    """Frozen scientific and operational parameters for C/T (t34-cf-1028)."""

    study: str = CT_STUDY_SLUG
    source_commit: str = ""
    allocation_index: int = 0
    seed_offset: int = 0
    worlds: int = 1024
    collection_ticks: int = 5000
    collection_base_seconds: float = 6600.0
    collection_cap_seconds: float = 13200.0
    ordinary_overlap_allowance: int = 64
    max_body_capacity: int = MAX_LEGAL_BODY_CAPACITY
    max_deaths: int = 160
    lookbacks: tuple[int, ...] = (15, 35, 65, 100, 140, 185, 235, 285)
    max_ordinary_roots: int = 1280
    anchor_rows: int = 50_000
    min_death_length: int = 600
    min_held_out_deaths: int = 32
    min_held_out_death_worlds: int = 32
    min_held_out_ordinary_roots: int = 256
    min_held_out_anchor_rows: int = 10_000
    branch_horizon: int = 300
    selection_suffixes: int = 4
    scoring_suffixes: int = 4
    alpha: float = 0.15
    forbidden_gap: float = 0.50
    logit_margin: float = 0.25
    reference_kl_beta: float = 0.05
    minibatches: int = 2000
    searched_batch_size: int = 256
    anchor_batch_size: int = 768
    trunk_lr: float = 1e-5
    head_lr: float = 1e-4
    fit_seed: int = 20261028
    bootstrap_seed: int = 3432
    bootstrap_resamples: int = 10_000
    scoring_chunks: int = 8
    branch_batch_size: int = 512
    snapshot_store_max_bytes: int = 512 * 1024 * 1024
    anchor_archive_max_bytes: int = 1536 * 1024 * 1024
    max_stage_export_bytes: int = 896 * 1024 * 1024
    max_total_export_bytes: int = 1024 * 1024 * 1024
    tiny: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.collection_cap_seconds)
            or self.collection_cap_seconds <= 0
            or self.collection_cap_seconds != 2 * self.collection_base_seconds
            or self.ordinary_overlap_allowance < 0
        ):
            raise ValueError("finite positive 2x collection cap and overlap allowance required")
        if self.study != CT_STUDY_SLUG:
            raise ValueError(f"CTControl study must equal {CT_STUDY_SLUG!r}")
        if self.worlds < 6 or self.collection_ticks < 1:
            raise ValueError("CTControl requires worlds >= 6 and collection_ticks >= 1")
        if self.max_body_capacity < MAX_LEGAL_BODY_CAPACITY and not self.tiny:
            raise ValueError("production C/T requires full 10,400 legal body capacity")
        if len(self.lookbacks) < 1 or any(lb < 1 for lb in self.lookbacks):
            raise ValueError("lookbacks must be positive integers")
        if tuple(sorted(self.lookbacks)) != tuple(self.lookbacks):
            raise ValueError("lookbacks must be strictly sorted ascending")
        if self.selection_suffixes != 4 or self.scoring_suffixes != 4:
            raise ValueError("C/T freezes 4 selection + 4 independent scoring suffixes")
        if self.scoring_chunks < 1:
            raise ValueError("scoring_chunks must be >= 1")
        if (
            self.snapshot_store_max_bytes < 1
            or self.anchor_archive_max_bytes < 1
            or self.max_stage_export_bytes < 1
            or self.max_total_export_bytes < self.max_stage_export_bytes
        ):
            raise ValueError("positive bounded store/archive/export byte caps required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lookbacks"] = list(self.lookbacks)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CTControl":
        kwargs = dict(payload)
        if "lookbacks" in kwargs:
            kwargs["lookbacks"] = tuple(int(x) for x in kwargs["lookbacks"])
        return cls(**kwargs)


def world_partition_table(
    worlds: int = 1024,
) -> tuple[tuple[str, ...], dict[str, list[int]], dict[str, dict[str, int]]]:
    """Deterministic stratified whole-world 80/20 split frozen before collection.

    For ``worlds=1024`` across balanced ``(scripted=342, frozen=341, mixed=341)``:
    - ``train``: 820 worlds (274 scripted, 273 frozen, 273 mixed)
    - ``assessment``: 204 worlds (68 scripted, 68 frozen, 68 mixed)
    """
    rosters = training_rosters(worlds)
    by_mix: dict[str, list[int]] = {mix: [] for mix in MIXES}
    for w_idx, roster in enumerate(rosters):
        by_mix[roster.mix].append(w_idx)

    labels = ["train"] * worlds
    for mix in MIXES:
        w_list = by_mix[mix]
        m = len(w_list)
        n_assess = max(1, int(round(0.20 * m)))
        # Evenly spaced deterministic stratification preserving roster parity balance.
        chosen_ranks: set[int] = set()
        for k in range(n_assess):
            rank = int(math.floor(((k + 0.5) * m) / n_assess))
            rank = min(m - 1, max(0, rank))
            while rank in chosen_ranks and rank + 1 < m:
                rank += 1
            while rank in chosen_ranks and rank - 1 >= 0:
                rank -= 1
            chosen_ranks.add(rank)
        for rank in chosen_ranks:
            labels[w_list[rank]] = "assessment"

    partitions: dict[str, list[int]] = {"train": [], "assessment": []}
    per_mix_counts: dict[str, dict[str, int]] = {
        "train": {mix: 0 for mix in MIXES},
        "assessment": {mix: 0 for mix in MIXES},
    }
    for w_idx, part in enumerate(labels):
        partitions[part].append(w_idx)
        per_mix_counts[part][rosters[w_idx].mix] += 1
    return tuple(labels), partitions, per_mix_counts


def ct_seed_allocations(control: Mapping[str, Any] | CTControl) -> list[dict[str, Any]]:
    """Return disjoint canonical seed allocations for training, validation, assessment."""
    ctrl = control if isinstance(control, CTControl) else CTControl.from_dict(control)
    raw_offset = int(ctrl.seed_offset)
    if raw_offset < 0:
        raise ValueError("seed_offset must be non-negative")
    _, partitions, _ = world_partition_table(ctrl.worlds)
    n_train = len(partitions["train"])
    n_assess = len(partitions["assessment"])
    max_ep = int(ctrl.collection_ticks) + 1
    rows = [
        {
            "phase": "training",
            "offset": raw_offset,
            "count": max_ep * n_train,
        },
        {
            "phase": "validation",
            "offset": 10_000_000 + raw_offset,
            "count": max(1024, int(ctrl.worlds)),
        },
        {
            "phase": "assessment",
            "offset": 20_000_000 + raw_offset,
            "count": max_ep * n_assess,
        },
    ]
    ordered = sorted(rows, key=lambda r: r["offset"])
    if any(a["offset"] + a["count"] > b["offset"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("overlapping C/T seed allocations")
    return rows


def canonical_low63_seed(study: str, index: int) -> int:
    """Compute canonical 63-bit effective seed under ``a100-dev/{study}/{index}``."""
    raw = hashlib.sha256(f"a100-dev/{study}/{index}".encode("utf-8")).digest()[:8]
    val = int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF
    if val == 0:
        raise ValueError(f"forbidden zero 63-bit seed for study={study} index={index}")
    return val


def train01_protected_eval_seeds() -> np.ndarray:
    """Enumerate all TRAIN-01 monitor and nomination 63-bit seeds to exclude."""
    indices = list(
        range(TRAIN01_MONITOR_OFFSET, TRAIN01_MONITOR_OFFSET + TRAIN01_MONITOR_COUNT)
    ) + list(range(TRAIN01_NOMINATION_OFFSET, TRAIN01_NOMINATION_OFFSET + TRAIN01_NOMINATION_COUNT))
    out = np.empty(len(indices), dtype=np.uint64)
    for pos, idx in enumerate(indices):
        raw = hashlib.sha256(f"{TRAIN01_NAMESPACE}/{idx}".encode("utf-8")).digest()[:8]
        out[pos] = int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF
    return out


def _length_bin_index(length: int) -> int:
    idx = 0
    for edge in ORDINARY_LENGTH_BIN_EDGES[1:]:
        if length >= edge:
            idx += 1
        else:
            break
    return idx


def _row_view(sim: CTDeviceSim, rows: torch.Tensor) -> CTDeviceSim:
    out = object.__new__(type(sim))
    out.__dict__ = sim.__dict__.copy()
    out.E = int(rows.numel())
    out.cfg = replace(sim.cfg, num_envs=out.E)
    for name in _ENV_TENSOR_FIELDS:
        setattr(out, name, getattr(sim, name).index_select(0, rows))
    r2_cache.inherit(out, sim)
    return out


def _copy_rows(target: CTDeviceSim, source: CTDeviceSim, rows: torch.Tensor) -> None:
    for name in _ENV_TENSOR_FIELDS:
        getattr(target, name).index_copy_(0, rows, getattr(source, name))


@dataclass
class AnchorArchive:
    """Unsearched ordinary anchor/flip archive partitioned strictly by whole world."""

    num_rows: int
    world_id: np.ndarray  # (N,) int32
    mix_id: np.ndarray  # (N,) int8 (0=scripted, 1=frozen, 2=mixed)
    is_assessment: np.ndarray  # (N,) bool
    life_id: np.ndarray  # (N,) int32
    tick: np.ndarray  # (N,) int32
    hero_length: np.ndarray  # (N,) int32
    local: torch.Tensor  # (N, 7, 31, 31) uint8
    glob: torch.Tensor  # (N, 4, 37, 37) uint8
    scalars: torch.Tensor  # (N, 60) float32
    legal_mask: torch.Tensor  # (N, 6) bool
    q_b8: torch.Tensor  # (N, 6) float32
    b8_action: np.ndarray  # (N,) int8
    topology: SparseTopologyBatch

    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.is_assessment)

    def assessment_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_assessment)

    @property
    def nbytes(self) -> int:
        return int(
            self.world_id.nbytes
            + self.mix_id.nbytes
            + self.is_assessment.nbytes
            + self.life_id.nbytes
            + self.tick.nbytes
            + self.hero_length.nbytes
            + int(self.local.numel() * self.local.element_size())
            + int(self.glob.numel() * self.glob.element_size())
            + int(self.scalars.numel() * self.scalars.element_size())
            + int(self.legal_mask.numel() * self.legal_mask.element_size())
            + int(self.q_b8.numel() * self.q_b8.element_size())
            + self.b8_action.nbytes
            + self.topology.nbytes
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ct-anchor-archive/v2",
            "num_rows": int(self.num_rows),
            "world_id": compress_array_to_u8_tensor(self.world_id, byte_shuffle=True),
            "mix_id": compress_array_to_u8_tensor(self.mix_id),
            "is_assessment": compress_array_to_u8_tensor(self.is_assessment),
            "life_id": compress_array_to_u8_tensor(self.life_id, byte_shuffle=True),
            "tick": compress_array_to_u8_tensor(self.tick, byte_shuffle=True),
            "hero_length": compress_array_to_u8_tensor(self.hero_length, byte_shuffle=True),
            "local": compress_array_to_u8_tensor(self.local.detach().cpu().numpy()),
            "glob": compress_array_to_u8_tensor(self.glob.detach().cpu().numpy()),
            "scalars": compress_array_to_u8_tensor(
                self.scalars.detach().cpu().numpy(), byte_shuffle=True
            ),
            "legal_mask": compress_array_to_u8_tensor(self.legal_mask.detach().cpu().numpy()),
            "q_b8": compress_array_to_u8_tensor(
                self.q_b8.detach().cpu().numpy(), byte_shuffle=True
            ),
            "b8_action": compress_array_to_u8_tensor(self.b8_action),
            "topology": self.topology.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AnchorArchive":
        schema = state.get("schema")
        if schema == "ct-anchor-archive/v1":
            return cls(
                num_rows=int(state["num_rows"]),
                world_id=state["world_id"].detach().cpu().numpy().astype(np.int32),
                mix_id=state["mix_id"].detach().cpu().numpy().astype(np.int8),
                is_assessment=state["is_assessment"].detach().cpu().numpy().astype(bool),
                life_id=state["life_id"].detach().cpu().numpy().astype(np.int32),
                tick=state["tick"].detach().cpu().numpy().astype(np.int32),
                hero_length=state["hero_length"].detach().cpu().numpy().astype(np.int32),
                local=state["local"].detach().cpu().to(dtype=torch.uint8),
                glob=state["glob"].detach().cpu().to(dtype=torch.uint8),
                scalars=state["scalars"].detach().cpu().to(dtype=torch.float32),
                legal_mask=state["legal_mask"].detach().cpu().to(dtype=torch.bool),
                q_b8=state["q_b8"].detach().cpu().to(dtype=torch.float32),
                b8_action=state["b8_action"].detach().cpu().numpy().astype(np.int8),
                topology=SparseTopologyBatch.from_state_dict(state["topology"]),
            )
        if schema != "ct-anchor-archive/v2":
            raise ValueError("foreign AnchorArchive schema")
        return cls(
            num_rows=int(state["num_rows"]),
            world_id=decompress_u8_tensor_to_array(state["world_id"]).astype(np.int32),
            mix_id=decompress_u8_tensor_to_array(state["mix_id"]).astype(np.int8),
            is_assessment=decompress_u8_tensor_to_array(state["is_assessment"]).astype(bool),
            life_id=decompress_u8_tensor_to_array(state["life_id"]).astype(np.int32),
            tick=decompress_u8_tensor_to_array(state["tick"]).astype(np.int32),
            hero_length=decompress_u8_tensor_to_array(state["hero_length"]).astype(np.int32),
            local=torch.from_numpy(decompress_u8_tensor_to_array(state["local"]).copy()).to(
                dtype=torch.uint8
            ),
            glob=torch.from_numpy(decompress_u8_tensor_to_array(state["glob"]).copy()).to(
                dtype=torch.uint8
            ),
            scalars=torch.from_numpy(decompress_u8_tensor_to_array(state["scalars"]).copy()).to(
                dtype=torch.float32
            ),
            legal_mask=torch.from_numpy(
                decompress_u8_tensor_to_array(state["legal_mask"]).copy()
            ).to(dtype=torch.bool),
            q_b8=torch.from_numpy(decompress_u8_tensor_to_array(state["q_b8"]).copy()).to(
                dtype=torch.float32
            ),
            b8_action=decompress_u8_tensor_to_array(state["b8_action"]).astype(np.int8),
            topology=SparseTopologyBatch.from_state_dict(state["topology"]),
        )


@dataclass
class CollectedCTDataset:
    """Complete Stage-1 output: frozen world split, root snapshot store, anchor archive."""

    control: CTControl
    world_partitions: tuple[str, ...]
    partition_mix_worlds: dict[str, dict[str, int]]
    store: LosslessBoundedSnapshotStore
    anchors: AnchorArchive
    collection_telemetry: dict[str, Any]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ct-collected-dataset/v1",
            "control": self.control.to_dict(),
            "world_partitions": list(self.world_partitions),
            "partition_mix_worlds": self.partition_mix_worlds,
            "store": self.store.to_state_dict(),
            "anchors": self.anchors.to_state_dict(),
            "collection_telemetry": self.collection_telemetry,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CollectedCTDataset":
        if state.get("schema") != "ct-collected-dataset/v1":
            raise ValueError("foreign CollectedCTDataset schema")
        return cls(
            control=CTControl.from_dict(state["control"]),
            world_partitions=tuple(str(x) for x in state["world_partitions"]),
            partition_mix_worlds={
                str(part): {str(m): int(c) for m, c in mix_map.items()}
                for part, mix_map in state["partition_mix_worlds"].items()
            },
            store=LosslessBoundedSnapshotStore.from_state_dict(state["store"]),
            anchors=AnchorArchive.from_state_dict(state["anchors"]),
            collection_telemetry=dict(state["collection_telemetry"]),
        )


def _preschedule_anchor_slots(
    *,
    partitions: Mapping[str, Sequence[int]],
    collection_ticks: int,
    target_anchor_rows: int,
    oversample_extra: int,
) -> dict[int, list[int]]:
    """Build deterministic stratified (tick -> [world_ids]) anchor capture schedule.

    Draws 80% of target rows from ``train`` worlds and 20% from ``assessment``
    worlds across all collection ticks so anchor states span the natural B8
    occupancy distribution without post-hoc outcome filtering.
    """
    n_assess_target = int(round(0.20 * target_anchor_rows))
    n_train_target = int(target_anchor_rows) - n_assess_target
    n_assess_sched = n_assess_target + max(64, oversample_extra // 4)
    n_train_sched = n_train_target + max(256, oversample_extra)

    rng = np.random.default_rng(
        derive_63bit_key(
            f"{CT_DOMAIN_PREFIX}/anchor-schedule/v1", collection_ticks, target_anchor_rows
        )
    )
    by_tick: dict[int, list[int]] = {t: [] for t in range(collection_ticks)}

    for part_name, count_sched in (("train", n_train_sched), ("assessment", n_assess_sched)):
        worlds_arr = np.asarray(partitions[part_name], dtype=np.int32)
        if worlds_arr.size == 0 or count_sched <= 0:
            continue
        total_pairs = collection_ticks * int(worlds_arr.size)
        actual = min(count_sched, total_pairs)
        flat_choices = rng.choice(total_pairs, size=actual, replace=False)
        flat_choices.sort()
        for flat_idx in flat_choices.tolist():
            t = int(flat_idx // int(worlds_arr.size))
            w = int(worlds_arr[flat_idx % int(worlds_arr.size)])
            by_tick[t].append(w)
    return by_tick


@dataclass
class _CandidateDeathEvent:
    world_id: int
    partition: str
    mix: str
    roster_index: int
    life_id: int
    death_tick: int
    death_length: int
    lookback_roots: list[
        tuple[
            int,  # lookback
            int,  # target_tick
            CompactWorldSnapshot,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            np.ndarray,
            np.ndarray,
        ]
    ]


@dataclass
class _CandidateOrdinaryRoot:
    world_id: int
    partition: str
    mix: str
    roster_index: int
    life_id: int
    tick: int
    hero_length: int
    length_bin: int
    snapshot: CompactWorldSnapshot
    local_obs: torch.Tensor
    glob_obs: torch.Tensor
    scalars_obs: torch.Tensor
    legal_mask: torch.Tensor
    q_b8: torch.Tensor
    b8_action: int
    topo_planes_u16: np.ndarray
    topo_summaries_f32: np.ndarray


def _select_deaths_deterministic(
    candidates: Sequence[_CandidateDeathEvent],
    *,
    max_deaths: int,
) -> list[_CandidateDeathEvent]:
    """Select up to ``max_deaths`` eligible deaths prioritizing distinct worlds & mixes.

    Respects the frozen 80/20 partition quota (128 train / 32 assessment for 160)
    when both partitions exceed quota, while allowing unused capacity to be filled
    if one partition has fewer deaths (never crossing world partitions).
    """
    target_assess = int(round(0.20 * max_deaths))
    target_train = int(max_deaths) - target_assess

    def rank_within_partition(pool: Sequence[_CandidateDeathEvent]) -> list[_CandidateDeathEvent]:
        # Group by world_id sorted chronologically, then round-robin by mix over
        # pass_idx = 0 (first death per distinct world), pass_idx = 1, ...
        by_world: dict[int, list[_CandidateDeathEvent]] = {}
        for ev in sorted(pool, key=lambda e: (e.death_tick, e.world_id)):
            by_world.setdefault(ev.world_id, []).append(ev)
        max_passes = max((len(v) for v in by_world.values()), default=0)
        ordered: list[_CandidateDeathEvent] = []
        for pass_idx in range(max_passes):
            by_mix_pass: dict[str, list[_CandidateDeathEvent]] = {m: [] for m in MIXES}
            for w_events in by_world.values():
                if pass_idx < len(w_events):
                    ev = w_events[pass_idx]
                    by_mix_pass[ev.mix].append(ev)
            for m in MIXES:
                by_mix_pass[m].sort(key=lambda e: (e.death_tick, e.world_id))
            cursors = {m: 0 for m in MIXES}
            while any(cursors[m] < len(by_mix_pass[m]) for m in MIXES):
                for m in MIXES:
                    if cursors[m] < len(by_mix_pass[m]):
                        ordered.append(by_mix_pass[m][cursors[m]])
                        cursors[m] += 1
        return ordered

    ranked_train = rank_within_partition([c for c in candidates if c.partition == "train"])
    ranked_assess = rank_within_partition([c for c in candidates if c.partition == "assessment"])

    take_assess = min(len(ranked_assess), target_assess)
    take_train = min(len(ranked_train), target_train)
    rem = int(max_deaths) - (take_assess + take_train)
    if rem > 0 and take_assess < len(ranked_assess):
        extra = min(rem, len(ranked_assess) - take_assess)
        take_assess += extra
        rem -= extra
    if rem > 0 and take_train < len(ranked_train):
        extra = min(rem, len(ranked_train) - take_train)
        take_train += extra
        rem -= extra

    selected = ranked_train[:take_train] + ranked_assess[:take_assess]
    selected.sort(key=lambda e: (0 if e.partition == "train" else 1, e.death_tick, e.world_id))
    return selected


def compute_required_ordinary_cell_quotas(
    death_cell_counts: Mapping[tuple[str, int], int],
    *,
    quota: int,
    min_partition_ordinary: int = 0,
) -> dict[tuple[str, int], int]:
    """Frozen per-``(mix, length_bin)`` target quotas without cross-mix/bin substitution.

    - Target partition total is ``min(quota, max(total_death_roots, min_partition_ordinary))``
      whenever ``total_death_roots > 0`` and ``quota > 0`` (else 0).
    - When ``target_total == total_death_roots``: exact 1:1 ``(mix, length_bin)`` matching.
    - Otherwise: largest-remainder scaling of ``death_cell_counts`` to ``target_total``
      strictly over cells with ``count[k] > 0`` (cells with zero deaths always get 0).
    """
    n_bins = len(ORDINARY_LENGTH_BIN_EDGES)
    all_keys = [(m, b) for m in MIXES for b in range(n_bins)]
    counts = {k: int(death_cell_counts.get(k, 0)) for k in all_keys}
    total_d = sum(counts.values())
    if quota <= 0 or total_d <= 0:
        return {k: 0 for k in all_keys}
    target_total = min(int(quota), max(int(total_d), int(min_partition_ordinary)))
    if target_total <= 0:
        return {k: 0 for k in all_keys}
    if target_total == total_d:
        return dict(counts)

    desired_raw = {k: (float(target_total) * float(c) / float(total_d)) for k, c in counts.items()}
    desired = {k: (int(math.floor(val)) if counts[k] > 0 else 0) for k, val in desired_raw.items()}
    rem = int(target_total) - sum(desired.values())
    positive_keys = [k for k in all_keys if counts[k] > 0]
    remainders = sorted(
        positive_keys,
        key=lambda k: (-(desired_raw[k] - math.floor(desired_raw[k])), MIXES.index(k[0]), k[1]),
    )
    for k in remainders:
        if rem <= 0:
            break
        desired[k] += 1
        rem -= 1
    return desired


def _select_matched_ordinary_roots(
    reservoirs: Mapping[tuple[str, str, int], list[_CandidateOrdinaryRoot]],
    selected_deaths: Sequence[_CandidateDeathEvent],
    *,
    max_ordinary_roots: int,
    forbidden_world_life_ticks: set[tuple[int, int, int]],
    min_train_ordinary_roots: int = 0,
    min_assess_ordinary_roots: int = 0,
    allow_tiny_zero_death_fallback: bool = False,
) -> tuple[list[_CandidateOrdinaryRoot], dict[str, Any]]:
    """Select ordinary roots matched strictly to death ``(mix, length_bin)`` quotas per partition.

    Never substitutes another mix or length bin to fill shortages; any cell shortfall is
    left unfilled and reported so ``validate_ct_support`` flags ``INCOMPLETE_SUPPORT``.
    """
    target_assess = int(round(0.20 * max_ordinary_roots))
    target_train = int(max_ordinary_roots) - target_assess
    n_bins = len(ORDINARY_LENGTH_BIN_EDGES)

    # Filter out any tick already retained as a death-lookback root.
    cleaned: dict[tuple[str, str, int], list[_CandidateOrdinaryRoot]] = {}
    for key, items in reservoirs.items():
        valid = [
            c for c in items if (c.world_id, c.life_id, c.tick) not in forbidden_world_life_ticks
        ]
        valid = quota_sample(valid, len(valid), key=key)
        cleaned[key] = valid

    report_partitions: dict[str, Any] = {}

    def pick_for_partition(
        part: str, quota: int, min_part_ord: int
    ) -> list[_CandidateOrdinaryRoot]:
        death_counts: dict[tuple[str, int], int] = {(m, b): 0 for m in MIXES for b in range(n_bins)}
        for ev in selected_deaths:
            if ev.partition != part:
                continue
            for _lb, _t, snap, *_rest in ev.lookback_roots:
                hlen = int(snap.small_tensors["length"][0].item())
                b = _length_bin_index(hlen)
                death_counts[(ev.mix, b)] += 1

        total_d = sum(death_counts.values())
        mode = "matched_to_deaths"
        if quota <= 0:
            desired_per_cell = {k: 0 for k in death_counts}
        elif total_d == 0 and allow_tiny_zero_death_fallback:
            mode = "tiny_zero_death_smoke"
            desired_per_cell = {k: 0 for k in death_counts}
            rem_smoke = int(quota)
            while rem_smoke > 0:
                progressed = False
                for m in MIXES:
                    for b in range(n_bins):
                        if rem_smoke <= 0:
                            break
                        avail = len(cleaned.get((part, m, b), []))
                        if desired_per_cell[(m, b)] < avail:
                            desired_per_cell[(m, b)] += 1
                            rem_smoke -= 1
                            progressed = True
                if not progressed:
                    break
        else:
            desired_per_cell = compute_required_ordinary_cell_quotas(
                death_counts, quota=quota, min_partition_ordinary=min_part_ord
            )

        taken_per_cell: dict[tuple[str, int], int] = {}
        shortfall_per_cell: dict[tuple[str, int], int] = {}
        cells_telemetry: dict[str, dict[str, dict[str, int]]] = {m: {} for m in MIXES}
        chosen: list[_CandidateOrdinaryRoot] = []

        for m in MIXES:
            for b in range(n_bins):
                avail_list = cleaned.get((part, m, b), [])
                req = int(desired_per_cell[(m, b)])
                taken = min(len(avail_list), req)
                taken_per_cell[(m, b)] = taken
                shortfall_per_cell[(m, b)] = req - taken
                if taken > 0:
                    chosen.extend(quota_sample(avail_list, taken, key=(part, m, b)))
                cells_telemetry[m][ORDINARY_LENGTH_BIN_LABELS[b]] = {
                    "death_lookback_roots": int(death_counts[(m, b)]),
                    "required_ordinary_roots": req,
                    "available_candidates": len(avail_list),
                    "achieved_ordinary_roots": taken,
                    "shortfall": req - taken,
                }

        chosen.sort(key=lambda c: (c.tick, c.world_id))
        report_partitions[part] = {
            "mode": mode,
            "partition_quota_cap": int(quota),
            "death_lookback_roots_total": int(total_d),
            "required_ordinary_roots_total": int(sum(desired_per_cell.values())),
            "achieved_ordinary_roots_total": int(sum(taken_per_cell.values())),
            "shortfall_total": int(sum(shortfall_per_cell.values())),
            "per_mix_bin": cells_telemetry,
        }
        return chosen

    train_picked = pick_for_partition("train", target_train, min_train_ordinary_roots)
    assess_picked = pick_for_partition("assessment", target_assess, min_assess_ordinary_roots)
    return train_picked + assess_picked, {
        "length_bin_edges": list(ORDINARY_LENGTH_BIN_EDGES),
        "length_bin_labels": list(ORDINARY_LENGTH_BIN_LABELS),
        "cross_mix_or_bin_spill_allowed": False,
        "partitions": report_partitions,
    }


@torch.inference_mode()
def collect_ct_dataset(
    control: CTControl,
    *,
    b8_network: nn.Module,
    checkpoint_networks: Mapping[str, nn.Module],
    device: str | torch.device = "cpu",
    stop_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[str], None] | None = None,
    log_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = 45.0,
    diagnostic: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> CollectedCTDataset | None:
    """Run ``worlds x collection_ticks`` bare-B8 collection and freeze Stage-1 roots.

    Returns ``None`` cleanly if ``stop_requested()`` fires (e.g. ``SIGUSR1``)
    mid-collection so no partial stage snapshot is published.
    """
    t0 = time.perf_counter()
    completed_ticks = 0

    def check_collection_cap() -> None:
        if (
            completed_ticks < control.collection_ticks
            and time.perf_counter() - t0 >= control.collection_cap_seconds
        ):
            raise CollectionCapReached(completed_ticks, control.worlds, time.perf_counter() - t0)

    dev = torch.device("cuda:0" if str(device) == "cuda" else device)
    b8_eval = b8_network.to(dev).eval()
    world_labels, partitions, partition_mix_worlds = world_partition_table(control.worlds)
    allocations = {row["phase"]: row for row in ct_seed_allocations(control)}

    # Map each world to its monotonic local rank inside its pre-assigned partition
    # so train worlds draw strictly from "training" and assessment worlds draw
    # strictly from "assessment".
    partition_rank: dict[int, tuple[str, int, int]] = {}
    for phase_name, part_key in (("training", "train"), ("assessment", "assessment")):
        base_offset = int(allocations[phase_name]["offset"])
        part_worlds = partitions[part_key]
        for rank_idx, w_id in enumerate(part_worlds):
            partition_rank[w_id] = (phase_name, base_offset, len(part_worlds) * 0 + rank_idx)
    part_sizes = {"training": len(partitions["train"]), "assessment": len(partitions["assessment"])}

    cfg = BatchSimConfig(
        num_envs=control.worlds,
        num_snakes=6,
        max_capacity=control.max_body_capacity,
        body_storage_capacity=control.max_body_capacity,
        mechanics_version=1,
        **(
            dict(game_width=240, game_height=240, initial_food=10, max_food=18)
            if control.tiny
            else {}
        ),
    )
    sim = CTDeviceSim(
        cfg,
        device=dev,
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if dev.type == "cuda" else "reference",
    )
    r2_cache.enable(sim)
    rosters: tuple[TrainingRoster, ...] = training_rosters(control.worlds)
    mix_to_id = {m: idx for idx, m in enumerate(MIXES)}
    roster_index_by_identity: dict[tuple[Any, ...], int] = {}
    world_roster_idx: list[int] = []
    for r in rosters:
        if r.identity not in roster_index_by_identity:
            roster_index_by_identity[r.identity] = len(roster_index_by_identity)
        world_roster_idx.append(roster_index_by_identity[r.identity])

    episode_counts = [0] * control.worlds
    life_ids = [0] * control.worlds

    def reset_world_rows(mask_t: torch.Tensor, *, opponents_rt: CTOpponentRuntime | None) -> None:
        rows = mask_t.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            return
        rows_cpu = [int(r) for r in rows.detach().cpu().tolist()]
        sim_keys: list[int] = []
        opp_keys: dict[int, int] = {}
        for w_id in rows_cpu:
            phase_name, base_offset, local_rank = partition_rank[w_id]
            ep = episode_counts[w_id]
            max_ep = int(control.collection_ticks) + 1
            if ep >= max_ep:
                raise RuntimeError("world reset exceeded audited C/T seed allocation")
            seed_index = base_offset + ep * part_sizes[phase_name] + local_rank
            s_key = canonical_low63_seed(control.study, seed_index)
            sim_keys.append(s_key)
            opp_keys[w_id] = derive_63bit_key(
                f"{CT_DOMAIN_PREFIX}/opp-world-seed/v1", phase_name, seed_index
            )
        for name in (f for f in _ENV_TENSOR_FIELDS if f != "_last_death_cause"):
            getattr(sim, name)[rows] = 0
        sim.alive[rows] = False
        sim.length[rows] = 1
        sim._reward_prev_length[rows] = 1
        sim.rng_key.index_copy_(0, rows, torch.tensor(sim_keys, dtype=torch.long, device=dev))
        sim.terminal_slots[rows] = False
        sim._prepared = False
        local_view = _row_view(sim, rows)
        local_view.reset_fresh_worlds()
        _copy_rows(sim, local_view, rows)
        sim.terminal_slots[rows, 0] = True
        sim._prepared = False
        if opponents_rt is not None:
            opponents_rt.reset(mask_t, opp_keys)
        r2_cache.bump(sim)

    init_mask = torch.ones(control.worlds, dtype=torch.bool, device=dev)
    reset_world_rows(init_mask, opponents_rt=None)
    opponents = CTOpponentRuntime(
        sim,
        rosters,
        b8_network=b8_eval,
        checkpoint_networks=checkpoint_networks,
    )
    # Sync initial opponent keys to audited reset keys.
    init_opp_keys = {
        w_id: derive_63bit_key(
            f"{CT_DOMAIN_PREFIX}/opp-world-seed/v1",
            partition_rank[w_id][0],
            partition_rank[w_id][1] + partition_rank[w_id][2],
        )
        for w_id in range(control.worlds)
    }
    opponents.reset(init_mask, init_opp_keys)

    max_lb = max(control.lookbacks)
    ring = LookbackRing(
        num_worlds=control.worlds,
        stride=5,
        max_lookback=max_lb,
        max_body_capacity=control.max_body_capacity,
        diagnostic=diagnostic,
    )
    replay_cache: dict[tuple[Any, ...], tuple[CTDeviceSim, CTOpponentRuntime]] = {}

    # Candidate death pool (bounded generously so memory stays small while
    # allowing distinct-world priority selection across the full 5,000 ticks).
    max_death_candidates = max(control.max_deaths * 4, 640)
    death_candidates: list[_CandidateDeathEvent] = []
    death_seen_worlds: set[int] = set()
    total_hero_deaths = 0
    total_eligible_deaths_observed = 0

    # Stratified per-(partition, mix, length_bin) reservoir for alive ordinary roots
    # across all length bins (including [1,600) for early lookback states, never
    # conditioned on future survival).
    n_bins = len(ORDINARY_LENGTH_BIN_EDGES)
    ord_cap_assess = (
        int(round(0.20 * control.max_ordinary_roots)) + control.ordinary_overlap_allowance
    )
    ord_cap_train = (
        control.max_ordinary_roots
        - int(round(0.20 * control.max_ordinary_roots))
        + control.ordinary_overlap_allowance
    )
    ord_reservoirs: dict[tuple[str, str, int], list[_CandidateOrdinaryRoot]] = {
        (part, mix, b): []
        for part in ("train", "assessment")
        for mix in MIXES
        for b in range(n_bins)
    }
    ord_seen_counts: dict[tuple[str, str, int], int] = {k: 0 for k in ord_reservoirs}
    ord_rng = np.random.default_rng(
        derive_63bit_key(
            f"{CT_DOMAIN_PREFIX}/ordinary-reservoir/v1", control.worlds, control.collection_ticks
        )
    )

    # Pre-scheduled unsearched anchor slots (with slight oversampling so we can
    # drop any exact (world_id, life_id, tick) collision with searched roots).
    oversample_extra = min(
        4_096, max(256, control.max_deaths * len(control.lookbacks) + control.max_ordinary_roots)
    )
    anchor_schedule = _preschedule_anchor_slots(
        partitions=partitions,
        collection_ticks=control.collection_ticks,
        target_anchor_rows=control.anchor_rows,
        oversample_extra=oversample_extra,
    )
    raw_anchor_meta: list[tuple[int, int, bool, int, int, int, int]] = []
    raw_anchor_loc: list[torch.Tensor] = []
    raw_anchor_glb: list[torch.Tensor] = []
    raw_anchor_scl: list[torch.Tensor] = []
    raw_anchor_msk: list[torch.Tensor] = []
    raw_anchor_qb8: list[torch.Tensor] = []
    raw_anchor_topo_builder = IncrementalSparseTopologyBuilder(int(sim.grid_h), int(sim.grid_w))
    raw_anchor_obs_bytes = 0

    last_log = time.perf_counter()

    completed_ticks = 0
    for tick in range(control.collection_ticks):
        if (
            completed_ticks < control.collection_ticks
            and time.perf_counter() - t0 >= control.collection_cap_seconds
        ):
            raise CollectionCapReached(completed_ticks, control.worlds, time.perf_counter() - t0)
        if stop_requested is not None and stop_requested():
            return None
        if heartbeat is not None and (tick % 16 == 0):
            heartbeat(f"collect:tick={tick}/{control.collection_ticks}")

        sim.prepare()
        ring.maybe_capture_checkpoint(sim, opponents, tick=tick, life_ids=life_ids)
        loc_obs, glb_obs, scl_obs, leg_msk, q_b8, a_b8 = observe_hero_bv8(sim, b8_eval)
        a_b8_np = a_b8.detach().cpu().numpy().astype(np.int16)
        pre_lengths_np = sim.length[:, 0].detach().cpu().numpy().astype(np.int64)
        alive_np = sim.alive[:, 0].detach().cpu().numpy()

        ring.record_tick_actions(tick=tick, hero_actions=a_b8_np, life_ids=life_ids, q_b8=q_b8)
        if diagnostic is not None:
            for mix in MIXES:
                ids = [w for w, roster in enumerate(rosters) if roster.mix == mix and alive_np[w]]
                counts = np.bincount(
                    np.searchsorted(ORDINARY_LENGTH_BIN_EDGES, pre_lengths_np[ids], side="right")
                    - 1,
                    minlength=len(ORDINARY_LENGTH_BIN_EDGES),
                )
                diagnostic(
                    "length-histograms",
                    {
                        "tick": tick,
                        "mix": mix,
                        "alive_worlds": len(ids),
                        "bin_edges": list(ORDINARY_LENGTH_BIN_EDGES),
                        "counts": counts.tolist(),
                    },
                )

        # 1. Determine which rows on this tick need topology extraction right now
        #    (scheduled anchor rows + reservoir-accepted L >= min_death_length ordinary rows).
        sched_worlds = [w for w in anchor_schedule.get(tick, []) if alive_np[w]]
        eligible_ord_worlds = np.flatnonzero(alive_np & (pre_lengths_np >= 1))
        proposals = []
        for w_int in eligible_ord_worlds.tolist():
            w = int(w_int)
            cell = (world_labels[w], rosters[w].mix, _length_bin_index(int(pre_lengths_np[w])))
            proposals.append((w, cell))
        ord_accepts = reservoir_admissions(
            proposals,
            reservoirs=ord_reservoirs,
            seen_counts=ord_seen_counts,
            capacities={"train": ord_cap_train, "assessment": ord_cap_assess},
            rng=ord_rng,
        )

        topo_needed = sorted(set(sched_worlds) | {w for w, _ck, _slot in ord_accepts})
        topo_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if topo_needed:
            t_planes, t_sums = compute_topology_features(sim, topo_needed)
            for idx_pos, w in enumerate(topo_needed):
                topo_map[w] = (t_planes[idx_pos], t_sums[idx_pos])

        for w in sched_worlds:
            check_collection_cap()
            pl_w, sm_w = topo_map[w]
            loc_w = loc_obs[w].detach().to("cpu", dtype=torch.uint8).clone()
            glb_w = glb_obs[w].detach().to("cpu", dtype=torch.uint8).clone()
            scl_w = scl_obs[w].detach().to("cpu", dtype=torch.float32).clone()
            msk_w = leg_msk[w].detach().to("cpu", dtype=torch.bool).clone()
            qb8_w = q_b8[w].detach().to("cpu", dtype=torch.float32).clone()
            row_obs_bytes = (
                int(loc_w.numel())
                + int(glb_w.numel())
                + int(scl_w.numel() * 4)
                + int(msk_w.numel())
                + int(qb8_w.numel() * 4)
                + 20
            )
            raw_anchor_topo_builder.append_batch(pl_w[None, ...], sm_w[None, ...])
            projected_anc_bytes = (
                raw_anchor_obs_bytes + row_obs_bytes + raw_anchor_topo_builder.nbytes
            )
            if projected_anc_bytes > int(control.anchor_archive_max_bytes):
                raise StageExportOverflowError(
                    f"anchor archive in-memory byte cap exceeded during collection: "
                    f"{projected_anc_bytes} > {control.anchor_archive_max_bytes}"
                )
            raw_anchor_obs_bytes += row_obs_bytes
            raw_anchor_meta.append(
                (
                    w,
                    mix_to_id[rosters[w].mix],
                    world_labels[w] == "assessment",
                    life_ids[w],
                    tick,
                    int(pre_lengths_np[w]),
                    int(a_b8_np[w]),
                )
            )
            raw_anchor_loc.append(loc_w)
            raw_anchor_glb.append(glb_w)
            raw_anchor_scl.append(scl_w)
            raw_anchor_msk.append(msk_w)
            raw_anchor_qb8.append(qb8_w)

        for w, cell_key, replace_slot in ord_accepts:
            check_collection_cap()
            pl_w, sm_w = topo_map[w]
            cand = _CandidateOrdinaryRoot(
                world_id=w,
                partition=cell_key[0],
                mix=cell_key[1],
                roster_index=world_roster_idx[w],
                life_id=life_ids[w],
                tick=tick,
                hero_length=int(pre_lengths_np[w]),
                length_bin=cell_key[2],
                snapshot=capture_world_snapshot(
                    sim,
                    opponents,
                    w,
                    world_id=w,
                    life_id=life_ids[w],
                    tick=tick,
                    max_body_capacity=control.max_body_capacity,
                ),
                local_obs=loc_obs[w].detach().to("cpu", dtype=torch.uint8).clone(),
                glob_obs=glb_obs[w].detach().to("cpu", dtype=torch.uint8).clone(),
                scalars_obs=scl_obs[w].detach().to("cpu", dtype=torch.float32).clone(),
                legal_mask=leg_msk[w].detach().to("cpu", dtype=torch.bool).clone(),
                q_b8=q_b8[w].detach().to("cpu", dtype=torch.float32).clone(),
                b8_action=int(a_b8_np[w]),
                topo_planes_u16=pl_w.copy(),
                topo_summaries_f32=sm_w.copy(),
            )
            if replace_slot is None:
                ord_reservoirs[cell_key].append(cand)
            else:
                ord_reservoirs[cell_key][replace_slot] = cand

        # 2. Step all worlds by 1 frame under bare B8 + rosters.
        full_actions = opponents.actions(sim, a_b8)
        sim.step(full_actions)
        opponents.after_step(sim)

        # 3. Inspect hero deaths BEFORE world reset and reconstruct 8 lookbacks.
        died_mask = sim._last_done[:, 0]
        if bool(died_mask.any().item()):
            dead_worlds = [
                int(w) for w in died_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            ]
            total_hero_deaths += len(dead_worlds)
            for w in dead_worlds:
                d_len = int(pre_lengths_np[w])
                same_life = ring.has_same_life_lookbacks(
                    w, death_tick=tick, life_id=life_ids[w], lookbacks=control.lookbacks
                )
                death_record = {
                    "world": w,
                    "partition": world_labels[w],
                    "mix": rosters[w].mix,
                    "life": life_ids[w],
                    "length": d_len,
                    "tick": tick,
                    "below_min_length": d_len < control.min_death_length,
                    "missing_same_life_lookbacks": not same_life,
                    "candidate_pool_rejected": False,
                    "candidate_admitted": False,
                }
                if d_len >= control.min_death_length and same_life:
                    total_eligible_deaths_observed += 1
                    # Always admit a newly discovered distinct world (replacing a
                    # duplicate-world death if the candidate buffer is full).
                    admit = len(death_candidates) < max_death_candidates
                    replace_idx: int | None = None
                    if not admit and w not in death_seen_worlds:
                        seen_tmp: set[int] = set()
                        for idx_c in range(len(death_candidates) - 1, -1, -1):
                            cw = death_candidates[idx_c].world_id
                            if cw in seen_tmp:
                                replace_idx = idx_c
                                admit = True
                                break
                            seen_tmp.add(cw)
                    death_record["candidate_pool_rejected"] = not admit
                    death_record["candidate_admitted"] = admit
                    if diagnostic is not None:
                        diagnostic("deaths", death_record)
                    if admit:
                        lb_list = []
                        for lb in control.lookbacks:
                            t_target = tick - int(lb)
                            rec = ring.reconstruct_lookback_root(
                                world_id=w,
                                target_tick=t_target,
                                expected_life_id=life_ids[w],
                                roster=rosters[w],
                                cfg=cfg,
                                b8_network=b8_eval,
                                checkpoint_networks=checkpoint_networks,
                                device=dev,
                                replay_cache=replay_cache,
                            )
                            lb_list.append((int(lb), t_target, *rec))
                        ev = _CandidateDeathEvent(
                            world_id=w,
                            partition=world_labels[w],
                            mix=rosters[w].mix,
                            roster_index=world_roster_idx[w],
                            life_id=life_ids[w],
                            death_tick=tick,
                            death_length=d_len,
                            lookback_roots=lb_list,
                        )
                        death_seen_worlds.add(w)
                        if replace_idx is None:
                            death_candidates.append(ev)
                        else:
                            death_candidates[replace_idx] = ev

                if diagnostic is not None and (d_len < control.min_death_length or not same_life):
                    diagnostic("deaths", death_record)
                episode_counts[w] += 1
                life_ids[w] += 1

            reset_world_rows(died_mask, opponents_rt=opponents)

        completed_ticks = tick + 1
        now = time.perf_counter()
        if (
            completed_ticks < control.collection_ticks
            and now - t0 >= control.collection_cap_seconds
        ):
            raise CollectionCapReached(completed_ticks, control.worlds, now - t0)
        if log_progress is not None and (
            now - last_log >= progress_interval_seconds or tick + 1 == control.collection_ticks
        ):
            last_log = now
            log_progress(
                {
                    "phase": "collection",
                    "tick": tick + 1,
                    "total_ticks": control.collection_ticks,
                    "charged_rows": (tick + 1) * control.worlds,
                    "hero_deaths": total_hero_deaths,
                    "eligible_deaths_observed": total_eligible_deaths_observed,
                    "retained_death_candidates": len(death_candidates),
                    "elapsed_sec": round(now - t0, 2),
                }
            )

    if stop_requested is not None and stop_requested():
        return None

    natural_collection_wall_sec = time.perf_counter() - t0
    # Finalize deterministic death selection, ordinary root matching, and anchor archive.
    selected_deaths = _select_deaths_deterministic(death_candidates, max_deaths=control.max_deaths)
    max_roots_cap = control.max_deaths * len(control.lookbacks) + control.max_ordinary_roots
    store = LosslessBoundedSnapshotStore(
        grid_h=int(sim.grid_h),
        grid_w=int(sim.grid_w),
        max_roots=max_roots_cap,
        max_bytes=control.snapshot_store_max_bytes,
        max_body_capacity=control.max_body_capacity,
    )

    searched_keys: set[tuple[int, int, int]] = set()
    for death_event_id, ev in enumerate(selected_deaths):
        check_collection_cap()
        for (
            lb,
            t_target,
            snap,
            loc_r,
            glb_r,
            scl_r,
            msk_r,
            qb8_r,
            b8_act,
            pl_r,
            sm_r,
        ) in ev.lookback_roots:
            uid = (
                f"death{death_event_id:04d}-w{ev.world_id:04d}-"
                f"l{ev.life_id:03d}-t{t_target:05d}-lb{lb:03d}"
            )
            store.add_root(
                meta=OmitIndexRootMeta(
                    root_uid=uid,
                    kind="death_lookback",
                    partition=ev.partition,
                    world_id=ev.world_id,
                    mix=ev.mix,
                    roster_index=ev.roster_index,
                    life_id=ev.life_id,
                    tick=t_target,
                    hero_length=int(snap.small_tensors["length"][0].item()),
                    death_event_id=death_event_id,
                    lookback=lb,
                    death_tick=ev.death_tick,
                    death_length=ev.death_length,
                    b8_action=b8_act,
                ),
                snapshot=snap,
                local_obs=loc_r,
                glob_obs=glb_r,
                scalars_obs=scl_r,
                legal_mask=msk_r,
                q_b8=qb8_r,
                topo_planes_u16=pl_r,
                topo_summaries_f32=sm_r,
            )
            searched_keys.add((ev.world_id, ev.life_id, t_target))

    selected_ordinary, ordinary_matching_report = _select_matched_ordinary_roots(
        ord_reservoirs,
        selected_deaths,
        max_ordinary_roots=control.max_ordinary_roots,
        forbidden_world_life_ticks=searched_keys,
        min_train_ordinary_roots=0 if control.tiny else 1,
        min_assess_ordinary_roots=int(control.min_held_out_ordinary_roots),
        allow_tiny_zero_death_fallback=bool(control.tiny and control.min_held_out_deaths == 0),
    )
    for ord_idx, cand in enumerate(selected_ordinary):
        uid = f"ord{ord_idx:04d}-w{cand.world_id:04d}-l{cand.life_id:03d}-t{cand.tick:05d}"
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=uid,
                kind="ordinary",
                partition=cand.partition,
                world_id=cand.world_id,
                mix=cand.mix,
                roster_index=cand.roster_index,
                life_id=cand.life_id,
                tick=cand.tick,
                hero_length=cand.hero_length,
                death_event_id=None,
                lookback=None,
                death_tick=None,
                death_length=None,
                b8_action=cand.b8_action,
            ),
            snapshot=cand.snapshot,
            local_obs=cand.local_obs,
            glob_obs=cand.glob_obs,
            scalars_obs=cand.scalars_obs,
            legal_mask=cand.legal_mask,
            q_b8=cand.q_b8,
            topo_planes_u16=cand.topo_planes_u16,
            topo_summaries_f32=cand.topo_summaries_f32,
        )
        searched_keys.add((cand.world_id, cand.life_id, cand.tick))

    store.finalize_topology()

    # Finalize strictly unsearched anchor archive (80% train / 20% assessment).
    target_assess_anc = int(round(0.20 * control.anchor_rows))
    target_train_anc = int(control.anchor_rows) - target_assess_anc
    keep_train: list[int] = []
    keep_assess: list[int] = []
    for idx_a, (w, _mid, is_assess, lid, t_a, _hlen, _act) in enumerate(raw_anchor_meta):
        if (w, lid, t_a) in searched_keys:
            continue
        if is_assess:
            if len(keep_assess) < target_assess_anc:
                keep_assess.append(idx_a)
        else:
            if len(keep_train) < target_train_anc:
                keep_train.append(idx_a)
    keep_all = keep_train + keep_assess

    anc_topo = raw_anchor_topo_builder.build().slice_rows(np.asarray(keep_all, dtype=np.int64))
    n_anc = len(keep_all)
    if n_anc > 0:
        meta_arr = [raw_anchor_meta[i] for i in keep_all]
        anc_archive = AnchorArchive(
            num_rows=n_anc,
            world_id=np.array([m[0] for m in meta_arr], dtype=np.int32),
            mix_id=np.array([m[1] for m in meta_arr], dtype=np.int8),
            is_assessment=np.array([m[2] for m in meta_arr], dtype=bool),
            life_id=np.array([m[3] for m in meta_arr], dtype=np.int32),
            tick=np.array([m[4] for m in meta_arr], dtype=np.int32),
            hero_length=np.array([m[5] for m in meta_arr], dtype=np.int32),
            local=torch.stack([raw_anchor_loc[i] for i in keep_all], dim=0),
            glob=torch.stack([raw_anchor_glb[i] for i in keep_all], dim=0),
            scalars=torch.stack([raw_anchor_scl[i] for i in keep_all], dim=0),
            legal_mask=torch.stack([raw_anchor_msk[i] for i in keep_all], dim=0),
            q_b8=torch.stack([raw_anchor_qb8[i] for i in keep_all], dim=0),
            b8_action=np.array([m[6] for m in meta_arr], dtype=np.int8),
            topology=anc_topo,
        )
    else:
        anc_archive = AnchorArchive(
            num_rows=0,
            world_id=np.zeros(0, dtype=np.int32),
            mix_id=np.zeros(0, dtype=np.int8),
            is_assessment=np.zeros(0, dtype=bool),
            life_id=np.zeros(0, dtype=np.int32),
            tick=np.zeros(0, dtype=np.int32),
            hero_length=np.zeros(0, dtype=np.int32),
            local=torch.empty((0, 7, 31, 31), dtype=torch.uint8),
            glob=torch.empty((0, 4, 37, 37), dtype=torch.uint8),
            scalars=torch.empty((0, 60), dtype=torch.float32),
            legal_mask=torch.empty((0, 6), dtype=torch.bool),
            q_b8=torch.empty((0, 6), dtype=torch.float32),
            b8_action=np.zeros(0, dtype=np.int8),
            topology=anc_topo,
        )

    elapsed = time.perf_counter() - t0
    telemetry = {
        "completed_ticks": completed_ticks,
        "collection_complete": True,
        "collection_cap_seconds": control.collection_cap_seconds,
        "bank_assembly_crossed_collection_cap": elapsed >= control.collection_cap_seconds,
        "ordinary_cell_capacities": {"train": ord_cap_train, "assessment": ord_cap_assess},
        "ordinary_reservoir_sizes": {str(k): len(v) for k, v in ord_reservoirs.items()},
        "task_identity": task_separation_report(),
        "topology_spec": TOPOLOGY_SPEC,
        "charged_collection_rows": int(control.worlds * control.collection_ticks),
        "total_hero_deaths": int(total_hero_deaths),
        "total_eligible_deaths_observed": int(total_eligible_deaths_observed),
        "selected_deaths_total": len(selected_deaths),
        "selected_deaths_train": sum(1 for e in selected_deaths if e.partition == "train"),
        "selected_deaths_assessment": sum(
            1 for e in selected_deaths if e.partition == "assessment"
        ),
        "selected_death_lookback_roots": len(selected_deaths) * len(control.lookbacks),
        "selected_ordinary_roots_total": len(selected_ordinary),
        "selected_ordinary_roots_train": sum(
            1 for c in selected_ordinary if c.partition == "train"
        ),
        "selected_ordinary_roots_assessment": sum(
            1 for c in selected_ordinary if c.partition == "assessment"
        ),
        "ordinary_matching": ordinary_matching_report,
        "anchor_rows_total": int(anc_archive.num_rows),
        "anchor_rows_train": int((~anc_archive.is_assessment).sum()),
        "anchor_rows_assessment": int(anc_archive.is_assessment.sum()),
        "anchor_archive_bytes": int(anc_archive.nbytes),
        "snapshot_store_bytes": int(store.bytes_used),
        "collection_wall_sec": round(elapsed, 3),
        "natural_collection_wall_sec": round(natural_collection_wall_sec, 3),
        "bank_assembly_wall_sec": round(elapsed - natural_collection_wall_sec, 3),
    }
    return CollectedCTDataset(
        control=control,
        world_partitions=world_labels,
        partition_mix_worlds=partition_mix_worlds,
        store=store,
        anchors=anc_archive,
        collection_telemetry=telemetry,
    )
