"""Counterfactual H300 branching engine with 4 selection + 4 independent scoring suffixes.

Implements Ruling §4:
- Preserves prepared root observation/Q/mask.
- Scores all canonical legal actions for H300 steps under bare B8 continuation
  across 4 selection suffixes (k in 0..3) plus 4 independent scoring suffixes
  (k in 4..7).
- Domain-separated keys omit action ID so candidate actions share Common Random
  Numbers (CRN) within each suffix across every active post-prepare RNG consumer
  (simulator food/respawn + random_safe opponent tie-breaking).
- Freezes ``a*(s)`` and forbidden set ``F_sel(s)`` on selection suffixes ``0..3``
  *before* running scoring suffixes ``4..7``.
- Exact dead-hero slab compaction (verified bitwise identical to uncompacted
  stepping because dead hero ``terminal_slots[:, 0] == True`` never respawns and
  every world row owns row-independent RNG counters).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import (
    RNG_CONSUMER_INVENTORY,
    CTDeviceSim,
    CTOpponentRuntime,
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_snapshots import (
    LosslessBoundedSnapshotStore,
    observe_hero_bv8,
)
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.sandbox_a_resident.training_rosters import TrainingRoster, training_rosters
from src.simd_env.batch_sim import BatchSimConfig


def compute_q_from_survival_and_mass(
    survive: np.ndarray, mass: np.ndarray, legal_mask: np.ndarray
) -> np.ndarray:
    """Compute ``Q(s, a) = S(s, a) + 0.25 * (M(s, a) / 1000.0)`` in float64."""
    q = survive.astype(np.float64) + 0.25 * (mass.astype(np.float64) / 1000.0)
    return np.where(legal_mask, q, np.nan)


def select_a_star_and_forbidden(
    *,
    s_sel: np.ndarray,
    m_sel: np.ndarray,
    legal_mask: np.ndarray,
    b8_actions: np.ndarray,
    forbidden_gap: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically freeze ``(q_sel, a_star, forbidden_sel)`` on selection suffixes.

    Tie-breaking rule for ``a*(s) = argmax_{a legal} Q_sel(s, a)``:
    If ``a_B8(s)`` is legal and attains the exact float64 maximum ``Q_sel``,
    prefer ``a_B8(s)`` (no unnecessary intervention on zero-benefit ties);
    otherwise choose the lowest action index attaining the maximum.
    """
    n_roots = s_sel.shape[0]
    q_sel = compute_q_from_survival_and_mass(s_sel, m_sel, legal_mask)
    a_star = np.zeros(n_roots, dtype=np.int64)
    forbidden = np.zeros((n_roots, 6), dtype=bool)

    for i in range(n_roots):
        leg = legal_mask[i]
        legal_idxs = np.flatnonzero(leg)
        if legal_idxs.size == 0:
            raise ValueError(f"root row {i} has empty canonical legal mask")
        q_row = q_sel[i, legal_idxs]
        q_max = float(np.max(q_row))
        tied = legal_idxs[q_row == q_max]
        b8_a = int(b8_actions[i])
        if b8_a in tied:
            chosen = b8_a
        else:
            chosen = int(tied[0])
        a_star[i] = chosen
        s_best = float(s_sel[i, chosen])
        for a in legal_idxs.tolist():
            if s_best - float(s_sel[i, a]) >= float(forbidden_gap):
                forbidden[i, a] = True
    return q_sel, a_star, forbidden


def _compact_sim_in_place(sim: CTDeviceSim, keep: torch.Tensor) -> None:
    keep = keep.to(device=sim.device, dtype=torch.long).flatten()
    new_e = int(keep.numel())
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            setattr(sim, name, val.index_select(0, keep))
    sim.E = new_e
    sim.cfg = replace(sim.cfg, num_envs=new_e)
    r2_cache.bump(sim)


@dataclass
class ScoredRootChunk:
    """Immutable counterfactual label table for one root chunk (selection + scoring)."""

    chunk_index: int
    root_indices: np.ndarray  # (C,) int64 indices into LosslessBoundedSnapshotStore
    legal_mask: np.ndarray  # (C, 6) bool
    b8_action: np.ndarray  # (C,) int64
    survive_sel_raw: np.ndarray  # (C, 6, 4) float32 (NaN off legal)
    mass_sel_raw: np.ndarray  # (C, 6, 4) float32 (NaN off legal)
    s_sel: np.ndarray  # (C, 6) float64
    m_sel: np.ndarray  # (C, 6) float64
    q_sel: np.ndarray  # (C, 6) float64
    a_star: np.ndarray  # (C,) int64 frozen BEFORE scoring suffixes
    forbidden_sel: np.ndarray  # (C, 6) bool frozen BEFORE scoring suffixes
    selection_freeze_sha256: str
    survive_score_raw: np.ndarray  # (C, 6, 4) float32 (NaN off legal)
    mass_score_raw: np.ndarray  # (C, 6, 4) float32 (NaN off legal)
    s_score: np.ndarray  # (C, 6) float64
    m_score: np.ndarray  # (C, 6) float64
    q_score: np.ndarray  # (C, 6) float64
    telemetry: dict[str, Any]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ct-scored-root-chunk/v1",
            "chunk_index": int(self.chunk_index),
            "root_indices": torch.from_numpy(self.root_indices.copy()),
            "legal_mask": torch.from_numpy(self.legal_mask.copy()),
            "b8_action": torch.from_numpy(self.b8_action.copy()),
            "survive_sel_raw": torch.from_numpy(self.survive_sel_raw.copy()),
            "mass_sel_raw": torch.from_numpy(self.mass_sel_raw.copy()),
            "s_sel": torch.from_numpy(self.s_sel.copy()),
            "m_sel": torch.from_numpy(self.m_sel.copy()),
            "q_sel": torch.from_numpy(self.q_sel.copy()),
            "a_star": torch.from_numpy(self.a_star.copy()),
            "forbidden_sel": torch.from_numpy(self.forbidden_sel.copy()),
            "selection_freeze_sha256": str(self.selection_freeze_sha256),
            "survive_score_raw": torch.from_numpy(self.survive_score_raw.copy()),
            "mass_score_raw": torch.from_numpy(self.mass_score_raw.copy()),
            "s_score": torch.from_numpy(self.s_score.copy()),
            "m_score": torch.from_numpy(self.m_score.copy()),
            "q_score": torch.from_numpy(self.q_score.copy()),
            "telemetry": dict(self.telemetry),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ScoredRootChunk":
        if state.get("schema") != "ct-scored-root-chunk/v1":
            raise ValueError("foreign ScoredRootChunk schema")
        return cls(
            chunk_index=int(state["chunk_index"]),
            root_indices=state["root_indices"].detach().cpu().numpy().astype(np.int64),
            legal_mask=state["legal_mask"].detach().cpu().numpy().astype(bool),
            b8_action=state["b8_action"].detach().cpu().numpy().astype(np.int64),
            survive_sel_raw=state["survive_sel_raw"].detach().cpu().numpy().astype(np.float32),
            mass_sel_raw=state["mass_sel_raw"].detach().cpu().numpy().astype(np.float32),
            s_sel=state["s_sel"].detach().cpu().numpy().astype(np.float64),
            m_sel=state["m_sel"].detach().cpu().numpy().astype(np.float64),
            q_sel=state["q_sel"].detach().cpu().numpy().astype(np.float64),
            a_star=state["a_star"].detach().cpu().numpy().astype(np.int64),
            forbidden_sel=state["forbidden_sel"].detach().cpu().numpy().astype(bool),
            selection_freeze_sha256=str(state["selection_freeze_sha256"]),
            survive_score_raw=state["survive_score_raw"].detach().cpu().numpy().astype(np.float32),
            mass_score_raw=state["mass_score_raw"].detach().cpu().numpy().astype(np.float32),
            s_score=state["s_score"].detach().cpu().numpy().astype(np.float64),
            m_score=state["m_score"].detach().cpu().numpy().astype(np.float64),
            q_score=state["q_score"].detach().cpu().numpy().astype(np.float64),
            telemetry=dict(state["telemetry"]),
        )


@dataclass
class MergedScoredRoots:
    """Complete scored root table aligned 1:1 with ``store.metadata``."""

    num_roots: int
    legal_mask: np.ndarray  # (N, 6) bool
    b8_action: np.ndarray  # (N,) int64
    s_sel: np.ndarray  # (N, 6) float64
    m_sel: np.ndarray  # (N, 6) float64
    q_sel: np.ndarray  # (N, 6) float64
    a_star: np.ndarray  # (N,) int64
    forbidden_sel: np.ndarray  # (N, 6) bool
    s_score: np.ndarray  # (N, 6) float64
    m_score: np.ndarray  # (N, 6) float64
    q_score: np.ndarray  # (N, 6) float64
    chunk_sha256s: list[str]
    total_logical_branch_steps: int
    total_executed_sim_steps: int
    total_scoring_wall_sec: float


def merge_scored_chunks(
    chunks: Sequence[ScoredRootChunk], *, expected_num_roots: int
) -> MergedScoredRoots:
    """Merge and verify disjoint complete coverage over ``0..expected_num_roots-1``."""
    N = int(expected_num_roots)
    if N == 0:
        return MergedScoredRoots(
            num_roots=0,
            legal_mask=np.zeros((0, 6), dtype=bool),
            b8_action=np.zeros(0, dtype=np.int64),
            s_sel=np.zeros((0, 6), dtype=np.float64),
            m_sel=np.zeros((0, 6), dtype=np.float64),
            q_sel=np.zeros((0, 6), dtype=np.float64),
            a_star=np.zeros(0, dtype=np.int64),
            forbidden_sel=np.zeros((0, 6), dtype=bool),
            s_score=np.zeros((0, 6), dtype=np.float64),
            m_score=np.zeros((0, 6), dtype=np.float64),
            q_score=np.zeros((0, 6), dtype=np.float64),
            chunk_sha256s=[],
            total_logical_branch_steps=0,
            total_executed_sim_steps=0,
            total_scoring_wall_sec=0.0,
        )

    seen = np.zeros(N, dtype=bool)
    legal_mask = np.zeros((N, 6), dtype=bool)
    b8_action = np.zeros(N, dtype=np.int64)
    s_sel = np.full((N, 6), np.nan, dtype=np.float64)
    m_sel = np.full((N, 6), np.nan, dtype=np.float64)
    q_sel = np.full((N, 6), np.nan, dtype=np.float64)
    a_star = np.zeros(N, dtype=np.int64)
    forbidden_sel = np.zeros((N, 6), dtype=bool)
    s_score = np.full((N, 6), np.nan, dtype=np.float64)
    m_score = np.full((N, 6), np.nan, dtype=np.float64)
    q_score = np.full((N, 6), np.nan, dtype=np.float64)

    shas: list[str] = []
    logical_steps = 0
    exec_steps = 0
    wall_sec = 0.0
    for chunk in sorted(chunks, key=lambda c: c.chunk_index):
        idx = chunk.root_indices
        if idx.size > 0:
            if np.any(idx < 0) or np.any(idx >= N) or np.any(seen[idx]):
                raise ValueError("duplicate or out-of-range root_indices across scored chunks")
            seen[idx] = True
            legal_mask[idx] = chunk.legal_mask
            b8_action[idx] = chunk.b8_action
            s_sel[idx] = chunk.s_sel
            m_sel[idx] = chunk.m_sel
            q_sel[idx] = chunk.q_sel
            a_star[idx] = chunk.a_star
            forbidden_sel[idx] = chunk.forbidden_sel
            s_score[idx] = chunk.s_score
            m_score[idx] = chunk.m_score
            q_score[idx] = chunk.q_score
        shas.append(chunk.selection_freeze_sha256)
        logical_steps += int(chunk.telemetry.get("logical_branch_steps", 0))
        exec_steps += int(chunk.telemetry.get("executed_sim_steps", 0))
        wall_sec += float(chunk.telemetry.get("wall_sec", 0.0))

    if not np.all(seen):
        missing = int(np.sum(~seen))
        raise ValueError(f"incomplete scored root coverage: {missing} roots missing")

    return MergedScoredRoots(
        num_roots=N,
        legal_mask=legal_mask,
        b8_action=b8_action,
        s_sel=s_sel,
        m_sel=m_sel,
        q_sel=q_sel,
        a_star=a_star,
        forbidden_sel=forbidden_sel,
        s_score=s_score,
        m_score=m_score,
        q_score=q_score,
        chunk_sha256s=shas,
        total_logical_branch_steps=logical_steps,
        total_executed_sim_steps=exec_steps,
        total_scoring_wall_sec=round(wall_sec, 3),
    )


def _selection_freeze_digest(
    root_indices: np.ndarray,
    a_star: np.ndarray,
    forbidden_sel: np.ndarray,
    q_sel: np.ndarray,
    s_sel: np.ndarray,
    m_sel: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(root_indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a_star, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(forbidden_sel, dtype=np.uint8).tobytes())
    h.update(np.nan_to_num(q_sel, nan=-999.0).astype(np.float64).tobytes())
    h.update(np.nan_to_num(s_sel, nan=-999.0).astype(np.float64).tobytes())
    h.update(np.nan_to_num(m_sel, nan=-999.0).astype(np.float64).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class _BranchJob:
    local_pos: int
    store_root_idx: int
    root_uid: str
    action: int
    suffix_offset: int  # 0..3 within current phase
    suffix_global: int  # 0..3 selection, 4..7 scoring
    sim_rng_key: int
    opp_rng_key: int


def partition_roots_into_chunks(num_roots: int, num_chunks: int) -> list[np.ndarray]:
    """Deterministically partition ``0..num_roots-1`` into ``num_chunks`` contiguous slices."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    all_idx = np.arange(int(num_roots), dtype=np.int64)
    return [arr.astype(np.int64) for arr in np.array_split(all_idx, int(num_chunks))]


@torch.inference_mode()
def score_root_chunk(
    *,
    chunk_index: int,
    root_indices: Sequence[int] | np.ndarray,
    store: LosslessBoundedSnapshotStore,
    control: CTControl,
    b8_network: nn.Module,
    checkpoint_networks: Mapping[str, nn.Module],
    device: str | torch.device = "cpu",
    compaction_interval: int = 10,
    stop_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[str], None] | None = None,
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> ScoredRootChunk | None:
    """Score one root chunk across 4 selection suffixes, freeze a*/F_sel, then 4 scoring suffixes.

    Returns ``None`` cleanly if ``stop_requested()`` fires so no partial chunk is committed.
    """
    t0 = time.perf_counter()
    dev = torch.device("cuda:0" if str(device) == "cuda" else device)
    b8_eval = b8_network.to(dev).eval()
    idx_arr = np.asarray(root_indices, dtype=np.int64).reshape(-1)
    C = int(idx_arr.size)
    H = int(control.branch_horizon)

    if C == 0:
        empty_b = np.zeros((0, 6), dtype=bool)
        empty_i = np.zeros(0, dtype=np.int64)
        empty_r = np.zeros((0, 6, 4), dtype=np.float32)
        empty_f = np.zeros((0, 6), dtype=np.float64)
        sha = _selection_freeze_digest(empty_i, empty_i, empty_b, empty_f, empty_f, empty_f)
        return ScoredRootChunk(
            chunk_index=int(chunk_index),
            root_indices=idx_arr,
            legal_mask=empty_b,
            b8_action=empty_i,
            survive_sel_raw=empty_r,
            mass_sel_raw=empty_r,
            s_sel=empty_f,
            m_sel=empty_f,
            q_sel=empty_f,
            a_star=empty_i,
            forbidden_sel=empty_b,
            selection_freeze_sha256=sha,
            survive_score_raw=empty_r,
            mass_score_raw=empty_r,
            s_score=empty_f,
            m_score=empty_f,
            q_score=empty_f,
            telemetry={
                "chunk_index": int(chunk_index),
                "roots": 0,
                "logical_branch_steps": 0,
                "executed_sim_steps": 0,
                "wall_sec": 0.0,
            },
        )

    all_rosters = training_rosters(control.worlds)
    unique_rosters_by_id: dict[tuple[Any, ...], TrainingRoster] = {}
    for r in all_rosters:
        unique_rosters_by_id.setdefault(r.identity, r)

    obs = store.stacked_observations()
    legal_chunk = obs["legal_mask"][idx_arr].numpy().astype(bool)
    b8_act_chunk = np.array([store.metadata[int(r)].b8_action for r in idx_arr], dtype=np.int64)

    base_cfg = BatchSimConfig(
        num_envs=1,
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

    executed_sim_steps = 0
    logical_branch_steps = 0
    slabs_completed = 0
    last_log = time.perf_counter()

    def run_phase(
        phase_name: str, suffix_start: int, suffix_count: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        nonlocal executed_sim_steps, logical_branch_steps, slabs_completed, last_log
        surv_out = np.full((C, 6, suffix_count), np.nan, dtype=np.float32)
        mass_out = np.full((C, 6, suffix_count), np.nan, dtype=np.float32)

        # Group jobs by exact roster.identity for single-group homogeneous GPU slabs.
        jobs_by_roster: dict[tuple[Any, ...], list[_BranchJob]] = {}
        for local_pos, r_idx_val in enumerate(idx_arr.tolist()):
            r_idx = int(r_idx_val)
            meta = store.metadata[r_idx]
            roster = all_rosters[meta.world_id]
            legal_actions = np.flatnonzero(legal_chunk[local_pos]).tolist()
            for s_off in range(suffix_count):
                k_global = suffix_start + s_off
                # Domain-separated keys omit action_id -> exact CRN across legal actions!
                bkeys = derive_branch_keys(meta.root_uid, k_global)
                for act in legal_actions:
                    jobs_by_roster.setdefault(roster.identity, []).append(
                        _BranchJob(
                            local_pos=local_pos,
                            store_root_idx=r_idx,
                            root_uid=meta.root_uid,
                            action=int(act),
                            suffix_offset=s_off,
                            suffix_global=k_global,
                            sim_rng_key=bkeys.sim_rng_key,
                            opp_rng_key=bkeys.opp_rng_key,
                        )
                    )

        batch_cap = max(1, int(control.branch_batch_size))
        for r_identity, jobs in jobs_by_roster.items():
            roster_proto = unique_rosters_by_id[r_identity]
            for start in range(0, len(jobs), batch_cap):
                if stop_requested is not None and stop_requested():
                    return None
                slab_jobs = jobs[start : start + batch_cap]
                B = len(slab_jobs)
                if heartbeat is not None:
                    heartbeat(
                        f"score:chunk={chunk_index}:{phase_name}:slab={slabs_completed}:B={B}"
                    )

                slab_cfg = replace(base_cfg, num_envs=B)
                slab_sim = CTDeviceSim(
                    slab_cfg,
                    device=dev,
                    train_mode=False,
                    allow_respawn=True,
                    move_backend="indexed" if dev.type == "cuda" else "reference",
                )
                r2_cache.enable(slab_sim)
                slab_opp = CTOpponentRuntime(
                    slab_sim,
                    [roster_proto] * B,
                    b8_network=b8_eval,
                    checkpoint_networks=checkpoint_networks,
                )

                # Restore exact prepared root states and rekey ALL post-prepare RNG consumers.
                store.restore_batch_into([j.store_root_idx for j in slab_jobs], slab_sim, slab_opp)
                slab_sim.terminal_slots[:, 0] = True
                sim_keys_t = torch.tensor(
                    [j.sim_rng_key for j in slab_jobs], dtype=torch.long, device=dev
                )
                opp_keys_t = torch.tensor(
                    [j.opp_rng_key for j in slab_jobs], dtype=torch.long, device=dev
                )
                slab_sim.rng_key.copy_(sim_keys_t)
                slab_sim.rng_counter.zero_()
                slab_opp.rekey_branches(opp_keys_t)

                active_job_slots = torch.arange(B, dtype=torch.long, device=dev)
                survive_slab = torch.zeros(B, dtype=torch.float64, device=dev)
                mass_sum_slab = torch.zeros(B, dtype=torch.float64, device=dev)
                forced_step0 = torch.tensor(
                    [j.action for j in slab_jobs], dtype=torch.long, device=dev
                )

                for h in range(H):
                    if h == 0:
                        hero_act = forced_step0
                    else:
                        if (
                            compaction_interval > 0
                            and (h % compaction_interval == 0)
                            and not bool(slab_sim.alive[:, 0].all().item())
                        ):
                            keep = slab_sim.alive[:, 0].nonzero(as_tuple=False).flatten()
                            if keep.numel() == 0:
                                break
                            _compact_sim_in_place(slab_sim, keep)
                            slab_opp.compact_rows(keep)
                            active_job_slots = active_job_slots.index_select(0, keep)
                        slab_sim.prepare()
                        _, _, _, _, _, hero_act = observe_hero_bv8(slab_sim, b8_eval)

                    full_actions = slab_opp.actions(slab_sim, hero_act)
                    slab_sim.step(full_actions)
                    slab_opp.after_step(slab_sim)
                    executed_sim_steps += int(slab_sim.E)

                    alive_now = slab_sim.alive[:, 0]
                    step_mass = torch.where(
                        alive_now,
                        slab_sim.length[:, 0].to(dtype=torch.float64),
                        torch.zeros(slab_sim.E, dtype=torch.float64, device=dev),
                    )
                    mass_sum_slab.index_add_(0, active_job_slots, step_mass)
                    if h == H - 1:
                        survive_slab.index_copy_(
                            0, active_job_slots, alive_now.to(dtype=torch.float64)
                        )

                logical_branch_steps += B * H
                slabs_completed += 1
                surv_np = survive_slab.detach().cpu().numpy().astype(np.float32)
                mass_np = (mass_sum_slab / float(H)).detach().cpu().numpy().astype(np.float32)
                for b_pos, job in enumerate(slab_jobs):
                    surv_out[job.local_pos, job.action, job.suffix_offset] = surv_np[b_pos]
                    mass_out[job.local_pos, job.action, job.suffix_offset] = mass_np[b_pos]

                now = time.perf_counter()
                if log_progress is not None and (now - last_log >= 45.0):
                    last_log = now
                    log_progress(
                        {
                            "phase": f"scoring_chunk_{chunk_index}_{phase_name}",
                            "slabs_completed": slabs_completed,
                            "logical_branch_steps": logical_branch_steps,
                            "executed_sim_steps": executed_sim_steps,
                            "elapsed_sec": round(now - t0, 2),
                        }
                    )
        return surv_out, mass_out

    # PHASE 1: Selection suffixes k in {0, 1, 2, 3} -> freeze a*(s) and F_sel(s) FIRST.
    phase1 = run_phase("selection", 0, control.selection_suffixes)
    if phase1 is None:
        return None
    survive_sel_raw, mass_sel_raw = phase1
    s_sel = np.where(
        legal_chunk,
        np.nan_to_num(survive_sel_raw.astype(np.float64), nan=0.0).mean(axis=2),
        np.nan,
    )
    m_sel = np.where(
        legal_chunk,
        np.nan_to_num(mass_sel_raw.astype(np.float64), nan=0.0).mean(axis=2),
        np.nan,
    )
    q_sel, a_star, forbidden_sel = select_a_star_and_forbidden(
        s_sel=s_sel,
        m_sel=m_sel,
        legal_mask=legal_chunk,
        b8_actions=b8_act_chunk,
        forbidden_gap=control.forbidden_gap,
    )
    freeze_sha = _selection_freeze_digest(idx_arr, a_star, forbidden_sel, q_sel, s_sel, m_sel)
    freeze_monotonic_ns = time.monotonic_ns()

    # PHASE 2: Independent scoring suffixes k in {4, 5, 6, 7} (a* and F_sel already frozen!).
    scoring_start_ns = time.monotonic_ns()
    if scoring_start_ns < freeze_monotonic_ns:
        raise RuntimeError("monotonic clock inversion across selection freeze boundary")
    phase2 = run_phase("scoring", control.selection_suffixes, control.scoring_suffixes)
    if phase2 is None:
        return None
    survive_score_raw, mass_score_raw = phase2
    s_score = np.where(
        legal_chunk,
        np.nan_to_num(survive_score_raw.astype(np.float64), nan=0.0).mean(axis=2),
        np.nan,
    )
    m_score = np.where(
        legal_chunk,
        np.nan_to_num(mass_score_raw.astype(np.float64), nan=0.0).mean(axis=2),
        np.nan,
    )
    q_score = compute_q_from_survival_and_mass(s_score, m_score, legal_chunk)

    # Assert post-Phase-2 immutability of frozen selection artifacts.
    if _selection_freeze_digest(idx_arr, a_star, forbidden_sel, q_sel, s_sel, m_sel) != freeze_sha:
        raise RuntimeError("frozen selection labels mutated during scoring phase")

    wall_sec = time.perf_counter() - t0
    telemetry = {
        "chunk_index": int(chunk_index),
        "roots": C,
        "horizon": H,
        "rng_consumer_inventory": json.loads(json.dumps(list(RNG_CONSUMER_INVENTORY))),
        "selection_frozen_before_scoring": True,
        "selection_freeze_monotonic_ns": int(freeze_monotonic_ns),
        "scoring_start_monotonic_ns": int(scoring_start_ns),
        "logical_branch_steps": int(logical_branch_steps),
        "executed_sim_steps": int(executed_sim_steps),
        "compaction_step_savings_ratio": (
            round(1.0 - (executed_sim_steps / max(1, logical_branch_steps)), 4)
        ),
        "slabs_completed": int(slabs_completed),
        "wall_sec": round(wall_sec, 3),
    }
    return ScoredRootChunk(
        chunk_index=int(chunk_index),
        root_indices=idx_arr,
        legal_mask=legal_chunk,
        b8_action=b8_act_chunk,
        survive_sel_raw=survive_sel_raw,
        mass_sel_raw=mass_sel_raw,
        s_sel=s_sel,
        m_sel=m_sel,
        q_sel=q_sel,
        a_star=a_star,
        forbidden_sel=forbidden_sel,
        selection_freeze_sha256=freeze_sha,
        survive_score_raw=survive_score_raw,
        mass_score_raw=mass_score_raw,
        s_score=s_score,
        m_score=m_score,
        q_score=q_score,
        telemetry=telemetry,
    )
