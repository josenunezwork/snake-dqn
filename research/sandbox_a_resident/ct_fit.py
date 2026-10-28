"""Three-arm counterfactual distillation trainer and detached D100-D300 linear probe.

Implements Ruling §4:
- Target: ``pi_imp(a|s) proportional to pi_B8(a|s) * exp(Q_sel(s, a) / 0.15)``
  over canonical legal actions in float64 log-space.
- Forbidden set: ``F_sel(s) = {a legal : S_sel(s, a*) - S_sel(s, a) >= 0.50}``.
- Searched loss:
  ``D_KL(pi_imp(.|s) || pi_theta(.|s))`` plus
  ``sum_{a in F_sel(s)} max(0, z_theta(s,a) - z_theta(s,a*) + 0.25)``.
- Anchor loss: ``0.05 * D_KL(pi_B8(.|s) || pi_theta(.|s))`` on unsearched ordinary archive.
- Optimization: 2,000 minibatches of 256 searched train roots + 768 unsearched train
  anchor rows, Adam ``trunk_lr = 1e-5``, ``head_lr = 1e-4``, fixed seed, identical
  split/labels/minibatch/anchor stream across ``frozen_phi``, ``full_bv8``, and
  ``full_bv8_plus_topology``, with zero assessment leakage.
- Detached post-hoc train-only linear logistic probe for ``D100-D300`` danger
  separability AUC evaluation on held-out assessment worlds (never attached to trunk
  gradients during fitting).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from research.gpu_sim_spike.device_policy import _masked_argmax
from research.sandbox_a_resident.ct_branching import MergedScoredRoots
from research.sandbox_a_resident.ct_models import (
    CT_ARM_NAMES,
    CTPolicyArm,
    verify_zero_residual_init,
)
from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore
from research.sandbox_a_resident.ct_task import AnchorArchive, CTControl
from research.sandbox_a_resident.ct_topology import SparseTopologyBatch
from research.sandbox_a_resident.training_model import B8Actor, canonical_legal_mask
from src.model.ego2s_network import Ego2sNet

D100_D300_LOOKBACKS: frozenset[int] = frozenset({100, 140, 185, 235, 285})


def float64_masked_kl_from_logits(
    target_logits_64: torch.Tensor,
    student_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute ``D_KL(softmax(target_logits_64) || softmax(student_logits))`` over legal support.

    Promotes student logits to float64, sets illegal actions to ``-inf``, and uses
    ``torch.where`` so illegal entries contribute exact ``0.0`` without ``0 * inf``.
    """
    legal = canonical_legal_mask(legal_mask)
    tgt_masked = torch.where(legal, target_logits_64.to(dtype=torch.float64), float("-inf"))
    stu_masked = torch.where(legal, student_logits.to(dtype=torch.float64), float("-inf"))
    log_q_raw = F.log_softmax(tgt_masked, dim=-1)
    log_p_raw = F.log_softmax(stu_masked, dim=-1)
    zeros = torch.zeros_like(log_q_raw)
    log_q = torch.where(legal, log_q_raw, zeros)
    log_p = torch.where(legal, log_p_raw, zeros)
    q = torch.where(legal, log_q.exp(), zeros)
    kl = torch.where(legal, q * (log_q - log_p), zeros).sum(dim=-1)
    return kl.to(dtype=student_logits.dtype)


def forbidden_margin_loss(
    student_logits: torch.Tensor,
    a_star: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    margin: float = 0.25,
) -> torch.Tensor:
    """Compute ``sum_{a in F_sel(s)} max(0, z_theta(s, a) - z_theta(s, a*) + margin)``."""
    z_star = student_logits.gather(1, a_star.long()[:, None])
    hinge = F.relu(student_logits - z_star + float(margin))
    return torch.where(forbidden_mask.bool(), hinge, torch.zeros_like(hinge)).sum(dim=-1)


@dataclass
class TrainFitDataset:
    """Leak-free training partition tensors (physically excludes assessment worlds & k=4..7)."""

    train_root_store_indices: np.ndarray  # (N_s,) int64
    searched_local: torch.Tensor  # (N_s, 7, 31, 31) uint8
    searched_glob: torch.Tensor  # (N_s, 4, 37, 37) uint8
    searched_scalars: torch.Tensor  # (N_s, 60) float32
    searched_legal_mask: torch.Tensor  # (N_s, 6) bool
    searched_z_b8: torch.Tensor  # (N_s, 6) float32
    searched_q_sel: torch.Tensor  # (N_s, 6) float64 (0.0 off legal)
    searched_a_star: torch.Tensor  # (N_s,) int64
    searched_forbidden_sel: torch.Tensor  # (N_s, 6) bool
    searched_is_d100_d300: np.ndarray  # (N_s,) bool
    searched_is_ordinary: np.ndarray  # (N_s,) bool
    searched_topology: SparseTopologyBatch

    train_anchor_indices: np.ndarray  # (N_a,) int64
    anchor_local: torch.Tensor  # (N_a, 7, 31, 31) uint8
    anchor_glob: torch.Tensor  # (N_a, 4, 37, 37) uint8
    anchor_scalars: torch.Tensor  # (N_a, 60) float32
    anchor_legal_mask: torch.Tensor  # (N_a, 6) bool
    anchor_z_b8: torch.Tensor  # (N_a, 6) float32
    anchor_topology: SparseTopologyBatch


def _slice_sparse_topology(batch: SparseTopologyBatch, indices: np.ndarray) -> SparseTopologyBatch:
    return batch.slice_rows(indices)


@torch.no_grad()
def _compute_reference_logits_batched(
    ref_actor: B8Actor,
    local: torch.Tensor,
    glob: torch.Tensor,
    scalars: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    N = int(local.shape[0])
    if N == 0:
        return torch.empty((0, 6), dtype=torch.float32)
    ref_dev = ref_actor.to(device).eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        z = ref_dev(
            local[start:end].to(device),
            glob[start:end].to(device),
            scalars[start:end].to(device),
        )
        chunks.append(z.detach().cpu().float())
    return torch.cat(chunks, dim=0)


def build_train_fit_dataset(
    *,
    store: LosslessBoundedSnapshotStore,
    scored: MergedScoredRoots,
    anchors: AnchorArchive,
    world_partitions: Sequence[str],
    b8_model: Ego2sNet,
    device: torch.device,
) -> TrainFitDataset:
    """Extract strictly train-partition data; verify zero world overlap with assessment."""
    assess_worlds = {w for w, part in enumerate(world_partitions) if part == "assessment"}
    train_root_idxs: list[int] = []
    is_d100_d300_list: list[bool] = []
    is_ord_list: list[bool] = []

    for r_idx, meta in enumerate(store.metadata):
        if meta.partition == "train":
            if meta.world_id in assess_worlds:
                raise RuntimeError(
                    f"partition leak: train root {meta.root_uid} in assessment world"
                )
            train_root_idxs.append(r_idx)
            is_d100_d300_list.append(
                meta.kind == "death_lookback" and meta.lookback in D100_D300_LOOKBACKS
            )
            is_ord_list.append(meta.kind == "ordinary")
        elif meta.partition != "assessment":
            raise ValueError(f"unknown root partition {meta.partition!r}")

    train_anc_idxs = anchors.train_indices().astype(np.int64)
    if train_anc_idxs.size > 0:
        leaked_anc = set(anchors.world_id[train_anc_idxs].tolist()) & assess_worlds
        if leaked_anc:
            raise RuntimeError(
                f"partition leak: train anchors intersect assessment worlds {leaked_anc}"
            )

    t_roots = np.asarray(train_root_idxs, dtype=np.int64)
    obs = store.stacked_observations()
    store_topo = store.finalize_topology()
    ref_actor = B8Actor(b8_model=b8_model, temperature=1.0).eval()

    if t_roots.size > 0:
        t_roots_t = torch.from_numpy(t_roots)
        s_loc = obs["local"].index_select(0, t_roots_t)
        s_glb = obs["glob"].index_select(0, t_roots_t)
        s_scl = obs["scalars"].index_select(0, t_roots_t)
        s_leg = canonical_legal_mask(obs["legal_mask"].index_select(0, t_roots_t))
        s_zb8 = _compute_reference_logits_batched(ref_actor, s_loc, s_glb, s_scl, device=device)
        q_sel_sub = np.nan_to_num(scored.q_sel[t_roots], nan=0.0)
        s_qsel = torch.from_numpy(q_sel_sub.astype(np.float64))
        s_astar = torch.from_numpy(scored.a_star[t_roots].astype(np.int64))
        s_forb = torch.from_numpy(scored.forbidden_sel[t_roots].astype(bool)) & s_leg
    else:
        s_loc = torch.empty((0, 7, 31, 31), dtype=torch.uint8)
        s_glb = torch.empty((0, 4, 37, 37), dtype=torch.uint8)
        s_scl = torch.empty((0, 60), dtype=torch.float32)
        s_leg = torch.empty((0, 6), dtype=torch.bool)
        s_zb8 = torch.empty((0, 6), dtype=torch.float32)
        s_qsel = torch.empty((0, 6), dtype=torch.float64)
        s_astar = torch.empty((0,), dtype=torch.long)
        s_forb = torch.empty((0, 6), dtype=torch.bool)

    if train_anc_idxs.size > 0:
        t_anc_t = torch.from_numpy(train_anc_idxs)
        a_loc = anchors.local.index_select(0, t_anc_t)
        a_glb = anchors.glob.index_select(0, t_anc_t)
        a_scl = anchors.scalars.index_select(0, t_anc_t)
        a_leg = canonical_legal_mask(anchors.legal_mask.index_select(0, t_anc_t))
        a_zb8 = _compute_reference_logits_batched(ref_actor, a_loc, a_glb, a_scl, device=device)
    else:
        a_loc = torch.empty((0, 7, 31, 31), dtype=torch.uint8)
        a_glb = torch.empty((0, 4, 37, 37), dtype=torch.uint8)
        a_scl = torch.empty((0, 60), dtype=torch.float32)
        a_leg = torch.empty((0, 6), dtype=torch.bool)
        a_zb8 = torch.empty((0, 6), dtype=torch.float32)

    return TrainFitDataset(
        train_root_store_indices=t_roots,
        searched_local=s_loc,
        searched_glob=s_glb,
        searched_scalars=s_scl,
        searched_legal_mask=s_leg,
        searched_z_b8=s_zb8,
        searched_q_sel=s_qsel,
        searched_a_star=s_astar,
        searched_forbidden_sel=s_forb,
        searched_is_d100_d300=np.asarray(is_d100_d300_list, dtype=bool),
        searched_is_ordinary=np.asarray(is_ord_list, dtype=bool),
        searched_topology=_slice_sparse_topology(store_topo, t_roots),
        train_anchor_indices=train_anc_idxs,
        anchor_local=a_loc,
        anchor_glob=a_glb,
        anchor_scalars=a_scl,
        anchor_legal_mask=a_leg,
        anchor_z_b8=a_zb8,
        anchor_topology=_slice_sparse_topology(anchors.topology, train_anc_idxs),
    )


def build_shared_minibatch_schedule(
    *,
    num_train_searched: int,
    num_train_anchors: int,
    minibatches: int,
    searched_batch_size: int,
    anchor_batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate identical deterministic ``(minibatches, B_s)`` and ``(minibatches, B_a)`` arrays."""
    if num_train_searched < 1 or num_train_anchors < 1:
        raise ValueError("training fit requires non-empty searched and anchor train pools")
    rng = np.random.default_rng(int(seed))

    def epoch_stream(pool_size: int, total_needed: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        collected = 0
        while collected < total_needed:
            perm = rng.permutation(pool_size).astype(np.int64)
            chunks.append(perm)
            collected += pool_size
        return np.concatenate(chunks)[:total_needed]

    s_flat = epoch_stream(num_train_searched, minibatches * searched_batch_size)
    a_flat = epoch_stream(num_train_anchors, minibatches * anchor_batch_size)
    return (
        s_flat.reshape(minibatches, searched_batch_size),
        a_flat.reshape(minibatches, anchor_batch_size),
    )


@dataclass
class DetachedLinearDangerProbe:
    """Standardized L2 logistic regression fit strictly on detached train representations."""

    mean: np.ndarray  # (D,) float64
    std: np.ndarray  # (D,) float64
    weight: np.ndarray  # (D,) float64
    bias: float
    train_pos_count: int
    train_neg_count: int
    fitted: bool

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted or features.shape[0] == 0:
            return np.zeros(features.shape[0], dtype=np.float64)
        z = (features.astype(np.float64) - self.mean[None, :]) / self.std[None, :]
        return z @ self.weight + self.bias


def fit_detached_danger_probe(
    features: np.ndarray,
    is_pos: np.ndarray,
    is_neg: np.ndarray,
    *,
    seed: int = 20261028,
    l2_reg: float = 1e-2,
    steps: int = 300,
) -> DetachedLinearDangerProbe:
    """Fit L2-regularized logistic linear probe on detached train D100-D300 vs ordinary features."""
    mask = is_pos | is_neg
    pos_n = int(np.sum(is_pos))
    neg_n = int(np.sum(is_neg))
    D = int(features.shape[1]) if features.ndim == 2 else 1
    if pos_n < 1 or neg_n < 1:
        return DetachedLinearDangerProbe(
            mean=np.zeros(D, dtype=np.float64),
            std=np.ones(D, dtype=np.float64),
            weight=np.zeros(D, dtype=np.float64),
            bias=0.0,
            train_pos_count=pos_n,
            train_neg_count=neg_n,
            fitted=False,
        )
    X_raw = features[mask].astype(np.float64)
    y_np = is_pos[mask].astype(np.float64)
    mean = X_raw.mean(axis=0)
    std = np.maximum(X_raw.std(axis=0), 1e-6)
    X_norm = (X_raw - mean[None, :]) / std[None, :]

    torch.manual_seed(int(seed))
    X_t = torch.from_numpy(X_norm)
    y_t = torch.from_numpy(y_np)
    # Class-balanced weights so imbalanced root counts do not bias hyperplane orientation.
    pos_weight = torch.tensor(float(neg_n) / float(max(1, pos_n)), dtype=torch.float64)
    w = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = X_t @ w + b
        bce = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        reg = 0.5 * float(l2_reg) * (w * w).mean()
        loss = bce + reg
        loss.backward()
        return loss

    opt.step(closure)
    return DetachedLinearDangerProbe(
        mean=mean,
        std=std,
        weight=w.detach().cpu().numpy().astype(np.float64),
        bias=float(b.detach().cpu().item()),
        train_pos_count=pos_n,
        train_neg_count=neg_n,
        fitted=True,
    )


@dataclass
class ArmEvaluationPredictions:
    """Arm training telemetry and held-out / train predictions for gate evaluation."""

    arm_name: str
    root_logits: np.ndarray  # (N_roots, 6) float32
    root_greedy_actions: np.ndarray  # (N_roots,) int64
    root_b8_neg_log_prob: np.ndarray  # (N_roots,) float64 (-log pi_theta(a_B8|s))
    root_danger_probe_scores: np.ndarray  # (N_roots,) float64
    danger_probe_fitted: bool
    assessment_anchor_greedy_actions: np.ndarray  # (N_assess_anc,) int64
    train_telemetry: dict[str, Any]


@torch.no_grad()
def _evaluate_arm_predictions(
    arm: CTPolicyArm,
    *,
    store: LosslessBoundedSnapshotStore,
    anchors: AnchorArchive,
    train_fit: TrainFitDataset,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    arm_eval = arm.to(device).eval()
    needs_topo = arm.arm_name == "full_bv8_plus_topology"

    # 1. Extract detached train features to fit the post-hoc D100-D300 linear probe.
    N_train_s = int(train_fit.searched_local.shape[0])
    train_reps: list[np.ndarray] = []
    for start in range(0, N_train_s, batch_size):
        end = min(N_train_s, start + batch_size)
        tp, ts = (
            train_fit.searched_topology.unpack_rows(np.arange(start, end), device=device)
            if needs_topo
            else (None, None)
        )
        _, rep = arm_eval.forward_with_features(
            train_fit.searched_local[start:end].to(device),
            train_fit.searched_glob[start:end].to(device),
            train_fit.searched_scalars[start:end].to(device),
            topo_planes=tp,
            topo_summaries=ts,
        )
        train_reps.append(rep.detach().cpu().numpy().astype(np.float32))
    train_rep_mat = (
        np.concatenate(train_reps, axis=0) if train_reps else np.zeros((0, 256), dtype=np.float32)
    )
    probe = fit_detached_danger_probe(
        train_rep_mat,
        train_fit.searched_is_d100_d300,
        train_fit.searched_is_ordinary,
        seed=seed,
    )

    # 2. Evaluate all store roots (both train & assessment, without updating any parameter).
    obs = store.stacked_observations()
    store_topo = store.finalize_topology()
    N_roots = len(store)
    root_logits_list: list[np.ndarray] = []
    root_greedy_list: list[np.ndarray] = []
    root_neg_logp_b8_list: list[np.ndarray] = []
    root_probe_list: list[np.ndarray] = []

    for start in range(0, N_roots, batch_size):
        end = min(N_roots, start + batch_size)
        tp, ts = (
            store_topo.unpack_rows(np.arange(start, end), device=device)
            if needs_topo
            else (None, None)
        )
        loc_b = obs["local"][start:end].to(device)
        glb_b = obs["glob"][start:end].to(device)
        scl_b = obs["scalars"][start:end].to(device)
        leg_b = canonical_legal_mask(obs["legal_mask"][start:end].to(device))
        logits_b, rep_b = arm_eval.forward_with_features(
            loc_b, glb_b, scl_b, topo_planes=tp, topo_summaries=ts
        )
        greedy_b = _masked_argmax(logits_b, leg_b)
        logits64 = torch.where(leg_b, logits_b.to(torch.float64), float("-inf"))
        logp64 = F.log_softmax(logits64, dim=-1)
        b8_acts_b = torch.tensor(
            [store.metadata[i].b8_action for i in range(start, end)],
            dtype=torch.long,
            device=device,
        )
        neg_lp = -logp64.gather(1, b8_acts_b[:, None]).squeeze(1)

        root_logits_list.append(logits_b.detach().cpu().numpy().astype(np.float32))
        root_greedy_list.append(greedy_b.detach().cpu().numpy().astype(np.int64))
        root_neg_logp_b8_list.append(neg_lp.detach().cpu().numpy().astype(np.float64))
        root_probe_list.append(probe.score(rep_b.detach().cpu().numpy()))

    root_logits = (
        np.concatenate(root_logits_list, axis=0)
        if root_logits_list
        else np.zeros((0, 6), dtype=np.float32)
    )
    root_greedy = (
        np.concatenate(root_greedy_list, axis=0)
        if root_greedy_list
        else np.zeros(0, dtype=np.int64)
    )
    root_neg_logp = (
        np.concatenate(root_neg_logp_b8_list, axis=0)
        if root_neg_logp_b8_list
        else np.zeros(0, dtype=np.float64)
    )
    root_probe_scores = (
        np.concatenate(root_probe_list, axis=0)
        if root_probe_list
        else np.zeros(0, dtype=np.float64)
    )

    # 3. Evaluate held-out assessment unsearched anchor archive greedy actions for Gate (d).
    assess_anc_idx = anchors.assessment_indices().astype(np.int64)
    assess_greedy_list: list[np.ndarray] = []
    for start in range(0, int(assess_anc_idx.size), batch_size):
        end = min(int(assess_anc_idx.size), start + batch_size)
        sub_idx = assess_anc_idx[start:end]
        sub_idx_t = torch.from_numpy(sub_idx)
        tp, ts = (
            anchors.topology.unpack_rows(sub_idx, device=device) if needs_topo else (None, None)
        )
        logits_a = arm_eval(
            anchors.local.index_select(0, sub_idx_t).to(device),
            anchors.glob.index_select(0, sub_idx_t).to(device),
            anchors.scalars.index_select(0, sub_idx_t).to(device),
            topo_planes=tp,
            topo_summaries=ts,
        )
        leg_a = canonical_legal_mask(anchors.legal_mask.index_select(0, sub_idx_t).to(device))
        greedy_a = _masked_argmax(logits_a, leg_a)
        assess_greedy_list.append(greedy_a.detach().cpu().numpy().astype(np.int64))

    assess_anc_greedy = (
        np.concatenate(assess_greedy_list, axis=0)
        if assess_greedy_list
        else np.zeros(0, dtype=np.int64)
    )
    return (
        root_logits,
        root_greedy,
        root_neg_logp,
        root_probe_scores,
        probe.fitted,
        assess_anc_greedy,
    )


def fit_all_three_arms(
    *,
    control: CTControl,
    store: LosslessBoundedSnapshotStore,
    scored: MergedScoredRoots,
    anchors: AnchorArchive,
    world_partitions: Sequence[str],
    b8_model: Ego2sNet,
    device: str | torch.device = "cpu",
    stop_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[str], None] | None = None,
    log_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = 45.0,
) -> tuple[dict[str, ArmEvaluationPredictions], dict[str, Any]] | None:
    """Train all 3 C/T arms on identical minibatch streams and evaluate predictions."""
    dev = torch.device("cuda:0" if str(device) == "cuda" else device)
    zero_init_check = verify_zero_residual_init(
        b8_model,
        grid_h=store.grid_h,
        grid_w=store.grid_w,
        seed=control.fit_seed,
        device=dev,
    )
    train_fit = build_train_fit_dataset(
        store=store,
        scored=scored,
        anchors=anchors,
        world_partitions=world_partitions,
        b8_model=b8_model,
        device=dev,
    )
    s_sched, a_sched = build_shared_minibatch_schedule(
        num_train_searched=int(train_fit.searched_local.shape[0]),
        num_train_anchors=int(train_fit.anchor_local.shape[0]),
        minibatches=control.minibatches,
        searched_batch_size=control.searched_batch_size,
        anchor_batch_size=control.anchor_batch_size,
        seed=control.fit_seed,
    )

    # Precompute target_logits_64 = z_B8_64 + Q_sel_64 / alpha once for all arms.
    searched_target_logits_64 = train_fit.searched_z_b8.to(
        dtype=torch.float64
    ) + train_fit.searched_q_sel / float(control.alpha)
    anchor_target_logits_64 = train_fit.anchor_z_b8.to(dtype=torch.float64)

    results: dict[str, ArmEvaluationPredictions] = {}
    last_log = time.perf_counter()

    for arm_name in CT_ARM_NAMES:
        t_arm0 = time.perf_counter()
        torch.manual_seed(int(control.fit_seed))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(int(control.fit_seed))
        arm = CTPolicyArm(b8_model, arm_name).to(dev).train()
        optimizer = arm.build_optimizer(trunk_lr=control.trunk_lr, head_lr=control.head_lr)
        needs_topo = arm_name == "full_bv8_plus_topology"

        initial_losses: dict[str, float] = {}
        final_window_kl: list[float] = []
        final_window_margin: list[float] = []
        final_window_anchor: list[float] = []

        for step in range(control.minibatches):
            if stop_requested is not None and stop_requested():
                return None
            if heartbeat is not None and (step % 25 == 0):
                heartbeat(f"fit:{arm_name}:step={step}/{control.minibatches}")

            s_idx = s_sched[step]
            a_idx = a_sched[step]
            s_idx_t = torch.from_numpy(s_idx)
            a_idx_t = torch.from_numpy(a_idx)

            s_tp, s_ts = (
                train_fit.searched_topology.unpack_rows(s_idx, device=dev)
                if needs_topo
                else (None, None)
            )
            a_tp, a_ts = (
                train_fit.anchor_topology.unpack_rows(a_idx, device=dev)
                if needs_topo
                else (None, None)
            )

            optimizer.zero_grad(set_to_none=True)
            z_s = arm(
                train_fit.searched_local.index_select(0, s_idx_t).to(dev, non_blocking=True),
                train_fit.searched_glob.index_select(0, s_idx_t).to(dev, non_blocking=True),
                train_fit.searched_scalars.index_select(0, s_idx_t).to(dev, non_blocking=True),
                topo_planes=s_tp,
                topo_summaries=s_ts,
            )
            leg_s = train_fit.searched_legal_mask.index_select(0, s_idx_t).to(
                dev, non_blocking=True
            )
            tgt_s = searched_target_logits_64.index_select(0, s_idx_t).to(dev, non_blocking=True)
            astar_s = train_fit.searched_a_star.index_select(0, s_idx_t).to(dev, non_blocking=True)
            forb_s = train_fit.searched_forbidden_sel.index_select(0, s_idx_t).to(
                dev, non_blocking=True
            )

            kl_s = float64_masked_kl_from_logits(tgt_s, z_s, leg_s).mean()
            m_s = forbidden_margin_loss(z_s, astar_s, forb_s, margin=control.logit_margin).mean()

            z_a = arm(
                train_fit.anchor_local.index_select(0, a_idx_t).to(dev, non_blocking=True),
                train_fit.anchor_glob.index_select(0, a_idx_t).to(dev, non_blocking=True),
                train_fit.anchor_scalars.index_select(0, a_idx_t).to(dev, non_blocking=True),
                topo_planes=a_tp,
                topo_summaries=a_ts,
            )
            leg_a = train_fit.anchor_legal_mask.index_select(0, a_idx_t).to(dev, non_blocking=True)
            tgt_a = anchor_target_logits_64.index_select(0, a_idx_t).to(dev, non_blocking=True)
            kl_a = float64_masked_kl_from_logits(tgt_a, z_a, leg_a).mean()
            anc_loss = float(control.reference_kl_beta) * kl_a

            loss = kl_s + m_s + anc_loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite C/T fit loss on arm={arm_name} step={step}")
            loss.backward()
            optimizer.step()

            kl_s_val = float(kl_s.detach().cpu().item())
            m_s_val = float(m_s.detach().cpu().item())
            anc_val = float(anc_loss.detach().cpu().item())
            if step == 0:
                initial_losses = {
                    "searched_kl": round(kl_s_val, 6),
                    "forbidden_margin": round(m_s_val, 6),
                    "weighted_anchor_kl": round(anc_val, 6),
                    "total_loss": round(kl_s_val + m_s_val + anc_val, 6),
                }
            if step >= max(0, control.minibatches - 50):
                final_window_kl.append(kl_s_val)
                final_window_margin.append(m_s_val)
                final_window_anchor.append(anc_val)

            now = time.perf_counter()
            if log_progress is not None and (
                now - last_log >= progress_interval_seconds or step + 1 == control.minibatches
            ):
                last_log = now
                log_progress(
                    {
                        "phase": f"fit_{arm_name}",
                        "step": step + 1,
                        "minibatches": control.minibatches,
                        "searched_kl": round(kl_s_val, 5),
                        "forbidden_margin": round(m_s_val, 5),
                        "weighted_anchor_kl": round(anc_val, 6),
                        "arm_elapsed_sec": round(now - t_arm0, 2),
                    }
                )

        if stop_requested is not None and stop_requested():
            return None

        (
            root_logits,
            root_greedy,
            root_neg_logp,
            root_probe_scores,
            probe_fitted,
            assess_anc_greedy,
        ) = _evaluate_arm_predictions(
            arm,
            store=store,
            anchors=anchors,
            train_fit=train_fit,
            seed=control.fit_seed,
            device=dev,
        )

        arm_wall = time.perf_counter() - t_arm0
        results[arm_name] = ArmEvaluationPredictions(
            arm_name=arm_name,
            root_logits=root_logits,
            root_greedy_actions=root_greedy,
            root_b8_neg_log_prob=root_neg_logp,
            root_danger_probe_scores=root_probe_scores,
            danger_probe_fitted=probe_fitted,
            assessment_anchor_greedy_actions=assess_anc_greedy,
            train_telemetry={
                "minibatches": int(control.minibatches),
                "searched_batch_size": int(control.searched_batch_size),
                "anchor_batch_size": int(control.anchor_batch_size),
                "trunk_lr": float(control.trunk_lr),
                "head_lr": float(control.head_lr),
                "initial_step0_losses": initial_losses,
                "final50_mean_losses": {
                    "searched_kl": round(float(np.mean(final_window_kl)), 6),
                    "forbidden_margin": round(float(np.mean(final_window_margin)), 6),
                    "weighted_anchor_kl": round(float(np.mean(final_window_anchor)), 6),
                    "total_loss": round(
                        float(
                            np.mean(final_window_kl)
                            + np.mean(final_window_margin)
                            + np.mean(final_window_anchor)
                        ),
                        6,
                    ),
                },
                "wall_sec": round(arm_wall, 3),
            },
        )

    shared_meta = {
        "zero_residual_init_verification": zero_init_check,
        "train_searched_roots": int(train_fit.searched_local.shape[0]),
        "train_anchor_rows": int(train_fit.anchor_local.shape[0]),
        "shared_schedule_seed": int(control.fit_seed),
        "assessment_tuned_lambda_grid": False,
    }
    return results, shared_meta
