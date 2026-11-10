"""Step B1b online GPU SITL training loop (`B1-A100-Pilot`, `t34-gpu-sitl-train-20261109-v1`).

Implements the pre-registered online SITL training recipe from
``docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md`` §3.1, §3.3, §3.4:
- ``E = 256`` ``CTDeviceSim`` Watch6-v2 worlds (`t34-gpu-sitl-train-20261109-v1`) with
  ``5x frp3-s12 + v8`` opponents.
- Student ``Ego2sNet`` initialized from ``b8-s3`` with ``value.weight`` and ``value.bias``
  frozen via :func:`freeze_ego2s_value_head`; student acts on-sim (`a_theta(s)`).
- Online labeler: :func:`batched_open_stop_h32` evaluating all legal actions over ``H = 32``
  under ``suffix 0``.
- Strict partition by ``bare_is_safe``:
  - Mined rescue ring buffer ``D_rescue^train``: ``a_b8(s) not in A_safe(s)`` and
    ``A_safe(s) != empty``.
  - Ordinary safe buffer ``D_ordinary^train``: ``a_b8(s) in A_safe(s)``.
- Balanced ``B = 256`` minibatch: ``128`` mined rescue + ``128`` ordinary (`64` uniform +
  ``64`` hard-negative where ``a_theta(s) != a_b8(s)``, prioritizing
  ``a_theta(s) not in A_safe(s)``).
- Pure hinge sum objective (`delta_rescue = 0.3002`, `delta_ordinary = 0.0632`, no TD loss).
- ``512`` unique rescue warmup, ``>= 5,000`` unique mined rescue roots, ``1,000`` updates.
- Every-200-update fit-drift & selectivity tripwires on ``t34-gpu-sitl-probe32-20261109-v1``
  plus ``100k``-decision window fault checks (`0` non-finite loss/Q/grad, `0` illegal actions,
  `< 90.0%` one-action share).
- End-of-pilot training-set aliasing diagnostic emitting ``STOP_EGO2S_BV8_INPUTS_INSUFFICIENT``
  when training rescue agreement ``< 80.0%`` at training ordinary flips ``<= 2.0%``.
- Atomic intra-unit checkpointing with ``.sha256`` and ``.complete`` markers and deterministic
  resume (`resumed == uninterrupted`).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.gpu_sim_spike import r2_cache
from research.gpu_sitl.open_stop_gpu import (
    _VECTOR_RUNTIME_ENV_FIELDS,
    ORDINARY_MARGIN_DELTA_DEFAULT,
    RESCUE_MARGIN_DELTA_DEFAULT,
    batched_open_stop_h32,
    compact_sim_and_policy,
    concat_sims_and_policies,
    freeze_ego2s_value_head,
    verify_cuda_native_extensions,
)
from research.gpu_sitl.preflight import (
    MATURE_EXTENSION_SPECS,
    build_extension_worlds,
    resolve_product_checkpoints,
    snapshot_worlds_to_device,
)
from research.gpu_sitl.probe_bank import (
    build_watch6_ct_device_bundle,
    evaluate_probe_bank_tripwires,
    extract_hero_root_observations,
    harvest_probe32_bank,
    load_probe32_bank,
    sha256_file,
    verify_file_sha256,
    write_atomic_with_sha256,
)
from research.gpu_sitl.seed_check import (
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    train_seeds,
)
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.watch_replay.open_stop import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
)
from src.model.ego2s_network import (
    Ego2sNet,
    load_ego2s_checkpoint,
    save_ego2s_checkpoint,
)
from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

REGISTERED_ENVS = 256
REGISTERED_TOTAL_UPDATES = 1000
REGISTERED_WARMUP_RESCUE_ROOTS = 512
REGISTERED_MIN_UNIQUE_RESCUE_ROOTS = 5000
REGISTERED_RESCUE_BATCH_SIZE = 128
REGISTERED_ORDINARY_BATCH_SIZE = 128
REGISTERED_ORDINARY_UNIFORM_HALF = 64
REGISTERED_ORDINARY_HARD_NEG_HALF = 64
REGISTERED_TRIPWIRE_INTERVAL = 200
REGISTERED_DECISION_WINDOW_SIZE = 100_000
REGISTERED_MAX_SINGLE_ACTION_SHARE = 0.900
REGISTERED_DELIVERED_BOOST_MIN = 0.005
REGISTERED_DELIVERED_BOOST_MAX = 0.100

ALIASING_RESCUE_AGREEMENT_THRESHOLD = 0.800
ALIASING_ORDINARY_FLIP_THRESHOLD = 0.020
VERDICT_INPUTS_INSUFFICIENT = "STOP_EGO2S_BV8_INPUTS_INSUFFICIENT"
VERDICT_TRIPWIRE_STOP = "STOP_TRIPWIRE_BREACH"
VERDICT_PROCEED_TO_GATE_B1 = "PROCEED_TO_GATE_B1_EVALUATION"

DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_RESCUE_BUFFER_CAPACITY = 16384
DEFAULT_ORDINARY_BUFFER_CAPACITY = 32768
DEFAULT_HARD_NEG_PROBE_POOL = 512


class SitlRingBuffer:
    """CPU-backed ring buffer for mined rescue roots or ordinary safe states."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.size = 0
        self.write_ptr = 0
        self.total_added = 0
        self.local_u8 = torch.zeros((self.capacity, 7, 31, 31), dtype=torch.uint8)
        self.global_u8 = torch.zeros((self.capacity, 4, 37, 37), dtype=torch.uint8)
        self.scalars_f32 = torch.zeros((self.capacity, 60), dtype=torch.float32)
        self.legal_mask = torch.zeros((self.capacity, 6), dtype=torch.bool)
        self.safe_mask = torch.zeros((self.capacity, 6), dtype=torch.bool)
        self.unsafe_mask = torch.zeros((self.capacity, 6), dtype=torch.bool)
        self.bare_action = torch.zeros((self.capacity,), dtype=torch.long)
        self.a_star_action = torch.zeros((self.capacity,), dtype=torch.long)
        self.world_seed = torch.zeros((self.capacity,), dtype=torch.long)
        self.decision_frame = torch.zeros((self.capacity,), dtype=torch.long)
        # Priority tier for ordinary hard-negative tracking:
        # 2 = student action unsafe (a_theta not in A_safe)
        # 1 = student action differs from bare but is safe (a_theta != a_b8, a_theta in A_safe)
        # 0 = student action equals bare (a_theta == a_b8)
        self.hard_neg_tier = torch.zeros((self.capacity,), dtype=torch.int8)

    def add_single(
        self,
        *,
        local_u8: torch.Tensor,
        global_u8: torch.Tensor,
        scalars_f32: torch.Tensor,
        legal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        bare_action: int,
        a_star_action: int,
        world_seed: int,
        decision_frame: int,
        hard_neg_tier: int = 0,
    ) -> int:
        idx = int(self.write_ptr)
        self.local_u8[idx].copy_(local_u8.detach().cpu())
        self.global_u8[idx].copy_(global_u8.detach().cpu())
        self.scalars_f32[idx].copy_(scalars_f32.detach().cpu())
        self.legal_mask[idx].copy_(legal_mask.detach().cpu())
        self.safe_mask[idx].copy_(safe_mask.detach().cpu())
        self.unsafe_mask[idx].copy_(unsafe_mask.detach().cpu())
        self.bare_action[idx] = int(bare_action)
        self.a_star_action[idx] = int(a_star_action)
        self.world_seed[idx] = int(world_seed)
        self.decision_frame[idx] = int(decision_frame)
        self.hard_neg_tier[idx] = int(hard_neg_tier)
        self.write_ptr = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        self.total_added += 1
        return idx

    def state_dict(self) -> dict[str, Any]:
        n = int(self.size)
        return {
            "capacity": int(self.capacity),
            "size": n,
            "write_ptr": int(self.write_ptr),
            "total_added": int(self.total_added),
            "local_u8": self.local_u8[:n].clone(),
            "global_u8": self.global_u8[:n].clone(),
            "scalars_f32": self.scalars_f32[:n].clone(),
            "legal_mask": self.legal_mask[:n].clone(),
            "safe_mask": self.safe_mask[:n].clone(),
            "unsafe_mask": self.unsafe_mask[:n].clone(),
            "bare_action": self.bare_action[:n].clone(),
            "a_star_action": self.a_star_action[:n].clone(),
            "world_seed": self.world_seed[:n].clone(),
            "decision_frame": self.decision_frame[:n].clone(),
            "hard_neg_tier": self.hard_neg_tier[:n].clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError(f"Saved buffer size {n} exceeds capacity {self.capacity}")
        self.size = n
        self.write_ptr = int(state["write_ptr"])
        self.total_added = int(state["total_added"])
        if n > 0:
            self.local_u8[:n].copy_(state["local_u8"])
            self.global_u8[:n].copy_(state["global_u8"])
            self.scalars_f32[:n].copy_(state["scalars_f32"])
            self.legal_mask[:n].copy_(state["legal_mask"])
            self.safe_mask[:n].copy_(state["safe_mask"])
            self.unsafe_mask[:n].copy_(state["unsafe_mask"])
            self.bare_action[:n].copy_(state["bare_action"])
            self.a_star_action[:n].copy_(state["a_star_action"])
            self.world_seed[:n].copy_(state["world_seed"])
            self.decision_frame[:n].copy_(state["decision_frame"])
            self.hard_neg_tier[:n].copy_(state["hard_neg_tier"])

    def gather_batch(
        self, indices: Sequence[int] | np.ndarray, device: torch.device
    ) -> dict[str, torch.Tensor]:
        idx_t = torch.as_tensor(list(indices), dtype=torch.long)
        return {
            "local_u8": self.local_u8.index_select(0, idx_t).to(device),
            "global_u8": self.global_u8.index_select(0, idx_t).to(device),
            "scalars_f32": self.scalars_f32.index_select(0, idx_t).to(device),
            "legal_mask": self.legal_mask.index_select(0, idx_t).to(device),
            "safe_mask": self.safe_mask.index_select(0, idx_t).to(device),
            "unsafe_mask": self.unsafe_mask.index_select(0, idx_t).to(device),
            "bare_action": self.bare_action.index_select(0, idx_t).to(device),
            "a_star_action": self.a_star_action.index_select(0, idx_t).to(device),
        }


def compute_sitl_hinge_loss(
    student_net: Ego2sNet,
    rescue_batch: dict[str, torch.Tensor],
    ordinary_batch: dict[str, torch.Tensor],
    *,
    delta_rescue: float = RESCUE_MARGIN_DELTA_DEFAULT,
    delta_ordinary: float = ORDINARY_MARGIN_DELTA_DEFAULT,
) -> dict[str, torch.Tensor]:
    """Compute the registered pure hinge sum objective (no TD loss).

    Sums the rescue pairwise margin hinge ``delta_rescue`` and the ordinary DQfD
    large-margin hinge ``delta_ordinary``.
    """
    n_res = int(rescue_batch["local_u8"].shape[0])
    n_ord = int(ordinary_batch["local_u8"].shape[0])
    if n_res < 1 or n_ord < 1:
        raise ValueError(f"Rescue ({n_res}) and ordinary ({n_ord}) batches must be non-empty")

    # Concatenate for a single fused forward pass through Ego2sNet
    loc_all = torch.cat((rescue_batch["local_u8"], ordinary_batch["local_u8"]), dim=0)
    glb_all = torch.cat((rescue_batch["global_u8"], ordinary_batch["global_u8"]), dim=0)
    sca_all = torch.cat((rescue_batch["scalars_f32"], ordinary_batch["scalars_f32"]), dim=0)
    q_all = student_net(loc_all, glb_all, sca_all)
    q_res = q_all[:n_res]
    q_ord = q_all[n_res:]

    # 1. Rescue hinge term: max(0, delta_rescue - (Q(s, a*) - max_{a in legal \ safe} Q(s, a)))
    res_arange = torch.arange(n_res, device=q_res.device, dtype=torch.long)
    a_star = rescue_batch["a_star_action"].clamp(0, 5)
    unsafe_mask = rescue_batch["unsafe_mask"] & rescue_batch["legal_mask"]
    if not bool(unsafe_mask.any(dim=1).all().item()):
        raise RuntimeError("Every mined rescue root must have at least one legal unsafe action")
    q_unsafe_masked = q_res.masked_fill(~unsafe_mask, -torch.inf)
    q_best_unsafe, a_unsafe = q_unsafe_masked.max(dim=1)
    q_star = q_res[res_arange, a_star]
    res_margins = float(delta_rescue) - (q_star - q_best_unsafe)
    res_losses = torch.clamp(res_margins, min=0.0)
    loss_rescue = res_losses.mean()

    # 2. Ordinary DQfD hinge term:
    # max_{a in legal}(Q(s, a) + delta_ordinary * 1[a != a_b8]) - Q(s, a_b8)
    ord_arange = torch.arange(n_ord, device=q_ord.device, dtype=torch.long)
    a_b8 = ordinary_batch["bare_action"].clamp(0, 5)
    ord_legal = ordinary_batch["legal_mask"]
    if not bool(ord_legal[ord_arange, a_b8].all().item()):
        raise RuntimeError("Every ordinary state must have a_b8(s) in legal(s)")
    action_ids = torch.arange(6, device=q_ord.device, dtype=torch.long).unsqueeze(0)
    margin_bonus = torch.where(
        action_ids != a_b8.unsqueeze(1),
        torch.full_like(q_ord, float(delta_ordinary)),
        torch.zeros_like(q_ord),
    )
    q_ord_augmented = (q_ord + margin_bonus).masked_fill(~ord_legal, -torch.inf)
    q_ord_max = q_ord_augmented.max(dim=1).values
    q_ord_b8 = q_ord[ord_arange, a_b8]
    ord_losses = torch.clamp(q_ord_max - q_ord_b8, min=0.0)
    loss_ordinary = ord_losses.mean()

    total_loss = loss_rescue + loss_ordinary
    return {
        "total_loss": total_loss,
        "loss_rescue": loss_rescue.detach(),
        "loss_ordinary": loss_ordinary.detach(),
        "q_all_finite": torch.isfinite(q_all).all().detach(),
        "active_rescue_frac": (res_losses.detach() > 0).float().mean(),
        "active_ordinary_frac": (ord_losses.detach() > 0).float().mean(),
        "a_unsafe": a_unsafe.detach(),
    }


@torch.no_grad()
def sample_ordinary_minibatch_indices(
    ordinary_buf: SitlRingBuffer,
    student_net: Ego2sNet,
    rng: np.random.Generator,
    *,
    uniform_count: int = REGISTERED_ORDINARY_UNIFORM_HALF,
    hard_neg_count: int = REGISTERED_ORDINARY_HARD_NEG_HALF,
    probe_pool_size: int = DEFAULT_HARD_NEG_PROBE_POOL,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample ``uniform_count`` uniform ordinary states + ``hard_neg_count`` hard-negative
    ordinary states where ``a_theta(s) != a_b8(s)``, prioritizing ``a_theta(s) not in A_safe(s)``.
    """
    n = int(ordinary_buf.size)
    if n < 1:
        raise ValueError("ordinary_buf is empty")

    replace_u = n < int(uniform_count)
    uniform_idx = rng.choice(n, size=int(uniform_count), replace=replace_u).astype(np.int64)

    # Refresh hard_neg_tier on flagged rows + a random probe slice under current student_net
    flagged = np.flatnonzero(ordinary_buf.hard_neg_tier[:n].numpy() > 0)
    k_probe = min(n, max(int(probe_pool_size), int(hard_neg_count) * 2))
    random_probe = rng.choice(n, size=k_probe, replace=False).astype(np.int64)
    eval_indices = np.unique(np.concatenate([flagged, random_probe])).astype(np.int64)

    if len(eval_indices) > 0:
        was_training = student_net.training
        student_net.eval()
        try:
            idx_t = torch.as_tensor(eval_indices, dtype=torch.long)
            loc = ordinary_buf.local_u8.index_select(0, idx_t).to(device)
            glb = ordinary_buf.global_u8.index_select(0, idx_t).to(device)
            sca = ordinary_buf.scalars_f32.index_select(0, idx_t).to(device)
            leg = ordinary_buf.legal_mask.index_select(0, idx_t).to(device)
            saf = ordinary_buf.safe_mask.index_select(0, idx_t).to(device)
            b8_a = ordinary_buf.bare_action.index_select(0, idx_t).to(device).clamp(0, 5)

            q = student_net(loc, glb, sca)
            a_theta = q.masked_fill(~leg, -torch.inf).argmax(dim=1)
            row_i = torch.arange(len(eval_indices), device=device, dtype=torch.long)
            theta_safe = saf[row_i, a_theta]
            diff_b8 = a_theta != b8_a

            new_tier = torch.where(
                diff_b8 & (~theta_safe),
                torch.full_like(a_theta, 2, dtype=torch.int8),
                torch.where(
                    diff_b8,
                    torch.full_like(a_theta, 1, dtype=torch.int8),
                    torch.zeros_like(a_theta, dtype=torch.int8),
                ),
            ).cpu()
            ordinary_buf.hard_neg_tier[idx_t] = new_tier
        finally:
            if was_training:
                student_net.train()

    tiers = ordinary_buf.hard_neg_tier[:n].numpy()
    tier2_idx = np.flatnonzero(tiers == 2)
    tier1_idx = np.flatnonzero(tiers == 1)

    chosen_hn: list[int] = []
    n_tier2_used = 0
    n_tier1_used = 0
    n_uniform_fallback = 0

    if len(tier2_idx) > 0:
        take2 = min(int(hard_neg_count), len(tier2_idx))
        picked2 = rng.choice(tier2_idx, size=take2, replace=False).tolist()
        chosen_hn.extend(int(x) for x in picked2)
        n_tier2_used = take2

    rem = int(hard_neg_count) - len(chosen_hn)
    if rem > 0 and len(tier1_idx) > 0:
        take1 = min(rem, len(tier1_idx))
        picked1 = rng.choice(tier1_idx, size=take1, replace=False).tolist()
        chosen_hn.extend(int(x) for x in picked1)
        n_tier1_used = take1

    rem = int(hard_neg_count) - len(chosen_hn)
    if rem > 0:
        # If fewer than hard_neg_count distinct hard negatives exist, oversample existing hard
        # negatives if any exist (prioritizing tier 2 then tier 1), else fill from uniform ordinary.
        if len(tier2_idx) > 0:
            extra2 = rng.choice(tier2_idx, size=rem, replace=True).tolist()
            chosen_hn.extend(int(x) for x in extra2)
            n_tier2_used += rem
        elif len(tier1_idx) > 0:
            extra1 = rng.choice(tier1_idx, size=rem, replace=True).tolist()
            chosen_hn.extend(int(x) for x in extra1)
            n_tier1_used += rem
        else:
            extra_u = rng.choice(n, size=rem, replace=(n < rem)).tolist()
            chosen_hn.extend(int(x) for x in extra_u)
            n_uniform_fallback = rem

    hn_idx = np.asarray(chosen_hn, dtype=np.int64)
    all_ord_idx = np.concatenate([uniform_idx, hn_idx])
    stats = {
        "uniform_sampled": int(len(uniform_idx)),
        "hard_neg_unsafe_sampled": int(n_tier2_used),
        "hard_neg_safe_sampled": int(n_tier1_used),
        "hard_neg_uniform_fallback": int(n_uniform_fallback),
        "buffer_tier2_unsafe_available": int(len(tier2_idx)),
        "buffer_tier1_safe_available": int(len(tier1_idx)),
    }
    return all_ord_idx, stats


@torch.no_grad()
def evaluate_training_set_aliasing_diagnostic(
    student_net: Ego2sNet,
    rescue_buf: SitlRingBuffer,
    ordinary_buf: SitlRingBuffer,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Evaluate in-sample training-set fit quality on ``D_rescue^train`` and ``D_ordinary^train``
    per registration §3.1(3) and §3.4:

    If training-set rescue agreement (``argmax_{a in legal(s)} Q_theta(s, a) in A_safe(s)``)
    is ``< 80.0%`` while training-set ordinary flips
    (``argmax_{a in legal(s)} Q_theta(s, a) != a_b8(s)``) are ``<= 2.0%``, trigger
    ``STOP_EGO2S_BV8_INPUTS_INSUFFICIENT``.
    """
    was_training = student_net.training
    student_net.eval()
    try:
        n_res = int(rescue_buf.size)
        n_ord = int(ordinary_buf.size)
        if n_res < 1 or n_ord < 1:
            raise ValueError(
                f"Both rescue ({n_res}) and ordinary ({n_ord}) buffers must be non-empty"
            )

        res_in_safe_count = 0
        res_eq_astar_count = 0
        for start in range(0, n_res, int(batch_size)):
            end = min(n_res, start + int(batch_size))
            loc = rescue_buf.local_u8[start:end].to(device)
            glb = rescue_buf.global_u8[start:end].to(device)
            sca = rescue_buf.scalars_f32[start:end].to(device)
            leg = rescue_buf.legal_mask[start:end].to(device)
            saf = rescue_buf.safe_mask[start:end].to(device)
            ast = rescue_buf.a_star_action[start:end].to(device).clamp(0, 5)
            q = student_net(loc, glb, sca)
            a_theta = q.masked_fill(~leg, -torch.inf).argmax(dim=1)
            row_i = torch.arange(end - start, device=device, dtype=torch.long)
            res_in_safe_count += int(saf[row_i, a_theta].sum().item())
            res_eq_astar_count += int((a_theta == ast).sum().item())

        ord_flip_count = 0
        ord_unsafe_flip_count = 0
        for start in range(0, n_ord, int(batch_size)):
            end = min(n_ord, start + int(batch_size))
            loc = ordinary_buf.local_u8[start:end].to(device)
            glb = ordinary_buf.global_u8[start:end].to(device)
            sca = ordinary_buf.scalars_f32[start:end].to(device)
            leg = ordinary_buf.legal_mask[start:end].to(device)
            saf = ordinary_buf.safe_mask[start:end].to(device)
            b8_a = ordinary_buf.bare_action[start:end].to(device).clamp(0, 5)
            q = student_net(loc, glb, sca)
            a_theta = q.masked_fill(~leg, -torch.inf).argmax(dim=1)
            row_i = torch.arange(end - start, device=device, dtype=torch.long)
            flips = a_theta != b8_a
            ord_flip_count += int(flips.sum().item())
            ord_unsafe_flip_count += int((flips & (~saf[row_i, a_theta])).sum().item())

        rescue_agreement_rate = float(res_in_safe_count) / float(n_res)
        rescue_exact_a_star_rate = float(res_eq_astar_count) / float(n_res)
        ordinary_flip_rate = float(ord_flip_count) / float(n_ord)
        ordinary_unsafe_flip_rate = float(ord_unsafe_flip_count) / float(n_ord)

        inputs_insufficient = bool(
            rescue_agreement_rate < ALIASING_RESCUE_AGREEMENT_THRESHOLD
            and ordinary_flip_rate <= ALIASING_ORDINARY_FLIP_THRESHOLD
        )
        return {
            "rescue_buffer_evaluated_count": n_res,
            "ordinary_buffer_evaluated_count": n_ord,
            "train_rescue_safe_agreement_count": int(res_in_safe_count),
            "train_rescue_safe_agreement_rate": rescue_agreement_rate,
            "train_rescue_exact_a_star_count": int(res_eq_astar_count),
            "train_rescue_exact_a_star_rate": rescue_exact_a_star_rate,
            "train_ordinary_flip_count": int(ord_flip_count),
            "train_ordinary_flip_rate": ordinary_flip_rate,
            "train_ordinary_unsafe_flip_count": int(ord_unsafe_flip_count),
            "train_ordinary_unsafe_flip_rate": ordinary_unsafe_flip_rate,
            "rescue_agreement_threshold": ALIASING_RESCUE_AGREEMENT_THRESHOLD,
            "ordinary_flip_threshold": ALIASING_ORDINARY_FLIP_THRESHOLD,
            "stop_ego2s_bv8_inputs_insufficient": inputs_insufficient,
            "verdict": (
                VERDICT_INPUTS_INSUFFICIENT
                if inputs_insufficient
                else "TRAIN_FIT_NOT_ALIASED_OR_ORDINARY_UNCLAMPED"
            ),
        }
    finally:
        if was_training:
            student_net.train()


def required_unique_rescues_for_update(
    update_index_1based: int,
    *,
    warmup_rescue_roots: int,
    min_unique_rescue_roots: int,
    total_updates: int,
) -> int:
    """Return required cumulative unique mined rescue roots before ``update_index_1based``."""
    u = max(1, int(update_index_1based))
    if total_updates <= 1:
        return max(int(warmup_rescue_roots), int(min_unique_rescue_roots))
    span = max(0, int(min_unique_rescue_roots) - int(warmup_rescue_roots))
    return int(warmup_rescue_roots) + ((u - 1) * span) // (int(total_updates) - 1)


def _capture_sim_and_runtime_state(sim: Any, policy: Any) -> dict[str, Any]:
    sim_state: dict[str, Any] = {"E": int(sim.E)}
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            sim_state[name] = val.detach().cpu().clone()
    rt = policy.vector_runtime
    rt_state: dict[str, Any] = {
        "E": int(rt.E),
        "vector_prepared": bool(policy.vector_prepared),
    }
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        val = getattr(rt, name, None)
        if isinstance(val, torch.Tensor):
            rt_state[name] = val.detach().cpu().clone()
    return {"sim": sim_state, "vector_runtime": rt_state}


@torch.inference_mode()
def _restore_sim_and_runtime_state(sim: Any, policy: Any, state: dict[str, Any]) -> None:
    dev = sim.device
    sim_state = state["sim"]
    for name in _ENV_TENSOR_FIELDS:
        if name in sim_state and isinstance(getattr(sim, name, None), torch.Tensor):
            setattr(sim, name, sim_state[name].to(dev).clone())
    sim.E = int(sim_state["E"])
    r2_cache.bump(sim)

    rt = policy.vector_runtime
    rt_state = state["vector_runtime"]
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        if name in rt_state and isinstance(getattr(rt, name, None), torch.Tensor):
            setattr(rt, name, rt_state[name].to(dev).clone())
    rt.E = int(rt_state["E"])
    policy.vector_prepared = bool(rt_state["vector_prepared"])


def run_sitl_training(
    output_dir: Path | str,
    *,
    probe_bank_path: Path | str | None = None,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    device: str | torch.device = "cpu",
    num_envs: int = REGISTERED_ENVS,
    total_updates: int = REGISTERED_TOTAL_UPDATES,
    warmup_rescue_roots: int = REGISTERED_WARMUP_RESCUE_ROOTS,
    min_unique_rescue_roots: int = REGISTERED_MIN_UNIQUE_RESCUE_ROOTS,
    rescue_batch_size: int = REGISTERED_RESCUE_BATCH_SIZE,
    ordinary_batch_size: int = REGISTERED_ORDINARY_BATCH_SIZE,
    ordinary_uniform_half: int = REGISTERED_ORDINARY_UNIFORM_HALF,
    ordinary_hard_neg_half: int = REGISTERED_ORDINARY_HARD_NEG_HALF,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    delta_rescue: float = RESCUE_MARGIN_DELTA_DEFAULT,
    delta_ordinary: float = ORDINARY_MARGIN_DELTA_DEFAULT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    tripwire_interval: int = REGISTERED_TRIPWIRE_INTERVAL,
    decision_window_size: int = REGISTERED_DECISION_WINDOW_SIZE,
    checkpoint_every_updates: int = 100,
    checkpoint_every_seconds: float = 900.0,
    max_collection_steps: int = 200_000,
    seed: int = 20261109,
    cpu_fast_flood: bool = True,
    resume: bool = True,
    stop_after_updates: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the Step B1b online GPU SITL training loop (or dry-run on CPU)."""
    t_start = time.perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda":
        verify_cuda_native_extensions(dev)

    if int(ordinary_uniform_half) + int(ordinary_hard_neg_half) != int(ordinary_batch_size):
        raise ValueError(
            f"ordinary_uniform_half ({ordinary_uniform_half}) + ordinary_hard_neg_half "
            f"({ordinary_hard_neg_half}) must equal ordinary_batch_size ({ordinary_batch_size})"
        )

    if not dry_run:
        if (
            int(num_envs) != REGISTERED_ENVS
            or int(total_updates) != REGISTERED_TOTAL_UPDATES
            or int(warmup_rescue_roots) != REGISTERED_WARMUP_RESCUE_ROOTS
            or int(min_unique_rescue_roots) != REGISTERED_MIN_UNIQUE_RESCUE_ROOTS
            or int(rescue_batch_size) != REGISTERED_RESCUE_BATCH_SIZE
            or int(ordinary_batch_size) != REGISTERED_ORDINARY_BATCH_SIZE
            or int(ordinary_uniform_half) != REGISTERED_ORDINARY_UNIFORM_HALF
            or int(ordinary_hard_neg_half) != REGISTERED_ORDINARY_HARD_NEG_HALF
            or int(branch_horizon) != OPEN_STOP_BRANCH_HORIZON
            or int(tripwire_interval) != REGISTERED_TRIPWIRE_INTERVAL
            or int(decision_window_size) != REGISTERED_DECISION_WINDOW_SIZE
            or not math.isclose(
                float(delta_rescue), RESCUE_MARGIN_DELTA_DEFAULT, rel_tol=0.0, abs_tol=1e-9
            )
            or not math.isclose(
                float(delta_ordinary), ORDINARY_MARGIN_DELTA_DEFAULT, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError(
                "Non-dry-run SITL training must use exact registered hyperparameters: "
                f"E={REGISTERED_ENVS}, updates={REGISTERED_TOTAL_UPDATES}, "
                f"warmup={REGISTERED_WARMUP_RESCUE_ROOTS}, "
                f"min_unique_rescues={REGISTERED_MIN_UNIQUE_RESCUE_ROOTS}, "
                f"B_rescue={REGISTERED_RESCUE_BATCH_SIZE}, "
                f"B_ordinary={REGISTERED_ORDINARY_BATCH_SIZE} "
                f"({REGISTERED_ORDINARY_UNIFORM_HALF}+{REGISTERED_ORDINARY_HARD_NEG_HALF}), "
                f"H={OPEN_STOP_BRANCH_HORIZON}, "
                f"delta_rescue={RESCUE_MARGIN_DELTA_DEFAULT}, "
                f"delta_ordinary={ORDINARY_MARGIN_DELTA_DEFAULT}, "
                f"tripwire_interval={REGISTERED_TRIPWIRE_INTERVAL}, "
                f"decision_window_size={REGISTERED_DECISION_WINDOW_SIZE}"
            )

    seed_audit = audit_b1b_seeds(scan_code_roots=[])
    if not bool(seed_audit["ok"]):
        raise RuntimeError(f"Step B1b seed audit failed: {seed_audit}")

    if b8_checkpoint is None or opponent_checkpoint is None:
        default_b8, default_opp = resolve_product_checkpoints()
        b8_checkpoint = b8_checkpoint or default_b8
        opponent_checkpoint = opponent_checkpoint or default_opp
    b8_path = Path(b8_checkpoint)
    opp_path = Path(opponent_checkpoint)
    b8_sha = sha256_file(b8_path)
    opp_sha = sha256_file(opp_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Harvest or load probe32 bank before training starts
    if probe_bank_path is None:
        probe_bank_path = out_dir / "probe32_bank.pt"
    probe_bank_file = Path(probe_bank_path)
    if not probe_bank_file.is_file() or not Path(f"{probe_bank_file}.sha256").is_file():
        harvest_probe32_bank(
            probe_bank_file,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device=dev,
            num_worlds=min(4, int(num_envs)) if dry_run else 32,
            horizon=6 if dry_run else 5000,
            branch_horizon=int(branch_horizon),
            ordinary_per_world=4 if dry_run else 210,
            harvest_batch_roots=max(4, int(num_envs)),
            cpu_fast_flood=cpu_fast_flood,
            dry_run=dry_run,
        )
    probe_bank = load_probe32_bank(probe_bank_file)
    if not dry_run and bool(probe_bank["meta"].get("dry_run", False)):
        raise ValueError("Refusing to use a dry_run probe32 bank for non-dry-run SITL training")

    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    world_seeds = train_seeds(int(num_envs))
    sim, policy = build_watch6_ct_device_bundle(
        world_seeds,
        b8_path,
        opp_path,
        device=dev,
        hero_respawn=True,
    )
    if dry_run and int(num_envs) >= 2:
        # In dry-run mode (CPU, 0 u), seed the first min(2, num_envs // 2) worlds from
        # MATURE_EXTENSION_SPECS doorway rescue states so a tiny dry-run horizon mines
        # real rescue roots and ordinary roots without needing thousands of rollout ticks.
        n_door = min(2, int(num_envs) // 2)
        ext_worlds, ext_pol_tpl, _ = build_extension_worlds(
            b8_path,
            opp_path,
            specs=MATURE_EXTENSION_SPECS[:n_door],
        )
        ext_sim, ext_pol = snapshot_worlds_to_device(
            ext_worlds,
            ext_pol_tpl,
            device=dev,
            vector_prepared=False,
        )
        ext_sim.terminal_slots[:, 0] = False
        keep_rest = torch.arange(n_door, int(num_envs), device=dev, dtype=torch.long)
        sim_rest = sim.clone()
        pol_rest = policy.repeat_interleave(1)
        compact_sim_and_policy(sim_rest, pol_rest, keep_rest)
        sim, policy = concat_sims_and_policies([ext_sim, sim_rest], [ext_pol, pol_rest])
        r2_cache.enable(sim)

    student_net = load_ego2s_checkpoint(str(b8_path), device=dev)
    if not isinstance(student_net, Ego2sNet):
        raise TypeError(f"Expected Ego2sNet from {b8_path}, got {type(student_net)!r}")
    freeze_info = freeze_ego2s_value_head(student_net)
    trainable_params = [p for p in student_net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=float(learning_rate))

    rescue_buf = SitlRingBuffer(capacity=DEFAULT_RESCUE_BUFFER_CAPACITY)
    ordinary_buf = SitlRingBuffer(capacity=DEFAULT_ORDINARY_BUFFER_CAPACITY)
    unique_rescue_keys: set[tuple[int, int]] = set()

    updates_completed = 0
    collection_steps = 0
    total_hero_decisions = 0
    total_branches_evaluated = 0
    non_finite_loss_count = 0
    non_finite_q_count = 0
    non_finite_grad_count = 0
    illegal_action_count = 0
    total_action_counts = [0, 0, 0, 0, 0, 0]
    window_action_counts = [0, 0, 0, 0, 0, 0]
    window_decisions = 0
    completed_windows: list[dict[str, Any]] = []
    tripwire_history: list[dict[str, Any]] = []
    update_history: list[dict[str, Any]] = []
    tripwire_breached = False
    tripwire_breach_reason: str | None = None

    latest_ckpt_meta_path = ckpt_dir / "latest_checkpoint.json"
    resumed_from_checkpoint: str | None = None

    if resume and latest_ckpt_meta_path.is_file():
        verify_file_sha256(latest_ckpt_meta_path)
        latest_meta = json.loads(latest_ckpt_meta_path.read_text(encoding="utf-8"))
        ckpt_pt_path = Path(latest_meta["checkpoint_path"])
        verify_file_sha256(ckpt_pt_path)
        state = torch.load(ckpt_pt_path, map_location="cpu", weights_only=False)
        if (
            state["b8_checkpoint_sha256"] != b8_sha
            or state["opponent_checkpoint_sha256"] != opp_sha
        ):
            raise ValueError("Refusing to resume SITL checkpoint with mismatched model SHA256")
        if state["world_seeds"] != [int(s) for s in world_seeds]:
            raise ValueError("Refusing to resume SITL checkpoint with mismatched world_seeds")
        student_net.load_state_dict(state["student_state_dict"])
        freeze_ego2s_value_head(student_net)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _restore_sim_and_runtime_state(sim, policy, state["sim_and_runtime"])
        rescue_buf.load_state_dict(state["rescue_buffer"])
        ordinary_buf.load_state_dict(state["ordinary_buffer"])
        unique_rescue_keys = {(int(a), int(b)) for a, b in state["unique_rescue_keys"]}
        updates_completed = int(state["updates_completed"])
        collection_steps = int(state["collection_steps"])
        total_hero_decisions = int(state["total_hero_decisions"])
        total_branches_evaluated = int(state["total_branches_evaluated"])
        non_finite_loss_count = int(state["non_finite_loss_count"])
        non_finite_q_count = int(state["non_finite_q_count"])
        non_finite_grad_count = int(state["non_finite_grad_count"])
        illegal_action_count = int(state["illegal_action_count"])
        total_action_counts = [int(x) for x in state["total_action_counts"]]
        window_action_counts = [int(x) for x in state["window_action_counts"]]
        window_decisions = int(state["window_decisions"])
        completed_windows = list(state["completed_windows"])
        tripwire_history = list(state["tripwire_history"])
        update_history = list(state["update_history"])
        tripwire_breached = bool(state["tripwire_breached"])
        tripwire_breach_reason = state["tripwire_breach_reason"]
        random.setstate(state["py_random_state"])
        rng.bit_generator.state = state["np_rng_state"]
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if dev.type == "cuda" and state.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
        resumed_from_checkpoint = str(ckpt_pt_path)

    last_ckpt_wall = time.perf_counter()
    signal_requested = {"stop": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        signal_requested["stop"] = True

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigusr1 = signal.getsignal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else None
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    def _save_training_checkpoint(tag: str) -> str:
        ckpt_file = ckpt_dir / f"sitl_state_{tag}.pt"
        payload_ckpt = {
            "schema": "t34-gpu-sitl-train-checkpoint/v1",
            "namespace": TRAIN_NAMESPACE,
            "dry_run": bool(dry_run),
            "b8_checkpoint_sha256": b8_sha,
            "opponent_checkpoint_sha256": opp_sha,
            "probe_bank_sha256": probe_bank["verified_sha256"],
            "world_seeds": [int(s) for s in world_seeds],
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "total_hero_decisions": int(total_hero_decisions),
            "total_branches_evaluated": int(total_branches_evaluated),
            "unique_rescue_keys": sorted(list(unique_rescue_keys)),
            "non_finite_loss_count": int(non_finite_loss_count),
            "non_finite_q_count": int(non_finite_q_count),
            "non_finite_grad_count": int(non_finite_grad_count),
            "illegal_action_count": int(illegal_action_count),
            "total_action_counts": list(total_action_counts),
            "window_action_counts": list(window_action_counts),
            "window_decisions": int(window_decisions),
            "completed_windows": list(completed_windows),
            "tripwire_history": list(tripwire_history),
            "update_history": list(update_history),
            "tripwire_breached": bool(tripwire_breached),
            "tripwire_breach_reason": tripwire_breach_reason,
            "student_state_dict": {
                k: v.detach().cpu() for k, v in student_net.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "sim_and_runtime": _capture_sim_and_runtime_state(sim, policy),
            "rescue_buffer": rescue_buf.state_dict(),
            "ordinary_buffer": ordinary_buf.state_dict(),
            "py_random_state": random.getstate(),
            "np_rng_state": rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
            ),
        }
        ckpt_sha = write_atomic_with_sha256(
            ckpt_file,
            lambda tmp: torch.save(payload_ckpt, tmp),
            write_complete_marker=True,
        )
        latest_info = {
            "checkpoint_path": str(ckpt_file),
            "checkpoint_sha256": ckpt_sha,
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
        }
        write_atomic_with_sha256(
            latest_ckpt_meta_path,
            lambda tmp: tmp.write_text(
                json.dumps(latest_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            ),
            write_complete_marker=True,
        )
        return str(ckpt_file)

    def _flush_decision_window(*, is_partial_final: bool = False) -> dict[str, Any] | None:
        nonlocal window_action_counts, window_decisions, tripwire_breached, tripwire_breach_reason
        if window_decisions <= 0:
            return None
        max_share = max(window_action_counts) / float(window_decisions)
        win_pass = bool(
            non_finite_loss_count == 0
            and non_finite_q_count == 0
            and non_finite_grad_count == 0
            and illegal_action_count == 0
            and max_share < REGISTERED_MAX_SINGLE_ACTION_SHARE
        )
        rec = {
            "window_index": len(completed_windows),
            "is_partial_final": bool(is_partial_final),
            "decisions": int(window_decisions),
            "action_counts": list(window_action_counts),
            "max_single_action_share": float(max_share),
            "max_single_action_share_threshold": REGISTERED_MAX_SINGLE_ACTION_SHARE,
            "non_finite_loss_count": int(non_finite_loss_count),
            "non_finite_q_count": int(non_finite_q_count),
            "non_finite_grad_count": int(non_finite_grad_count),
            "illegal_action_count": int(illegal_action_count),
            "pass": win_pass,
        }
        completed_windows.append(rec)
        if not win_pass and not tripwire_breached:
            tripwire_breached = True
            tripwire_breach_reason = f"decision_window_fault:{rec}"
        if not is_partial_final:
            window_action_counts = [0, 0, 0, 0, 0, 0]
            window_decisions = 0
        return rec

    def _run_single_learner_update(next_u: int) -> None:
        nonlocal updates_completed, non_finite_loss_count, non_finite_q_count, non_finite_grad_count
        nonlocal tripwire_breached, tripwire_breach_reason, last_ckpt_wall

        student_net.train()
        optimizer.zero_grad(set_to_none=True)

        n_res_avail = int(rescue_buf.size)
        res_idx = rng.choice(
            n_res_avail,
            size=int(rescue_batch_size),
            replace=(n_res_avail < int(rescue_batch_size)),
        ).astype(np.int64)
        ord_idx, hn_stats = sample_ordinary_minibatch_indices(
            ordinary_buf,
            student_net,
            rng,
            uniform_count=int(ordinary_uniform_half),
            hard_neg_count=int(ordinary_hard_neg_half),
            device=dev,
        )
        res_batch = rescue_buf.gather_batch(res_idx, dev)
        ord_batch = ordinary_buf.gather_batch(ord_idx, dev)

        loss_out = compute_sitl_hinge_loss(
            student_net,
            res_batch,
            ord_batch,
            delta_rescue=float(delta_rescue),
            delta_ordinary=float(delta_ordinary),
        )
        total_loss = loss_out["total_loss"]
        if not bool(loss_out["q_all_finite"].item()):
            non_finite_q_count += 1
        if not bool(torch.isfinite(total_loss).item()):
            non_finite_loss_count += 1
            tripwire_breached = True
            tripwire_breach_reason = f"non_finite_loss_at_update_{next_u}"
            return

        total_loss.backward()

        # Verify value.weight and value.bias stay frozen with None gradient
        if (
            student_net.value.weight.requires_grad
            or student_net.value.bias.requires_grad
            or student_net.value.weight.grad is not None
            or student_net.value.bias.grad is not None
        ):
            raise RuntimeError("value.* parameters must remain frozen with None gradient")

        grads_finite = True
        for p in trainable_params:
            if p.grad is None or not bool(torch.isfinite(p.grad).all().item()):
                grads_finite = False
                break
        if not grads_finite:
            non_finite_grad_count += 1
            tripwire_breached = True
            tripwire_breach_reason = f"non_finite_grad_at_update_{next_u}"
            return

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm)).item()
        )
        optimizer.step()
        updates_completed = int(next_u)

        update_history.append(
            {
                "update": int(updates_completed),
                "collection_steps": int(collection_steps),
                "unique_mined_rescue_roots": len(unique_rescue_keys),
                "rescue_buffer_size": int(rescue_buf.size),
                "ordinary_buffer_size": int(ordinary_buf.size),
                "total_loss": float(total_loss.detach().item()),
                "loss_rescue": float(loss_out["loss_rescue"].item()),
                "loss_ordinary": float(loss_out["loss_ordinary"].item()),
                "active_rescue_frac": float(loss_out["active_rescue_frac"].item()),
                "active_ordinary_frac": float(loss_out["active_ordinary_frac"].item()),
                "grad_norm": grad_norm,
                **hn_stats,
            }
        )

        # Tripwire check every tripwire_interval updates and at final update
        if updates_completed % int(tripwire_interval) == 0 or updates_completed == int(
            total_updates
        ):
            check_rescue_bar = bool(updates_completed == int(total_updates))
            tw_res = evaluate_probe_bank_tripwires(
                student_net,
                probe_bank,
                update_step=updates_completed,
                check_decisive_rescue_bar=check_rescue_bar,
                device=dev,
            )
            tripwire_history.append(tw_res)
            # Note: at intermediate updates (200, 400, 600, 800), decile ordinary disagreement
            # > 5% or boost share > 1.5x breaches the fit-drift tripwire immediately.
            # At update 1000, we record the tripwire result and still run the end-of-pilot
            # training-set aliasing diagnostic so STOP_EGO2S_BV8_INPUTS_INSUFFICIENT is
            # evaluated and reported alongside the update-1000 probe tripwire.
            if not bool(tw_res["all_pass"]) and updates_completed < int(total_updates):
                tripwire_breached = True
                tripwire_breach_reason = f"probe_tripwire_breach_at_update_{updates_completed}"

        now_wall = time.perf_counter()
        if updates_completed % int(checkpoint_every_updates) == 0 or (
            now_wall - last_ckpt_wall
        ) >= float(checkpoint_every_seconds):
            _save_training_checkpoint(f"u{updates_completed:04d}")
            last_ckpt_wall = now_wall

    try:
        while updates_completed < int(total_updates) or len(unique_rescue_keys) < int(
            min_unique_rescue_roots
        ):
            if tripwire_breached:
                break
            if stop_after_updates is not None and updates_completed >= int(stop_after_updates):
                _save_training_checkpoint(f"u{updates_completed:04d}")
                break
            if signal_requested["stop"]:
                _save_training_checkpoint(f"sig_u{updates_completed:04d}")
                raise RuntimeError(
                    "Training interrupted by signal after saving verified checkpoint"
                )

            # Run learners updates if warmup and pacing targets are met
            while (
                updates_completed < int(total_updates)
                and not tripwire_breached
                and (stop_after_updates is None or updates_completed < int(stop_after_updates))
            ):
                next_u = updates_completed + 1
                req_rescues = required_unique_rescues_for_update(
                    next_u,
                    warmup_rescue_roots=int(warmup_rescue_roots),
                    min_unique_rescue_roots=int(min_unique_rescue_roots),
                    total_updates=int(total_updates),
                )
                if (
                    len(unique_rescue_keys) >= req_rescues
                    and rescue_buf.size >= 1
                    and ordinary_buf.size >= 1
                ):
                    _run_single_learner_update(next_u)
                else:
                    break

            if updates_completed >= int(total_updates) and len(unique_rescue_keys) >= int(
                min_unique_rescue_roots
            ):
                break
            if tripwire_breached:
                break
            if stop_after_updates is not None and updates_completed >= int(stop_after_updates):
                _save_training_checkpoint(f"u{updates_completed:04d}")
                break
            if collection_steps >= int(max_collection_steps):
                raise RuntimeError(
                    f"Exceeded max_collection_steps={max_collection_steps} at "
                    f"updates_completed={updates_completed}, "
                    f"unique_rescues={len(unique_rescue_keys)}"
                )

            # Online collection step: student acts on-sim, batched_open_stop_h32 labels alive roots
            with torch.inference_mode():
                sim.prepare()
                policy.vector_runtime.prepare(sim)
                policy.vector_prepared = True

            obs = extract_hero_root_observations(sim, cpu_fast_flood=cpu_fast_flood)
            alive_mask = obs["hero_alive"]
            alive_rows = alive_mask.nonzero(as_tuple=False).squeeze(1)
            n_alive = int(alive_rows.numel())

            student_hero_actions = torch.ones(int(sim.E), dtype=torch.long, device=dev)
            if n_alive > 0:
                student_net.eval()
                with torch.no_grad():
                    loc_alive = obs["local_u8"].index_select(0, alive_rows)
                    glb_alive = obs["global_u8"].index_select(0, alive_rows)
                    sca_alive = obs["scalars_f32"].index_select(0, alive_rows)
                    leg_alive = obs["legal_mask"].index_select(0, alive_rows)
                    q_student_alive = student_net(loc_alive, glb_alive, sca_alive)
                    if not bool(torch.isfinite(q_student_alive).all().item()):
                        non_finite_q_count += 1
                        tripwire_breached = True
                        tripwire_breach_reason = "non_finite_student_q_online"
                        break
                    act_student_alive = q_student_alive.masked_fill(~leg_alive, -torch.inf).argmax(
                        dim=1
                    )
                    row_arange = torch.arange(n_alive, device=dev, dtype=torch.long)
                    illegal_now = int((~leg_alive[row_arange, act_student_alive]).sum().item())
                    if illegal_now > 0:
                        illegal_action_count += illegal_now
                        tripwire_breached = True
                        tripwire_breach_reason = "illegal_student_action_online"
                        break
                    student_hero_actions[alive_rows] = act_student_alive

                # Update 100k decision window counters
                for a_val in act_student_alive.detach().cpu().tolist():
                    ai = int(a_val)
                    total_action_counts[ai] += 1
                    window_action_counts[ai] += 1
                    total_hero_decisions += 1
                    window_decisions += 1
                    if window_decisions >= int(decision_window_size):
                        _flush_decision_window(is_partial_final=False)

                # Evaluate batched_open_stop_h32 on alive roots
                if n_alive == int(sim.E):
                    sim_eval = sim
                    pol_eval = policy
                    seeds_eval = [int(s) for s in world_seeds]
                    decs_eval = [int(x) for x in obs["decision_frame"].detach().cpu().tolist()]
                else:
                    with torch.inference_mode():
                        sim_eval = sim.clone()
                        pol_eval = policy.repeat_interleave(1)
                        compact_sim_and_policy(sim_eval, pol_eval, alive_rows)
                    alive_list = alive_rows.detach().cpu().tolist()
                    seeds_eval = [int(world_seeds[int(i)]) for i in alive_list]
                    decs_eval = [int(obs["decision_frame"][int(i)].item()) for i in alive_list]

                label_res = batched_open_stop_h32(
                    sim_eval,
                    pol_eval,
                    world_seeds=seeds_eval,
                    decisions=decs_eval,
                    suffix_id=0,
                    branch_horizon=int(branch_horizon),
                    namespace=OPEN_STOP_CRN_NAMESPACE,
                    compact_dead=True,
                    enable_r2=True,
                    cpu_fast_flood=cpu_fast_flood,
                    return_host_records=False,
                )
                total_branches_evaluated += n_alive * 6

                safe_m = label_res["safe_mask"].detach().cpu()
                unsafe_m = label_res["unsafe_mask"].detach().cpu()
                rescue_m = label_res["rescue_mask"].detach().cpu()
                ordinary_m = label_res["ordinary_mask"].detach().cpu()
                bare_acts = label_res["bare_actions"].detach().cpu()
                astar_acts = label_res["a_star_actions"].detach().cpu()
                stu_acts_cpu = act_student_alive.detach().cpu()
                loc_cpu = obs["local_u8"].index_select(0, alive_rows).detach().cpu()
                glb_cpu = obs["global_u8"].index_select(0, alive_rows).detach().cpu()
                sca_cpu = obs["scalars_f32"].index_select(0, alive_rows).detach().cpu()
                leg_cpu = obs["legal_mask"].index_select(0, alive_rows).detach().cpu()

                for local_i in range(n_alive):
                    ws_i = int(seeds_eval[local_i])
                    dec_i = int(decs_eval[local_i])
                    b_act = int(bare_acts[local_i].item())
                    s_act = int(astar_acts[local_i].item())
                    stu_a = int(stu_acts_cpu[local_i].item())

                    if bool(rescue_m[local_i].item()):
                        unique_rescue_keys.add((ws_i, dec_i))
                        rescue_buf.add_single(
                            local_u8=loc_cpu[local_i],
                            global_u8=glb_cpu[local_i],
                            scalars_f32=sca_cpu[local_i],
                            legal_mask=leg_cpu[local_i],
                            safe_mask=safe_m[local_i],
                            unsafe_mask=unsafe_m[local_i],
                            bare_action=b_act,
                            a_star_action=s_act,
                            world_seed=ws_i,
                            decision_frame=dec_i,
                            hard_neg_tier=0,
                        )
                    elif bool(ordinary_m[local_i].item()):
                        if stu_a != b_act:
                            hn_tier = 2 if not bool(safe_m[local_i, stu_a].item()) else 1
                        else:
                            hn_tier = 0
                        ordinary_buf.add_single(
                            local_u8=loc_cpu[local_i],
                            global_u8=glb_cpu[local_i],
                            scalars_f32=sca_cpu[local_i],
                            legal_mask=leg_cpu[local_i],
                            safe_mask=safe_m[local_i],
                            unsafe_mask=unsafe_m[local_i],
                            bare_action=b_act,
                            a_star_action=s_act,
                            world_seed=ws_i,
                            decision_frame=dec_i,
                            hard_neg_tier=hn_tier,
                        )

            # Step online simulator under student's action
            with torch.inference_mode():
                step_actions = policy.actions(sim, student_hero_actions)
                sim.step(step_actions)
                policy.after_step(sim)
            collection_steps += 1
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if hasattr(signal, "SIGUSR1") and prev_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, prev_sigusr1)

    if stop_after_updates is not None and updates_completed < int(total_updates):
        return {
            "status": "PAUSED_FOR_RESUME",
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
            "latest_checkpoint_meta": str(latest_ckpt_meta_path),
        }

    # Flush final partial decision window
    _flush_decision_window(is_partial_final=True)

    # Save final student checkpoint in served Ego2sNet format + full resumable state
    final_student_ckpt = out_dir / "b8_sitl_student_final.pth"
    write_atomic_with_sha256(
        final_student_ckpt,
        lambda tmp: save_ego2s_checkpoint(
            tmp,
            student_net,
            meta={
                "study": "t34-gpu-sitl-b1b-20261109",
                "namespace": TRAIN_NAMESPACE,
                "dry_run": bool(dry_run),
                "updates_completed": int(updates_completed),
                "unique_mined_rescue_roots": len(unique_rescue_keys),
                "parent_b8_sha256": b8_sha,
            },
        ),
        write_complete_marker=True,
    )
    final_state_path = _save_training_checkpoint("final")

    # End-of-pilot training-set aliasing diagnostic
    aliasing_diag = evaluate_training_set_aliasing_diagnostic(
        student_net,
        rescue_buf,
        ordinary_buf,
        device=dev,
    )

    total_decisions_denom = max(1, sum(total_action_counts))
    delivered_boost_share = float(sum(total_action_counts[3:6])) / float(total_decisions_denom)
    delivered_boost_band_pass = bool(
        REGISTERED_DELIVERED_BOOST_MIN <= delivered_boost_share <= REGISTERED_DELIVERED_BOOST_MAX
    )
    overall_max_single_action_share = float(max(total_action_counts)) / float(total_decisions_denom)
    all_windows_pass = all(bool(w["pass"]) for w in completed_windows)
    all_probe_tripwires_pass = bool(
        tripwire_history and all(bool(t["all_pass"]) for t in tripwire_history)
    )

    if bool(aliasing_diag["stop_ego2s_bv8_inputs_insufficient"]):
        pilot_verdict = VERDICT_INPUTS_INSUFFICIENT
    elif tripwire_breached or not all_windows_pass or not all_probe_tripwires_pass:
        pilot_verdict = VERDICT_TRIPWIRE_STOP
    else:
        pilot_verdict = VERDICT_PROCEED_TO_GATE_B1

    elapsed_s = float(time.perf_counter() - t_start)
    result_summary: dict[str, Any] = {
        "schema": "t34-gpu-sitl-b1b-pilot-result/v1",
        "namespace": TRAIN_NAMESPACE,
        "probe_namespace": probe_bank["meta"]["namespace"],
        "dry_run": bool(dry_run),
        "evidence_class": (
            "DRY_RUN_NON_EVIDENCE" if dry_run else "REGISTERED_STEP_B1B_PILOT_EVIDENCE"
        ),
        "verdict": pilot_verdict,
        "stop_ego2s_bv8_inputs_insufficient": bool(
            aliasing_diag["stop_ego2s_bv8_inputs_insufficient"]
        ),
        "device": str(dev),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "b8_checkpoint_sha256": b8_sha,
        "b8_expected_sha256": B8_S3_SHA256,
        "opponent_checkpoint_sha256": opp_sha,
        "opponent_expected_sha256": FRP3_S12_SHA256,
        "probe_bank_path": str(probe_bank_file),
        "probe_bank_sha256": probe_bank["verified_sha256"],
        "final_student_checkpoint_path": str(final_student_ckpt),
        "final_student_checkpoint_sha256": sha256_file(final_student_ckpt),
        "final_state_checkpoint_path": str(final_state_path),
        "value_head_freeze_summary": freeze_info,
        "hyperparameters": {
            "num_envs": int(num_envs),
            "total_updates": int(total_updates),
            "warmup_rescue_roots": int(warmup_rescue_roots),
            "min_unique_rescue_roots": int(min_unique_rescue_roots),
            "rescue_batch_size": int(rescue_batch_size),
            "ordinary_batch_size": int(ordinary_batch_size),
            "ordinary_uniform_half": int(ordinary_uniform_half),
            "ordinary_hard_neg_half": int(ordinary_hard_neg_half),
            "branch_horizon": int(branch_horizon),
            "delta_rescue": float(delta_rescue),
            "delta_ordinary": float(delta_ordinary),
            "learning_rate": float(learning_rate),
            "max_grad_norm": float(max_grad_norm),
            "tripwire_interval": int(tripwire_interval),
            "decision_window_size": int(decision_window_size),
            "td_loss_weight": 0.0,
        },
        "counts": {
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "total_hero_decisions": int(total_hero_decisions),
            "total_branches_evaluated": int(total_branches_evaluated),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
            "rescue_buffer_size": int(rescue_buf.size),
            "rescue_buffer_total_added": int(rescue_buf.total_added),
            "ordinary_buffer_size": int(ordinary_buf.size),
            "ordinary_buffer_total_added": int(ordinary_buf.total_added),
        },
        "unconditional_fault_checks": {
            "non_finite_loss_count": int(non_finite_loss_count),
            "non_finite_q_count": int(non_finite_q_count),
            "non_finite_grad_count": int(non_finite_grad_count),
            "illegal_action_count": int(illegal_action_count),
            "total_action_counts": list(total_action_counts),
            "delivered_boost_share": float(delivered_boost_share),
            "delivered_boost_band": [
                REGISTERED_DELIVERED_BOOST_MIN,
                REGISTERED_DELIVERED_BOOST_MAX,
            ],
            "delivered_boost_band_pass": delivered_boost_band_pass,
            "overall_max_single_action_share": float(overall_max_single_action_share),
            "decision_windows": completed_windows,
            "all_decision_windows_pass": bool(all_windows_pass),
        },
        "probe_bank_tripwires": {
            "history": tripwire_history,
            "all_pass": bool(all_probe_tripwires_pass),
            "tripwire_breached": bool(tripwire_breached),
            "tripwire_breach_reason": tripwire_breach_reason,
        },
        "training_set_aliasing_diagnostic": aliasing_diag,
        "update_history": update_history,
        "elapsed_seconds": elapsed_s,
    }

    result_json_path = out_dir / "sitl_pilot_result.json"
    result_sha = write_atomic_with_sha256(
        result_json_path,
        lambda tmp: tmp.write_text(
            json.dumps(result_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
        write_complete_marker=True,
    )
    result_summary["result_json_path"] = str(result_json_path)
    result_summary["result_json_sha256"] = result_sha
    return result_summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step B1b online GPU SITL pilot (t34-gpu-sitl-train-20261109-v1)."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-bank-path", type=Path, default=None)
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-envs", type=int, default=REGISTERED_ENVS)
    parser.add_argument("--total-updates", type=int, default=REGISTERED_TOTAL_UPDATES)
    parser.add_argument("--warmup-rescue-roots", type=int, default=REGISTERED_WARMUP_RESCUE_ROOTS)
    parser.add_argument(
        "--min-unique-rescue-roots", type=int, default=REGISTERED_MIN_UNIQUE_RESCUE_ROOTS
    )
    parser.add_argument("--rescue-batch-size", type=int, default=REGISTERED_RESCUE_BATCH_SIZE)
    parser.add_argument("--ordinary-batch-size", type=int, default=REGISTERED_ORDINARY_BATCH_SIZE)
    parser.add_argument(
        "--ordinary-uniform-half", type=int, default=REGISTERED_ORDINARY_UNIFORM_HALF
    )
    parser.add_argument(
        "--ordinary-hard-neg-half", type=int, default=REGISTERED_ORDINARY_HARD_NEG_HALF
    )
    parser.add_argument("--branch-horizon", type=int, default=OPEN_STOP_BRANCH_HORIZON)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--tripwire-interval", type=int, default=REGISTERED_TRIPWIRE_INTERVAL)
    parser.add_argument("--decision-window-size", type=int, default=REGISTERED_DECISION_WINDOW_SIZE)
    parser.add_argument("--checkpoint-every-updates", type=int, default=100)
    parser.add_argument("--max-collection-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_sitl_training(
        args.output_dir,
        probe_bank_path=args.probe_bank_path,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        total_updates=args.total_updates,
        warmup_rescue_roots=args.warmup_rescue_roots,
        min_unique_rescue_roots=args.min_unique_rescue_roots,
        rescue_batch_size=args.rescue_batch_size,
        ordinary_batch_size=args.ordinary_batch_size,
        ordinary_uniform_half=args.ordinary_uniform_half,
        ordinary_hard_neg_half=args.ordinary_hard_neg_half,
        branch_horizon=args.branch_horizon,
        learning_rate=args.learning_rate,
        tripwire_interval=args.tripwire_interval,
        decision_window_size=args.decision_window_size,
        checkpoint_every_updates=args.checkpoint_every_updates,
        max_collection_steps=args.max_collection_steps,
        seed=args.seed,
        resume=not bool(args.no_resume),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
