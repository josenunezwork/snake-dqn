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
import io
import json
import math
import random
import shutil
import signal
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.gpu_sim_spike import r2_cache
from research.gpu_sitl.open_stop_gpu import (
    _VECTOR_RUNTIME_ENV_FIELDS,
    ORDINARY_MARGIN_DELTA_DEFAULT,
    RESCUE_MARGIN_DELTA_DEFAULT,
    FastDeviceVectorRuntime,
    assert_sim_and_policy_on_device,
    batched_open_stop_h32,
    compact_sim_and_policy,
    concat_sims_and_policies,
    ensure_cuda_initialized,
    freeze_ego2s_value_head,
    move_device_policy_to_device,
    normalize_device,
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
    PROBE32_NAMESPACE,
    SHAKEOUT_PROBE_NAMESPACE,
    SHAKEOUT_TRAIN_NAMESPACE,
    TRAIN_NAMESPACE,
    assert_allowed_training_seeds,
    audit_b1b_seeds,
    shakeout_train_seeds,
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
VERDICT_RUNAWAY_GUARD_STOP = "STOP_RUNAWAY_GUARD_REACHED"
VERDICT_PROCEED_TO_GATE_B1 = "PROCEED_TO_GATE_B1_EVALUATION"

RESCUE_QUOTA_SCORING_METRIC = "world_seed_decision_frame"
REGISTRATION_VERBATIM_QUOTES: tuple[dict[str, Any], ...] = (
    {
        "section": "§3.1",
        "line": 116,
        "quote": (
            "At the end of `Step B1b` (after training on `>= 5,000` unique mined rescue roots "
            "under the fixed 50%/50% balanced minibatch rule), evaluate **in-sample training-set "
            "fit quality** on the student's **OWN mined training rescue roots** "
            "($\\mathcal{D}_{\\text{rescue}}^{\\text{train}}$, counting only true rescue roots "
            "where $a_{\\text{b8}}(s) \\notin \\mathcal{A}_{\\text{safe}}(s)$ and "
            "$\\mathcal{A}_{\\text{safe}}(s) \\neq \\emptyset$) and paired training ordinary "
            "anchor buffer ($\\mathcal{D}_{\\text{ordinary}}^{\\text{train}}$, where "
            "$a_{\\text{b8}}(s) \\in \\mathcal{A}_{\\text{safe}}(s)$):"
        ),
    },
    {
        "section": "§3.3 item 2",
        "line": 143,
        "quote": (
            "Training must accumulate **`>= 5,000` unique mined rescue roots** (counting only "
            "true rescue roots $s \\in \\mathcal{D}_{\\text{rescue}}^{\\text{train}}$ with "
            "$a_{\\text{b8}}(s) \\notin \\mathcal{A}_{\\text{safe}}(s)$ and "
            "$\\mathcal{A}_{\\text{safe}}(s) \\neq \\emptyset$, just as the `< 80.0%` diagnostic "
            "in §3.1(3) and §3.4 counts only true rescue roots in "
            "$\\mathcal{D}_{\\text{rescue}}^{\\text{train}}$) and complete **`1,000` balanced "
            "learner updates** (after an initial buffer warmup of `512` unique mined rescue roots "
            "in $\\mathcal{D}_{\\text{rescue}}^{\\text{train}}$) before Gate B1 evaluation."
        ),
    },
    {
        "section": "§3.3 item 4",
        "line": 156,
        "quote": (
            "On untouched 32-world probe bank `t34-gpu-sitl-probe32-20261109-v1` ($H = 5,000$), "
            "decisive rescue roots and stratified ordinary roots are harvested **once before "
            "training** from frozen `bare b8-s3` trajectories using the Stage 1B certification "
            "rule (`suffix 0` + independent scoring `suffixes 4..7`, `n_diff == 0`, as in §3.3 "
            "One-Look Gate B1 Evaluation item 2 / §3.4 Bar 1)."
        ),
    },
)
RESCUE_QUOTA_REGISTRATION_QUOTES = REGISTRATION_VERBATIM_QUOTES[1]["quote"]

REQUIRED_PROBE_TRIPWIRE_KEYS: tuple[str, ...] = (
    "all_pass",
    "decile_tripwire_pass",
    "boost_tripwire_pass",
    "decisive_rescue_bar_pass",
    "all_q_finite",
    "illegal_action_count",
)


def compute_step_b1b_pilot_verdict(
    *,
    runaway_guard_reached: bool,
    updates_completed: int,
    total_updates: int,
    completed_windows: Sequence[dict[str, Any]],
    delivered_boost_band_pass: bool,
    non_finite_loss_count: int,
    non_finite_q_count: int,
    non_finite_grad_count: int,
    illegal_action_count: int,
    tripwire_history: Sequence[dict[str, Any]],
    tripwire_breached: bool,
    stop_ego2s_bv8_inputs_insufficient: bool,
    overall_max_single_action_share: float | None = None,
) -> str:
    """Resolve Step B1b pilot terminal verdict with strict tripwire key validation.

    Raises :class:`KeyError` if any entry in ``tripwire_history`` is missing a required key
    produced by :func:`evaluate_probe_bank_tripwires` or if any entry in ``completed_windows``
    is missing ``"pass"``.
    """
    for w in completed_windows:
        _ = w["pass"]
    for t in tripwire_history:
        for req_k in REQUIRED_PROBE_TRIPWIRE_KEYS:
            _ = t[req_k]

    all_windows_pass = all(bool(w["pass"]) for w in completed_windows)
    short_run_action_collapse = bool(
        not completed_windows
        and overall_max_single_action_share is not None
        and float(overall_max_single_action_share) >= REGISTERED_MAX_SINGLE_ACTION_SHARE
    )
    all_probe_tripwires_pass = bool(
        tripwire_history and all(bool(t["all_pass"]) for t in tripwire_history)
    )
    any_probe_fault_or_drift = any(
        (not bool(t["decile_tripwire_pass"]))
        or (not bool(t["boost_tripwire_pass"]))
        or (not bool(t["all_q_finite"]))
        or (int(t["illegal_action_count"]) > 0)
        for t in tripwire_history
    )
    fault_or_drift_tripwire_failed = bool(
        int(updates_completed) < int(total_updates)
        or not all_windows_pass
        or short_run_action_collapse
        or not bool(delivered_boost_band_pass)
        or int(non_finite_loss_count) > 0
        or int(non_finite_q_count) > 0
        or int(non_finite_grad_count) > 0
        or int(illegal_action_count) > 0
        or any_probe_fault_or_drift
    )

    if runaway_guard_reached:
        return VERDICT_RUNAWAY_GUARD_STOP
    if fault_or_drift_tripwire_failed:
        return VERDICT_TRIPWIRE_STOP
    if bool(stop_ego2s_bv8_inputs_insufficient):
        return VERDICT_INPUTS_INSUFFICIENT
    if tripwire_breached or not all_probe_tripwires_pass:
        return VERDICT_TRIPWIRE_STOP
    return VERDICT_PROCEED_TO_GATE_B1


DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_RESCUE_BUFFER_CAPACITY = 16384
DEFAULT_ORDINARY_BUFFER_CAPACITY = 32768
DEFAULT_HARD_NEG_PROBE_POOL = 2048


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
        tensors_dict = {
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
        buf = io.BytesIO()
        torch.save(tensors_dict, buf)
        return {
            "capacity": int(self.capacity),
            "size": n,
            "write_ptr": int(self.write_ptr),
            "total_added": int(self.total_added),
            "compressed_tensors_zlib": zlib.compress(buf.getvalue(), level=1),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError(f"Saved buffer size {n} exceeds capacity {self.capacity}")
        self.size = n
        self.write_ptr = int(state["write_ptr"])
        self.total_added = int(state["total_added"])
        if n > 0:
            if "compressed_tensors_zlib" in state:
                raw_bytes = zlib.decompress(state["compressed_tensors_zlib"])
                t_state = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
            else:
                t_state = state
            self.local_u8[:n].copy_(t_state["local_u8"])
            self.global_u8[:n].copy_(t_state["global_u8"])
            self.scalars_f32[:n].copy_(t_state["scalars_f32"])
            self.legal_mask[:n].copy_(t_state["legal_mask"])
            self.safe_mask[:n].copy_(t_state["safe_mask"])
            self.unsafe_mask[:n].copy_(t_state["unsafe_mask"])
            self.bare_action[:n].copy_(t_state["bare_action"])
            self.a_star_action[:n].copy_(t_state["a_star_action"])
            self.world_seed[:n].copy_(t_state["world_seed"])
            self.decision_frame[:n].copy_(t_state["decision_frame"])
            self.hard_neg_tier[:n].copy_(t_state["hard_neg_tier"])

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
    a_star = rescue_batch["a_star_action"].to(dtype=torch.long)
    if not bool(((a_star >= 0) & (a_star < 6)).all().item()):
        raise RuntimeError(
            f"Every mined rescue root must have legal a_star_action in [0, 5], got {a_star.tolist()}"
        )
    if not bool(rescue_batch["legal_mask"][res_arange, a_star].all().item()):
        raise RuntimeError("Every mined rescue root must have a_star_action in legal_mask")
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
    a_b8 = ordinary_batch["bare_action"].to(dtype=torch.long)
    if not bool(((a_b8 >= 0) & (a_b8 < 6)).all().item()):
        raise RuntimeError(
            f"Every ordinary state must have bare_action in [0, 5], got {a_b8.tolist()}"
        )
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
            chunk_size = 512
            for start_i in range(0, len(eval_indices), chunk_size):
                chunk_arr = eval_indices[start_i : start_i + chunk_size]
                idx_t = torch.as_tensor(chunk_arr, dtype=torch.long)
                loc = ordinary_buf.local_u8.index_select(0, idx_t).to(device)
                glb = ordinary_buf.global_u8.index_select(0, idx_t).to(device)
                sca = ordinary_buf.scalars_f32.index_select(0, idx_t).to(device)
                leg = ordinary_buf.legal_mask.index_select(0, idx_t).to(device)
                saf = ordinary_buf.safe_mask.index_select(0, idx_t).to(device)
                b8_a = ordinary_buf.bare_action.index_select(0, idx_t).to(device).clamp(0, 5)

                q = student_net(loc, glb, sca)
                a_theta = q.masked_fill(~leg, -torch.inf).argmax(dim=1)
                row_i = torch.arange(len(chunk_arr), device=device, dtype=torch.long)
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
        # When fewer than hard_neg_count distinct hard negatives exist, fill the remaining
        # slots uniformly without replacement from the remaining ordinary buffer rows (never
        # duplicating a hard negative up to 64 times).
        excluded_hn = np.asarray(chosen_hn, dtype=np.int64)
        preferred_excluded = np.unique(np.concatenate([excluded_hn, uniform_idx]))
        preferred_pool = np.setdiff1d(
            np.arange(n, dtype=np.int64), preferred_excluded, assume_unique=True
        )
        if len(preferred_pool) >= rem:
            extra_u = rng.choice(preferred_pool, size=rem, replace=False).tolist()
        else:
            fallback_pool = np.setdiff1d(
                np.arange(n, dtype=np.int64), excluded_hn, assume_unique=True
            )
            if len(fallback_pool) >= rem:
                extra_u = rng.choice(fallback_pool, size=rem, replace=False).tolist()
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
    sim_state: dict[str, Any] = {
        "E": int(sim.E),
        "_prepared": bool(getattr(sim, "_prepared", False)),
    }
    for name in _ENV_TENSOR_FIELDS:
        val = getattr(sim, name, None)
        if isinstance(val, torch.Tensor):
            sim_state[name] = val.detach().cpu().clone()
    rt = policy.vector_runtime
    rt_state: dict[str, Any] = {
        "E": int(getattr(rt, "E", sim.E)),
        "vector_prepared": bool(policy.vector_prepared),
        "fresh_rows": (
            rt.fresh_rows.detach().cpu().clone()
            if isinstance(getattr(rt, "fresh_rows", None), torch.Tensor)
            else int(getattr(rt, "fresh_rows", 0))
        ),
        "carried_rows": (
            rt.carried_rows.detach().cpu().clone()
            if isinstance(getattr(rt, "carried_rows", None), torch.Tensor)
            else int(getattr(rt, "carried_rows", 0))
        ),
    }
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        val = getattr(rt, name, None)
        if isinstance(val, torch.Tensor):
            rt_state[name] = val.detach().cpu().clone()
    raw_obj = {"sim": sim_state, "vector_runtime": rt_state}
    buf = io.BytesIO()
    torch.save(raw_obj, buf)
    return {
        "E": int(sim.E),
        "compressed_zlib": zlib.compress(buf.getvalue(), level=1),
    }


@torch.no_grad()
def _restore_sim_and_runtime_state(sim: Any, policy: Any, state: dict[str, Any]) -> None:
    if "compressed_zlib" in state:
        raw_bytes = zlib.decompress(state["compressed_zlib"])
        state = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
    dev = normalize_device(sim.device)
    sim.device = dev
    move_device_policy_to_device(policy, dev)
    sim_state = state["sim"]
    for name in _ENV_TENSOR_FIELDS:
        if name in sim_state and isinstance(sim_state[name], torch.Tensor):
            setattr(sim, name, sim_state[name].to(dev).clone())
    sim.E = int(sim_state["E"])
    sim._prepared = bool(sim_state.get("_prepared", False))
    r2_cache.bump(sim)

    rt = policy.vector_runtime
    rt_state = state["vector_runtime"]
    for name in _VECTOR_RUNTIME_ENV_FIELDS:
        if name in rt_state and isinstance(rt_state[name], torch.Tensor):
            setattr(rt, name, rt_state[name].to(dev).clone())
    if "fresh_rows" in rt_state:
        fr = rt_state["fresh_rows"]
        rt.fresh_rows = (
            fr.to(dev).clone()
            if isinstance(fr, torch.Tensor)
            else torch.as_tensor(int(fr), dtype=torch.long, device=dev)
        )
    if "carried_rows" in rt_state:
        cr = rt_state["carried_rows"]
        rt.carried_rows = (
            cr.to(dev).clone()
            if isinstance(cr, torch.Tensor)
            else torch.as_tensor(int(cr), dtype=torch.long, device=dev)
        )
    rt.E = int(rt_state.get("E", sim_state["E"]))
    rt.__class__ = FastDeviceVectorRuntime
    policy.vector_prepared = bool(rt_state["vector_prepared"])
    assert_sim_and_policy_on_device(sim, policy, dev)


def _resolve_staged_input_file(
    relpath_or_path: str,
    *,
    input_dir: Path | None,
    ledger_file: Path | None,
    out_dir: Path,
) -> Path:
    cand = Path(relpath_or_path)
    if cand.is_absolute() and cand.is_file():
        return cand
    search_roots: list[Path] = []
    if input_dir is not None:
        search_roots.append(Path(input_dir))
    if ledger_file is not None:
        search_roots.append(ledger_file.parent)
        if ledger_file.parent.name == "inputs":
            search_roots.append(ledger_file.parent.parent)
    search_roots.append(out_dir)
    for root in search_roots:
        p1 = root / relpath_or_path
        if p1.is_file():
            return p1
        p2 = root / Path(relpath_or_path).name
        if p2.is_file():
            return p2
    raise FileNotFoundError(
        f"Staged input {relpath_or_path!r} not found in {[str(r) for r in search_roots]}"
    )


def run_sitl_training(
    output_dir: Path | str,
    *,
    probe_bank_path: Path | str | None = None,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    ledger_path: Path | str | None = None,
    input_dir: Path | str | None = None,
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
    max_wall_seconds: float | None = None,
    seed: int = 20261109,
    cpu_fast_flood: bool = True,
    resume: bool = True,
    stop_after_updates: int | None = None,
    stop_after_collection_steps: int | None = None,
    pause_after_seconds: float | None = None,
    dry_run: bool = False,
    shakeout: bool = False,
) -> dict[str, Any]:
    """Run the Step B1b online GPU SITL training loop (or dry-run / L4 shake-out)."""
    t_start = time.perf_counter()
    dev = ensure_cuda_initialized(device)
    native_extensions = verify_cuda_native_extensions(dev)

    is_shakeout = bool(shakeout)
    eff_dry_run = bool(dry_run or is_shakeout)
    active_train_namespace = SHAKEOUT_TRAIN_NAMESPACE if is_shakeout else TRAIN_NAMESPACE
    active_probe_namespace = SHAKEOUT_PROBE_NAMESPACE if is_shakeout else PROBE32_NAMESPACE

    if int(ordinary_uniform_half) + int(ordinary_hard_neg_half) != int(ordinary_batch_size):
        raise ValueError(
            f"ordinary_uniform_half ({ordinary_uniform_half}) + ordinary_hard_neg_half "
            f"({ordinary_hard_neg_half}) must equal ordinary_batch_size ({ordinary_batch_size})"
        )

    if not eff_dry_run:
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

    seed_audit = audit_b1b_seeds()
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
    if not eff_dry_run:
        if b8_sha != B8_S3_SHA256:
            raise ValueError(
                f"Non-dry-run SITL training requires canonical b8-s3 SHA256 {B8_S3_SHA256}, "
                f"got {b8_sha} at {b8_path}"
            )
        if opp_sha != FRP3_S12_SHA256:
            raise ValueError(
                f"Non-dry-run SITL training requires canonical frp3-s12 SHA256 {FRP3_S12_SHA256}, "
                f"got {opp_sha} at {opp_path}"
            )

    out_dir = Path(output_dir)
    result_json_path = out_dir / (
        "shakeout_sitl_subresult.json" if is_shakeout else "sitl_pilot_result.json"
    )
    if Path(f"{result_json_path}.complete").is_file():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite completed SITL training output at {result_json_path}"
            )
        existing_sha = verify_file_sha256(result_json_path, require_complete=True)
        existing_summary = json.loads(result_json_path.read_text(encoding="utf-8"))
        if existing_summary.get("namespace") != active_train_namespace:
            raise ValueError(
                f"Refusing completed output at {result_json_path} with foreign namespace "
                f"{existing_summary.get('namespace')!r} (expected {active_train_namespace!r})"
            )
        existing_summary["result_json_path"] = str(result_json_path)
        existing_summary["result_json_sha256"] = existing_sha
        return existing_summary

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(input_dir) if input_dir is not None else None

    world_seeds = shakeout_train_seeds(int(num_envs)) if is_shakeout else train_seeds(int(num_envs))
    assert_allowed_training_seeds(
        namespace=active_train_namespace,
        seeds=world_seeds,
        context="run_sitl_training.world_seeds",
    )

    def _validate_ledger_file(l_path: Path) -> dict[str, Any]:
        obj = json.loads(l_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid verified ledger at {l_path}")
        if obj.get("namespace") != active_train_namespace:
            raise ValueError(
                f"Refusing foreign training namespace in ledger {l_path}: "
                f"expected {active_train_namespace!r}, got {obj.get('namespace')!r}"
            )
        if obj.get("probe_namespace", active_probe_namespace) != active_probe_namespace:
            raise ValueError(
                f"Refusing foreign probe namespace in ledger {l_path}: "
                f"expected {active_probe_namespace!r}, got {obj.get('probe_namespace')!r}"
            )
        l_seeds = obj.get("world_seeds")
        if l_seeds and [int(s) for s in l_seeds] != [int(s) for s in world_seeds]:
            raise ValueError(f"Refusing ledger {l_path} with mismatched world_seeds allocation")
        return obj

    # Resolve and validate verified-ledger.json if present (external staged + local working/part)
    ext_ledger_file: Path | None = None
    if ledger_path is not None:
        ext_ledger_file = Path(ledger_path)
    elif in_dir is not None:
        for cand_l in (in_dir / "inputs" / "verified-ledger.json", in_dir / "verified-ledger.json"):
            if cand_l.is_file():
                ext_ledger_file = cand_l
                break

    local_ledger_file: Path | None = None
    if resume:
        working_l = out_dir / "verified-ledger.working.json"
        part_ls = sorted(
            p for p in out_dir.glob("verified-ledger.part*.json") if Path(f"{p}.complete").is_file()
        )
        base_l = out_dir / "verified-ledger.json"
        if working_l.is_file() and Path(f"{working_l}.sha256").is_file():
            verify_file_sha256(working_l, require_complete=False)
            local_ledger_file = working_l
        elif part_ls:
            verify_file_sha256(part_ls[-1], require_complete=True)
            local_ledger_file = part_ls[-1]
        elif base_l.is_file():
            verify_file_sha256(base_l, require_complete=Path(f"{base_l}.complete").is_file())
            local_ledger_file = base_l

    active_ledger_file: Path | None = ext_ledger_file or local_ledger_file
    loaded_ledger: dict[str, Any] = {}
    verified_units_ledger: list[dict[str, Any]] = []
    if ext_ledger_file is not None and ext_ledger_file.is_file():
        loaded_ledger = _validate_ledger_file(ext_ledger_file)
        if isinstance(loaded_ledger.get("verified_units"), list):
            verified_units_ledger = list(loaded_ledger["verified_units"])
    if (
        local_ledger_file is not None
        and local_ledger_file.is_file()
        and local_ledger_file != ext_ledger_file
    ):
        local_obj = _validate_ledger_file(local_ledger_file)
        for k, v in local_obj.items():
            if k == "verified_units" and isinstance(v, list):
                for u_item in v:
                    if not any(
                        u_existing.get("sha256") == u_item.get("sha256")
                        for u_existing in verified_units_ledger
                    ):
                        verified_units_ledger.append(u_item)
            else:
                loaded_ledger[k] = v

    out_ledger_path = out_dir / "verified-ledger.json"
    working_ledger_path = out_dir / "verified-ledger.working.json"

    def _publish_ledger_payload(
        ledger_obj: dict[str, Any],
        *,
        mark_complete: bool = False,
    ) -> Path:
        payload_txt = json.dumps(ledger_obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
        write_atomic_with_sha256(
            working_ledger_path,
            lambda tmp: tmp.write_text(payload_txt, encoding="utf-8"),
            write_complete_marker=False,
        )
        if not mark_complete:
            return working_ledger_path
        if not Path(f"{out_ledger_path}.complete").exists():
            write_atomic_with_sha256(
                out_ledger_path,
                lambda tmp: tmp.write_text(payload_txt, encoding="utf-8"),
                write_complete_marker=True,
            )
            return out_ledger_path
        part_idx = 2
        while True:
            cand_part = out_dir / f"verified-ledger.part{part_idx:02d}.json"
            if not cand_part.exists() and not Path(f"{cand_part}.complete").exists():
                write_atomic_with_sha256(
                    cand_part,
                    lambda tmp: tmp.write_text(payload_txt, encoding="utf-8"),
                    write_complete_marker=True,
                )
                return cand_part
            part_idx += 1

    # Harvest or load probe32 bank before training starts
    if probe_bank_path is None:
        probe_bank_path = out_dir / ("shakeout_probe_bank.pt" if is_shakeout else "probe32_bank.pt")
    probe_bank_file = Path(probe_bank_path)

    def _has_complete_probe_bank() -> bool:
        return probe_bank_file.is_file() and Path(f"{probe_bank_file}.complete").is_file()

    def _clean_incomplete_probe_bank() -> None:
        if not Path(f"{probe_bank_file}.complete").is_file():
            for p_rem in (probe_bank_file, Path(f"{probe_bank_file}.sha256")):
                if p_rem.is_file():
                    p_rem.unlink()

    if resume and not _has_complete_probe_bank():
        staged_probe_info = loaded_ledger.get("probe32_bank")
        if isinstance(staged_probe_info, dict) and staged_probe_info.get("relpath"):
            _clean_incomplete_probe_bank()
            staged_pb = _resolve_staged_input_file(
                str(staged_probe_info["relpath"]),
                input_dir=in_dir,
                ledger_file=active_ledger_file,
                out_dir=out_dir,
            )
            expected_pb_sha = str(staged_probe_info["sha256"])
            actual_pb_sha = sha256_file(staged_pb)
            if actual_pb_sha != expected_pb_sha:
                raise ValueError(
                    f"Staged probe32_bank SHA256 mismatch: expected {expected_pb_sha}, "
                    f"got {actual_pb_sha}"
                )
            published_sha = write_atomic_with_sha256(
                probe_bank_file,
                lambda tmp: shutil.copyfile(staged_pb, tmp),
                write_complete_marker=True,
            )
            if published_sha != expected_pb_sha:
                raise RuntimeError("Published probe32_bank SHA256 mismatch after staging")

    staged_harvest_ckpt: Path | None = None
    if resume and not _has_complete_probe_bank():
        staged_hckpt_info = loaded_ledger.get("probe32_harvest_ckpt")
        if isinstance(staged_hckpt_info, dict) and staged_hckpt_info.get("relpath"):
            cand_hckpt = _resolve_staged_input_file(
                str(staged_hckpt_info["relpath"]),
                input_dir=in_dir,
                ledger_file=active_ledger_file,
                out_dir=out_dir,
            )
            expected_h_sha = str(staged_hckpt_info["sha256"])
            if sha256_file(cand_hckpt) != expected_h_sha:
                raise ValueError("Staged probe32 harvest checkpoint SHA256 mismatch")
            staged_harvest_ckpt = cand_hckpt

    signal_requested = {"stop": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        signal_requested["stop"] = True

    if not _has_complete_probe_bank():
        _clean_incomplete_probe_bank()
        prev_sigterm_h = signal.getsignal(signal.SIGTERM)
        prev_sigusr1_h = signal.getsignal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else None
        signal.signal(signal.SIGTERM, _on_signal)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, _on_signal)
        try:
            rem_pause = (
                max(0.0, float(pause_after_seconds) - (time.perf_counter() - t_start))
                if pause_after_seconds is not None
                else None
            )
            harvest_res = harvest_probe32_bank(
                probe_bank_file,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev,
                num_worlds=min(4, int(num_envs)) if eff_dry_run else 32,
                horizon=6 if eff_dry_run else 5000,
                branch_horizon=int(branch_horizon),
                ordinary_per_world=4 if eff_dry_run else 210,
                harvest_batch_roots=max(4, int(num_envs)),
                checkpoint_every_seconds=float(checkpoint_every_seconds),
                pause_after_seconds=rem_pause,
                should_pause_fn=lambda: bool(signal_requested["stop"]),
                resume_ckpt_path=staged_harvest_ckpt,
                cpu_fast_flood=cpu_fast_flood,
                dry_run=eff_dry_run,
                shakeout=is_shakeout,
            )
        finally:
            signal.signal(signal.SIGTERM, prev_sigterm_h)
            if hasattr(signal, "SIGUSR1") and prev_sigusr1_h is not None:
                signal.signal(signal.SIGUSR1, prev_sigusr1_h)
        if isinstance(harvest_res, dict) and harvest_res.get("status") == "PAUSED_FOR_RESUME":
            harvest_ledger_obj: dict[str, Any] = {
                "schema": "t34-gpu-sitl-verified-ledger/v1",
                "namespace": active_train_namespace,
                "probe_namespace": active_probe_namespace,
                "world_seeds": [int(s) for s in world_seeds],
                "unit_kind": "update",
                "skip_verified": True,
                "probe32_harvest_ckpt": {
                    "relpath": Path(str(harvest_res["harvest_ckpt_path"])).name,
                    "sha256": str(harvest_res["harvest_ckpt_sha256"]),
                    "size_bytes": int(harvest_res["harvest_ckpt_size_bytes"]),
                    "completed_tick": int(harvest_res["completed_tick"]),
                    "horizon": int(harvest_res["horizon"]),
                },
                "verified_units": list(verified_units_ledger),
            }
            pub_l = _publish_ledger_payload(harvest_ledger_obj, mark_complete=True)
            return {
                "status": "PAUSED_FOR_RESUME",
                "phase": "probe32_harvest",
                "completed_tick": int(harvest_res["completed_tick"]),
                "horizon": int(harvest_res["horizon"]),
                "harvest_ckpt_path": str(harvest_res["harvest_ckpt_path"]),
                "verified_ledger_path": str(pub_l),
            }

    probe_bank = load_probe32_bank(probe_bank_file, expected_namespace=active_probe_namespace)
    probe_meta_file = out_dir / f"{probe_bank_file.name}.meta.json"
    if not Path(f"{probe_meta_file}.complete").is_file():
        for p_rem in (probe_meta_file, Path(f"{probe_meta_file}.sha256")):
            if p_rem.is_file():
                p_rem.unlink()
        meta_to_write = dict(probe_bank["meta"])
        meta_to_write.setdefault("probe_bank_path", str(probe_bank_file))
        meta_to_write.setdefault("probe_bank_sha256", probe_bank["verified_sha256"])
        write_atomic_with_sha256(
            probe_meta_file,
            lambda tmp: tmp.write_text(
                json.dumps(meta_to_write, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            ),
            write_complete_marker=True,
        )
    if not eff_dry_run and bool(probe_bank["meta"].get("dry_run", False)):
        raise ValueError("Refusing to use a dry_run probe32 bank for non-dry-run SITL training")
    if probe_bank.get("meta", {}).get("namespace") != active_probe_namespace:
        raise ValueError(
            f"probe_bank namespace must be {active_probe_namespace!r}, "
            f"got {probe_bank.get('meta', {}).get('namespace')!r}"
        )
    probe_seed_set: set[int] = {int(s) for s in probe_bank.get("meta", {}).get("world_seeds", [])}
    if "rescue" in probe_bank and isinstance(probe_bank["rescue"].get("world_seed"), torch.Tensor):
        probe_seed_set.update(int(x) for x in probe_bank["rescue"]["world_seed"].tolist())
    if "ordinary" in probe_bank and isinstance(
        probe_bank["ordinary"].get("world_seed"), torch.Tensor
    ):
        probe_seed_set.update(int(x) for x in probe_bank["ordinary"]["world_seed"].tolist())
    if not probe_seed_set:
        raise ValueError("probe_bank world_seeds set is empty")
    assert_allowed_training_seeds(
        namespace=active_probe_namespace,
        seeds=sorted(probe_seed_set),
        context="run_sitl_training.probe_bank",
    )
    seed_overlap = probe_seed_set.intersection(int(s) for s in world_seeds)
    if seed_overlap:
        raise ValueError(
            f"Seed contamination between probe32 bank and training world_seeds: "
            f"{sorted(seed_overlap)}"
        )

    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        ensure_cuda_initialized(dev)
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    sim, policy = build_watch6_ct_device_bundle(
        world_seeds,
        b8_path,
        opp_path,
        device=dev,
        hero_respawn=True,
    )
    if eff_dry_run and int(num_envs) >= 2:
        # In dry-run / shake-out mode, seed the first min(2, num_envs // 2) worlds from
        # MATURE_EXTENSION_SPECS[9:9+n_door] (disjoint from MATURE_EXTENSION_SPECS[0] used in
        # probe_bank.py dry-run) so a tiny horizon mines rescue and ordinary roots with L >= 80.
        n_door = min(2, int(num_envs) // 2)
        ext_worlds, ext_pol_tpl, _ = build_extension_worlds(
            b8_path,
            opp_path,
            specs=MATURE_EXTENSION_SPECS[9 : 9 + n_door],
        )
        ext_sim, ext_pol = snapshot_worlds_to_device(
            ext_worlds,
            ext_pol_tpl,
            device=dev,
            vector_prepared=False,
        )
        ext_sim.hero_respawn = True
        ext_sim.terminal_slots[:, 0] = False
        keep_rest = torch.arange(n_door, int(num_envs), device=dev, dtype=torch.long)
        sim_rest = sim.clone()
        pol_rest = policy.repeat_interleave(1)
        if pol_rest.vector_runtime is not None:
            pol_rest.vector_runtime.__class__ = FastDeviceVectorRuntime
        compact_sim_and_policy(sim_rest, pol_rest, keep_rest)
        sim, policy = concat_sims_and_policies([ext_sim, sim_rest], [ext_pol, pol_rest])
        sim.hero_respawn = True
        sim.terminal_slots[:, 0] = False
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
    unique_rescue_episode_keys: set[tuple[int, int, int]] = set()
    hero_life_id_by_env: list[int] = [0] * int(num_envs)
    rescue_episode_seq_by_env: list[int] = [0] * int(num_envs)
    in_rescue_run_by_env: list[bool] = [False] * int(num_envs)

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
    runaway_guard_reached = False
    runaway_guard_reason: str | None = None

    latest_ckpt_meta_path = ckpt_dir / "latest_checkpoint.json"
    resumed_from_checkpoint: str | None = None
    last_latest_ckpt_info: dict[str, Any] | None = None

    def _write_verified_ledger(
        latest_ckpt_info: dict[str, Any] | None = None,
        *,
        mark_complete: bool = False,
    ) -> Path:
        ledger_obj: dict[str, Any] = {
            "schema": "t34-gpu-sitl-verified-ledger/v1",
            "namespace": active_train_namespace,
            "probe_namespace": active_probe_namespace,
            "world_seeds": [int(s) for s in world_seeds],
            "unit_kind": "update",
            "skip_verified": True,
            "probe32_bank": {
                "relpath": probe_bank_file.name,
                "sha256": probe_bank["verified_sha256"],
                "size_bytes": int(probe_bank_file.stat().st_size),
            },
            "verified_units": list(verified_units_ledger),
        }
        if latest_ckpt_info is not None:
            ledger_obj["resume_checkpoint"] = dict(latest_ckpt_info)
        elif loaded_ledger.get("resume_checkpoint") is not None:
            ledger_obj["resume_checkpoint"] = dict(loaded_ledger["resume_checkpoint"])
        return _publish_ledger_payload(ledger_obj, mark_complete=mark_complete)

    def _load_training_checkpoint_file(ckpt_pt_path: Path) -> None:
        nonlocal unique_rescue_keys, unique_rescue_episode_keys
        nonlocal hero_life_id_by_env, rescue_episode_seq_by_env, in_rescue_run_by_env
        nonlocal updates_completed, collection_steps, total_hero_decisions, total_branches_evaluated
        nonlocal non_finite_loss_count, non_finite_q_count, non_finite_grad_count
        nonlocal illegal_action_count, total_action_counts, window_action_counts, window_decisions
        nonlocal completed_windows, tripwire_history, update_history
        nonlocal tripwire_breached, tripwire_breach_reason, resumed_from_checkpoint
        nonlocal last_latest_ckpt_info

        state = torch.load(ckpt_pt_path, map_location="cpu", weights_only=False)
        if state.get("namespace") != active_train_namespace:
            raise ValueError(
                f"Refusing to resume SITL checkpoint with foreign namespace: "
                f"{state.get('namespace')!r}"
            )
        if (
            state["b8_checkpoint_sha256"] != b8_sha
            or state["opponent_checkpoint_sha256"] != opp_sha
        ):
            raise ValueError("Refusing to resume SITL checkpoint with mismatched model SHA256")
        if state.get("probe_bank_sha256") != probe_bank["verified_sha256"]:
            raise ValueError("Refusing to resume SITL checkpoint with mismatched probe_bank_sha256")
        if state["world_seeds"] != [int(s) for s in world_seeds]:
            raise ValueError("Refusing to resume SITL checkpoint with mismatched world_seeds")
        student_net.load_state_dict(state["student_state_dict"])
        freeze_ego2s_value_head(student_net)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for opt_st in optimizer.state.values():
            for opt_k, opt_v in list(opt_st.items()):
                if isinstance(opt_v, torch.Tensor) and opt_k != "step":
                    opt_st[opt_k] = opt_v.to(dev)
        _restore_sim_and_runtime_state(sim, policy, state["sim_and_runtime"])
        rescue_buf.load_state_dict(state["rescue_buffer"])
        ordinary_buf.load_state_dict(state["ordinary_buffer"])
        unique_rescue_keys = {(int(a), int(b)) for a, b in state["unique_rescue_keys"]}
        unique_rescue_episode_keys = {
            (int(a), int(b), int(c)) for a, b, c in state.get("unique_rescue_episode_keys", [])
        }
        hero_life_id_by_env = [
            int(x) for x in state.get("hero_life_id_by_env", [0] * int(num_envs))
        ]
        rescue_episode_seq_by_env = [
            int(x) for x in state.get("rescue_episode_seq_by_env", [0] * int(num_envs))
        ]
        in_rescue_run_by_env = [
            bool(x) for x in state.get("in_rescue_run_by_env", [False] * int(num_envs))
        ]
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
        if state.get("np_global_rng_state") is not None:
            np.random.set_state(state["np_global_rng_state"])
        rng.bit_generator.state = state["np_rng_state"]
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if dev.type == "cuda" and state.get("torch_cuda_rng_state") is not None:
            ensure_cuda_initialized(dev)
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
        resumed_from_checkpoint = str(ckpt_pt_path)
        ckpt_sha_loaded = sha256_file(ckpt_pt_path)
        last_latest_ckpt_info = {
            "checkpoint_path": str(ckpt_pt_path),
            "relpath": (
                f"checkpoints/{ckpt_pt_path.name}"
                if ckpt_pt_path.parent == ckpt_dir
                else ckpt_pt_path.name
            ),
            "sha256": ckpt_sha_loaded,
            "checkpoint_sha256": ckpt_sha_loaded,
            "size_bytes": int(ckpt_pt_path.stat().st_size),
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
            "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
        }

    if resume:
        if latest_ckpt_meta_path.is_file():
            verify_file_sha256(
                latest_ckpt_meta_path,
                require_complete=Path(f"{latest_ckpt_meta_path}.complete").is_file(),
            )
            latest_meta = json.loads(latest_ckpt_meta_path.read_text(encoding="utf-8"))
            ckpt_pt_path = Path(latest_meta["checkpoint_path"])
            if not ckpt_pt_path.is_file():
                ckpt_pt_path = ckpt_dir / ckpt_pt_path.name
            verify_file_sha256(ckpt_pt_path)
            _load_training_checkpoint_file(ckpt_pt_path)
        elif isinstance(loaded_ledger.get("resume_checkpoint"), dict):
            rc_info = loaded_ledger["resume_checkpoint"]
            staged_ckpt = _resolve_staged_input_file(
                str(rc_info["relpath"]),
                input_dir=in_dir,
                ledger_file=active_ledger_file,
                out_dir=out_dir,
            )
            expected_ckpt_sha = str(rc_info["sha256"])
            actual_ckpt_sha = sha256_file(staged_ckpt)
            if actual_ckpt_sha != expected_ckpt_sha:
                raise ValueError(
                    f"Staged resume_checkpoint SHA256 mismatch: expected {expected_ckpt_sha}, "
                    f"got {actual_ckpt_sha}"
                )
            _load_training_checkpoint_file(staged_ckpt)

    _write_verified_ledger(last_latest_ckpt_info, mark_complete=False)

    last_ckpt_wall = time.perf_counter()
    last_ckpt_state: tuple[int, int] = (-1, -1)
    paused_early = False
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigusr1 = signal.getsignal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else None
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    def _save_training_checkpoint(tag: str) -> str:
        nonlocal last_ckpt_state, last_latest_ckpt_info
        eff_tag = tag
        ckpt_file = ckpt_dir / f"sitl_state_{eff_tag}.pt"
        if ckpt_file.exists() or Path(f"{ckpt_file}.complete").exists():
            eff_tag = f"{tag}_s{int(collection_steps):06d}"
            ckpt_file = ckpt_dir / f"sitl_state_{eff_tag}.pt"
            seq_idx = 1
            while ckpt_file.exists() or Path(f"{ckpt_file}.complete").exists():
                eff_tag = f"{tag}_s{int(collection_steps):06d}_{seq_idx}"
                ckpt_file = ckpt_dir / f"sitl_state_{eff_tag}.pt"
                seq_idx += 1
        payload_ckpt = {
            "schema": "t34-gpu-sitl-train-checkpoint/v1",
            "namespace": active_train_namespace,
            "dry_run": bool(eff_dry_run),
            "shakeout": bool(is_shakeout),
            "not_evidence": bool(eff_dry_run or is_shakeout),
            "b8_checkpoint_sha256": b8_sha,
            "opponent_checkpoint_sha256": opp_sha,
            "probe_bank_sha256": probe_bank["verified_sha256"],
            "world_seeds": [int(s) for s in world_seeds],
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "total_hero_decisions": int(total_hero_decisions),
            "total_branches_evaluated": int(total_branches_evaluated),
            "unique_rescue_keys": sorted(list(unique_rescue_keys)),
            "unique_rescue_episode_keys": sorted(list(unique_rescue_episode_keys)),
            "hero_life_id_by_env": list(hero_life_id_by_env),
            "rescue_episode_seq_by_env": list(rescue_episode_seq_by_env),
            "in_rescue_run_by_env": list(in_rescue_run_by_env),
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
            "np_global_rng_state": np.random.get_state(),
            "np_rng_state": rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                (ensure_cuda_initialized(dev), torch.cuda.get_rng_state_all())[1]
                if dev.type == "cuda"
                else None
            ),
        }
        ckpt_sha = write_atomic_with_sha256(
            ckpt_file,
            lambda tmp: torch.save(payload_ckpt, tmp),
            write_complete_marker=True,
        )
        ckpt_size = int(ckpt_file.stat().st_size)
        latest_info = {
            "checkpoint_path": str(ckpt_file),
            "relpath": f"checkpoints/{ckpt_file.name}",
            "sha256": ckpt_sha,
            "checkpoint_sha256": ckpt_sha,
            "size_bytes": ckpt_size,
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
            "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
        }
        write_atomic_with_sha256(
            latest_ckpt_meta_path,
            lambda tmp: tmp.write_text(
                json.dumps(latest_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            ),
            write_complete_marker=False,
        )
        unit_entry = {
            "unit_id": f"tag:{eff_tag}",
            "relpath": f"checkpoints/{ckpt_file.name}",
            "sha256": ckpt_sha,
            "size_bytes": ckpt_size,
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
        }
        if not any(u.get("sha256") == ckpt_sha for u in verified_units_ledger):
            verified_units_ledger.append(unit_entry)
        ckpt_meta_file = ckpt_dir / f"sitl_state_{eff_tag}.meta.json"
        if not ckpt_meta_file.exists() and not Path(f"{ckpt_meta_file}.complete").exists():
            immutable_meta = {
                **latest_info,
                "namespace": active_train_namespace,
                "probe_namespace": active_probe_namespace,
                "world_seeds": [int(s) for s in world_seeds],
                "verified_units": list(verified_units_ledger),
            }
            write_atomic_with_sha256(
                ckpt_meta_file,
                lambda tmp: tmp.write_text(
                    json.dumps(immutable_meta, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                ),
                write_complete_marker=True,
            )
        last_latest_ckpt_info = dict(latest_info)
        _write_verified_ledger(latest_info, mark_complete=False)
        last_ckpt_state = (int(updates_completed), int(collection_steps))
        return str(ckpt_file)

    def _flush_decision_window(*, is_partial_final: bool = False) -> dict[str, Any] | None:
        nonlocal window_action_counts, window_decisions, tripwire_breached, tripwire_breach_reason
        if window_decisions <= 0:
            return None
        max_share = max(window_action_counts) / float(window_decisions)
        enforce_action_share = bool(
            not is_partial_final
            or window_decisions >= int(decision_window_size)
            or len(completed_windows) == 0
        )
        win_pass = bool(
            non_finite_loss_count == 0
            and non_finite_q_count == 0
            and non_finite_grad_count == 0
            and illegal_action_count == 0
            and (not enforce_action_share or max_share < REGISTERED_MAX_SINGLE_ACTION_SHARE)
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
            tripwire_breached = True
            tripwire_breach_reason = f"non_finite_q_at_update_{next_u}"
            return
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

        req_for_u = required_unique_rescues_for_update(
            updates_completed,
            warmup_rescue_roots=int(warmup_rescue_roots),
            min_unique_rescue_roots=int(min_unique_rescue_roots),
            total_updates=int(total_updates),
        )
        update_history.append(
            {
                "update": int(updates_completed),
                "collection_steps": int(collection_steps),
                "required_unique_rescue_roots": int(req_for_u),
                "unique_mined_rescue_roots": len(unique_rescue_keys),
                "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
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
            if not bool(tw_res["all_pass"]):
                tripwire_breached = True
                tripwire_breach_reason = f"probe_tripwire_breach_at_update_{updates_completed}"

        now_wall = time.perf_counter()
        if updates_completed % int(checkpoint_every_updates) == 0 or (
            now_wall - last_ckpt_wall
        ) >= float(checkpoint_every_seconds):
            _save_training_checkpoint(f"u{updates_completed:04d}")
            last_ckpt_wall = time.perf_counter()

    try:
        while updates_completed < int(total_updates):
            if tripwire_breached:
                break
            if signal_requested["stop"] or (
                pause_after_seconds is not None
                and (time.perf_counter() - t_start) >= float(pause_after_seconds)
            ):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(
                        f"sig_u{updates_completed:04d}_s{collection_steps:06d}"
                    )
                paused_early = True
                break
            if stop_after_updates is not None and updates_completed >= int(stop_after_updates):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(f"u{updates_completed:04d}")
                break
            if stop_after_collection_steps is not None and collection_steps >= int(
                stop_after_collection_steps
            ):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(f"u{updates_completed:04d}_s{collection_steps:06d}")
                break

            # Run learner updates strictly paced by required unique rescue roots (512 at update 1
            # rising linearly to 5,000 at update 1,000); no step-count fallback.
            while (
                updates_completed < int(total_updates)
                and not tripwire_breached
                and not signal_requested["stop"]
                and (
                    pause_after_seconds is None
                    or (time.perf_counter() - t_start) < float(pause_after_seconds)
                )
                and (stop_after_updates is None or updates_completed < int(stop_after_updates))
            ):
                next_u = updates_completed + 1
                req_rescues = required_unique_rescues_for_update(
                    next_u,
                    warmup_rescue_roots=int(warmup_rescue_roots),
                    min_unique_rescue_roots=int(min_unique_rescue_roots),
                    total_updates=int(total_updates),
                )
                warmup_ok = (
                    len(unique_rescue_keys) >= int(warmup_rescue_roots)
                    and rescue_buf.size >= 1
                    and ordinary_buf.size >= 1
                )
                pacing_ok = len(unique_rescue_keys) >= req_rescues
                if warmup_ok and pacing_ok:
                    _run_single_learner_update(next_u)
                else:
                    break

            if updates_completed >= int(total_updates):
                break
            if tripwire_breached:
                break
            if signal_requested["stop"] or (
                pause_after_seconds is not None
                and (time.perf_counter() - t_start) >= float(pause_after_seconds)
            ):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(
                        f"sig_u{updates_completed:04d}_s{collection_steps:06d}"
                    )
                paused_early = True
                break
            if stop_after_updates is not None and updates_completed >= int(stop_after_updates):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(f"u{updates_completed:04d}")
                break
            if stop_after_collection_steps is not None and collection_steps >= int(
                stop_after_collection_steps
            ):
                if (
                    last_ckpt_state != (int(updates_completed), int(collection_steps))
                    or last_latest_ckpt_info is None
                ):
                    _save_training_checkpoint(f"u{updates_completed:04d}_s{collection_steps:06d}")
                break
            if collection_steps >= int(max_collection_steps):
                runaway_guard_reached = True
                runaway_guard_reason = (
                    f"max_collection_steps_reached:{collection_steps}>={max_collection_steps}"
                )
                _save_training_checkpoint(f"guard_u{updates_completed:04d}_s{collection_steps:06d}")
                break
            if max_wall_seconds is not None and (time.perf_counter() - t_start) >= float(
                max_wall_seconds
            ):
                runaway_guard_reached = True
                elapsed_wall = time.perf_counter() - t_start
                runaway_guard_reason = (
                    f"max_wall_seconds_reached:{elapsed_wall:.1f}>={max_wall_seconds}"
                )
                _save_training_checkpoint(f"guard_u{updates_completed:04d}_s{collection_steps:06d}")
                break

            # Online collection step: student acts on-sim, batched_open_stop_h32 labels alive roots
            with torch.inference_mode():
                sim.prepare()
                policy.vector_runtime.prepare(sim)
                policy.vector_prepared = True

            obs = extract_hero_root_observations(sim, cpu_fast_flood=cpu_fast_flood)
            alive_mask = obs["hero_alive"]
            alive_rows = alive_mask.nonzero(as_tuple=False).squeeze(1)
            n_alive = int(alive_rows.numel())

            for env_i in range(int(sim.E)):
                if not bool(alive_mask[env_i].item()):
                    in_rescue_run_by_env[env_i] = False

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
                alive_list = [int(i) for i in alive_rows.detach().cpu().tolist()]
                if n_alive == int(sim.E):
                    sim_eval = sim
                    pol_eval = policy
                    seeds_eval = [int(s) for s in world_seeds]
                    decs_eval = [int(x) for x in obs["decision_frame"].detach().cpu().tolist()]
                else:
                    with torch.inference_mode():
                        sim_eval = sim.clone()
                        pol_eval = policy.repeat_interleave(1)
                        if pol_eval.vector_runtime is not None:
                            pol_eval.vector_runtime.__class__ = FastDeviceVectorRuntime
                        compact_sim_and_policy(sim_eval, pol_eval, alive_rows)
                    seeds_eval = [int(world_seeds[i]) for i in alive_list]
                    decs_eval = [int(obs["decision_frame"][i].item()) for i in alive_list]

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
                    env_idx = alive_list[local_i]
                    ws_i = int(seeds_eval[local_i])
                    dec_i = int(decs_eval[local_i])
                    if ws_i in probe_seed_set:
                        raise RuntimeError(
                            f"Refusing to insert probe32 seed {ws_i} into training buffer"
                        )
                    b_act = int(bare_acts[local_i].item())
                    s_act = int(astar_acts[local_i].item())
                    stu_a = int(stu_acts_cpu[local_i].item())

                    if bool(rescue_m[local_i].item()):
                        unique_rescue_keys.add((ws_i, dec_i))
                        if not in_rescue_run_by_env[env_idx]:
                            in_rescue_run_by_env[env_idx] = True
                            ep_seq = rescue_episode_seq_by_env[env_idx]
                            rescue_episode_seq_by_env[env_idx] = ep_seq + 1
                            unique_rescue_episode_keys.add(
                                (ws_i, hero_life_id_by_env[env_idx], ep_seq)
                            )
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
                    else:
                        in_rescue_run_by_env[env_idx] = False
                        if bool(ordinary_m[local_i].item()):
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
                done_hero = sim._last_done[:, 0].detach().cpu().tolist()
                for env_i, is_done in enumerate(done_hero):
                    if bool(is_done):
                        hero_life_id_by_env[env_i] += 1
                        in_rescue_run_by_env[env_i] = False
            collection_steps += 1

            # Periodic wall-clock checkpointing (<= 15 minutes) during warmup and collection
            now_wall = time.perf_counter()
            if (now_wall - last_ckpt_wall) >= float(checkpoint_every_seconds):
                _save_training_checkpoint(f"u{updates_completed:04d}_s{collection_steps:06d}")
                last_ckpt_wall = time.perf_counter()
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if hasattr(signal, "SIGUSR1") and prev_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, prev_sigusr1)

    if (
        paused_early
        or (
            stop_after_updates is not None
            and (
                updates_completed < int(total_updates)
                or len(unique_rescue_keys) < int(min_unique_rescue_roots)
            )
        )
        or (
            stop_after_collection_steps is not None
            and (
                updates_completed < int(total_updates)
                or len(unique_rescue_keys) < int(min_unique_rescue_roots)
            )
        )
    ) and not runaway_guard_reached:
        pub_ledger = _write_verified_ledger(last_latest_ckpt_info, mark_complete=True)
        return {
            "status": "PAUSED_FOR_RESUME",
            "updates_completed": int(updates_completed),
            "collection_steps": int(collection_steps),
            "unique_mined_rescue_roots": len(unique_rescue_keys),
            "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
            "latest_checkpoint_meta": str(latest_ckpt_meta_path),
            "verified_ledger_path": str(pub_ledger),
        }

    # Save full resumable state BEFORE flushing the terminal partial decision window so
    # completed_windows inside sitl_state_final.pt remains clean if ever resumed.
    final_state_path = _save_training_checkpoint("final")

    # Flush final partial decision window
    _flush_decision_window(is_partial_final=True)

    # Save final student checkpoint in served Ego2sNet format
    final_student_ckpt = out_dir / (
        "shakeout_student_final.pth" if is_shakeout else "b8_sitl_student_final.pth"
    )
    write_atomic_with_sha256(
        final_student_ckpt,
        lambda tmp: save_ego2s_checkpoint(
            tmp,
            student_net,
            meta={
                "study": (
                    "t34-gpu-sitl-b1b-shakeout-not-evidence"
                    if is_shakeout
                    else "t34-gpu-sitl-b1b-20261109"
                ),
                "namespace": active_train_namespace,
                "shakeout": is_shakeout,
                "not_evidence": bool(is_shakeout or dry_run),
                "dry_run": bool(dry_run),
                "updates_completed": int(updates_completed),
                "unique_mined_rescue_roots": len(unique_rescue_keys),
                "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
                "parent_b8_sha256": b8_sha,
            },
        ),
        write_complete_marker=True,
    )
    pub_final_ledger = _write_verified_ledger(last_latest_ckpt_info, mark_complete=True)

    # End-of-pilot training-set aliasing diagnostic
    if rescue_buf.size >= 1 and ordinary_buf.size >= 1:
        aliasing_diag = evaluate_training_set_aliasing_diagnostic(
            student_net,
            rescue_buf,
            ordinary_buf,
            device=dev,
        )
    else:
        aliasing_diag = {
            "rescue_buffer_evaluated_count": int(rescue_buf.size),
            "ordinary_buffer_evaluated_count": int(ordinary_buf.size),
            "train_rescue_safe_agreement_count": 0,
            "train_rescue_safe_agreement_rate": 0.0,
            "train_rescue_exact_a_star_count": 0,
            "train_rescue_exact_a_star_rate": 0.0,
            "train_ordinary_flip_count": 0,
            "train_ordinary_flip_rate": 0.0,
            "train_ordinary_unsafe_flip_count": 0,
            "train_ordinary_unsafe_flip_rate": 0.0,
            "rescue_agreement_threshold": ALIASING_RESCUE_AGREEMENT_THRESHOLD,
            "ordinary_flip_threshold": ALIASING_ORDINARY_FLIP_THRESHOLD,
            "stop_ego2s_bv8_inputs_insufficient": False,
            "verdict": "INSUFFICIENT_SAMPLES_DUE_TO_EARLY_STOP",
        }

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
    pilot_verdict = compute_step_b1b_pilot_verdict(
        runaway_guard_reached=bool(runaway_guard_reached),
        updates_completed=int(updates_completed),
        total_updates=int(total_updates),
        completed_windows=completed_windows,
        delivered_boost_band_pass=bool(delivered_boost_band_pass),
        non_finite_loss_count=int(non_finite_loss_count),
        non_finite_q_count=int(non_finite_q_count),
        non_finite_grad_count=int(non_finite_grad_count),
        illegal_action_count=int(illegal_action_count),
        tripwire_history=tripwire_history,
        tripwire_breached=bool(tripwire_breached),
        stop_ego2s_bv8_inputs_insufficient=bool(
            aliasing_diag["stop_ego2s_bv8_inputs_insufficient"]
        ),
        overall_max_single_action_share=float(overall_max_single_action_share),
    )

    elapsed_s = float(time.perf_counter() - t_start)
    result_summary: dict[str, Any] = {
        "schema": (
            "t34-gpu-sitl-shakeout-train-subresult/v1"
            if is_shakeout
            else "t34-gpu-sitl-b1b-pilot-result/v1"
        ),
        "namespace": active_train_namespace,
        "probe_namespace": probe_bank["meta"]["namespace"],
        "shakeout": is_shakeout,
        "not_evidence": bool(is_shakeout or dry_run),
        "dry_run": bool(dry_run),
        "evidence_class": (
            "SHAKEOUT_NOT_EVIDENCE"
            if is_shakeout
            else ("DRY_RUN_NON_EVIDENCE" if dry_run else "REGISTERED_STEP_B1B_PILOT_EVIDENCE")
        ),
        "verdict": pilot_verdict,
        "runaway_guard_reached": bool(runaway_guard_reached),
        "runaway_guard_reason": runaway_guard_reason,
        "stop_ego2s_bv8_inputs_insufficient": bool(
            aliasing_diag["stop_ego2s_bv8_inputs_insufficient"]
        ),
        "device": str(dev),
        "native_extensions": native_extensions,
        "declared_torch_cuda_arch_list": native_extensions.get("torch_cuda_arch_list"),
        "effective_build_cuda_arch": native_extensions.get("effective_build_cuda_arch"),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "verified_ledger_path": str(pub_final_ledger),
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
            "unique_mined_rescue_roots_by_world_frame": len(unique_rescue_keys),
            "unique_mined_rescue_episodes": len(unique_rescue_episode_keys),
            "rescue_buffer_size": int(rescue_buf.size),
            "rescue_buffer_total_added": int(rescue_buf.total_added),
            "ordinary_buffer_size": int(ordinary_buf.size),
            "ordinary_buffer_total_added": int(ordinary_buf.total_added),
        },
        "update_pacing_schedule": [
            {
                "update": int(u["update"]),
                "collection_steps": int(u["collection_steps"]),
                "required_unique_rescue_roots": int(u["required_unique_rescue_roots"]),
                "unique_mined_rescue_roots": int(u["unique_mined_rescue_roots"]),
            }
            for u in update_history
        ],
        "rescue_quota_scoring": {
            "scored_metric": RESCUE_QUOTA_SCORING_METRIC,
            "scored_count": len(unique_rescue_keys),
            "distinct_episode_count": len(unique_rescue_episode_keys),
            "target_min_unique_rescues": int(min_unique_rescue_roots),
            "scored_quota_met": bool(len(unique_rescue_keys) >= int(min_unique_rescue_roots)),
            "distinct_episode_quota_met": bool(
                len(unique_rescue_episode_keys) >= int(min_unique_rescue_roots)
            ),
            "registration_quote": RESCUE_QUOTA_REGISTRATION_QUOTES,
            "registration_quote_line": 143,
            "registration_verbatim_quotes": [dict(q) for q in REGISTRATION_VERBATIM_QUOTES],
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

    result_json_path = out_dir / (
        "shakeout_sitl_subresult.json" if is_shakeout else "sitl_pilot_result.json"
    )
    result_sha = write_atomic_with_sha256(
        result_json_path,
        lambda tmp: tmp.write_text(
            json.dumps(result_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
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
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--warmup-rescue-roots", type=int, default=None)
    parser.add_argument("--min-unique-rescue-roots", type=int, default=None)
    parser.add_argument("--rescue-batch-size", type=int, default=None)
    parser.add_argument("--ordinary-batch-size", type=int, default=None)
    parser.add_argument("--ordinary-uniform-half", type=int, default=None)
    parser.add_argument("--ordinary-hard-neg-half", type=int, default=None)
    parser.add_argument("--branch-horizon", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--tripwire-interval", type=int, default=None)
    parser.add_argument("--decision-window-size", type=int, default=None)
    parser.add_argument("--checkpoint-every-updates", type=int, default=None)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=900.0)
    parser.add_argument("--max-collection-steps", type=int, default=200_000)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--stop-after-updates", type=int, default=None)
    parser.add_argument("--stop-after-collection-steps", type=int, default=None)
    parser.add_argument("--pause-after-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shakeout", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    is_dry = bool(args.dry_run or args.shakeout)
    summary = run_sitl_training(
        args.output_dir,
        probe_bank_path=args.probe_bank_path,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        ledger_path=args.ledger,
        input_dir=args.input_dir,
        device=args.device,
        num_envs=args.num_envs if args.num_envs is not None else (4 if is_dry else REGISTERED_ENVS),
        total_updates=(
            args.total_updates
            if args.total_updates is not None
            else (4 if is_dry else REGISTERED_TOTAL_UPDATES)
        ),
        warmup_rescue_roots=(
            args.warmup_rescue_roots
            if args.warmup_rescue_roots is not None
            else (1 if is_dry else REGISTERED_WARMUP_RESCUE_ROOTS)
        ),
        min_unique_rescue_roots=(
            args.min_unique_rescue_roots
            if args.min_unique_rescue_roots is not None
            else (2 if is_dry else REGISTERED_MIN_UNIQUE_RESCUE_ROOTS)
        ),
        rescue_batch_size=(
            args.rescue_batch_size
            if args.rescue_batch_size is not None
            else (2 if is_dry else REGISTERED_RESCUE_BATCH_SIZE)
        ),
        ordinary_batch_size=(
            args.ordinary_batch_size
            if args.ordinary_batch_size is not None
            else (4 if is_dry else REGISTERED_ORDINARY_BATCH_SIZE)
        ),
        ordinary_uniform_half=(
            args.ordinary_uniform_half
            if args.ordinary_uniform_half is not None
            else (2 if is_dry else REGISTERED_ORDINARY_UNIFORM_HALF)
        ),
        ordinary_hard_neg_half=(
            args.ordinary_hard_neg_half
            if args.ordinary_hard_neg_half is not None
            else (2 if is_dry else REGISTERED_ORDINARY_HARD_NEG_HALF)
        ),
        branch_horizon=(
            args.branch_horizon
            if args.branch_horizon is not None
            else (2 if is_dry else OPEN_STOP_BRANCH_HORIZON)
        ),
        learning_rate=args.learning_rate,
        tripwire_interval=(
            args.tripwire_interval
            if args.tripwire_interval is not None
            else (2 if is_dry else REGISTERED_TRIPWIRE_INTERVAL)
        ),
        decision_window_size=(
            args.decision_window_size
            if args.decision_window_size is not None
            else (10 if is_dry else REGISTERED_DECISION_WINDOW_SIZE)
        ),
        checkpoint_every_updates=(
            args.checkpoint_every_updates
            if args.checkpoint_every_updates is not None
            else (2 if is_dry else 100)
        ),
        checkpoint_every_seconds=args.checkpoint_every_seconds,
        max_collection_steps=args.max_collection_steps,
        max_wall_seconds=args.max_wall_seconds,
        stop_after_updates=args.stop_after_updates,
        stop_after_collection_steps=args.stop_after_collection_steps,
        pause_after_seconds=args.pause_after_seconds,
        seed=args.seed,
        resume=not bool(args.no_resume),
        dry_run=bool(args.dry_run),
        shakeout=bool(args.shakeout),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
