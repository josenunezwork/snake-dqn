"""End-to-end deterministic world-disjoint Gate-1 pipeline with resume-by-default unit exports.

Temporal anti-leakage invariant within each world-disjoint replicate ``r``:
  1. Harvest ``fit`` and ``val`` Watch6-v2 worlds + branch ``fit`` mature roots;
  2. Fit ``LengthHorizonBaseline`` on ``fit`` (regularization selected on ``val``) and train
     ``PrivilegedFactoredCritic`` on ``fit`` (best epoch selected on ``val``);
  3. Freeze critic weights + record ``frozen_critic_sha256`` BEFORE ``test`` worlds exist in RAM;
  4. Harvest held-out ``test`` worlds + branch held-out ``test`` roots across disjoint CRN
     selection and scoring suffixes, and run a single frozen evaluation pass;
  5. Export compact immutable per-world sufficient statistics & per-root H300 records with
     ``.sha256`` and ``.complete`` published last.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from research.sandbox_a_resident.ct_branching import compute_q_from_survival_and_mass
from research.t4_gate1 import (
    ALLOCATION_INDEX,
    ALLOCATION_NAME,
    B8_S3_SHA256,
    FRP3_S12_SHA256,
    STUDY,
    UNIT_KIND,
    replicate_unit_id,
    replicate_unit_stem,
)
from research.t4_gate1.checkpoints import (
    ReplicateCheckpointManager,
    branch_config_fingerprint,
    critic_config_fingerprint,
    harvest_config_fingerprint,
)
from research.t4_gate1.featurizer import (
    CompactBoardDataset,
    DeviceBoardDataset,
    featurize_device_batch,
    slice_featurized_batch,
)
from research.t4_gate1.harvest import (
    BranchedRootsTable,
    PreparedRootSnapshot,
    WorldPartitionSpec,
    branch_prepared_roots,
    harvest_world_partition,
    harvest_world_partitions_batched,
)
from research.t4_gate1.model import (
    LengthHorizonBaseline,
    PrivilegedFactoredCritic,
    _float32_no_autocast,
    discrete_survival_nll,
)
from research.t4_gate1.seed_audit import world_allocation
from tools.ops.package_spec import digest, publish_json, verify_immutable

LEDGER_SCHEMA = "t34-t4-gate1-verified-ledger/v1"
UNIT_SCHEMA = "t34-t4-gate1-replicate-unit/v1"

EVAL_STRATA: tuple[str, ...] = (
    "l_ge_600",
    "l_ge_800",
    "l_lt_600",
    "doomed_l_ge_600",
    "doomed_all",
    "all_lengths",
)


def publish_verified_ledger(
    path: Path,
    entries: Mapping[int, Mapping[str, Any]],
    *,
    namespace: str,
) -> dict[str, Any]:
    """Publish an immutable resume-by-default verified ledger with sidecars."""
    normalized: dict[str, dict[str, Any]] = {}
    for rep_idx, info in sorted(entries.items(), key=lambda kv: int(kv[0])):
        r_int = int(rep_idx)
        uid = replicate_unit_id(namespace, r_int)
        if str(info.get("unit_id", uid)) != uid:
            raise ValueError(f"Ledger unit_id mismatch for replicate {r_int}")
        normalized[str(r_int)] = {
            "unit_id": uid,
            "replicate_index": r_int,
            "sha256": str(info["sha256"]),
            "size_bytes": int(info["size_bytes"]),
        }
    payload = {
        "schema": LEDGER_SCHEMA,
        "study": STUDY,
        "namespace": namespace,
        "allocation": ALLOCATION_NAME,
        "allocation_index": ALLOCATION_INDEX,
        "unit_kind": UNIT_KIND,
        "entries": normalized,
    }
    return publish_json(path, payload)


def load_and_validate_ledger(
    ledger_path: Path,
    *,
    expected_namespace: str,
    valid_replicates: set[int],
) -> set[int]:
    """Verify ledger sidecars and return verified replicate indices to skip."""
    verify_immutable(ledger_path)
    doc = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"Foreign or malformed ledger schema in {ledger_path}")
    if (
        doc.get("study") != STUDY
        or doc.get("namespace") != expected_namespace
        or doc.get("allocation") != ALLOCATION_NAME
        or doc.get("allocation_index") != ALLOCATION_INDEX
        or doc.get("unit_kind") != UNIT_KIND
    ):
        raise ValueError(
            f"Refusing foreign ledger namespace/allocation in {ledger_path}: "
            f"got namespace={doc.get('namespace')!r}, expected {expected_namespace!r}"
        )
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Ledger entries must be a JSON object")

    skipped: set[int] = set()
    for key, item in entries.items():
        if not isinstance(item, dict):
            raise ValueError(f"Malformed ledger entry for {key!r}")
        r_idx = int(item.get("replicate_index", -1))
        if str(r_idx) != str(key) or r_idx not in valid_replicates:
            raise ValueError(f"Ledger entry {key!r} outside frozen replicate allocation")
        expected_uid = replicate_unit_id(expected_namespace, r_idx)
        if item.get("unit_id") != expected_uid:
            raise ValueError(f"Ledger entry unit_id mismatch for replicate {r_idx}")
        sha = str(item.get("sha256", ""))
        size_b = int(item.get("size_bytes", 0))
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or size_b <= 0:
            raise ValueError(f"Invalid digest/size in ledger entry {r_idx}")
        skipped.add(r_idx)
    return skipped


@contextlib.contextmanager
def _critic_training_precision_scope(device: torch.device) -> Iterator[None]:
    """Enable CUDA TF32 during neural critic training and restore exact DeviceSim flags on exit."""
    if device.type != "cuda":
        yield
        return
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_fp16_red = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    prev_bf16_red = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    prev_prec = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev_fp16_red
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_bf16_red
        torch.set_float32_matmul_precision(prev_prec)


def _select_critic_action(
    s_hat_row: np.ndarray,
    m_hat_row: np.ndarray,
    legal_row: np.ndarray,
    b8_act: int,
) -> tuple[int, np.ndarray]:
    """Select ``argmax_{a legal} [S_hat + 0.25 * M_hat / 1000]`` with B8 tie-break."""
    q_row = compute_q_from_survival_and_mass(
        s_hat_row[None, :], m_hat_row[None, :], legal_row[None, :]
    )[0]
    legal_idxs = np.flatnonzero(legal_row)
    if legal_idxs.size == 0:
        return int(b8_act), q_row
    q_max = float(np.max(q_row[legal_idxs]))
    tied = legal_idxs[q_row[legal_idxs] == q_max]
    chosen = int(b8_act) if int(b8_act) in tied else int(tied[0])
    return chosen, q_row


def _assert_finite_dataset_targets(dataset: CompactBoardDataset, *, label: str) -> None:
    """Reject any non-finite supervised targets before critic/baseline training or evaluation."""
    if dataset.num_rows == 0:
        return
    checks = (
        ("h_rem", dataset.h_rem),
        ("tau_alive", dataset.tau_alive),
        ("return_g", dataset.return_g),
        ("forage_short", dataset.forage_short),
        ("forage_short_weight", dataset.forage_short_weight),
    )
    for name, arr in checks:
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"Non-finite {name} values in {label}")
    valid_long = np.asarray(dataset.forage_long_valid, dtype=bool)
    if bool(valid_long.any()) and not bool(np.isfinite(dataset.forage_long[valid_long]).all()):
        raise ValueError(f"Non-finite forage_long values on valid rows in {label}")


def _assert_valid_branched_roots(
    branched: BranchedRootsTable,
    *,
    require_score: bool,
    label: str,
) -> None:
    """Ensure every root has >=1 legal action and strictly finite legal-action branch outcomes."""
    if branched.num_roots == 0:
        return
    leg = np.asarray(branched.legal_mask, dtype=bool)
    if leg.shape != (branched.num_roots, 6) or not bool(np.all(np.any(leg, axis=1))):
        raise ValueError(f"Root with zero legal actions in {label}")
    if not (
        bool(np.isfinite(branched.s_sel[leg]).all())
        and bool(np.isfinite(branched.m_sel[leg]).all())
        and bool(np.isfinite(branched.q_sel[leg]).all())
    ):
        raise ValueError(f"Non-finite selection branch target on legal action in {label}")
    if require_score and not (
        bool(np.isfinite(branched.s_score[leg]).all())
        and bool(np.isfinite(branched.m_score[leg]).all())
        and bool(np.isfinite(branched.q_score[leg]).all())
    ):
        raise ValueError(f"Non-finite scoring branch outcome on legal action in {label}")


@torch.inference_mode()
def predict_critic_on_dataset(
    critic: PrivilegedFactoredCritic,
    dataset: CompactBoardDataset | DeviceBoardDataset,
    *,
    batch_size: int = 256,
    device: torch.device,
    h_rem_override: float | None = None,
) -> dict[str, np.ndarray]:
    """Run frozen ``critic`` over ``dataset`` in mini-batches and collect predictions."""
    critic.eval()
    n = dataset.num_rows
    if n == 0:
        return {
            "v_pred": np.zeros(0, dtype=np.float64),
            "q_all": np.zeros((0, 6), dtype=np.float64),
            "greedy_action": np.zeros(0, dtype=np.int64),
            "s_hrem_all": np.zeros((0, 6), dtype=np.float64),
            "s_hrem_logits_all": np.zeros((0, 6), dtype=np.float64),
            "s_300_all": np.zeros((0, 6), dtype=np.float64),
            "s_300_logits_all": np.zeros((0, 6), dtype=np.float64),
            "m_300_all": np.zeros((0, 6), dtype=np.float64),
            "hazard_nll": np.zeros(0, dtype=np.float64),
        }

    dev_ds = (
        dataset
        if isinstance(dataset, DeviceBoardDataset) and dataset.device == device
        else (
            DeviceBoardDataset.from_compact(dataset, device)
            if isinstance(dataset, CompactBoardDataset)
            else pytest_safe_move_device_dataset(dataset, device)
        )
    )

    v_pred = np.zeros(n, dtype=np.float64)
    q_all = np.zeros((n, 6), dtype=np.float64)
    greedy_action = np.zeros(n, dtype=np.int64)
    s_hrem_all = np.zeros((n, 6), dtype=np.float64)
    s_hrem_logits_all = np.zeros((n, 6), dtype=np.float64)
    s_300_all = np.zeros((n, 6), dtype=np.float64)
    s_300_logits_all = np.zeros((n, 6), dtype=np.float64)
    m_300_all = np.zeros((n, 6), dtype=np.float64)
    haz_nll = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx_t = torch.arange(start, end, dtype=torch.long, device=device)
        fb = featurize_device_batch(dev_ds, idx_t, h_rem_override=h_rem_override)
        out = critic(fb, h_rem_override=h_rem_override)
        q_f = out["q_life"].float()
        greedy_t = q_f.argmax(dim=-1)
        b_arange = torch.arange(end - start, device=device)
        acts_t = dev_ds.b8_action[idx_t]
        v_pred_t = q_f[b_arange, acts_t]
        q_all[start:end] = q_f.detach().cpu().numpy().astype(np.float64)
        v_pred[start:end] = v_pred_t.detach().cpu().numpy().astype(np.float64)
        greedy_action[start:end] = greedy_t.detach().cpu().numpy().astype(np.int64)
        s_hrem_all[start:end] = out["s_hrem"].float().detach().cpu().numpy().astype(np.float64)
        s_hrem_logits_all[start:end] = (
            out["s_hrem_logits"].float().detach().cpu().numpy().astype(np.float64)
        )
        s_300_all[start:end] = out["s_300"].float().detach().cpu().numpy().astype(np.float64)
        s_300_logits_all[start:end] = (
            out["s_300_logits"].float().detach().cpu().numpy().astype(np.float64)
        )
        m_300_all[start:end] = out["m_300"].float().detach().cpu().numpy().astype(np.float64)

        haz_taken = out["hazard_logits"][b_arange, acts_t].float()
        nll_b = discrete_survival_nll(
            haz_taken,
            dev_ds.tau_alive[idx_t],
            dev_ds.died_in_session[idx_t],
        )
        haz_nll[start:end] = nll_b.float().detach().cpu().numpy().astype(np.float64)

    result = {
        "v_pred": v_pred,
        "q_all": q_all,
        "greedy_action": greedy_action,
        "s_hrem_all": s_hrem_all,
        "s_hrem_logits_all": s_hrem_logits_all,
        "s_300_all": s_300_all,
        "s_300_logits_all": s_300_logits_all,
        "m_300_all": m_300_all,
        "hazard_nll": haz_nll,
    }
    for key, arr in result.items():
        if key != "greedy_action" and not bool(np.isfinite(arr).all()):
            raise ValueError(f"Non-finite critic prediction in {key}")
    return result


def pytest_safe_move_device_dataset(
    dataset: DeviceBoardDataset, device: torch.device
) -> DeviceBoardDataset:
    """Move an existing ``DeviceBoardDataset`` if requested on a different target device."""
    return DeviceBoardDataset(
        num_rows=dataset.num_rows,
        device=device,
        h_rem=dataset.h_rem.to(device),
        b8_action=dataset.b8_action.to(device),
        legal_mask=dataset.legal_mask.to(device),
        snake_alive=dataset.snake_alive.to(device),
        snake_length=dataset.snake_length.to(device),
        snake_seg_count=dataset.snake_seg_count.to(device),
        snake_dir=dataset.snake_dir.to(device),
        snake_boost_frames=dataset.snake_boost_frames.to(device),
        snake_hunger=dataset.snake_hunger.to(device),
        snake_respawn=dataset.snake_respawn.to(device),
        snake_boosted=dataset.snake_boosted.to(device),
        head_xy=dataset.head_xy.to(device),
        tail_xy=dataset.tail_xy.to(device),
        seg_row_offsets=dataset.seg_row_offsets.to(device),
        seg_row_counts=dataset.seg_row_counts.to(device),
        seg_slot=dataset.seg_slot.to(device),
        seg_x=dataset.seg_x.to(device),
        seg_y=dataset.seg_y.to(device),
        seg_ttl=dataset.seg_ttl.to(device),
        food_row_offsets=dataset.food_row_offsets.to(device),
        food_row_counts=dataset.food_row_counts.to(device),
        food_x=dataset.food_x.to(device),
        food_y=dataset.food_y.to(device),
        food_is_corpse=dataset.food_is_corpse.to(device),
        tau_alive=dataset.tau_alive.to(device),
        died_in_session=dataset.died_in_session.to(device),
        return_g=dataset.return_g.to(device),
        forage_short=dataset.forage_short.to(device),
        forage_short_weight=dataset.forage_short_weight.to(device),
        forage_long=dataset.forage_long.to(device),
        forage_long_valid=dataset.forage_long_valid.to(device),
    )


def train_privileged_critic(
    *,
    fit_dataset: CompactBoardDataset,
    val_dataset: CompactBoardDataset,
    fit_roots_dataset: CompactBoardDataset,
    fit_branched_roots: BranchedRootsTable,
    branch_horizon: int,
    min_mature_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    use_bf16_autocast: bool | None = None,
    early_stop_patience: int | None = None,
    min_epochs: int = 1,
    resume_state: Mapping[str, Any] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[PrivilegedFactoredCritic, dict[str, Any]]:
    """Train clean-slate ``PrivilegedFactoredCritic`` strictly on ``fit`` and select on ``val``."""
    t0 = time.perf_counter()
    _assert_finite_dataset_targets(fit_dataset, label="fit_dataset")
    _assert_finite_dataset_targets(val_dataset, label="val_dataset")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    critic = PrivilegedFactoredCritic().to(device)
    n_fit = fit_dataset.num_rows
    if n_fit == 0:
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        return critic, {"epochs": 0, "best_epoch": 0, "wall_sec": 0.0}

    n_branch_roots = fit_roots_dataset.num_rows
    if n_branch_roots > 0:
        if fit_branched_roots.num_roots != n_branch_roots:
            raise ValueError(
                f"fit_branched_roots.num_roots ({fit_branched_roots.num_roots}) != "
                f"fit_roots_dataset.num_rows ({n_branch_roots})"
            )
        _assert_valid_branched_roots(
            fit_branched_roots, require_score=False, label="fit_branched_roots"
        )

    var_g_fit = float(np.var(fit_dataset.return_g.astype(np.float64)))
    if not math.isfinite(var_g_fit):
        raise ValueError("Non-finite return_g variance in fit_dataset")
    ret_scale = max(var_g_fit, 1e-3)

    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    steps_per_epoch = max(1, (n_fit + batch_size - 1) // batch_size)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(learning_rate) * 0.05
    )

    if use_bf16_autocast is None:
        use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported()) or (
            device.type == "cpu" and torch.is_autocast_enabled("cpu")
        )
    else:
        use_bf16 = bool(use_bf16_autocast) and device.type in ("cuda", "cpu")
    branch_batch_size = min(32, n_branch_roots) if n_branch_roots > 0 else 0

    # Stage compact CSR datasets once on target device (zero per-step CPU NumPy / H2D syncs).
    fit_dev = DeviceBoardDataset.from_compact(fit_dataset, device)
    val_dev = DeviceBoardDataset.from_compact(val_dataset, device)
    if branch_batch_size > 0:
        roots_dev = DeviceBoardDataset.from_compact(fit_roots_dataset, device)
        roots_fb = featurize_device_batch(
            roots_dev,
            torch.arange(n_branch_roots, dtype=torch.long, device=device),
            h_rem_override=float(branch_horizon),
        )
        branch_leg_dev = torch.as_tensor(
            np.asarray(fit_branched_roots.legal_mask, dtype=bool),
            dtype=torch.bool,
            device=device,
        )
        branch_s_targ_dev = torch.as_tensor(
            np.nan_to_num(fit_branched_roots.s_sel, nan=0.0),
            dtype=torch.float32,
            device=device,
        )
        branch_q_targ_dev = torch.as_tensor(
            np.nan_to_num(fit_branched_roots.m_sel, nan=0.0) * (float(branch_horizon) / 1000.0),
            dtype=torch.float32,
            device=device,
        )
    else:
        roots_fb = None
        branch_leg_dev = None
        branch_s_targ_dev = None
        branch_q_targ_dev = None

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
    epoch_history: list[dict[str, Any]] = []
    resumed_from_epoch = 0

    if resume_state is not None:
        critic.load_state_dict(
            {
                k: v.to(device=device, dtype=torch.float32)
                for k, v in resume_state["model_state"].items()
            }
        )
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        rng.bit_generator.state = copy.deepcopy(dict(resume_state["np_rng_state"]))
        torch.set_rng_state(resume_state["torch_cpu_rng_state"].detach().cpu().to(torch.uint8))
        best_epoch = int(resume_state["best_epoch"])
        best_score = float(resume_state["best_score"])
        best_state = {
            k: v.detach().cpu().float().clone() for k, v in resume_state["best_state"].items()
        }
        epoch_history = [dict(x) for x in resume_state["epoch_history"]]
        resumed_from_epoch = int(resume_state["epoch"])
        start_epoch = resumed_from_epoch + 1

    eff_min_epochs = max(1, int(min_epochs))
    eff_patience = (
        int(early_stop_patience)
        if early_stop_patience is not None and int(early_stop_patience) > 0
        else None
    )
    early_stopped = bool(
        eff_patience is not None
        and resumed_from_epoch >= eff_min_epochs
        and best_epoch > 0
        and (resumed_from_epoch - best_epoch) >= eff_patience
        and resumed_from_epoch < int(epochs)
    )

    with _critic_training_precision_scope(device):
        for epoch in range(start_epoch, int(epochs) + 1) if not early_stopped else ():
            ep_t0 = time.perf_counter()
            critic.train()
            perm = rng.permutation(n_fit)
            sum_loss = 0.0
            sum_ret = 0.0
            sum_haz = 0.0
            n_batches = 0

            for start in range(0, n_fit, batch_size):
                idx = perm[start : start + batch_size]
                b_cur = int(idx.size)
                if b_cur == 0:
                    continue
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                fb = featurize_device_batch(fit_dev, idx_t)

                acts_t = fit_dev.b8_action[idx_t]
                tau_t = fit_dev.tau_alive[idx_t]
                died_t = fit_dev.died_in_session[idx_t]
                ret_t = fit_dev.return_g[idx_t]
                fs_t = fit_dev.forage_short[idx_t]
                fw_t = fit_dev.forage_short_weight[idx_t]
                fl_t = fit_dev.forage_long[idx_t]
                flv_t = fit_dev.forage_long_valid[idx_t]

                optimizer.zero_grad(set_to_none=True)
                amp_ctx = (
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                    if use_bf16
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    out = critic(fb)
                    if branch_batch_size > 0 and roots_fb is not None:
                        r_idx = rng.choice(n_branch_roots, size=branch_batch_size, replace=False)
                        r_idx_t = torch.as_tensor(r_idx, dtype=torch.long, device=device)
                        fb_r = slice_featurized_batch(roots_fb, r_idx_t)
                        out_r = critic(fb_r, h_rem_override=float(branch_horizon))
                    else:
                        r_idx_t = None
                        out_r = None

                with _float32_no_autocast(device):
                    b_ar = torch.arange(b_cur, device=device)
                    q_taken = out["q_life"][b_ar, acts_t].float()
                    haz_taken = out["hazard_logits"][b_ar, acts_t].float()
                    gs_taken = out["g_short"][b_ar, acts_t].float()
                    gl_taken = out["g_long"][b_ar, acts_t].float()

                    mature_w = torch.where(
                        fb.hero_length >= float(min_mature_length),
                        torch.full_like(ret_t, 2.0),
                        torch.ones_like(ret_t),
                    )
                    loss_ret = (mature_w * (q_taken - ret_t).square()).mean() / ret_scale
                    loss_haz = (discrete_survival_nll(haz_taken, tau_t, died_t) / 25.0).mean()

                    huber_s = F.huber_loss(gs_taken, fs_t, reduction="none", delta=0.03)
                    loss_fs = (huber_s * fw_t).sum() / fw_t.sum().clamp_min(1.0) * 50.0

                    if bool(flv_t.any()):
                        huber_l = F.huber_loss(
                            gl_taken[flv_t], fl_t[flv_t], reduction="mean", delta=0.02
                        )
                        loss_fl = huber_l * 50.0
                    else:
                        loss_fl = torch.zeros((), dtype=torch.float32, device=device)

                    loss_branch = torch.zeros((), dtype=torch.float32, device=device)
                    if (
                        r_idx_t is not None
                        and out_r is not None
                        and branch_leg_dev is not None
                        and branch_s_targ_dev is not None
                        and branch_q_targ_dev is not None
                    ):
                        leg_r = branch_leg_dev[r_idx_t]
                        if not bool(leg_r.any()):
                            raise ValueError("Empty legal action slice in fit_branched_roots batch")
                        s_targ_leg = branch_s_targ_dev[r_idx_t][leg_r]
                        q_targ_leg = branch_q_targ_dev[r_idx_t][leg_r]
                        s_logit_leg = out_r["s_hrem_logits"][leg_r].float().clamp(-12.0, 12.0)
                        q_pred_leg = out_r["q_life"][leg_r].float()
                        bce_r = F.binary_cross_entropy_with_logits(s_logit_leg, s_targ_leg)
                        mse_r = F.huber_loss(q_pred_leg, q_targ_leg, delta=0.10)
                        loss_branch = bce_r + 2.0 * mse_r

                    total_loss = (
                        loss_ret
                        + 0.5 * loss_haz
                        + 1.5 * loss_fs
                        + 1.0 * loss_fl
                        + 0.5 * loss_branch
                    )
                    if not bool(torch.isfinite(total_loss).item()):
                        raise ValueError(
                            "Non-finite total_loss encountered in train_privileged_critic"
                        )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                sum_loss += float(total_loss.item())
                sum_ret += float(loss_ret.item())
                sum_haz += float(loss_haz.item())
                n_batches += 1

            # Validation checkpoint selection strictly on val_dataset.
            val_preds = predict_critic_on_dataset(
                critic, val_dev, batch_size=batch_size, device=device
            )
            if val_dataset.num_rows > 0:
                val_l = val_dataset.snake_length[:, 0]
                val_mature = val_l >= min_mature_length
                sel_mask = (
                    val_mature if bool(np.any(val_mature)) else np.ones_like(val_mature, bool)
                )
                val_mse = float(
                    np.mean(
                        np.square(
                            val_dataset.return_g[sel_mask].astype(np.float64)
                            - val_preds["v_pred"][sel_mask]
                        )
                    )
                )
                val_nll = float(np.mean(val_preds["hazard_nll"][sel_mask]))
                val_score = (val_mse / ret_scale) + 0.10 * (val_nll / 25.0)
            else:
                val_mse = 0.0
                val_nll = 0.0
                val_score = sum_loss / max(n_batches, 1)

            if not (math.isfinite(val_mse) and math.isfinite(val_nll) and math.isfinite(val_score)):
                raise ValueError("Non-finite validation metric in train_privileged_critic")

            rec = {
                "epoch": epoch,
                "train_loss": round(sum_loss / max(n_batches, 1), 6),
                "train_ret_loss": round(sum_ret / max(n_batches, 1), 6),
                "train_haz_loss": round(sum_haz / max(n_batches, 1), 6),
                "val_mature_mse": round(val_mse, 6),
                "val_mature_hazard_nll": round(val_nll, 6),
                "val_selection_score": round(val_score, 6),
                "epoch_wall_sec": round(time.perf_counter() - ep_t0, 3),
            }
            epoch_history.append(rec)

            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}

            if log_progress is not None:
                log_progress({"phase": "critic_train", **rec})

            will_early_stop = bool(
                eff_patience is not None
                and epoch >= eff_min_epochs
                and best_epoch > 0
                and (epoch - best_epoch) >= eff_patience
                and epoch < int(epochs)
            )
            if on_epoch_end is not None:
                on_epoch_end(
                    {
                        "epoch": epoch,
                        "is_final_or_stopped": bool(will_early_stop or epoch == int(epochs)),
                        "critic": critic,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "np_rng_state": copy.deepcopy(rng.bit_generator.state),
                        "torch_cpu_rng_state": torch.get_rng_state().clone(),
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "best_state": best_state,
                        "epoch_history": list(epoch_history),
                    }
                )
            if will_early_stop:
                early_stopped = True
                if log_progress is not None:
                    log_progress(
                        {
                            "phase": "critic_early_stop",
                            "stopped_epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_val_selection_score": round(best_score, 6),
                            "patience": eff_patience,
                        }
                    )
                break

    critic.load_state_dict(best_state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    num_params = sum(int(p.numel()) for p in critic.parameters())
    summary = {
        "seed": int(seed),
        "epochs": int(epochs),
        "epochs_completed": len(epoch_history),
        "resumed_from_epoch": resumed_from_epoch,
        "early_stopped": bool(early_stopped),
        "early_stop_patience": eff_patience,
        "best_epoch": int(best_epoch),
        "best_val_selection_score": round(best_score, 6),
        "num_parameters": num_params,
        "b8_weights_embedded": False,
        "fit_rows": int(n_fit),
        "val_rows": int(val_dataset.num_rows),
        "fit_branched_roots": int(n_branch_roots),
        "use_bf16_autocast": bool(use_bf16),
        "wall_sec": round(time.perf_counter() - t0, 3),
        "epoch_history": epoch_history,
    }
    return critic, summary


def _stratum_mask(dataset: CompactBoardDataset, stratum: str) -> np.ndarray:
    l0 = dataset.snake_length[:, 0]
    died = dataset.died_in_session
    if stratum == "l_ge_600":
        return l0 >= 600
    if stratum == "l_ge_800":
        return l0 >= 800
    if stratum == "l_lt_600":
        return l0 < 600
    if stratum == "doomed_l_ge_600":
        return (l0 >= 600) & died
    if stratum == "doomed_all":
        return died.copy()
    if stratum == "all_lengths":
        return np.ones(dataset.num_rows, dtype=bool)
    raise ValueError(f"Unknown evaluation stratum: {stratum!r}")


def compute_world_sufficient_stats(
    *,
    world_ids: Sequence[int],
    world_seeds: Sequence[int],
    dataset: CompactBoardDataset,
    v_base: np.ndarray,
    v_critic: np.ndarray,
    hazard_nll: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute exact per-test-world sufficient statistics across all strata."""
    y = dataset.return_g.astype(np.float64)
    vb = np.asarray(v_base, dtype=np.float64)
    vc = np.asarray(v_critic, dtype=np.float64)
    hn = np.asarray(hazard_nll, dtype=np.float64)
    if dataset.num_rows > 0 and not (
        bool(np.isfinite(y).all())
        and bool(np.isfinite(vb).all())
        and bool(np.isfinite(vc).all())
        and bool(np.isfinite(hn).all())
    ):
        raise ValueError("Non-finite target or prediction in compute_world_sufficient_stats")
    err_b = y - vb
    err_c = y - vc

    strata_masks = {s: _stratum_mask(dataset, s) for s in EVAL_STRATA}
    records: list[dict[str, Any]] = []
    for wid, wseed in zip(world_ids, world_seeds):
        w_mask = dataset.world_id == int(wid)
        s_dict: dict[str, dict[str, Any]] = {}
        for s_name, s_mask in strata_masks.items():
            m = w_mask & s_mask
            cnt = int(np.sum(m))
            if cnt > 0:
                ym = y[m]
                eb = err_b[m]
                ec = err_c[m]
                s_dict[s_name] = {
                    "n": cnt,
                    "sum_y": float(np.sum(ym)),
                    "sum_y2": float(np.sum(ym * ym)),
                    "sse_base": float(np.sum(eb * eb)),
                    "sse_critic": float(np.sum(ec * ec)),
                    "sae_base": float(np.sum(np.abs(eb))),
                    "sae_critic": float(np.sum(np.abs(ec))),
                    "sum_hazard_nll": float(np.sum(hn[m])),
                }
            else:
                s_dict[s_name] = {
                    "n": 0,
                    "sum_y": 0.0,
                    "sum_y2": 0.0,
                    "sse_base": 0.0,
                    "sse_critic": 0.0,
                    "sae_base": 0.0,
                    "sae_critic": 0.0,
                    "sum_hazard_nll": 0.0,
                }
        records.append(
            {
                "world_id": int(wid),
                "seed": int(wseed),
                "strata": s_dict,
            }
        )
    return records


def build_held_out_root_records(
    *,
    roots: Sequence[PreparedRootSnapshot],
    roots_dataset: CompactBoardDataset,
    branched: BranchedRootsTable,
    critic: PrivilegedFactoredCritic,
    baseline: LengthHorizonBaseline,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate frozen critic & baseline on held-out counterfactual branched roots."""
    n = len(roots)
    if n == 0:
        return []
    if branched.num_roots != n:
        raise ValueError(f"branched.num_roots ({branched.num_roots}) != len(roots) ({n})")
    _assert_valid_branched_roots(branched, require_score=True, label="held_out_branched_roots")

    horizon = float(branched.horizon)
    preds = predict_critic_on_dataset(
        critic,
        roots_dataset,
        batch_size=128,
        device=device,
        h_rem_override=horizon,
    )
    s_hat_all = preds["s_hrem_all"]  # (R, 6) exact at horizon (== s_300 when horizon == 300)
    s_hat_logits_all = preds["s_hrem_logits_all"]  # (R, 6) exact survival logits at horizon
    m_hat_all = (1000.0 / horizon) * preds["q_all"]  # (R, 6) exact (== m_300 when horizon == 300)

    lengths_np = np.asarray([r.hero_length for r in roots], dtype=np.float64)
    base_s, base_m, base_s_logits = baseline.predict_h300_with_logits(lengths_np, horizon=horizon)

    out_rows: list[dict[str, Any]] = []
    for i, snap in enumerate(roots):
        leg = np.asarray(branched.legal_mask[i], dtype=bool)
        if not bool(leg.any()):
            raise ValueError(f"Held-out root {snap.root_uid} has zero legal actions")
        b8_a = int(branched.b8_action[i])
        oracle_a = int(branched.oracle_a_star[i])
        critic_a, _ = _select_critic_action(s_hat_all[i], m_hat_all[i], leg, b8_a)
        # Board/action-blind (L_t, H_rem) baseline assigns identical value to every legal action,
        # so tie-breaking selects b8_a whenever b8_a is legal.
        base_a = b8_a if bool(leg[b8_a]) else int(np.flatnonzero(leg)[0])

        s_sc_leg = branched.s_score[i][leg]
        m_sc_leg = branched.m_score[i][leg]
        if not (bool(np.isfinite(s_sc_leg).all()) and bool(np.isfinite(m_sc_leg).all())):
            raise ValueError(f"Non-finite scoring branch outcome on root {snap.root_uid}")

        with _float32_no_autocast(torch.device("cpu")):
            s_sc_leg_t = torch.as_tensor(s_sc_leg, dtype=torch.float32)
            s_logit_leg_t = torch.as_tensor(s_hat_logits_all[i][leg], dtype=torch.float32).clamp(
                -12.0, 12.0
            )
            base_logit_leg_t = torch.full_like(
                s_sc_leg_t, float(np.clip(base_s_logits[i], -12.0, 12.0))
            )
            critic_s_bce = float(
                F.binary_cross_entropy_with_logits(s_logit_leg_t, s_sc_leg_t).item()
            )
            base_s_bce = float(
                F.binary_cross_entropy_with_logits(base_logit_leg_t, s_sc_leg_t).item()
            )

        s_sc = np.where(leg, branched.s_score[i], 0.0)
        m_sc = np.where(leg, branched.m_score[i], 0.0)
        out_rows.append(
            {
                "root_uid": snap.root_uid,
                "world_id": int(snap.world_id),
                "seed": int(snap.seed),
                "kind": snap.kind,
                "lookback": snap.lookback,
                "tick": int(snap.tick),
                "h_rem": int(snap.h_rem),
                "hero_length": int(snap.hero_length),
                "legal_mask": [bool(x) for x in leg.tolist()],
                "b8_action": b8_a,
                "oracle_a_star": oracle_a,
                "base_action": int(base_a),
                "critic_action": int(critic_a),
                "s_score": [round(float(x), 6) for x in s_sc.tolist()],
                "m_score": [round(float(x), 4) for x in m_sc.tolist()],
                "s_hat": [round(float(x), 6) for x in s_hat_all[i].tolist()],
                "s_hat_logit": [round(float(x), 6) for x in s_hat_logits_all[i].tolist()],
                "m_hat": [round(float(x), 4) for x in m_hat_all[i].tolist()],
                "base_s_hat": round(float(base_s[i]), 6),
                "base_s_hat_logit": round(float(base_s_logits[i]), 6),
                "base_m_hat": round(float(base_m[i]), 4),
                "critic_s_bce": round(critic_s_bce, 6),
                "base_s_bce": round(base_s_bce, 6),
            }
        )
    return out_rows


def summarize_world_stats_quick(
    world_records: Sequence[Mapping[str, Any]], stratum: str
) -> dict[str, float]:
    """Quick local diagnostic EV summary for one replicate's held-out worlds."""
    n = sum(int(w["strata"][stratum]["n"]) for w in world_records)
    if n == 0:
        return {"n": 0, "residual_ev": 0.0, "raw_ev_base": 0.0, "raw_ev_critic": 0.0}
    sum_y = sum(float(w["strata"][stratum]["sum_y"]) for w in world_records)
    sum_y2 = sum(float(w["strata"][stratum]["sum_y2"]) for w in world_records)
    sse_b = sum(float(w["strata"][stratum]["sse_base"]) for w in world_records)
    sse_c = sum(float(w["strata"][stratum]["sse_critic"]) for w in world_records)
    sst = max(sum_y2 - (sum_y * sum_y) / float(n), 1e-12)
    res_ev = 1.0 - (sse_c / max(sse_b, 1e-12))
    return {
        "n": n,
        "residual_ev": round(res_ev, 6),
        "raw_ev_base": round(1.0 - sse_b / sst, 6),
        "raw_ev_critic": round(1.0 - sse_c / sst, 6),
    }


def run_single_replicate(
    *,
    replicate_index: int,
    allocation_doc: Mapping[str, Any],
    config: Mapping[str, Any],
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    device: torch.device,
    checkpoint_mgr: ReplicateCheckpointManager | None = None,
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one complete world-disjoint replicate with temporal pre-test weight freezing."""
    t0 = time.perf_counter()
    namespace = str(allocation_doc["namespace"])
    rep_rows = [
        r for r in allocation_doc["rows"] if int(r["replicate_index"]) == int(replicate_index)
    ]
    fit_rows = [r for r in rep_rows if r["partition"] == "fit"]
    val_rows = [r for r in rep_rows if r["partition"] == "val"]
    test_rows = [r for r in rep_rows if r["partition"] == "test"]
    if not (fit_rows and val_rows and test_rows):
        raise ValueError(f"Empty fit/val/test partition in replicate {replicate_index}")

    fit_seeds = {int(r["seed"]) for r in fit_rows}
    val_seeds = {int(r["seed"]) for r in val_rows}
    test_seeds = {int(r["seed"]) for r in test_rows}
    if (fit_seeds & val_seeds) or (fit_seeds & test_seeds) or (val_seeds & test_seeds):
        raise AssertionError("World seed overlap detected inside replicate")

    session_ticks = int(config["session_ticks"])
    max_slab_envs = int(config.get("max_slab_envs", 128))
    branch_slab_envs = int(config.get("branch_max_slab_envs", 512))
    max_capacity = int(config.get("max_capacity", 3200))
    stride_mature = int(config.get("stride_mature", 4))
    stride_early = int(config.get("stride_early", 16))
    min_mature_length = int(config.get("min_mature_length", 600))
    branch_horizon = int(config.get("branch_horizon", 300))
    fit_max_roots = int(config.get("fit_max_roots", 160))
    test_max_roots = int(config.get("test_max_roots", 128))
    lookbacks = [int(x) for x in config.get("death_lookbacks", [16, 48, 112, 200])]
    ord_stride = int(config.get("ordinary_root_stride", 96))
    train_sel_suffixes = [int(x) for x in config.get("train_branch_suffixes", [0, 1])]
    test_sel_suffixes = [int(x) for x in config.get("test_selection_suffixes", [0, 1])]
    test_score_suffixes = [int(x) for x in config.get("test_scoring_suffixes", [4, 5])]
    critic_epochs = int(config.get("critic_epochs", 12))
    critic_batch_size = int(config.get("critic_batch_size", 512))
    critic_ckpt_every = max(1, int(config.get("critic_checkpoint_every", 3)))
    critic_seed = int(config.get("critic_base_seed", 17001)) + int(replicate_index) * 101

    fit_part_name = f"rep{replicate_index}_fit"
    val_part_name = f"rep{replicate_index}_val"
    test_part_name = f"rep{replicate_index}_test"

    fit_fp = harvest_config_fingerprint(
        config,
        study=STUDY,
        namespace=namespace,
        partition=fit_part_name,
        world_rows=fit_rows,
        max_roots=fit_max_roots,
    )
    val_fp = harvest_config_fingerprint(
        config,
        study=STUDY,
        namespace=namespace,
        partition=val_part_name,
        world_rows=val_rows,
        max_roots=0,
    )
    critic_fp = critic_config_fingerprint(
        config,
        study=STUDY,
        namespace=namespace,
        replicate_index=replicate_index,
        critic_seed=critic_seed,
        fit_fingerprint=fit_fp,
        val_fingerprint=val_fp,
        train_branch_suffixes=train_sel_suffixes,
    )
    test_fp = harvest_config_fingerprint(
        config,
        study=STUDY,
        namespace=namespace,
        partition=test_part_name,
        world_rows=test_rows,
        max_roots=test_max_roots,
    )

    def _emit(payload: Mapping[str, Any]) -> None:
        if log_progress is not None:
            log_progress({"replicate_index": int(replicate_index), **dict(payload)})

    frozen = (
        checkpoint_mgr.try_load_frozen_critic(fingerprint=critic_fp, device=device)
        if checkpoint_mgr is not None
        else None
    )
    if frozen is not None:
        critic = frozen["critic"]
        baseline = frozen["baseline"]
        baseline_summary = baseline.to_summary_dict()
        frozen_critic_sha256 = str(frozen["frozen_critic_sha256"])
        train_summary = dict(frozen["train_summary"])
        fit_telemetry = dict(frozen["fit_telemetry"])
        val_telemetry = dict(frozen["val_telemetry"])
        fit_branch_telemetry = dict(frozen["fit_branch_telemetry"])
        _emit(
            {
                "phase": "resume_stage",
                "stage": "critic-frozen",
                "frozen_critic_sha256": frozen_critic_sha256,
            }
        )
    else:
        # STEP 1: Co-harvest fit + val worlds (1 slab when both needed) & branch fit roots.
        fit_part = (
            checkpoint_mgr.try_load_harvest("fit-harvest", fingerprint=fit_fp)
            if checkpoint_mgr is not None
            else None
        )
        if fit_part is not None:
            _emit({"phase": "resume_stage", "stage": "fit-harvest"})
        val_part = (
            checkpoint_mgr.try_load_harvest("val-harvest", fingerprint=val_fp)
            if checkpoint_mgr is not None
            else None
        )
        if val_part is not None:
            _emit({"phase": "resume_stage", "stage": "val-harvest"})

        missing_specs: list[WorldPartitionSpec] = []
        if fit_part is None:
            missing_specs.append(
                WorldPartitionSpec(
                    partition=fit_part_name,
                    world_rows=fit_rows,
                    max_roots=fit_max_roots,
                )
            )
        if val_part is None:
            missing_specs.append(
                WorldPartitionSpec(
                    partition=val_part_name,
                    world_rows=val_rows,
                    max_roots=0,
                )
            )
        if missing_specs:
            harvested_list = harvest_world_partitions_batched(
                specs=missing_specs,
                b8_checkpoint=b8_checkpoint,
                frp3_checkpoint=frp3_checkpoint,
                session_ticks=session_ticks,
                max_slab_envs=max_slab_envs,
                max_capacity=max_capacity,
                stride_mature=stride_mature,
                stride_early=stride_early,
                min_mature_length=min_mature_length,
                branch_horizon=branch_horizon,
                death_lookbacks=lookbacks,
                ordinary_root_stride=ord_stride,
                device=device,
                log_progress=_emit,
            )
            harvested_map = {p.partition: p for p in harvested_list}
            if fit_part is None:
                fit_part = harvested_map[fit_part_name]
                if checkpoint_mgr is not None:
                    rec = checkpoint_mgr.save_harvest("fit-harvest", fit_part, fingerprint=fit_fp)
                    _emit({"phase": "checkpoint_stage", **rec})
            if val_part is None:
                val_part = harvested_map[val_part_name]
                if checkpoint_mgr is not None:
                    rec = checkpoint_mgr.save_harvest("val-harvest", val_part, fingerprint=val_fp)
                    _emit({"phase": "checkpoint_stage", **rec})

        assert fit_part is not None and val_part is not None
        fit_branch_fp = branch_config_fingerprint(
            config,
            study=STUDY,
            namespace=namespace,
            roots=fit_part.roots,
            horizon=branch_horizon,
            selection_suffixes=train_sel_suffixes,
            scoring_suffixes=(),
        )
        fit_branched = (
            checkpoint_mgr.try_load_branch("fit-branch", fingerprint=fit_branch_fp)
            if checkpoint_mgr is not None
            else None
        )
        if fit_branched is not None:
            _emit({"phase": "resume_stage", "stage": "fit-branch"})
        else:
            fit_branched = branch_prepared_roots(
                fit_part.roots,
                namespace=namespace,
                b8_checkpoint=b8_checkpoint,
                frp3_checkpoint=frp3_checkpoint,
                horizon=branch_horizon,
                selection_suffixes=train_sel_suffixes,
                scoring_suffixes=(),
                max_slab_envs=branch_slab_envs,
                max_capacity=max_capacity,
                compaction_interval=int(config.get("compaction_interval", 10)),
                device=device,
                log_progress=_emit,
            )
            if checkpoint_mgr is not None:
                rec = checkpoint_mgr.save_branch(
                    "fit-branch", fit_branched, fingerprint=fit_branch_fp
                )
                _emit({"phase": "checkpoint_stage", **rec})

        # STEP 2: Fit board-blind (L_t, H_rem) baseline & train PrivilegedFactoredCritic on fit/val.
        baseline = LengthHorizonBaseline.fit(
            fit_length=fit_part.dataset.snake_length[:, 0],
            fit_h_rem=fit_part.dataset.h_rem,
            fit_tau_alive=fit_part.dataset.tau_alive,
            fit_died=fit_part.dataset.died_in_session,
            fit_return_g=fit_part.dataset.return_g,
            fit_forage_short=fit_part.dataset.forage_short,
            fit_forage_weight=fit_part.dataset.forage_short_weight,
            val_length=val_part.dataset.snake_length[:, 0],
            val_h_rem=val_part.dataset.h_rem,
            val_return_g=val_part.dataset.return_g,
        )
        resume_ep = (
            checkpoint_mgr.try_load_latest_critic_epoch(
                max_epochs=critic_epochs,
                fingerprint=critic_fp,
            )
            if checkpoint_mgr is not None
            else None
        )
        if resume_ep is not None:
            _emit(
                {
                    "phase": "resume_stage",
                    "stage": f"critic-epoch-{int(resume_ep['epoch']):02d}",
                    "resumed_epoch": int(resume_ep["epoch"]),
                }
            )

        def _on_critic_epoch(ep_state: dict[str, Any]) -> None:
            if checkpoint_mgr is None:
                return
            ep_num = int(ep_state["epoch"])
            if ep_num % critic_ckpt_every != 0:
                return
            rec = checkpoint_mgr.save_critic_epoch(
                epoch=ep_num,
                fingerprint=critic_fp,
                critic=ep_state["critic"],
                optimizer=ep_state["optimizer"],
                scheduler=ep_state["scheduler"],
                np_rng_state=ep_state["np_rng_state"],
                torch_cpu_rng_state=ep_state["torch_cpu_rng_state"],
                best_epoch=int(ep_state["best_epoch"]),
                best_score=float(ep_state["best_score"]),
                best_state=ep_state["best_state"],
                epoch_history=ep_state["epoch_history"],
            )
            _emit({"phase": "checkpoint_stage", **rec})

        critic, train_summary = train_privileged_critic(
            fit_dataset=fit_part.dataset,
            val_dataset=val_part.dataset,
            fit_roots_dataset=fit_part.roots_dataset,
            fit_branched_roots=fit_branched,
            branch_horizon=branch_horizon,
            min_mature_length=min_mature_length,
            epochs=critic_epochs,
            batch_size=critic_batch_size,
            learning_rate=float(config.get("critic_lr", 6e-4)),
            weight_decay=float(config.get("critic_weight_decay", 1e-3)),
            seed=critic_seed,
            device=device,
            early_stop_patience=config.get("critic_early_stop_patience"),
            min_epochs=int(config.get("critic_min_epochs", 1)),
            resume_state=resume_ep,
            on_epoch_end=_on_critic_epoch,
            log_progress=_emit,
        )

        # STEP 3: Freeze baseline & critic digests BEFORE held-out test worlds are harvested.
        baseline_summary = baseline.to_summary_dict()
        frozen_critic_sha256 = critic.state_dict_sha256()

        fit_telemetry = copy.deepcopy(fit_part.telemetry)
        val_telemetry = copy.deepcopy(val_part.telemetry)
        fit_branch_telemetry = copy.deepcopy(fit_branched.telemetry)
        if checkpoint_mgr is not None:
            rec = checkpoint_mgr.save_frozen_critic(
                fingerprint=critic_fp,
                critic=critic,
                baseline=baseline,
                train_summary=train_summary,
                frozen_critic_sha256=frozen_critic_sha256,
                fit_telemetry=fit_telemetry,
                val_telemetry=val_telemetry,
                fit_branch_telemetry=fit_branch_telemetry,
            )
            _emit({"phase": "checkpoint_stage", **rec})
        del fit_part, val_part, fit_branched

    # STEP 4: Harvest held-out test worlds + branch held-out test roots + single frozen pass.
    test_part = (
        checkpoint_mgr.try_load_harvest(
            "test-harvest",
            fingerprint=test_fp,
            required_extra={"frozen_critic_sha256": frozen_critic_sha256},
        )
        if checkpoint_mgr is not None
        else None
    )
    if test_part is not None:
        _emit({"phase": "resume_stage", "stage": "test-harvest"})
    else:
        test_part = harvest_world_partition(
            partition=test_part_name,
            world_rows=test_rows,
            b8_checkpoint=b8_checkpoint,
            frp3_checkpoint=frp3_checkpoint,
            session_ticks=session_ticks,
            max_slab_envs=max_slab_envs,
            max_capacity=max_capacity,
            stride_mature=stride_mature,
            stride_early=stride_early,
            min_mature_length=min_mature_length,
            max_roots=test_max_roots,
            branch_horizon=branch_horizon,
            death_lookbacks=lookbacks,
            ordinary_root_stride=ord_stride,
            device=device,
            log_progress=_emit,
        )
        if checkpoint_mgr is not None:
            rec = checkpoint_mgr.save_harvest(
                "test-harvest",
                test_part,
                fingerprint=test_fp,
                extra_meta={"frozen_critic_sha256": frozen_critic_sha256},
            )
            _emit({"phase": "checkpoint_stage", **rec})

    test_branch_fp = branch_config_fingerprint(
        config,
        study=STUDY,
        namespace=namespace,
        roots=test_part.roots,
        horizon=branch_horizon,
        selection_suffixes=test_sel_suffixes,
        scoring_suffixes=test_score_suffixes,
    )
    test_branched = (
        checkpoint_mgr.try_load_branch("test-branch", fingerprint=test_branch_fp)
        if checkpoint_mgr is not None
        else None
    )
    if test_branched is not None:
        _emit({"phase": "resume_stage", "stage": "test-branch"})
    else:
        test_branched = branch_prepared_roots(
            test_part.roots,
            namespace=namespace,
            b8_checkpoint=b8_checkpoint,
            frp3_checkpoint=frp3_checkpoint,
            horizon=branch_horizon,
            selection_suffixes=test_sel_suffixes,
            scoring_suffixes=test_score_suffixes,
            max_slab_envs=branch_slab_envs,
            max_capacity=max_capacity,
            compaction_interval=int(config.get("compaction_interval", 10)),
            device=device,
            log_progress=_emit,
        )
        if checkpoint_mgr is not None:
            rec = checkpoint_mgr.save_branch(
                "test-branch", test_branched, fingerprint=test_branch_fp
            )
            _emit({"phase": "checkpoint_stage", **rec})

    v_base_test = baseline.predict_v(test_part.dataset.snake_length[:, 0], test_part.dataset.h_rem)
    critic_preds_test = predict_critic_on_dataset(
        critic,
        test_part.dataset,
        batch_size=critic_batch_size,
        device=device,
    )
    post_eval_critic_sha256 = critic.state_dict_sha256()
    if post_eval_critic_sha256 != frozen_critic_sha256:
        raise AssertionError("Frozen critic weights mutated during held-out test evaluation")

    world_stats = compute_world_sufficient_stats(
        world_ids=test_part.world_ids,
        world_seeds=test_part.seeds,
        dataset=test_part.dataset,
        v_base=v_base_test,
        v_critic=critic_preds_test["v_pred"],
        hazard_nll=critic_preds_test["hazard_nll"],
    )
    root_records = build_held_out_root_records(
        roots=test_part.roots,
        roots_dataset=test_part.roots_dataset,
        branched=test_branched,
        critic=critic,
        baseline=baseline,
        device=device,
    )

    return {
        "schema": UNIT_SCHEMA,
        "study": STUDY,
        "namespace": namespace,
        "allocation": ALLOCATION_NAME,
        "allocation_index": ALLOCATION_INDEX,
        "unit_kind": UNIT_KIND,
        "unit_id": replicate_unit_id(namespace, replicate_index),
        "replicate_index": int(replicate_index),
        "canary_only": bool(config.get("canary_only", False)),
        "world_disjointness": {
            "fit_world_count": len(fit_rows),
            "val_world_count": len(val_rows),
            "test_world_count": len(test_rows),
            "fit_seeds": sorted(fit_seeds),
            "val_seeds": sorted(val_seeds),
            "test_seeds": sorted(test_seeds),
            "pairwise_overlap_count": 0,
            "critic_frozen_before_test_harvest": True,
            "frozen_critic_sha256": frozen_critic_sha256,
            "post_eval_critic_sha256": post_eval_critic_sha256,
            "frozen_baseline_sha256": baseline_summary["sha256"],
        },
        "config_summary": {
            "session_ticks": session_ticks,
            "min_mature_length": min_mature_length,
            "branch_horizon": branch_horizon,
            "stride_mature": stride_mature,
            "stride_early": stride_early,
        },
        "telemetry": {
            "fit_harvest": fit_telemetry,
            "val_harvest": val_telemetry,
            "fit_branch": fit_branch_telemetry,
            "critic_train": train_summary,
            "test_harvest": test_part.telemetry,
            "test_branch": test_branched.telemetry,
            "durability": {
                "resumed_stages": (
                    list(checkpoint_mgr.resumed_stages) if checkpoint_mgr is not None else []
                ),
                "exported_stage_checkpoints": (
                    len(checkpoint_mgr.exported_files) if checkpoint_mgr is not None else 0
                ),
            },
            "replicate_wall_sec": round(time.perf_counter() - t0, 3),
        },
        "diagnostic_quick_ev": {
            "l_ge_600": summarize_world_stats_quick(world_stats, "l_ge_600"),
            "all_lengths": summarize_world_stats_quick(world_stats, "all_lengths"),
        },
        "held_out_test_worlds": world_stats,
        "held_out_test_roots": root_records,
    }


def run_pipeline(
    *,
    input_dir: Path,
    config_path: Path,
    ledger_path: Path,
    output_dir: Path,
    device: str = "cpu",
    max_export_bytes: int = 128 * 1024**2,
    resume_checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Gate-1 units with verified-ledger skip and atomic immutable sidecar exports."""
    t0 = time.perf_counter()
    verify_immutable(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "t34-t4-gate1-config/v1":
        raise ValueError(f"Foreign Gate-1 config schema: {config.get('schema')!r}")

    b8_ckpt = (input_dir / str(config["b8_checkpoint"])).resolve(strict=True)
    frp3_ckpt = (input_dir / str(config["frp3_checkpoint"])).resolve(strict=True)
    if bool(config.get("verify_checkpoint_hashes", True)):
        if digest(b8_ckpt) != B8_S3_SHA256:
            raise ValueError(f"B8 checkpoint SHA256 mismatch at {b8_ckpt}")
        if digest(frp3_ckpt) != FRP3_S12_SHA256:
            raise ValueError(f"FRP3 checkpoint SHA256 mismatch at {frp3_ckpt}")

    seed_manifest_name = config.get("seed_manifest")
    namespace = str(config["namespace"])
    if seed_manifest_name:
        sm_path = input_dir / str(seed_manifest_name)
        verify_immutable(sm_path)
        sm_doc = json.loads(sm_path.read_text(encoding="utf-8"))
        alloc_key = str(config.get("allocation_key", "main"))
        alloc_doc = sm_doc["allocations"][alloc_key]
        if str(alloc_doc["namespace"]) != namespace:
            raise ValueError("seed_manifest allocation namespace differs from config namespace")
    else:
        alloc_doc = world_allocation(
            namespace,
            num_replicates=int(config["num_replicates"]),
            fit_per_replicate=int(config["fit_per_replicate"]),
            val_per_replicate=int(config["val_per_replicate"]),
            test_per_replicate=int(config["test_per_replicate"]),
        )

    num_reps = int(alloc_doc["num_replicates"])
    valid_reps = set(range(num_reps))
    skipped_reps = load_and_validate_ledger(
        ledger_path,
        expected_namespace=namespace,
        valid_replicates=valid_reps,
    )

    target = torch.device("cuda:0" if device == "cuda" else device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested for Gate-1 pipeline, but torch.cuda.is_available() is False"
        )

    units_dir = output_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    persist_ckpts = bool(config.get("persist_stage_checkpoints", True))
    resume_roots: list[Path] = []
    if resume_checkpoint_dir is not None and resume_checkpoint_dir.is_dir():
        resume_roots.append(resume_checkpoint_dir.resolve())

    def _log(msg: Mapping[str, Any]) -> None:
        print(f"T4_GATE1_PROGRESS {json.dumps(dict(msg), sort_keys=True)}", flush=True)

    completed_units: list[dict[str, Any]] = []
    skipped_unit_ids: list[str] = []

    for rep_idx in range(num_reps):
        uid = replicate_unit_id(namespace, rep_idx)
        if rep_idx in skipped_reps:
            skipped_unit_ids.append(uid)
            _log({"phase": "skip_verified_unit", "replicate_index": rep_idx, "unit_id": uid})
            continue

        ckpt_mgr = ReplicateCheckpointManager(
            study=STUDY,
            namespace=namespace,
            replicate_index=rep_idx,
            export_root=(output_dir / "checkpoints" if persist_ckpts else None),
            resume_roots=resume_roots,
            max_export_bytes=max_export_bytes,
        )
        unit_payload = run_single_replicate(
            replicate_index=rep_idx,
            allocation_doc=alloc_doc,
            config=config,
            b8_checkpoint=b8_ckpt,
            frp3_checkpoint=frp3_ckpt,
            device=target,
            checkpoint_mgr=ckpt_mgr,
            log_progress=_log,
        )
        unit_file = units_dir / f"{replicate_unit_stem(rep_idx)}.json"
        pub = publish_json(unit_file, unit_payload)
        if int(pub["size"]) > max_export_bytes:
            raise ValueError(
                f"Unit export {unit_file} exceeds max_export_bytes: "
                f"{pub['size']} > {max_export_bytes}"
            )
        completed_units.append(
            {
                "replicate_index": rep_idx,
                "unit_id": uid,
                "file": unit_file.name,
                "sha256": pub["sha256"],
                "size_bytes": int(pub["size"]),
            }
        )
        _log(
            {
                "phase": "exported_unit",
                "replicate_index": rep_idx,
                "unit_id": uid,
                "size_bytes": int(pub["size"]),
                "sha256": pub["sha256"],
            }
        )

    # Emit skip-proof.json: in canary/preflight mode, run an isolated synthetic ledger check
    # proving that a verified ledger entry causes 0 recomputed units.
    skip_proof: dict[str, Any] = {
        "schema": "t34-t4-gate1-skip-proof/v1",
        "study": STUDY,
        "namespace": namespace,
        "initial_skipped_units": skipped_unit_ids,
        "executed_units_count": len(completed_units),
    }
    if bool(config.get("run_skip_selfcheck", config.get("canary_only", False))) and completed_units:
        orig_ledger_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="t4-gate1-skipcheck-") as tmp_s:
            tmp_d = Path(tmp_s)
            syn_ledger = tmp_d / "verified-ledger.json"
            all_entries: dict[int, dict[str, Any]] = {
                int(k): dict(v) for k, v in orig_ledger_doc.get("entries", {}).items()
            }
            for c in completed_units:
                all_entries[int(c["replicate_index"])] = {
                    "unit_id": c["unit_id"],
                    "sha256": c["sha256"],
                    "size_bytes": c["size_bytes"],
                }
            publish_verified_ledger(syn_ledger, all_entries, namespace=namespace)
            syn_cfg = tmp_d / "syn-config.json"
            cfg_no_self = dict(config)
            cfg_no_self["run_skip_selfcheck"] = False
            publish_json(syn_cfg, cfg_no_self)
            sub_res = run_pipeline(
                input_dir=input_dir,
                config_path=syn_cfg,
                ledger_path=syn_ledger,
                output_dir=tmp_d / "out",
                device=device,
                max_export_bytes=max_export_bytes,
                resume_checkpoint_dir=resume_checkpoint_dir,
            )
            if sub_res["executed_units_count"] != 0:
                raise RuntimeError(
                    "Resume-by-default skip self-check failed to skip verified units"
                )
            skip_proof["synthetic_full_ledger_selfcheck"] = {
                "status": "PASS",
                "recomputed_units": sub_res["executed_units_count"],
                "skipped_units": sub_res["skipped_units"],
            }

    publish_json(output_dir / "skip-proof.json", skip_proof)

    run_summary = {
        "schema": "t34-t4-gate1-attempt-diagnostic/v1",
        "diagnostic_only_not_the_look": True,
        "study": STUDY,
        "namespace": namespace,
        "canary_only": bool(config.get("canary_only", False)),
        "device": str(target),
        "cuda_device_name": (torch.cuda.get_device_name(target) if target.type == "cuda" else None),
        "num_replicates_total": num_reps,
        "skipped_units": skipped_unit_ids,
        "executed_units_count": len(completed_units),
        "completed_units": completed_units,
        "wall_sec": round(time.perf_counter() - t0, 3),
    }
    publish_json(output_dir / "run-summary.json", run_summary)
    return run_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-export-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--resume-checkpoint-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_pipeline(
        input_dir=args.input_dir.resolve(),
        config_path=args.config.resolve(),
        ledger_path=args.ledger.resolve(),
        output_dir=args.output.resolve(),
        device=args.device,
        max_export_bytes=int(args.max_export_bytes),
        resume_checkpoint_dir=(
            args.resume_checkpoint_dir.resolve() if args.resume_checkpoint_dir is not None else None
        ),
    )
    print(f"T4_GATE1_DONE {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
