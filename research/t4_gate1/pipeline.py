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
from research.t4_gate1.featurizer import CompactBoardDataset, featurize_dataset_batch
from research.t4_gate1.harvest import (
    BranchedRootsTable,
    PreparedRootSnapshot,
    branch_prepared_roots,
    harvest_world_partition,
)
from research.t4_gate1.model import (
    LengthHorizonBaseline,
    PrivilegedFactoredCritic,
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


@torch.inference_mode()
def predict_critic_on_dataset(
    critic: PrivilegedFactoredCritic,
    dataset: CompactBoardDataset,
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
            "s_300_all": np.zeros((0, 6), dtype=np.float64),
            "m_300_all": np.zeros((0, 6), dtype=np.float64),
            "hazard_nll": np.zeros(0, dtype=np.float64),
        }

    v_pred = np.zeros(n, dtype=np.float64)
    q_all = np.zeros((n, 6), dtype=np.float64)
    greedy_action = np.zeros(n, dtype=np.int64)
    s_hrem_all = np.zeros((n, 6), dtype=np.float64)
    s_300_all = np.zeros((n, 6), dtype=np.float64)
    m_300_all = np.zeros((n, 6), dtype=np.float64)
    haz_nll = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = np.arange(start, end, dtype=np.int64)
        fb = featurize_dataset_batch(dataset, idx, device=device, h_rem_override=h_rem_override)
        out = critic(fb, h_rem_override=h_rem_override)
        greedy_t = out["q_life"].argmax(dim=-1)
        q_b = out["q_life"].detach().cpu().numpy().astype(np.float64)
        acts_b = dataset.b8_action[idx]
        v_pred[start:end] = q_b[np.arange(end - start), acts_b]
        q_all[start:end] = q_b
        greedy_action[start:end] = greedy_t.detach().cpu().numpy().astype(np.int64)
        s_hrem_all[start:end] = out["s_hrem"].detach().cpu().numpy().astype(np.float64)
        s_300_all[start:end] = out["s_300"].detach().cpu().numpy().astype(np.float64)
        m_300_all[start:end] = out["m_300"].detach().cpu().numpy().astype(np.float64)

        b_arange = torch.arange(end - start, device=device)
        acts_t = torch.as_tensor(acts_b, dtype=torch.long, device=device)
        haz_taken = out["hazard_logits"][b_arange, acts_t]
        nll_b = discrete_survival_nll(
            haz_taken,
            torch.as_tensor(dataset.tau_alive[idx], dtype=torch.float32, device=device),
            torch.as_tensor(dataset.died_in_session[idx], dtype=torch.bool, device=device),
        )
        haz_nll[start:end] = nll_b.detach().cpu().numpy().astype(np.float64)

    return {
        "v_pred": v_pred,
        "q_all": q_all,
        "greedy_action": greedy_action,
        "s_hrem_all": s_hrem_all,
        "s_300_all": s_300_all,
        "m_300_all": m_300_all,
        "hazard_nll": haz_nll,
    }


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
    log_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[PrivilegedFactoredCritic, dict[str, Any]]:
    """Train clean-slate ``PrivilegedFactoredCritic`` strictly on ``fit`` and select on ``val``."""
    t0 = time.perf_counter()
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

    var_g_fit = float(np.var(fit_dataset.return_g.astype(np.float64)))
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

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    n_branch_roots = fit_roots_dataset.num_rows
    branch_batch_size = min(32, n_branch_roots) if n_branch_roots > 0 else 0

    best_score = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
    epoch_history: list[dict[str, Any]] = []

    with _critic_training_precision_scope(device):
        for epoch in range(1, int(epochs) + 1):
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
                fb = featurize_dataset_batch(fit_dataset, idx, device=device)

                acts_t = torch.as_tensor(
                    fit_dataset.b8_action[idx], dtype=torch.long, device=device
                )
                tau_t = torch.as_tensor(
                    fit_dataset.tau_alive[idx], dtype=torch.float32, device=device
                )
                died_t = torch.as_tensor(
                    fit_dataset.died_in_session[idx], dtype=torch.bool, device=device
                )
                ret_t = torch.as_tensor(
                    fit_dataset.return_g[idx], dtype=torch.float32, device=device
                )
                fs_t = torch.as_tensor(
                    fit_dataset.forage_short[idx], dtype=torch.float32, device=device
                )
                fw_t = torch.as_tensor(
                    fit_dataset.forage_short_weight[idx], dtype=torch.float32, device=device
                )
                fl_t = torch.as_tensor(
                    fit_dataset.forage_long[idx], dtype=torch.float32, device=device
                )
                flv_t = torch.as_tensor(
                    fit_dataset.forage_long_valid[idx], dtype=torch.bool, device=device
                )

                optimizer.zero_grad(set_to_none=True)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    out = critic(fb)
                    b_ar = torch.arange(b_cur, device=device)
                    q_taken = out["q_life"][b_ar, acts_t]
                    haz_taken = out["hazard_logits"][b_ar, acts_t]
                    gs_taken = out["g_short"][b_ar, acts_t]
                    gl_taken = out["g_long"][b_ar, acts_t]

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
                    if branch_batch_size > 0:
                        r_idx = rng.choice(n_branch_roots, size=branch_batch_size, replace=False)
                        fb_r = featurize_dataset_batch(
                            fit_roots_dataset,
                            r_idx,
                            device=device,
                            h_rem_override=float(branch_horizon),
                        )
                        out_r = critic(fb_r, h_rem_override=float(branch_horizon))
                        leg_r = torch.as_tensor(
                            fit_branched_roots.legal_mask[r_idx],
                            dtype=torch.bool,
                            device=device,
                        )
                        s_targ = torch.as_tensor(
                            np.nan_to_num(fit_branched_roots.s_sel[r_idx], nan=0.0),
                            dtype=torch.float32,
                            device=device,
                        )
                        q_targ = torch.as_tensor(
                            np.nan_to_num(fit_branched_roots.m_sel[r_idx], nan=0.0)
                            * (float(branch_horizon) / 1000.0),
                            dtype=torch.float32,
                            device=device,
                        )
                        s_pred = out_r["s_hrem"].clamp(1e-5, 1.0 - 1e-5)
                        q_pred = out_r["q_life"]
                        bce_r = F.binary_cross_entropy(s_pred[leg_r], s_targ[leg_r])
                        mse_r = F.huber_loss(q_pred[leg_r], q_targ[leg_r], delta=0.10)
                        loss_branch = bce_r + 2.0 * mse_r

                    total_loss = (
                        loss_ret
                        + 0.5 * loss_haz
                        + 1.5 * loss_fs
                        + 1.0 * loss_fl
                        + 0.5 * loss_branch
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
                critic, val_dataset, batch_size=batch_size, device=device
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

            rec = {
                "epoch": epoch,
                "train_loss": round(sum_loss / max(n_batches, 1), 6),
                "train_ret_loss": round(sum_ret / max(n_batches, 1), 6),
                "train_haz_loss": round(sum_haz / max(n_batches, 1), 6),
                "val_mature_mse": round(val_mse, 6),
                "val_mature_hazard_nll": round(val_nll, 6),
                "val_selection_score": round(val_score, 6),
            }
            epoch_history.append(rec)

            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}

            if log_progress is not None:
                log_progress({"phase": "critic_train", **rec})

    critic.load_state_dict(best_state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    num_params = sum(int(p.numel()) for p in critic.parameters())
    summary = {
        "seed": int(seed),
        "epochs": int(epochs),
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
    err_b = y - v_base.astype(np.float64)
    err_c = y - v_critic.astype(np.float64)

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
                    "sum_hazard_nll": float(np.sum(hazard_nll[m])),
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
    horizon = float(branched.horizon)
    preds = predict_critic_on_dataset(
        critic,
        roots_dataset,
        batch_size=128,
        device=device,
        h_rem_override=horizon,
    )
    s_hat_all = preds["s_hrem_all"]  # (R, 6) exact at horizon (== s_300 when horizon == 300)
    m_hat_all = (1000.0 / horizon) * preds["q_all"]  # (R, 6) exact (== m_300 when horizon == 300)

    lengths_np = np.asarray([r.hero_length for r in roots], dtype=np.float64)
    base_s, base_m = baseline.predict_h300(lengths_np)

    out_rows: list[dict[str, Any]] = []
    for i, snap in enumerate(roots):
        leg = branched.legal_mask[i]
        b8_a = int(branched.b8_action[i])
        oracle_a = int(branched.oracle_a_star[i])
        critic_a, _ = _select_critic_action(s_hat_all[i], m_hat_all[i], leg, b8_a)
        # Board/action-blind (L_t, H_rem) baseline assigns identical value to every legal action,
        # so tie-breaking selects b8_a whenever b8_a is legal.
        base_a = b8_a if bool(leg[b8_a]) else int(np.flatnonzero(leg)[0])

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
                "m_hat": [round(float(x), 4) for x in m_hat_all[i].tolist()],
                "base_s_hat": round(float(base_s[i]), 6),
                "base_m_hat": round(float(base_m[i]), 4),
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
    max_slab_envs = int(config.get("max_slab_envs", 64))
    branch_slab_envs = int(config.get("branch_max_slab_envs", 128))
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

    # STEP 1: Harvest fit + val worlds and branch fit roots (test worlds NOT harvested yet).
    fit_part = harvest_world_partition(
        partition=f"rep{replicate_index}_fit",
        world_rows=fit_rows,
        b8_checkpoint=b8_checkpoint,
        frp3_checkpoint=frp3_checkpoint,
        session_ticks=session_ticks,
        max_slab_envs=max_slab_envs,
        max_capacity=max_capacity,
        stride_mature=stride_mature,
        stride_early=stride_early,
        min_mature_length=min_mature_length,
        max_roots=fit_max_roots,
        branch_horizon=branch_horizon,
        death_lookbacks=lookbacks,
        ordinary_root_stride=ord_stride,
        device=device,
        log_progress=log_progress,
    )
    val_part = harvest_world_partition(
        partition=f"rep{replicate_index}_val",
        world_rows=val_rows,
        b8_checkpoint=b8_checkpoint,
        frp3_checkpoint=frp3_checkpoint,
        session_ticks=session_ticks,
        max_slab_envs=max_slab_envs,
        max_capacity=max_capacity,
        stride_mature=stride_mature,
        stride_early=stride_early,
        min_mature_length=min_mature_length,
        max_roots=0,
        branch_horizon=branch_horizon,
        death_lookbacks=lookbacks,
        ordinary_root_stride=ord_stride,
        device=device,
        log_progress=log_progress,
    )
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
        log_progress=log_progress,
    )

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
    critic_seed = int(config.get("critic_base_seed", 17001)) + int(replicate_index) * 101
    critic, train_summary = train_privileged_critic(
        fit_dataset=fit_part.dataset,
        val_dataset=val_part.dataset,
        fit_roots_dataset=fit_part.roots_dataset,
        fit_branched_roots=fit_branched,
        branch_horizon=branch_horizon,
        min_mature_length=min_mature_length,
        epochs=int(config.get("critic_epochs", 16)),
        batch_size=int(config.get("critic_batch_size", 256)),
        learning_rate=float(config.get("critic_lr", 6e-4)),
        weight_decay=float(config.get("critic_weight_decay", 1e-3)),
        seed=critic_seed,
        device=device,
        log_progress=log_progress,
    )

    # STEP 3: Freeze baseline & critic digests BEFORE held-out test worlds are harvested.
    baseline_summary = baseline.to_summary_dict()
    frozen_critic_sha256 = critic.state_dict_sha256()

    # Free fit/val bulk arrays before test rollout to keep peak RAM tiny.
    fit_telemetry = copy.deepcopy(fit_part.telemetry)
    val_telemetry = copy.deepcopy(val_part.telemetry)
    fit_branch_telemetry = copy.deepcopy(fit_branched.telemetry)
    del fit_part, val_part, fit_branched

    # STEP 4: Harvest held-out test worlds + branch held-out test roots + single frozen pass.
    test_part = harvest_world_partition(
        partition=f"rep{replicate_index}_test",
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
        log_progress=log_progress,
    )
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
        log_progress=log_progress,
    )

    v_base_test = baseline.predict_v(test_part.dataset.snake_length[:, 0], test_part.dataset.h_rem)
    critic_preds_test = predict_critic_on_dataset(
        critic,
        test_part.dataset,
        batch_size=int(config.get("critic_batch_size", 256)),
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
    max_export_bytes: int = 4 * 1024**2,
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

        unit_payload = run_single_replicate(
            replicate_index=rep_idx,
            allocation_doc=alloc_doc,
            config=config,
            b8_checkpoint=b8_ckpt,
            frp3_checkpoint=frp3_ckpt,
            device=target,
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_pipeline(
        input_dir=args.input_dir.resolve(),
        config_path=args.config.resolve(),
        ledger_path=args.ledger.resolve(),
        output_dir=args.output.resolve(),
        device=args.device,
    )
    print(f"T4_GATE1_DONE {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
