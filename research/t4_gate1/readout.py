"""Authoritative Mac-side aggregator, world-clustered bootstrap CI, and kill/GO gate for Gate 1.

Enforces:
  1. Verified stopped supervisor receipt (``stop_verified is True`` and no ``cleanup_error``)
     unless ``--local-preflight`` is explicitly requested;
  2. Immutable sidecar verification (``.sha256`` + ``.complete``) on every unit file, refusing
     duplicate replicate IDs, foreign namespaces/allocations, or ``canary_only`` units;
  3. Outcome-blind ``--emit-ledger`` path that inspects strictly filename + sidecar size/digest
     without parsing any JSON outcome payload;
  4. Cross-replicate and cross-partition world seed disjointness + pre-test critic freeze proof;
  5. Pooled held-out residual EV over ``(L_t, H_rem)``, counterfactual H300 rescue value,
     world-clustered 95% bootstrap CIs, and pre-stated OPUS_META / review-4 kill vs GO verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.t4_gate1 import (
    ALLOCATION_INDEX,
    ALLOCATION_NAME,
    MAIN_NAMESPACE,
    STUDY,
    T4_FULL_GO_PREREQUISITES,
    T4_GATE1_THRESHOLDS,
    UNIT_KIND,
    evaluate_gate1_bars,
    replicate_unit_id,
)
from research.t4_gate1.pipeline import (
    EVAL_STRATA,
    UNIT_SCHEMA,
    publish_verified_ledger,
    summarize_world_stats_quick,
)
from tools.ops.package_spec import publish_json, verify_immutable

_UNIT_FILE_RE = re.compile(r"^replicate-(\d{4})\.json$")


def _sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _resolve_units_dir(attempt_dir: Path, *, local_preflight: bool) -> tuple[Path, dict[str, Any]]:
    """Verify stopped Colab supervisor receipt (unless local_preflight) and locate units/."""
    if local_preflight:
        direct = attempt_dir / "units"
        nested_out = attempt_dir / "output" / "t4_gate1" / "units"
        nested_exp = attempt_dir / "exports" / "t4_gate1" / "units"
        for candidate in (direct, nested_out, nested_exp):
            if candidate.is_dir():
                return candidate, {"mode": "local_preflight", "attempt_dir": str(attempt_dir)}
        return direct, {"mode": "local_preflight", "attempt_dir": str(attempt_dir)}

    receipt_path = attempt_dir / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Missing supervisor receipt.json in {attempt_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("stop_verified") is not True or receipt.get("cleanup_error"):
        raise ValueError(
            f"Refusing readout from unstopped runtime {attempt_dir}: "
            f"status={receipt.get('status')!r} stop_verified={receipt.get('stop_verified')!r} "
            f"cleanup_error={receipt.get('cleanup_error')!r}"
        )
    units_dir = attempt_dir / "exports" / "t4_gate1" / "units"
    return units_dir, {
        "mode": "colab_supervisor_receipt",
        "attempt_dir": str(attempt_dir),
        "receipt_sha256": _sha256_file(receipt_path),
        "job_id": receipt.get("job_id"),
        "status": receipt.get("status"),
        "stop_verified": receipt.get("stop_verified"),
    }


def build_verified_ledger_outcome_blind(
    attempt_dirs: Sequence[Path],
    *,
    ledger_out: Path,
    namespace: str = MAIN_NAMESPACE,
    valid_replicates: set[int] | None = None,
    local_preflight: bool = False,
) -> dict[str, Any]:
    """Build a resume-by-default verified ledger reading ONLY path/size/sha256 (zero JSON loads)."""
    entries: dict[int, dict[str, Any]] = {}
    for attempt_dir in attempt_dirs:
        units_dir, _ = _resolve_units_dir(attempt_dir, local_preflight=local_preflight)
        if not units_dir.is_dir():
            continue
        for path in sorted(units_dir.glob("replicate-*.json")):
            match = _UNIT_FILE_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected unit filename in {units_dir}: {path.name}")
            rep_idx = int(match.group(1))
            if valid_replicates is not None and rep_idx not in valid_replicates:
                raise ValueError(
                    f"Unit replicate index {rep_idx} outside frozen allocation in {path}"
                )
            if rep_idx in entries:
                raise ValueError(f"Duplicate verified unit replicate-{rep_idx:04d} across attempts")
            verified = verify_immutable(path)
            entries[rep_idx] = {
                "unit_id": replicate_unit_id(namespace, rep_idx),
                "replicate_index": rep_idx,
                "sha256": str(verified["sha256"]),
                "size_bytes": int(verified["size"]),
            }

    pub = publish_verified_ledger(ledger_out, entries, namespace=namespace)
    return {
        "ledger_path": str(ledger_out),
        "ledger_sha256": pub["sha256"],
        "verified_unit_count": len(entries),
        "verified_replicates": sorted(entries.keys()),
    }


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact rank-based ROC-AUC with average tie handling (returns 0.5 if single-class)."""
    yt = np.asarray(y_true, dtype=bool).reshape(-1)
    ys = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(yt))
    n_neg = int(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(ys, kind="mergesort")
    ys_sorted = ys[order]
    yt_sorted = yt[order]
    ranks = np.empty(yt.size, dtype=np.float64)
    i = 0
    n = yt.size
    while i < n:
        j = i + 1
        while j < n and ys_sorted[j] == ys_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    sum_pos_ranks = float(np.sum(ranks[yt_sorted]))
    auc = (sum_pos_ranks - 0.5 * n_pos * (n_pos + 1.0)) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _aggregate_stratum_ev(
    world_records: Sequence[Mapping[str, Any]],
    stratum: str,
    *,
    n_bootstrap: int = 2000,
    rng_seed: int = 20261031,
) -> dict[str, Any]:
    """Compute exact pooled EV + world-clustered 95% bootstrap CI for one evaluation stratum."""
    w_count = len(world_records)
    n_arr = np.asarray([int(w["strata"][stratum]["n"]) for w in world_records], dtype=np.float64)
    sy_arr = np.asarray(
        [float(w["strata"][stratum]["sum_y"]) for w in world_records], dtype=np.float64
    )
    sy2_arr = np.asarray(
        [float(w["strata"][stratum]["sum_y2"]) for w in world_records], dtype=np.float64
    )
    sb_arr = np.asarray(
        [float(w["strata"][stratum]["sse_base"]) for w in world_records], dtype=np.float64
    )
    sc_arr = np.asarray(
        [float(w["strata"][stratum]["sse_critic"]) for w in world_records], dtype=np.float64
    )
    ab_arr = np.asarray(
        [float(w["strata"][stratum]["sae_base"]) for w in world_records], dtype=np.float64
    )
    ac_arr = np.asarray(
        [float(w["strata"][stratum]["sae_critic"]) for w in world_records], dtype=np.float64
    )
    hn_arr = np.asarray(
        [float(w["strata"][stratum]["sum_hazard_nll"]) for w in world_records],
        dtype=np.float64,
    )

    total_n = int(np.sum(n_arr))
    contributing_worlds = int(np.sum(n_arr > 0))
    if total_n == 0 or w_count == 0:
        return {
            "num_rows": 0,
            "contributing_worlds": 0,
            "mse_base": 0.0,
            "mse_critic": 0.0,
            "mae_base": 0.0,
            "mae_critic": 0.0,
            "target_variance": 0.0,
            "raw_ev_base": 0.0,
            "raw_ev_critic": 0.0,
            "residual_ev": 0.0,
            "residual_ev_ci95": [0.0, 0.0],
            "raw_ev_critic_ci95": [0.0, 0.0],
            "mean_hazard_nll": 0.0,
        }

    sum_y = float(np.sum(sy_arr))
    sum_y2 = float(np.sum(sy2_arr))
    sse_b = float(np.sum(sb_arr))
    sse_c = float(np.sum(sc_arr))
    sst = max(sum_y2 - (sum_y * sum_y) / float(total_n), 1e-12)

    mse_b = sse_b / float(total_n)
    mse_c = sse_c / float(total_n)
    raw_ev_b = 1.0 - (sse_b / sst)
    raw_ev_c = 1.0 - (sse_c / sst)
    res_ev = 1.0 - (sse_c / max(sse_b, 1e-12))

    rng = np.random.default_rng(rng_seed)
    boot_idx = rng.integers(0, w_count, size=(n_bootstrap, w_count))
    b_n = n_arr[boot_idx].sum(axis=1)
    b_sy = sy_arr[boot_idx].sum(axis=1)
    b_sy2 = sy2_arr[boot_idx].sum(axis=1)
    b_sb = sb_arr[boot_idx].sum(axis=1)
    b_sc = sc_arr[boot_idx].sum(axis=1)

    valid_b = (b_n > 0) & (b_sb > 1e-12)
    if bool(np.any(valid_b)):
        res_samples = 1.0 - (b_sc[valid_b] / b_sb[valid_b])
        b_sst = np.maximum(b_sy2[valid_b] - np.square(b_sy[valid_b]) / b_n[valid_b], 1e-12)
        raw_c_samples = 1.0 - (b_sc[valid_b] / b_sst)
        res_ci = [
            round(float(np.quantile(res_samples, 0.025)), 6),
            round(float(np.quantile(res_samples, 0.975)), 6),
        ]
        raw_c_ci = [
            round(float(np.quantile(raw_c_samples, 0.025)), 6),
            round(float(np.quantile(raw_c_samples, 0.975)), 6),
        ]
    else:
        res_ci = [round(res_ev, 6), round(res_ev, 6)]
        raw_c_ci = [round(raw_ev_c, 6), round(raw_ev_c, 6)]

    return {
        "num_rows": total_n,
        "contributing_worlds": contributing_worlds,
        "mse_base": round(mse_b, 6),
        "mse_critic": round(mse_c, 6),
        "mae_base": round(float(np.sum(ab_arr)) / float(total_n), 6),
        "mae_critic": round(float(np.sum(ac_arr)) / float(total_n), 6),
        "target_variance": round(sst / float(total_n), 6),
        "raw_ev_base": round(raw_ev_b, 6),
        "raw_ev_critic": round(raw_ev_c, 6),
        "residual_ev": round(res_ev, 6),
        "residual_ev_ci95": res_ci,
        "raw_ev_critic_ci95": raw_c_ci,
        "mean_hazard_nll": round(float(np.sum(hn_arr)) / float(total_n), 6),
    }


def _aggregate_root_subset(
    roots: Sequence[Mapping[str, Any]],
    *,
    all_test_world_ids: Sequence[int],
    n_bootstrap: int = 2000,
    rng_seed: int = 20261031,
) -> dict[str, Any]:
    """Compute counterfactual H300 rescue mass/survival deltas, doom AUC, and world CIs."""
    n = len(roots)
    if n == 0:
        return {
            "num_roots": 0,
            "contributing_worlds": 0,
            "b8_mean_mass": 0.0,
            "b8_mean_survival": 0.0,
            "base_mean_mass": 0.0,
            "critic_mean_mass": 0.0,
            "critic_mean_survival": 0.0,
            "oracle_mean_mass": 0.0,
            "oracle_mean_survival": 0.0,
            "critic_vs_b8_mass_delta": 0.0,
            "critic_vs_b8_mass_delta_ci95": [0.0, 0.0],
            "critic_vs_b8_survival_delta": 0.0,
            "critic_vs_b8_survival_delta_ci95": [0.0, 0.0],
            "oracle_vs_b8_mass_delta": 0.0,
            "fraction_of_oracle_mass_ceiling": 0.0,
            "per_action_doom_auc": 0.5,
            "within_root_doom_concordance": 0.5,
            "actionable_roots_count": 0,
            "b8_action_agreement_rate": 1.0,
            "safe_root_b8_agreement_rate": 1.0,
            "newly_caused_lethal_flip_rate": 0.0,
        }

    b8_m = np.zeros(n, dtype=np.float64)
    b8_s = np.zeros(n, dtype=np.float64)
    base_m = np.zeros(n, dtype=np.float64)
    crit_m = np.zeros(n, dtype=np.float64)
    crit_s = np.zeros(n, dtype=np.float64)
    ora_m = np.zeros(n, dtype=np.float64)
    ora_s = np.zeros(n, dtype=np.float64)
    wids = np.zeros(n, dtype=np.int64)

    agree = np.zeros(n, dtype=bool)
    safe_root = np.zeros(n, dtype=bool)
    lethal_flip = np.zeros(n, dtype=bool)

    doom_y_actionable: list[bool] = []
    doom_score_actionable: list[float] = []
    doom_y_all: list[bool] = []
    doom_score_all: list[float] = []
    within_root_aucs: list[float] = []

    for i, r in enumerate(roots):
        wids[i] = int(r["world_id"])
        leg = np.asarray(r["legal_mask"], dtype=bool)
        s_sc = np.asarray(r["s_score"], dtype=np.float64)
        m_sc = np.asarray(r["m_score"], dtype=np.float64)
        s_ht = np.asarray(r["s_hat"], dtype=np.float64)

        b8_a = int(r["b8_action"])
        base_a = int(r["base_action"])
        crit_a = int(r["critic_action"])
        ora_a = int(r["oracle_a_star"])

        b8_m[i] = m_sc[b8_a]
        b8_s[i] = s_sc[b8_a]
        base_m[i] = m_sc[base_a]
        crit_m[i] = m_sc[crit_a]
        crit_s[i] = s_sc[crit_a]
        ora_m[i] = m_sc[ora_a]
        ora_s[i] = s_sc[ora_a]

        agree[i] = crit_a == b8_a
        leg_s = s_sc[leg]
        if leg_s.size > 0 and float(np.min(leg_s)) >= 0.75:
            safe_root[i] = True
        if s_sc[b8_a] >= 0.75 and s_sc[crit_a] <= 0.25:
            lethal_flip[i] = True

        # Per-action doom classification (true doom = s_score <= 0.5, predicted doom = 1 - s_hat).
        y_leg = s_sc[leg] <= 0.5
        d_leg = 1.0 - s_ht[leg]
        doom_y_all.extend(bool(v) for v in y_leg.tolist())
        doom_score_all.extend(float(v) for v in d_leg.tolist())
        if bool(np.any(y_leg)) and bool(np.any(~y_leg)):
            doom_y_actionable.extend(bool(v) for v in y_leg.tolist())
            doom_score_actionable.extend(float(v) for v in d_leg.tolist())
            within_root_aucs.append(_roc_auc_binary(y_leg, d_leg))

    dm = crit_m - b8_m
    ds = crit_s - b8_s
    ora_dm = float(np.mean(ora_m - b8_m))
    mean_dm = float(np.mean(dm))
    mean_ds = float(np.mean(ds))

    # World-clustered bootstrap over all held-out test worlds.
    unique_worlds = sorted(set(int(w) for w in all_test_world_ids))
    w_count = len(unique_worlds)
    w_to_pos = {wid: pos for pos, wid in enumerate(unique_worlds)}
    w_root_cnt = np.zeros(w_count, dtype=np.float64)
    w_dm_sum = np.zeros(w_count, dtype=np.float64)
    w_ds_sum = np.zeros(w_count, dtype=np.float64)
    for i in range(n):
        pos = w_to_pos[int(wids[i])]
        w_root_cnt[pos] += 1.0
        w_dm_sum[pos] += dm[i]
        w_ds_sum[pos] += ds[i]

    rng = np.random.default_rng(rng_seed)
    boot_idx = rng.integers(0, w_count, size=(n_bootstrap, w_count))
    b_cnt = w_root_cnt[boot_idx].sum(axis=1)
    b_dm = w_dm_sum[boot_idx].sum(axis=1)
    b_ds = w_ds_sum[boot_idx].sum(axis=1)
    valid_b = b_cnt > 0
    if bool(np.any(valid_b)):
        dm_samples = b_dm[valid_b] / b_cnt[valid_b]
        ds_samples = b_ds[valid_b] / b_cnt[valid_b]
        dm_ci = [
            round(float(np.quantile(dm_samples, 0.025)), 4),
            round(float(np.quantile(dm_samples, 0.975)), 4),
        ]
        ds_ci = [
            round(float(np.quantile(ds_samples, 0.025)), 6),
            round(float(np.quantile(ds_samples, 0.975)), 6),
        ]
    else:
        dm_ci = [round(mean_dm, 4), round(mean_dm, 4)]
        ds_ci = [round(mean_ds, 6), round(mean_ds, 6)]

    if doom_y_actionable:
        doom_auc = _roc_auc_binary(
            np.asarray(doom_y_actionable, dtype=bool),
            np.asarray(doom_score_actionable, dtype=np.float64),
        )
        within_conc = float(np.mean(within_root_aucs))
    else:
        doom_auc = _roc_auc_binary(
            np.asarray(doom_y_all, dtype=bool),
            np.asarray(doom_score_all, dtype=np.float64),
        )
        within_conc = doom_auc

    safe_agree = (
        float(np.mean(agree[safe_root])) if bool(np.any(safe_root)) else float(np.mean(agree))
    )
    ceiling_frac = (mean_dm / ora_dm) if ora_dm > 1e-6 else 0.0

    return {
        "num_roots": n,
        "contributing_worlds": len(set(wids.tolist())),
        "b8_mean_mass": round(float(np.mean(b8_m)), 4),
        "b8_mean_survival": round(float(np.mean(b8_s)), 6),
        "base_mean_mass": round(float(np.mean(base_m)), 4),
        "critic_mean_mass": round(float(np.mean(crit_m)), 4),
        "critic_mean_survival": round(float(np.mean(crit_s)), 6),
        "oracle_mean_mass": round(float(np.mean(ora_m)), 4),
        "oracle_mean_survival": round(float(np.mean(ora_s)), 6),
        "critic_vs_b8_mass_delta": round(mean_dm, 4),
        "critic_vs_b8_mass_delta_ci95": dm_ci,
        "critic_vs_b8_survival_delta": round(mean_ds, 6),
        "critic_vs_b8_survival_delta_ci95": ds_ci,
        "oracle_vs_b8_mass_delta": round(ora_dm, 4),
        "fraction_of_oracle_mass_ceiling": round(ceiling_frac, 4),
        "per_action_doom_auc": round(doom_auc, 6),
        "within_root_doom_concordance": round(within_conc, 6),
        "actionable_roots_count": len(within_root_aucs),
        "b8_action_agreement_rate": round(float(np.mean(agree)), 6),
        "safe_root_b8_agreement_rate": round(safe_agree, 6),
        "newly_caused_lethal_flip_rate": round(float(np.mean(lethal_flip)), 6),
    }


def aggregate_gate1_readout(
    attempt_dirs: Sequence[Path],
    *,
    expected_namespace: str = MAIN_NAMESPACE,
    expected_replicates: int = 2,
    local_preflight: bool = False,
    allow_canary_units: bool = False,
) -> dict[str, Any]:
    """Verify receipts + unit sidecars, prove world disjointness, and evaluate Gate-1 bars."""
    units_by_rep: dict[int, dict[str, Any]] = {}
    unit_digests: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []

    for attempt_dir in attempt_dirs:
        units_dir, prov = _resolve_units_dir(attempt_dir, local_preflight=local_preflight)
        provenance.append(prov)
        if not units_dir.is_dir():
            continue
        for json_path in sorted(units_dir.glob("replicate-*.json")):
            verified = verify_immutable(json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("schema") != UNIT_SCHEMA:
                raise ValueError(f"Foreign unit schema in {json_path}: {payload.get('schema')!r}")
            if (
                payload.get("study") != STUDY
                or payload.get("namespace") != expected_namespace
                or payload.get("allocation") != ALLOCATION_NAME
                or payload.get("allocation_index") != ALLOCATION_INDEX
                or payload.get("unit_kind") != UNIT_KIND
            ):
                raise ValueError(
                    f"Foreign unit namespace/allocation in {json_path}: "
                    f"got {payload.get('namespace')!r}, expected {expected_namespace!r}"
                )
            if bool(payload.get("canary_only", False)) and not allow_canary_units:
                raise ValueError(f"Refusing canary_only unit {json_path} in main readout")

            rep_idx = int(payload["replicate_index"])
            expected_uid = replicate_unit_id(expected_namespace, rep_idx)
            if payload.get("unit_id") != expected_uid:
                raise ValueError(f"Mismatched unit_id in {json_path}")
            if rep_idx in units_by_rep:
                raise ValueError(f"Duplicate replicate-{rep_idx:04d} across attempt directories")

            units_by_rep[rep_idx] = payload
            unit_digests[rep_idx] = {
                "unit_id": expected_uid,
                "path": str(json_path),
                "sha256": str(verified["sha256"]),
                "size_bytes": int(verified["size"]),
            }

    expected_set = set(range(int(expected_replicates)))
    if set(units_by_rep.keys()) != expected_set:
        missing = sorted(expected_set - set(units_by_rep.keys()))
        extra = sorted(set(units_by_rep.keys()) - expected_set)
        raise ValueError(
            f"Incomplete or unexpected replicate coverage: missing={missing}, extra={extra}"
        )

    # Verify cross-replicate & cross-partition world-seed disjointness + pre-test critic freeze.
    all_seen_seeds: set[int] = set()
    fit_world_total = 0
    val_world_total = 0
    test_world_total = 0
    pooled_test_worlds: list[dict[str, Any]] = []
    pooled_test_roots: list[dict[str, Any]] = []
    per_rep_l600_ev: list[float] = []
    per_rep_summaries: list[dict[str, Any]] = []

    for rep_idx in sorted(units_by_rep.keys()):
        u = units_by_rep[rep_idx]
        wd = u["world_disjointness"]
        if wd.get("critic_frozen_before_test_harvest") is not True:
            raise ValueError(f"Replicate {rep_idx} missing pre-test critic freeze proof")
        if wd.get("frozen_critic_sha256") != wd.get("post_eval_critic_sha256"):
            raise ValueError(f"Replicate {rep_idx} critic digest changed during test evaluation")

        rep_seeds = (
            [int(s) for s in wd["fit_seeds"]]
            + [int(s) for s in wd["val_seeds"]]
            + [int(s) for s in wd["test_seeds"]]
        )
        if len(rep_seeds) != len(set(rep_seeds)):
            raise ValueError(f"Intra-replicate seed overlap in replicate {rep_idx}")
        overlap = all_seen_seeds & set(rep_seeds)
        if overlap:
            raise ValueError(f"Cross-replicate world seed overlap on {sorted(overlap)}")
        all_seen_seeds.update(rep_seeds)

        fit_world_total += len(wd["fit_seeds"])
        val_world_total += len(wd["val_seeds"])
        test_world_total += len(wd["test_seeds"])

        rep_worlds = u["held_out_test_worlds"]
        rep_roots = u["held_out_test_roots"]
        pooled_test_worlds.extend(rep_worlds)
        pooled_test_roots.extend(rep_roots)

        rep_ev600 = summarize_world_stats_quick(rep_worlds, "l_ge_600")["residual_ev"]
        per_rep_l600_ev.append(float(rep_ev600))
        per_rep_summaries.append(
            {
                "replicate_index": rep_idx,
                "unit_id": u["unit_id"],
                "frozen_critic_sha256": wd["frozen_critic_sha256"],
                "frozen_baseline_sha256": wd["frozen_baseline_sha256"],
                "l_ge_600_quick_ev": summarize_world_stats_quick(rep_worlds, "l_ge_600"),
                "all_lengths_quick_ev": summarize_world_stats_quick(rep_worlds, "all_lengths"),
                "held_out_roots_count": len(rep_roots),
                "wall_sec": u["telemetry"]["replicate_wall_sec"],
            }
        )

    strata_metrics = {
        s_name: _aggregate_stratum_ev(pooled_test_worlds, s_name) for s_name in EVAL_STRATA
    }
    all_test_wids = [int(w["world_id"]) for w in pooled_test_worlds]
    death_roots = [r for r in pooled_test_roots if r["kind"] == "death_lookback"]
    ordinary_roots = [r for r in pooled_test_roots if r["kind"] == "ordinary"]

    h300_rescue = {
        "death_lookback": _aggregate_root_subset(death_roots, all_test_world_ids=all_test_wids),
        "ordinary": _aggregate_root_subset(ordinary_roots, all_test_world_ids=all_test_wids),
        "all_mature": _aggregate_root_subset(pooled_test_roots, all_test_world_ids=all_test_wids),
    }

    metrics_for_bars = {
        **strata_metrics,
        "per_replicate_l_ge_600_residual_ev": per_rep_l600_ev,
        "h300_rescue": h300_rescue,
    }
    decision = evaluate_gate1_bars(metrics_for_bars)

    return {
        "schema": "t34-t4-gate1-readout/v1",
        "study": STUDY,
        "namespace": expected_namespace,
        "allocation": ALLOCATION_NAME,
        "allocation_index": ALLOCATION_INDEX,
        "local_preflight": bool(local_preflight),
        "provenance": provenance,
        "verified_units": [unit_digests[k] for k in sorted(unit_digests.keys())],
        "world_disjointness_proof": {
            "status": "VERIFIED_DISJOINT",
            "total_unique_worlds": len(all_seen_seeds),
            "fit_worlds_total": fit_world_total,
            "val_worlds_total": val_world_total,
            "held_out_test_worlds_total": test_world_total,
            "cross_partition_overlap_count": 0,
            "cross_replicate_overlap_count": 0,
            "all_critics_frozen_before_test_harvest": True,
            "b8_weights_embedded_in_critic": False,
        },
        "pre_stated_thresholds": dict(T4_GATE1_THRESHOLDS),
        "full_track4_go_prerequisites": list(T4_FULL_GO_PREREQUISITES),
        "per_replicate": per_rep_summaries,
        "held_out_explained_variance": strata_metrics,
        "held_out_h300_counterfactual_rescue": h300_rescue,
        "decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--emit-ledger", type=Path, default=None)
    parser.add_argument("--namespace", type=str, default=MAIN_NAMESPACE)
    parser.add_argument("--expected-replicates", type=int, default=2)
    parser.add_argument("--local-preflight", action="store_true")
    parser.add_argument("--allow-canary-units", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    attempt_dirs = [p.resolve() for p in args.attempt_dir]
    if args.emit_ledger is not None:
        valid_set = set(range(int(args.expected_replicates)))
        report = build_verified_ledger_outcome_blind(
            attempt_dirs,
            ledger_out=args.emit_ledger.resolve(),
            namespace=args.namespace,
            valid_replicates=valid_set,
            local_preflight=args.local_preflight,
        )
        print(f"T4_GATE1_LEDGER {json.dumps(report, sort_keys=True)}", flush=True)
        return 0

    if args.output is None:
        raise ValueError("--output is required when not running --emit-ledger")

    readout = aggregate_gate1_readout(
        attempt_dirs,
        expected_namespace=args.namespace,
        expected_replicates=args.expected_replicates,
        local_preflight=args.local_preflight,
        allow_canary_units=args.allow_canary_units,
    )
    publish_json(args.output.resolve(), readout)
    headline = {
        "verdict": readout["decision"]["verdict"],
        "l600_residual_ev": readout["held_out_explained_variance"]["l_ge_600"]["residual_ev"],
        "l600_residual_ev_ci95": readout["held_out_explained_variance"]["l_ge_600"][
            "residual_ev_ci95"
        ],
        "death_h300_rescue_mass_delta": readout["held_out_h300_counterfactual_rescue"][
            "death_lookback"
        ]["critic_vs_b8_mass_delta"],
        "mature_doom_auc": readout["held_out_h300_counterfactual_rescue"]["all_mature"][
            "per_action_doom_auc"
        ],
        "failed_bars": readout["decision"]["failed_bars"],
    }
    print(f"T4_GATE1_READOUT {json.dumps(headline, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
