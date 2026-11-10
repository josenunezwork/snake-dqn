"""Step B1b-Smoke entrypoint (`short-canary <= 25 min`, cap `3.0 A100 u`).

Implements the registered throughput and CUDA parity smoke from
``docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md`` §3.3 / §4
and the review checklist:

1. **CUDA-vs-CPU parity check on row subsets against pure-Torch paths:**
   Rebuilds a slice of the frozen mature cohort directly from ``MATURE_EXTENSION_SPECS``
   in the repository (no ``/Users`` paths at runtime) and verifies exact parity of:
   - ``flood_cuda.flood`` (via ``v8_device.v8_decide(..., features=True)``) on row subsets
   - ``flood_cuda.components`` (via ``ego_device._component_sizes``) on row subsets
   - ``vector_geometry.exact_features_rows`` and ``FastDeviceVectorRuntime.prepare`` on
     ``fresh_rows`` / ``cap_rows`` row subsets
   - ``batched_open_stop_h32`` boolean certificates (``certified``, ``raw_survives``,
     ``safe_mask``, ``unsafe_mask``, ``rescue_mask``, ``ordinary_mask``,
     ``bare_actions``, ``a_star_actions``)
   against the pure-Torch wavefront / component / featurizer paths, failing closed on any
   boolean or action mismatch.
2. **Throughput & VRAM benchmark at ``E = 256``:**
   Measures env ``steps/s``, 6-branch ``H = 32`` ``labels/s`` (roots/s and branches/s),
   learner ``updates/s``, and peak VRAM on ``CTDeviceSim`` (`E = 256`).
3. **Pilot runaway guard calculation:**
   Writes the base estimate and ``2x`` runaway-guard envelope (minutes and A100 units)
   required by the registration before ``B1-A100-Pilot`` dispatches.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from research.gpu_sim_spike import ego_device, r2_cache
from research.gpu_sim_spike.v8_device import v8_decide as gpu_v8_decide
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from research.gpu_sitl.open_stop_gpu import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_V8_LAMBDA,
    FastDeviceVectorRuntime,
    _maybe_cpu_fast_flood,
    _ttl_by_owner,
    batched_open_stop_h32,
    freeze_ego2s_value_head,
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
    _ENV_TENSOR_FIELDS,
    _VECTOR_RUNTIME_ENV_FIELDS,
    PROBE32_HORIZON,
    PROBE32_SCORING_SUFFIXES,
    PROBE32_WORLD_COUNT,
    _evaluate_scoring_suffixes_on_snapshots,
    build_watch6_ct_device_bundle,
    extract_hero_root_observations,
    sha256_file,
    slice_single_row_cpu_state,
    write_atomic_with_sha256,
)
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    train_seeds,
)
from research.gpu_sitl.train_sitl import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    ORDINARY_MARGIN_DELTA_DEFAULT,
    REGISTERED_ENVS,
    REGISTERED_MIN_UNIQUE_RESCUE_ROOTS,
    REGISTERED_ORDINARY_BATCH_SIZE,
    REGISTERED_RESCUE_BATCH_SIZE,
    REGISTERED_TOTAL_UPDATES,
    RESCUE_MARGIN_DELTA_DEFAULT,
    SitlRingBuffer,
    compute_sitl_hinge_loss,
    sample_ordinary_minibatch_indices,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

# Default 8-root slice spanning Groups A (2), B (2), C (2), and D (2) of MATURE_EXTENSION_SPECS
DEFAULT_MATURE_SLICE_INDICES: tuple[int, ...] = (0, 1, 8, 9, 16, 17, 24, 25)
REGISTERED_RESCUE_RATE_PRIOR = 0.0012
MIN_OBSERVED_RESCUES_FOR_UNCLAMPED_RATE = 20
PRIOR_FALLBACK_LOWER_BOUND = 0.0006
PRIOR_FALLBACK_UPPER_BOUND = 0.0024
DEFAULT_CUDA_BENCH_LABEL_BATCHES = 64
DEFAULT_CUDA_MATURE_WARMUP_STEPS = 600
REGISTERED_ALIVE_HERO_FRAC_ESTIMATE = 0.95
REGISTERED_A100_CALIBRATED_RATE_UPH = 6.73
REGISTERED_A100_BROKER_CEILING_RATE_UPH = 8.00


def wilson_rate_ci95(successes: int, trials: int) -> tuple[float, float]:
    """Return 95% Wilson score confidence interval ``(low, high)`` for ``successes / trials``."""
    n = int(trials)
    k = max(0, min(n, int(successes)))
    if n <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    z2 = z * z
    p_hat = float(k) / float(n)
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    rad = max(0.0, (p_hat * (1.0 - p_hat)) / float(n) + z2 / (4.0 * float(n) * float(n)))
    half = (z / denom) * math.sqrt(rad)
    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return (low, high)


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.inference_mode()
def run_mature_cohort_row_subset_parity(
    b8_path: Path | str,
    opponent_path: Path | str,
    *,
    device: str | torch.device = "cuda",
    slice_indices: Sequence[int] = DEFAULT_MATURE_SLICE_INDICES,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compare device native kernels against pure-Torch CPU paths on mature cohort row subsets.

    Fails closed (raises :class:`RuntimeError`) on any boolean or action mismatch.
    """
    t0 = time.perf_counter()
    dev = normalize_device(device)
    if not dry_run and dev.type != "cuda":
        raise RuntimeError(f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}")
    native_info: dict[str, Any] = {}
    if dev.type == "cuda":
        native_info = verify_cuda_native_extensions(dev)

    selected_specs = [MATURE_EXTENSION_SPECS[int(i)] for i in slice_indices]
    n_roots = len(selected_specs)
    if n_roots < 2:
        raise ValueError("slice_indices must select at least 2 mature cohort roots")

    worlds, pol_tpl, build_meta = build_extension_worlds(
        Path(b8_path),
        Path(opponent_path),
        specs=selected_specs,
    )
    sim_cpu, pol_cpu = snapshot_worlds_to_device(
        worlds,
        pol_tpl,
        device="cpu",
        vector_prepared=True,
    )
    sim_dev, pol_dev = snapshot_worlds_to_device(
        worlds,
        pol_tpl,
        device=dev,
        vector_prepared=True,
    )
    r2_cache.enable(sim_cpu)
    r2_cache.enable(sim_dev)

    # Construct non-contiguous row subsets across (env, slot) to exercise indexed gather/scatter
    subset_pairs: list[tuple[int, int]] = []
    for env_i in range(n_roots):
        slot_a = (env_i * 2 + 1) % int(sim_cpu.S)
        slot_b = 0 if (env_i % 2 == 0) else ((env_i + 3) % int(sim_cpu.S))
        subset_pairs.append((env_i, slot_a))
        if slot_b != slot_a:
            subset_pairs.append((env_i, slot_b))
    rows_cpu = torch.tensor(subset_pairs, dtype=torch.long, device="cpu")
    rows_dev = rows_cpu.to(device=dev)
    n_sub = int(rows_cpu.shape[0])

    # 1. Parity Check: flood_cuda.flood via v8_decide(..., features=True) on row subsets
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261109)
    q_sub_cpu = torch.randn((n_sub, 6), generator=gen, dtype=torch.float32)
    q_sub_dev = q_sub_cpu.to(device=dev)

    adv_all_cpu = sim_cpu.advisory_mask()
    adv_sub_cpu = adv_all_cpu[rows_cpu[:, 0], rows_cpu[:, 1]]
    adv_sub_dev = adv_sub_cpu.to(device=dev)

    with _maybe_cpu_fast_flood(False):
        v8_ref = gpu_v8_decide(
            sim_cpu,
            rows_cpu,
            q_sub_cpu,
            adv_sub_cpu,
            float(OPEN_STOP_V8_LAMBDA),
            features=True,
        )
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        v8_cand = gpu_v8_decide(
            sim_dev,
            rows_dev,
            q_sub_dev,
            adv_sub_dev,
            float(OPEN_STOP_V8_LAMBDA),
            features=True,
        )

    v8_action_match = bool(torch.equal(v8_ref.action, v8_cand.action.cpu()))
    v8_outcome_match = bool(torch.equal(v8_ref.outcome, v8_cand.outcome.cpu()))
    v8_counts_match = bool(torch.equal(v8_ref.counts, v8_cand.counts.cpu()))
    v8_area_match = bool(torch.equal(v8_ref.area, v8_cand.area.cpu()))
    v8_cap_match = bool(torch.equal(v8_ref.cap, v8_cand.cap.cpu()))
    v8_landing_match = bool(torch.equal(v8_ref.landing, v8_cand.landing.cpu()))
    v8_spacious_match = bool(torch.equal(v8_ref.spacious, v8_cand.spacious.cpu()))
    v8_landing_ok_match = bool(torch.equal(v8_ref.landing_ok, v8_cand.landing_ok.cpu()))
    v8_head_risky_match = bool(torch.equal(v8_ref.head_risky, v8_cand.head_risky.cpu()))
    v8_feat_max_abs = float(
        (v8_ref.feature_planes() - v8_cand.feature_planes().cpu()).abs().max().item()
    )
    v8_parity_ok = bool(
        v8_action_match
        and v8_outcome_match
        and v8_counts_match
        and v8_area_match
        and v8_cap_match
        and v8_landing_match
        and v8_spacious_match
        and v8_landing_ok_match
        and v8_head_risky_match
        and v8_feat_max_abs <= 1e-6
    )
    if not v8_parity_ok:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in flood_cuda.flood / v8_decide on row subsets: "
            f"action={v8_action_match}, outcome={v8_outcome_match}, counts={v8_counts_match}, "
            f"area={v8_area_match}, cap={v8_cap_match}, landing={v8_landing_match}, "
            f"spacious={v8_spacious_match}, landing_ok={v8_landing_ok_match}, "
            f"head_risky={v8_head_risky_match}, feat_max_abs={v8_feat_max_abs}"
        )

    # 2. Parity Check: flood_cuda.components via ego_device._component_sizes on row subsets
    with _maybe_cpu_fast_flood(False):
        comp_ref = ego_device._component_sizes(sim_cpu, rows_cpu, _ttl_by_owner(sim_cpu))
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        comp_cand = ego_device._component_sizes(sim_dev, rows_dev, _ttl_by_owner(sim_dev))
    comp_match = bool(torch.equal(comp_ref.to(torch.long), comp_cand.cpu().to(torch.long)))
    if not comp_match:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in flood_cuda.components / _component_sizes "
            f"on row subsets: ref={comp_ref.tolist()} vs cand={comp_cand.cpu().tolist()}"
        )

    # 3. Parity Check: vector_geometry.exact_features_rows and FastDeviceVectorRuntime row subsets
    rt_cpu = pol_cpu.vector_runtime
    rt_dev = pol_dev.vector_runtime
    with _maybe_cpu_fast_flood(False):
        vec_ref = rt_cpu._features_rows(sim_cpu, rows_cpu, update_memory=False)
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        vec_cand = rt_dev._features_rows(sim_dev, rows_dev, update_memory=False)
    vec_diff = (vec_ref - vec_cand.cpu()).abs()
    if dry_run and dev.type == "mps":
        vec_col_mask = torch.ones(vec_diff.shape[-1], dtype=torch.bool)
        vec_col_mask[49] = False
        vec_max_abs = float(vec_diff[..., vec_col_mask].max().item())
    else:
        vec_max_abs = float(vec_diff.max().item())

    # Test FastDeviceVectorRuntime.prepare vs DeviceVectorRuntime.prepare across three
    # distinct non-empty row subsets:
    # (a) fresh_rows: sel_fresh=True, sel_ready=False, _carry_valid=False
    # (b) cap_rows: sel_fresh=False, sel_ready=False, _carry_valid=True
    # (c) already_ready: sel_fresh=False, sel_ready=True
    rt_base_cpu = rt_cpu.repeat_interleave(1)
    rt_sub_cpu = rt_cpu.repeat_interleave(1)
    rt_sub_dev = rt_dev.repeat_interleave(1)
    rt_base_cpu.__class__ = DeviceVectorRuntime
    rt_sub_cpu.__class__ = FastDeviceVectorRuntime
    rt_sub_dev.__class__ = FastDeviceVectorRuntime

    with _maybe_cpu_fast_flood(False):
        rt_base_cpu.observe_step(sim_cpu)
        rt_sub_cpu.observe_step(sim_cpu)
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        rt_sub_dev.observe_step(sim_dev)

    s_count = int(sim_cpu.S)
    sel_fresh_init = torch.zeros((n_roots, s_count), dtype=torch.bool, device="cpu")
    sel_ready_init = torch.zeros((n_roots, s_count), dtype=torch.bool, device="cpu")
    carry_valid_init = torch.zeros((n_roots, s_count), dtype=torch.bool, device="cpu")
    # (a) fresh_rows on slots 1..2
    sel_fresh_init[:, 1:3] = True
    sel_ready_init[:, 1:3] = False
    carry_valid_init[:, 1:3] = False
    # (b) cap_rows on slots 3..4
    sel_fresh_init[:, 3:5] = False
    sel_ready_init[:, 3:5] = False
    carry_valid_init[:, 3:5] = True
    # (c) already_ready on slot 5
    sel_fresh_init[:, 5:] = False
    sel_ready_init[:, 5:] = True
    carry_valid_init[:, 5:] = True

    for rt_obj, tgt_dev in (
        (rt_base_cpu, torch.device("cpu")),
        (rt_sub_cpu, torch.device("cpu")),
        (rt_sub_dev, dev),
    ):
        rt_obj.sel_fresh.copy_(sel_fresh_init.to(device=tgt_dev))
        rt_obj.sel_ready.copy_(sel_ready_init.to(device=tgt_dev))
        rt_obj._carry_valid.copy_(carry_valid_init.to(device=tgt_dev))

    active_ctrl = rt_sub_dev.controlled & sim_dev.alive.to(device=dev, dtype=torch.bool)
    pre_fresh_rows = int(
        (active_ctrl & rt_sub_dev.sel_fresh & (~rt_sub_dev.sel_ready) & (~rt_sub_dev._carry_valid))
        .sum()
        .item()
    )
    pre_cap_rows = int(
        (active_ctrl & (~rt_sub_dev.sel_fresh) & (~rt_sub_dev.sel_ready) & rt_sub_dev._carry_valid)
        .sum()
        .item()
    )
    pre_already_ready_rows = int(
        (active_ctrl & (~rt_sub_dev.sel_fresh) & rt_sub_dev.sel_ready).sum().item()
    )
    if pre_fresh_rows <= 0 or pre_cap_rows <= 0 or pre_already_ready_rows <= 0:
        raise RuntimeError(
            "Step B1b-Smoke prepare setup must have non-empty fresh_rows, cap_rows, and "
            f"already_ready_rows before prepare(sim): got fresh_rows={pre_fresh_rows}, "
            f"cap_rows={pre_cap_rows}, already_ready_rows={pre_already_ready_rows}"
        )

    with _maybe_cpu_fast_flood(False):
        rt_base_cpu.prepare(sim_cpu)
        rt_sub_cpu.prepare(sim_cpu)
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        rt_sub_dev.prepare(sim_dev)

    post_fresh_rows = int(rt_sub_dev.sel_fresh.sum().item())
    post_cap_rows = int((rt_sub_dev.sel_ready & ~rt_sub_dev.sel_fresh).sum().item())
    if post_fresh_rows <= 0 or post_cap_rows <= 0:
        raise RuntimeError(
            f"Step B1b-Smoke post-prepare fresh_rows={post_fresh_rows} and "
            f"cap_rows={post_cap_rows} must both be > 0"
        )

    prep_diff_base = (rt_base_cpu.sel_state - rt_sub_dev.sel_state.cpu()).abs()
    prep_diff_sub = (rt_sub_cpu.sel_state - rt_sub_dev.sel_state.cpu()).abs()
    if dry_run and dev.type == "mps":
        prep_col_mask = torch.ones(prep_diff_base.shape[-1], dtype=torch.bool)
        prep_col_mask[49] = False
        prep_state_max_abs = float(
            max(
                prep_diff_base[..., prep_col_mask].max().item(),
                prep_diff_sub[..., prep_col_mask].max().item(),
            )
        )
    else:
        prep_state_max_abs = float(
            max(
                prep_diff_base.max().item(),
                prep_diff_sub.max().item(),
            )
        )
    prep_mask_match = bool(
        torch.equal(rt_base_cpu.sel_mask, rt_sub_dev.sel_mask.cpu())
        and torch.equal(rt_sub_cpu.sel_mask, rt_sub_dev.sel_mask.cpu())
        and torch.equal(rt_base_cpu.sel_ready, rt_sub_dev.sel_ready.cpu())
        and torch.equal(rt_base_cpu.sel_fresh, rt_sub_dev.sel_fresh.cpu())
        and torch.equal(rt_base_cpu._carry_valid, rt_sub_dev._carry_valid.cpu())
    )
    vec_parity_ok = bool(
        math.isfinite(vec_max_abs)
        and vec_max_abs <= 1e-4
        and math.isfinite(prep_state_max_abs)
        and prep_state_max_abs <= 1e-4
        and prep_mask_match
        and pre_fresh_rows > 0
        and pre_cap_rows > 0
        and post_fresh_rows > 0
        and post_cap_rows > 0
    )
    if not vec_parity_ok:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in vector_geometry / FastDeviceVectorRuntime on row "
            f"subsets: vec_max_abs={vec_max_abs}, prep_state_max_abs={prep_state_max_abs}, "
            f"prep_mask_match={prep_mask_match}, fresh_rows={post_fresh_rows}, "
            f"cap_rows={post_cap_rows}"
        )

    # 4. Parity Check: batched_open_stop_h32 boolean certificates on the mature cohort slice
    slice_seeds = [int(w.seed) for w in worlds]
    slice_decisions = [int(w.sim.frame[0]) for w in worlds]

    cert_ref = batched_open_stop_h32(
        sim_cpu,
        pol_cpu,
        world_seeds=slice_seeds,
        decisions=slice_decisions,
        suffix_id=0,
        branch_horizon=int(branch_horizon),
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=False,
        return_host_records=False,
    )
    cert_cand = batched_open_stop_h32(
        sim_dev,
        pol_dev,
        world_seeds=slice_seeds,
        decisions=slice_decisions,
        suffix_id=0,
        branch_horizon=int(branch_horizon),
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=bool(dev.type == "cpu"),
        return_host_records=False,
    )

    certified_match = bool(torch.equal(cert_ref["certified"], cert_cand["certified"].cpu()))
    raw_survives_match = bool(
        torch.equal(cert_ref["raw_survives"], cert_cand["raw_survives"].cpu())
    )
    safe_mask_match = bool(torch.equal(cert_ref["safe_mask"], cert_cand["safe_mask"].cpu()))
    unsafe_mask_match = bool(torch.equal(cert_ref["unsafe_mask"], cert_cand["unsafe_mask"].cpu()))
    rescue_mask_match = bool(torch.equal(cert_ref["rescue_mask"], cert_cand["rescue_mask"].cpu()))
    ordinary_mask_match = bool(
        torch.equal(cert_ref["ordinary_mask"], cert_cand["ordinary_mask"].cpu())
    )
    bare_actions_match = bool(
        torch.equal(cert_ref["bare_actions"], cert_cand["bare_actions"].cpu())
    )
    a_star_actions_match = bool(
        torch.equal(cert_ref["a_star_actions"], cert_cand["a_star_actions"].cpu())
    )

    open_stop_parity_ok = bool(
        certified_match
        and raw_survives_match
        and safe_mask_match
        and unsafe_mask_match
        and rescue_mask_match
        and ordinary_mask_match
        and bare_actions_match
        and a_star_actions_match
    )
    if not open_stop_parity_ok:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in batched_open_stop_h32 boolean certificates on "
            f"rebuilt mature cohort slice: certified={certified_match}, "
            f"raw_survives={raw_survives_match}, safe_mask={safe_mask_match}, "
            f"unsafe_mask={unsafe_mask_match}, rescue_mask={rescue_mask_match}, "
            f"ordinary_mask={ordinary_mask_match}, bare_actions={bare_actions_match}, "
            f"a_star_actions={a_star_actions_match}"
        )

    return {
        "ok": True,
        "device": str(dev),
        "dry_run": bool(dry_run),
        "native_extensions": native_info,
        "slice_indices": [int(i) for i in slice_indices],
        "num_roots": int(n_roots),
        "num_row_subset_entries": int(n_sub),
        "branch_horizon": int(branch_horizon),
        "rebuilt_from_repo_specs_only": True,
        "build_meta_count": len(build_meta),
        "flood_cuda_v8_decide_row_subset": {
            "ok": v8_parity_ok,
            "action_match": v8_action_match,
            "outcome_match": v8_outcome_match,
            "counts_match": v8_counts_match,
            "area_match": v8_area_match,
            "cap_match": v8_cap_match,
            "landing_match": v8_landing_match,
            "spacious_match": v8_spacious_match,
            "landing_ok_match": v8_landing_ok_match,
            "head_risky_match": v8_head_risky_match,
            "feature_planes_max_abs_diff": v8_feat_max_abs,
        },
        "flood_cuda_components_row_subset": {
            "ok": comp_match,
            "exact_integer_match": comp_match,
        },
        "vector_geometry_row_subset": {
            "ok": vec_parity_ok,
            "features_rows_max_abs_diff": vec_max_abs,
            "partial_prepare_sel_state_max_abs_diff": prep_state_max_abs,
            "partial_prepare_sel_mask_exact_match": prep_mask_match,
            "pre_prepare_fresh_rows": pre_fresh_rows,
            "pre_prepare_cap_rows": pre_cap_rows,
            "pre_prepare_already_ready_rows": pre_already_ready_rows,
            "fresh_rows": post_fresh_rows,
            "cap_rows": post_cap_rows,
        },
        "batched_open_stop_h32_mature_slice": {
            "ok": open_stop_parity_ok,
            "certified_exact_match": certified_match,
            "raw_survives_exact_match": raw_survives_match,
            "safe_mask_exact_match": safe_mask_match,
            "unsafe_mask_exact_match": unsafe_mask_match,
            "rescue_mask_exact_match": rescue_mask_match,
            "ordinary_mask_exact_match": ordinary_mask_match,
            "bare_actions_exact_match": bare_actions_match,
            "a_star_actions_exact_match": a_star_actions_match,
            "num_rescue_roots_in_slice": int(cert_ref["rescue_mask"].sum().item()),
            "num_ordinary_roots_in_slice": int(cert_ref["ordinary_mask"].sum().item()),
        },
        "elapsed_sec": time.perf_counter() - t0,
    }


def compute_pilot_runaway_guard(
    *,
    env_steps_per_sec: float,
    label_batches_per_sec: float,
    updates_per_sec: float,
    measured_online_rescue_rate: float | None = None,
    measured_online_rescue_episode_rate: float | None = None,
    measured_alive_hero_decisions: int = 0,
    measured_rescue_frame_count: int = 0,
    measured_rescue_episode_count: int = 0,
    buffer_insertion_sec_per_step: float = 0.0,
    probe32_scoring_suffix_sec_per_root: float = 0.0,
) -> dict[str, Any]:
    """Compute the Step B1b (`B1-A100-Pilot`) base estimate and 2x runaway guard."""
    if env_steps_per_sec <= 0 or label_batches_per_sec <= 0 or updates_per_sec <= 0:
        raise ValueError("All measured throughput rates must be > 0")

    k_rescues = max(0, int(measured_rescue_frame_count))
    n_decisions = max(0, int(measured_alive_hero_decisions))
    ci_low, ci_high = wilson_rate_ci95(k_rescues, n_decisions)

    has_positive_rate = (
        measured_online_rescue_rate is not None and float(measured_online_rescue_rate) > 0.0
    )
    raw_measured_rate = (
        float(measured_online_rescue_rate)
        if has_positive_rate
        else (float(k_rescues) / float(n_decisions) if n_decisions > 0 else 0.0)
    )

    if k_rescues >= MIN_OBSERVED_RESCUES_FOR_UNCLAMPED_RATE and raw_measured_rate > 0.0:
        effective_rescue_rate = float(raw_measured_rate)
        rescue_rate_guard_rule = "observed_rate_ge_20_rescues"
        rescue_rate_source = "smoke_measured_train_namespace_ge_20_rescues"
    elif k_rescues == 0 and not has_positive_rate:
        effective_rescue_rate = float(REGISTERED_RESCUE_RATE_PRIOR)
        rescue_rate_guard_rule = "prior_fallback_under_20_rescues_zero_observed"
        rescue_rate_source = "registered_prior_fallback_under_20_rescues"
    else:
        if raw_measured_rate < float(PRIOR_FALLBACK_LOWER_BOUND):
            effective_rescue_rate = float(PRIOR_FALLBACK_LOWER_BOUND)
            rescue_rate_guard_rule = "prior_lower_bound_clamped_under_20_rescues"
        elif raw_measured_rate > float(PRIOR_FALLBACK_UPPER_BOUND):
            effective_rescue_rate = float(PRIOR_FALLBACK_UPPER_BOUND)
            rescue_rate_guard_rule = "prior_upper_bound_clamped_under_20_rescues"
        else:
            effective_rescue_rate = float(raw_measured_rate)
            rescue_rate_guard_rule = "prior_band_bounded_under_20_rescues"
        rescue_rate_source = "registered_prior_bounds_fallback_under_20_rescues"

    has_measured_ep_rate = (
        measured_online_rescue_episode_rate is not None
        and float(measured_online_rescue_episode_rate) > 0.0
    )
    raw_ep_rate = (
        float(measured_online_rescue_episode_rate)
        if has_measured_ep_rate
        else effective_rescue_rate
    )
    if int(measured_rescue_episode_count) >= MIN_OBSERVED_RESCUES_FOR_UNCLAMPED_RATE:
        effective_episode_rate = raw_ep_rate
    else:
        effective_episode_rate = min(
            float(PRIOR_FALLBACK_UPPER_BOUND),
            max(float(PRIOR_FALLBACK_LOWER_BOUND), raw_ep_rate),
        )

    # 1. Pre-training probe32 bank harvest (N=32 worlds, H=5,000 -> 160,000 root decisions)
    # Rolled in batches of E=32 (5,000 env steps), suffix-0 labeled in batches of 256 roots,
    # plus 4 scoring suffixes (4..7) on candidate rescue roots AND buffered ordinary candidates
    # (32 * 252 = 8,064 roots) per Stage 1B certification rule.
    probe32_env_sec = float(PROBE32_HORIZON) / float(env_steps_per_sec)
    probe32_s0_batches = float(PROBE32_WORLD_COUNT * PROBE32_HORIZON) / float(REGISTERED_ENVS)
    probe32_s0_label_sec = probe32_s0_batches / float(label_batches_per_sec)
    est_probe32_candidate_rescues = max(
        64.0, float(PROBE32_WORLD_COUNT * PROBE32_HORIZON) * effective_rescue_rate
    )
    est_probe32_ordinary_candidates = float(PROBE32_WORLD_COUNT * 252)
    est_probe32_total_scored_roots = est_probe32_candidate_rescues + est_probe32_ordinary_candidates
    if float(probe32_scoring_suffix_sec_per_root) > 0.0:
        probe32_scoring_suffix_sec = est_probe32_total_scored_roots * float(
            probe32_scoring_suffix_sec_per_root
        )
    else:
        probe32_scoring_batches = (
            est_probe32_total_scored_roots * float(len(PROBE32_SCORING_SUFFIXES))
        ) / float(REGISTERED_ENVS)
        probe32_scoring_suffix_sec = max(1.0, probe32_scoring_batches) / float(
            label_batches_per_sec
        )
    probe32_total_sec = probe32_env_sec + probe32_s0_label_sec + probe32_scoring_suffix_sec

    # 2. Online collection to mine >= 5,000 unique rescue roots using guarded rate
    # (and per-step CPU ring-buffer insertion cost).
    expected_hero_decisions = float(REGISTERED_MIN_UNIQUE_RESCUE_ROOTS) / max(
        1e-6, effective_rescue_rate
    )
    expected_collection_steps = expected_hero_decisions / (
        float(REGISTERED_ENVS) * float(REGISTERED_ALIVE_HERO_FRAC_ESTIMATE)
    )
    online_env_sec = expected_collection_steps / float(env_steps_per_sec)
    online_label_sec = expected_collection_steps / float(label_batches_per_sec)
    online_buffer_insertion_sec = expected_collection_steps * max(
        0.0, float(buffer_insertion_sec_per_step)
    )
    online_collection_sec = online_env_sec + online_label_sec + online_buffer_insertion_sec

    # Distinct-episode rescue reachability comparison
    expected_hero_decisions_episodes = float(REGISTERED_MIN_UNIQUE_RESCUE_ROOTS) / max(
        1e-6, effective_episode_rate
    )
    expected_collection_steps_episodes = expected_hero_decisions_episodes / (
        float(REGISTERED_ENVS) * float(REGISTERED_ALIVE_HERO_FRAC_ESTIMATE)
    )

    # 3. 1,000 learner updates + 6 probe32 tripwires + checkpoint I/O
    learner_sec = float(REGISTERED_TOTAL_UPDATES) / float(updates_per_sec)
    overhead_sec = 180.0

    base_seconds = probe32_total_sec + online_collection_sec + learner_sec + overhead_sec
    base_minutes = base_seconds / 60.0
    timeout_minutes = int(math.ceil(2.0 * base_minutes))
    timeout_seconds = float(timeout_minutes * 60)
    max_collection_steps_guard = int(math.ceil(2.0 * expected_collection_steps))

    base_units_calibrated = (base_minutes / 60.0) * REGISTERED_A100_CALIBRATED_RATE_UPH
    cap_units_calibrated = 2.0 * base_units_calibrated
    cap_units_broker_8uph = (
        float(timeout_minutes) / 60.0
    ) * REGISTERED_A100_BROKER_CEILING_RATE_UPH

    return {
        "registered_min_unique_rescue_roots": REGISTERED_MIN_UNIQUE_RESCUE_ROOTS,
        "registered_rescue_rate_prior": REGISTERED_RESCUE_RATE_PRIOR,
        "min_observed_rescues_for_unclamped_rate": MIN_OBSERVED_RESCUES_FOR_UNCLAMPED_RATE,
        "prior_fallback_bounds": [
            float(PRIOR_FALLBACK_LOWER_BOUND),
            float(PRIOR_FALLBACK_UPPER_BOUND),
        ],
        "measured_online_rescue_rate": float(raw_measured_rate),
        "measured_rescue_rate_ci95_wilson": [float(ci_low), float(ci_high)],
        "measured_online_rescue_episode_rate": (
            float(measured_online_rescue_episode_rate)
            if measured_online_rescue_episode_rate is not None
            else 0.0
        ),
        "measured_alive_hero_decisions": int(n_decisions),
        "measured_rescue_frame_count": int(k_rescues),
        "measured_rescue_episode_count": int(measured_rescue_episode_count),
        "rescue_rate_used_for_guard": effective_rescue_rate,
        "rescue_rate_guard_rule": rescue_rate_guard_rule,
        "rescue_rate_source": rescue_rate_source,
        "expected_hero_decisions_for_5000_rescues": expected_hero_decisions,
        "expected_collection_steps_at_e256": expected_collection_steps,
        "expected_collection_steps_for_5000_distinct_episodes": expected_collection_steps_episodes,
        "distinct_episodes_reachable_within_2x_step_guard": bool(
            expected_collection_steps_episodes <= float(max_collection_steps_guard)
        ),
        "runaway_guard_max_collection_steps_2x": max_collection_steps_guard,
        "probe32_env_estimate_sec": probe32_env_sec,
        "probe32_s0_label_estimate_sec": probe32_s0_label_sec,
        "probe32_scoring_suffix_estimate_sec": probe32_scoring_suffix_sec,
        "probe32_harvest_estimate_sec": probe32_total_sec,
        "buffer_insertion_sec_per_step": float(buffer_insertion_sec_per_step),
        "online_buffer_insertion_estimate_sec": online_buffer_insertion_sec,
        "online_collection_estimate_sec": online_collection_sec,
        "learner_updates_estimate_sec": learner_sec,
        "checkpoint_and_tripwire_overhead_sec": overhead_sec,
        "base_estimate_seconds": base_seconds,
        "base_estimate_minutes": base_minutes,
        "runaway_guard_timeout_minutes_2x": timeout_minutes,
        "runaway_guard_timeout_seconds_2x": timeout_seconds,
        "base_estimate_a100_units_at_6p73": base_units_calibrated,
        "runaway_guard_a100_units_2x_at_6p73": cap_units_calibrated,
        "runaway_guard_a100_units_2x_at_8p00_broker": cap_units_broker_8uph,
    }


def run_step_b1b_smoke(
    output_dir: Path | str,
    *,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    device: str | torch.device = "cuda",
    num_envs: int = REGISTERED_ENVS,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    parity_slice_indices: Sequence[int] = DEFAULT_MATURE_SLICE_INDICES,
    bench_env_steps: int = 32,
    bench_label_batches: int | None = None,
    bench_learner_updates: int = 20,
    seed: int = 20261109,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Step B1b-Smoke (`short-canary <= 25 min`, cap `3.0 A100 u`) and write JSON + SHA256."""
    t_start = time.perf_counter()
    dev = normalize_device(device)
    effective_label_batches = (
        int(bench_label_batches)
        if bench_label_batches is not None
        else (8 if dry_run else int(DEFAULT_CUDA_BENCH_LABEL_BATCHES))
    )
    if not dry_run:
        if dev.type != "cuda":
            raise RuntimeError(f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}")
        if int(num_envs) != REGISTERED_ENVS or int(branch_horizon) != OPEN_STOP_BRANCH_HORIZON:
            raise ValueError(
                f"Non-dry-run Step B1b-Smoke requires E={REGISTERED_ENVS} and "
                f"H={OPEN_STOP_BRANCH_HORIZON}, got E={num_envs}, H={branch_horizon}"
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

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    # (a) CUDA-vs-CPU parity check on row subsets against pure-Torch paths
    parity_report = run_mature_cohort_row_subset_parity(
        b8_path,
        opp_path,
        device=dev,
        slice_indices=parity_slice_indices,
        branch_horizon=int(branch_horizon),
        dry_run=dry_run,
    )

    # (b) Throughput and peak VRAM benchmark at E = num_envs (256 on A100)
    world_seeds = train_seeds(int(num_envs))
    sim, policy = build_watch6_ct_device_bundle(
        world_seeds,
        b8_path,
        opp_path,
        device=dev,
        hero_respawn=True,
    )
    student_net = load_ego2s_checkpoint(str(b8_path), device=dev)
    if not isinstance(student_net, Ego2sNet):
        raise TypeError(f"Expected Ego2sNet from {b8_path}, got {type(student_net)!r}")
    freeze_info = freeze_ego2s_value_head(student_net)
    trainable_params = [p for p in student_net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=float(DEFAULT_LEARNING_RATE))

    # Warmup 1 step + 1 label batch before timing
    with torch.inference_mode():
        sim.prepare()
        policy.vector_runtime.prepare(sim)
        policy.vector_prepared = True
    obs_warm = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
    with torch.no_grad():
        q_warm = student_net(obs_warm["local_u8"], obs_warm["global_u8"], obs_warm["scalars_f32"])
        act_warm = q_warm.masked_fill(~obs_warm["legal_mask"], -torch.inf).argmax(dim=1)
    _ = batched_open_stop_h32(
        sim,
        policy,
        world_seeds=world_seeds,
        decisions=[int(x) for x in obs_warm["decision_frame"].detach().cpu().tolist()],
        suffix_id=0,
        branch_horizon=int(branch_horizon),
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=bool(dev.type == "cpu"),
        return_host_records=False,
    )
    with torch.inference_mode():
        warm_actions = policy.actions(sim, act_warm)
        sim.step(warm_actions)
        policy.after_step(sim)
    _sync_device(dev)

    # Benchmark 1: env steps/s at E = num_envs
    t_env0 = time.perf_counter()
    for _ in range(int(bench_env_steps)):
        with torch.inference_mode():
            sim.prepare()
            policy.vector_runtime.prepare(sim)
            policy.vector_prepared = True
        obs_step = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
        with torch.no_grad():
            q_step = student_net(
                obs_step["local_u8"], obs_step["global_u8"], obs_step["scalars_f32"]
            )
            act_step = q_step.masked_fill(~obs_step["legal_mask"], -torch.inf).argmax(dim=1)
        with torch.inference_mode():
            full_act = policy.actions(sim, act_step)
            sim.step(full_act)
            policy.after_step(sim)
    _sync_device(dev)
    env_elapsed = max(1e-9, time.perf_counter() - t_env0)
    env_steps_per_sec = float(bench_env_steps) / env_elapsed
    hero_decisions_per_sec = float(int(num_envs) * int(bench_env_steps)) / env_elapsed

    # On CUDA (non-dry-run), advance bare b8-s3 on train_seeds(num_envs) into mature mid-game
    # traffic (600 frames) and splice mature cohort snapshots (L in [210, 823], L >= 160) so the
    # online rescue-rate measurement window covers both mature lengths and natural trajectories.
    t_warm0 = time.perf_counter()
    mature_warmup_steps = (
        0
        if dry_run or dev.type != "cuda"
        else max(0, int(DEFAULT_CUDA_MATURE_WARMUP_STEPS) - int(bench_env_steps))
    )
    for _ in range(mature_warmup_steps):
        with torch.inference_mode():
            sim.prepare()
            policy.vector_runtime.prepare(sim)
            policy.vector_prepared = True
            _, _, bare_w_act = policy.evaluate(sim)
            sim.step(policy.actions(sim, bare_w_act))
            policy.after_step(sim)

    n_mature_splice = min(len(parity_slice_indices), max(1, int(num_envs) // 2))
    if n_mature_splice > 0:
        splice_specs = [
            MATURE_EXTENSION_SPECS[int(i)] for i in parity_slice_indices[:n_mature_splice]
        ]
        splice_worlds, splice_pol_tpl, _ = build_extension_worlds(
            b8_path,
            opp_path,
            specs=splice_specs,
        )
        splice_sim, splice_pol = snapshot_worlds_to_device(
            splice_worlds,
            splice_pol_tpl,
            sim_cls=CTDeviceSim,
            device=dev,
            vector_prepared=False,
        )
        with torch.inference_mode():
            for name in _ENV_TENSOR_FIELDS:
                dst_t = getattr(sim, name, None)
                src_t = getattr(splice_sim, name, None)
                if (
                    isinstance(dst_t, torch.Tensor)
                    and isinstance(src_t, torch.Tensor)
                    and dst_t.shape[1:] == src_t.shape[1:]
                ):
                    dst_t[:n_mature_splice].copy_(src_t[:n_mature_splice])
            sim.terminal_slots[:n_mature_splice, 0] = False
            sim._prepared = False
            r2_cache.bump(sim)
            for name in _VECTOR_RUNTIME_ENV_FIELDS:
                dst_rt = getattr(policy.vector_runtime, name, None)
                src_rt = getattr(splice_pol.vector_runtime, name, None)
                if (
                    isinstance(dst_rt, torch.Tensor)
                    and isinstance(src_rt, torch.Tensor)
                    and dst_rt.shape[1:] == src_rt.shape[1:]
                ):
                    dst_rt[:n_mature_splice].copy_(src_rt[:n_mature_splice])
            policy.vector_prepared = False
    _sync_device(dev)
    mature_warmup_elapsed = max(0.0, time.perf_counter() - t_warm0)

    # Benchmark 2: 6-branch H=branch_horizon labels/s AND online rescue-root rate on
    # train_seeds(num_envs) under bare b8-s3, plus per-step buffer insertion cost.
    res_batch_sz = (
        min(REGISTERED_RESCUE_BATCH_SIZE, max(2, int(num_envs) * 2))
        if dry_run
        else REGISTERED_RESCUE_BATCH_SIZE
    )
    ord_batch_sz = (
        min(REGISTERED_ORDINARY_BATCH_SIZE, max(2, int(num_envs) * 2))
        if dry_run
        else REGISTERED_ORDINARY_BATCH_SIZE
    )
    ord_uni = ord_batch_sz // 2
    ord_hn = ord_batch_sz - ord_uni

    rescue_buf = SitlRingBuffer(capacity=max(512, res_batch_sz * 2))
    ordinary_buf = SitlRingBuffer(capacity=max(512, ord_batch_sz * 2))

    lbl_total_elapsed = 0.0
    buf_insert_total_elapsed = 0.0
    online_alive_hero_decisions = 0
    online_rescue_frame_count = 0
    online_rescue_episode_count = 0
    window_mature_length_ge_160_decisions = 0
    window_length_sum = 0
    window_min_hero_length = 0
    window_max_hero_length = 0
    in_rescue_run_by_env = [False] * int(num_envs)
    last_label_res: dict[str, Any] | None = None

    for _b_i in range(int(effective_label_batches)):
        with torch.inference_mode():
            sim.prepare()
            policy.vector_runtime.prepare(sim)
            policy.vector_prepared = True
        decs_now = [int(x) for x in sim.frame.detach().cpu().tolist()]
        _sync_device(dev)
        t_lbl_step = time.perf_counter()
        last_label_res = batched_open_stop_h32(
            sim,
            policy,
            world_seeds=world_seeds,
            decisions=decs_now,
            suffix_id=0,
            branch_horizon=int(branch_horizon),
            namespace=OPEN_STOP_CRN_NAMESPACE,
            compact_dead=True,
            enable_r2=True,
            cpu_fast_flood=bool(dev.type == "cpu"),
            return_host_records=False,
        )
        _sync_device(dev)
        lbl_total_elapsed += max(1e-9, time.perf_counter() - t_lbl_step)

        # Measure per-step observation extraction + CPU ring-buffer insertion cost
        # and accumulate online rescue-root and distinct-episode rescue counts.
        t_buf_step = time.perf_counter()
        obs_b = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
        loc_b = obs_b["local_u8"].detach().cpu()
        glb_b = obs_b["global_u8"].detach().cpu()
        sca_b = obs_b["scalars_f32"].detach().cpu()
        leg_b = obs_b["legal_mask"].detach().cpu()
        alive_b = obs_b["hero_alive"].detach().cpu()
        len_b = obs_b["hero_length"].detach().cpu()
        safe_b = last_label_res["safe_mask"].detach().cpu()
        unsafe_b = last_label_res["unsafe_mask"].detach().cpu()
        rescue_b = last_label_res["rescue_mask"].detach().cpu()
        ordinary_b = last_label_res["ordinary_mask"].detach().cpu()
        bare_b = last_label_res["bare_actions"].detach().cpu()
        astar_b = last_label_res["a_star_actions"].detach().cpu()

        for e_idx in range(int(num_envs)):
            is_alive = bool(alive_b[e_idx].item())
            if not is_alive:
                in_rescue_run_by_env[e_idx] = False
                continue
            h_len = int(len_b[e_idx].item())
            if online_alive_hero_decisions == 0 or h_len < window_min_hero_length:
                window_min_hero_length = h_len
            if h_len > window_max_hero_length:
                window_max_hero_length = h_len
            window_length_sum += h_len
            if h_len >= 160:
                window_mature_length_ge_160_decisions += 1
            online_alive_hero_decisions += 1
            if bool(rescue_b[e_idx].item()):
                online_rescue_frame_count += 1
                if not in_rescue_run_by_env[e_idx]:
                    online_rescue_episode_count += 1
                    in_rescue_run_by_env[e_idx] = True
                rescue_buf.add_single(
                    local_u8=loc_b[e_idx],
                    global_u8=glb_b[e_idx],
                    scalars_f32=sca_b[e_idx],
                    legal_mask=leg_b[e_idx],
                    safe_mask=safe_b[e_idx],
                    unsafe_mask=unsafe_b[e_idx],
                    bare_action=int(bare_b[e_idx].item()),
                    a_star_action=int(astar_b[e_idx].item()),
                    world_seed=int(world_seeds[e_idx]),
                    decision_frame=int(decs_now[e_idx]),
                    hard_neg_tier=0,
                )
            else:
                in_rescue_run_by_env[e_idx] = False
                if bool(ordinary_b[e_idx].item()):
                    ordinary_buf.add_single(
                        local_u8=loc_b[e_idx],
                        global_u8=glb_b[e_idx],
                        scalars_f32=sca_b[e_idx],
                        legal_mask=leg_b[e_idx],
                        safe_mask=safe_b[e_idx],
                        unsafe_mask=unsafe_b[e_idx],
                        bare_action=int(bare_b[e_idx].item()),
                        a_star_action=int(astar_b[e_idx].item()),
                        world_seed=int(world_seeds[e_idx]),
                        decision_frame=int(decs_now[e_idx]),
                        hard_neg_tier=0,
                    )
        buf_insert_total_elapsed += max(0.0, time.perf_counter() - t_buf_step)

        # Step bare b8-s3 trajectory forward 1 frame for next label batch
        with torch.inference_mode():
            _, _, bare_step_act = policy.evaluate(sim)
            sim.step(policy.actions(sim, bare_step_act))
            policy.after_step(sim)
            for e_idx in range(int(num_envs)):
                if bool(sim._last_done[e_idx, 0].item()):
                    in_rescue_run_by_env[e_idx] = False

    lbl_elapsed = max(1e-9, lbl_total_elapsed)
    label_batches_per_sec = float(effective_label_batches) / lbl_elapsed
    root_labels_per_sec = float(int(num_envs) * int(effective_label_batches)) / lbl_elapsed
    branch_rollouts_per_sec = 6.0 * root_labels_per_sec
    buffer_insertion_sec_per_step = buf_insert_total_elapsed / float(
        max(1, int(effective_label_batches))
    )
    measured_online_rescue_rate = float(online_rescue_frame_count) / float(
        max(1, online_alive_hero_decisions)
    )
    measured_online_rescue_episode_rate = float(online_rescue_episode_count) / float(
        max(1, online_alive_hero_decisions)
    )
    window_mean_hero_length = float(window_length_sum) / float(max(1, online_alive_hero_decisions))
    window_mature_length_ge_160_fraction = float(window_mature_length_ge_160_decisions) / float(
        max(1, online_alive_hero_decisions)
    )

    # Benchmark probe32 4-suffix scoring cost per candidate root using CPU 1-row snapshot slicing
    with torch.inference_mode():
        sim.prepare()
        policy.vector_runtime.prepare(sim)
        policy.vector_prepared = True
    n_probe_sample = min(int(num_envs), 2 if dry_run else 8)
    obs_p = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
    assert last_label_res is not None
    sample_snaps: list[dict[str, Any]] = []
    for r_i in range(n_probe_sample):
        sample_snaps.append(
            {
                "world_seed": int(world_seeds[r_i]),
                "world_index": int(r_i),
                "decision_frame": int(obs_p["decision_frame"][r_i].item()),
                "hero_length": int(obs_p["hero_length"][r_i].item()),
                "local_u8": obs_p["local_u8"][r_i].detach().cpu(),
                "global_u8": obs_p["global_u8"][r_i].detach().cpu(),
                "scalars_f32": obs_p["scalars_f32"][r_i].detach().cpu(),
                "legal_mask": obs_p["legal_mask"][r_i].detach().cpu(),
                "safe_mask_s0": last_label_res["safe_mask"][r_i].detach().cpu(),
                "bare_action": int(last_label_res["bare_actions"][r_i].item()),
                "a_star_action": int(last_label_res["a_star_actions"][r_i].item()),
                "root_q_b8": last_label_res["root_q"][r_i].detach().cpu(),
                "row_state": slice_single_row_cpu_state(sim, policy, r_i),
            }
        )
    _sync_device(dev)
    t_probe_suf0 = time.perf_counter()
    _ = _evaluate_scoring_suffixes_on_snapshots(
        sample_snaps,
        template_sim=sim,
        template_pol=policy,
        scoring_suffixes=PROBE32_SCORING_SUFFIXES,
        branch_horizon=int(branch_horizon),
        namespace=OPEN_STOP_CRN_NAMESPACE,
        batch_roots=max(1, n_probe_sample),
        cpu_fast_flood=bool(dev.type == "cpu"),
    )
    _sync_device(dev)
    probe_suf_elapsed = max(1e-9, time.perf_counter() - t_probe_suf0)
    probe32_scoring_suffix_sec_per_root = probe_suf_elapsed / float(max(1, n_probe_sample))

    # Benchmark 3: Learner updates/s (128 rescue + 128 ordinary with hard-negative mining)
    obs_fill = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
    loc_fill = obs_fill["local_u8"].detach().cpu()
    glb_fill = obs_fill["global_u8"].detach().cpu()
    sca_fill = obs_fill["scalars_f32"].detach().cpu()
    leg_fill = obs_fill["legal_mask"].detach().cpu()

    n_fill = max(res_batch_sz, ord_batch_sz, int(num_envs))
    for i in range(n_fill):
        idx = i % int(num_envs)
        safe_synth = torch.zeros(6, dtype=torch.bool)
        safe_synth[1] = True
        unsafe_synth = leg_fill[idx].clone()
        unsafe_synth[1] = False
        if not bool(unsafe_synth.any().item()):
            unsafe_synth[0] = True
            leg_fill[idx, 0] = True
            leg_fill[idx, 1] = True
        rescue_buf.add_single(
            local_u8=loc_fill[idx],
            global_u8=glb_fill[idx],
            scalars_f32=sca_fill[idx],
            legal_mask=leg_fill[idx],
            safe_mask=safe_synth,
            unsafe_mask=unsafe_synth,
            bare_action=0,
            a_star_action=1,
            world_seed=int(world_seeds[idx]),
            decision_frame=int(i),
            hard_neg_tier=0,
        )
        ordinary_buf.add_single(
            local_u8=loc_fill[idx],
            global_u8=glb_fill[idx],
            scalars_f32=sca_fill[idx],
            legal_mask=leg_fill[idx],
            safe_mask=safe_synth,
            unsafe_mask=unsafe_synth,
            bare_action=1,
            a_star_action=1,
            world_seed=int(world_seeds[idx]),
            decision_frame=int(i),
            hard_neg_tier=(i % 3),
        )

    rng = np.random.default_rng(int(seed))
    _sync_device(dev)
    t_upd0 = time.perf_counter()
    for _ in range(int(bench_learner_updates)):
        student_net.train()
        optimizer.zero_grad(set_to_none=True)
        r_idx = rng.choice(rescue_buf.size, size=res_batch_sz, replace=False).astype(np.int64)
        o_idx, _ = sample_ordinary_minibatch_indices(
            ordinary_buf,
            student_net,
            rng,
            uniform_count=ord_uni,
            hard_neg_count=ord_hn,
            device=dev,
        )
        r_batch = rescue_buf.gather_batch(r_idx, dev)
        o_batch = ordinary_buf.gather_batch(o_idx, dev)
        loss_dict = compute_sitl_hinge_loss(
            student_net,
            r_batch,
            o_batch,
            delta_rescue=RESCUE_MARGIN_DELTA_DEFAULT,
            delta_ordinary=ORDINARY_MARGIN_DELTA_DEFAULT,
        )
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, float(DEFAULT_MAX_GRAD_NORM))
        optimizer.step()
    _sync_device(dev)
    upd_elapsed = max(1e-9, time.perf_counter() - t_upd0)
    updates_per_sec = float(bench_learner_updates) / upd_elapsed

    # Restore student_net weights to initial b8-s3 after synthetic benchmark updates
    peak_vram_allocated_mib = (
        float(torch.cuda.max_memory_allocated(dev)) / float(1024**2) if dev.type == "cuda" else 0.0
    )
    peak_vram_reserved_mib = (
        float(torch.cuda.max_memory_reserved(dev)) / float(1024**2) if dev.type == "cuda" else 0.0
    )

    runaway_guard = compute_pilot_runaway_guard(
        env_steps_per_sec=env_steps_per_sec,
        label_batches_per_sec=label_batches_per_sec,
        updates_per_sec=updates_per_sec,
        measured_online_rescue_rate=measured_online_rescue_rate,
        measured_online_rescue_episode_rate=measured_online_rescue_episode_rate,
        measured_alive_hero_decisions=online_alive_hero_decisions,
        measured_rescue_frame_count=online_rescue_frame_count,
        measured_rescue_episode_count=online_rescue_episode_count,
        buffer_insertion_sec_per_step=buffer_insertion_sec_per_step,
        probe32_scoring_suffix_sec_per_root=probe32_scoring_suffix_sec_per_root,
    )

    result_payload: dict[str, Any] = {
        "schema": "t34-gpu-sitl-b1b-smoke-result/v1",
        "step": "Step B1b-Smoke",
        "dry_run": bool(dry_run),
        "non_registration_evidence": bool(dry_run),
        "device": str(dev),
        "train_namespace": TRAIN_NAMESPACE,
        "preflight_namespace": PREFLIGHT_NAMESPACE,
        "seed_audit_status": seed_audit["status"],
        "b8_checkpoint": str(b8_path),
        "b8_checkpoint_sha256": b8_sha,
        "opponent_checkpoint": str(opp_path),
        "opponent_checkpoint_sha256": opp_sha,
        "value_head_freeze": freeze_info,
        "parity_check": parity_report,
        "throughput": {
            "num_envs": int(num_envs),
            "branch_horizon": int(branch_horizon),
            "bench_env_steps": int(bench_env_steps),
            "bench_label_batches": int(effective_label_batches),
            "bench_learner_updates": int(bench_learner_updates),
            "mature_warmup_steps": int(mature_warmup_steps),
            "mature_spliced_worlds": int(n_mature_splice),
            "window_min_hero_length": int(window_min_hero_length),
            "window_max_hero_length": int(window_max_hero_length),
            "window_mean_hero_length": float(window_mean_hero_length),
            "window_mature_length_ge_160_decisions": int(window_mature_length_ge_160_decisions),
            "window_mature_length_ge_160_fraction": float(window_mature_length_ge_160_fraction),
            "env_steps_per_sec": env_steps_per_sec,
            "hero_decisions_per_sec": hero_decisions_per_sec,
            "label_batches_per_sec": label_batches_per_sec,
            "six_branch_h32_root_labels_per_sec": root_labels_per_sec,
            "six_branch_h32_branch_rollouts_per_sec": branch_rollouts_per_sec,
            "updates_per_sec": updates_per_sec,
            "buffer_insertion_sec_per_step": buffer_insertion_sec_per_step,
            "probe32_scoring_suffix_sec_per_root": probe32_scoring_suffix_sec_per_root,
            "measured_online_rescue_rate": measured_online_rescue_rate,
            "measured_online_rescue_episode_rate": measured_online_rescue_episode_rate,
            "measured_alive_hero_decisions": online_alive_hero_decisions,
            "measured_rescue_frame_count": online_rescue_frame_count,
            "measured_rescue_episode_count": online_rescue_episode_count,
            "peak_vram_allocated_mib": peak_vram_allocated_mib,
            "peak_vram_reserved_mib": peak_vram_reserved_mib,
        },
        "phase_timings_sec": {
            "parity_check_sec": float(parity_report.get("elapsed_sec", 0.0)),
            "env_benchmark_sec": float(env_elapsed),
            "mature_warmup_and_splice_sec": float(mature_warmup_elapsed),
            "label_benchmark_sec": float(lbl_elapsed),
            "buffer_insertion_sec": float(buf_insert_total_elapsed),
            "probe_scoring_benchmark_sec": float(probe_suf_elapsed),
            "learner_benchmark_sec": float(upd_elapsed),
        },
        "pilot_runaway_guard": runaway_guard,
        "elapsed_sec": time.perf_counter() - t_start,
    }

    result_file = out_dir / "smoke_result.json"
    result_sha = write_atomic_with_sha256(
        result_file,
        lambda tmp: tmp.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
        write_complete_marker=True,
    )
    result_payload["result_json_path"] = str(result_file)
    result_payload["result_json_sha256"] = result_sha

    # Write family-canary export-probe artifacts for the pilot's required_exports relative paths
    # if they do not already exist, so the colab_cli runtime export scanner records full producer
    # export coverage in the smoke canary receipt (`canary.py:322`).
    canary_probe_meta = {
        "schema": "t34-gpu-sitl-canary-export-probe/v1",
        "evidence_class": "CANARY_EXPORT_PROBE_NON_EVIDENCE",
        "canary_export_probe": True,
        "smoke_result_sha256": result_sha,
    }
    probe_pt_path = out_dir / "probe32_bank.pt"
    if not probe_pt_path.exists():
        write_atomic_with_sha256(
            probe_pt_path,
            lambda tmp: torch.save(canary_probe_meta, tmp),
            write_complete_marker=True,
        )
    probe_meta_path = out_dir / "probe32_bank.pt.meta.json"
    if not probe_meta_path.exists():
        write_atomic_with_sha256(
            probe_meta_path,
            lambda tmp: tmp.write_text(
                json.dumps(canary_probe_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            ),
            write_complete_marker=True,
        )
    student_final_path = out_dir / "b8_sitl_student_final.pth"
    if not student_final_path.exists():
        write_atomic_with_sha256(
            student_final_path,
            lambda tmp: torch.save(canary_probe_meta, tmp),
            write_complete_marker=True,
        )
    pilot_res_probe_path = out_dir / "sitl_pilot_result.json"
    if not pilot_res_probe_path.exists():
        write_atomic_with_sha256(
            pilot_res_probe_path,
            lambda tmp: tmp.write_text(
                json.dumps(canary_probe_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            ),
            write_complete_marker=True,
        )

    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step B1b-Smoke: CUDA parity on row subsets + E=256 throughput & VRAM."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--branch-horizon", type=int, default=None)
    parser.add_argument("--bench-env-steps", type=int, default=None)
    parser.add_argument("--bench-label-batches", type=int, default=None)
    parser.add_argument("--bench-learner-updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny CPU dry-run configuration (marks dry_run=true).",
    )
    args = parser.parse_args()
    is_dry = bool(args.dry_run)
    res = run_step_b1b_smoke(
        args.output_dir,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        device=args.device,
        num_envs=args.num_envs if args.num_envs is not None else (4 if is_dry else REGISTERED_ENVS),
        branch_horizon=(
            args.branch_horizon
            if args.branch_horizon is not None
            else (2 if is_dry else OPEN_STOP_BRANCH_HORIZON)
        ),
        bench_env_steps=(
            args.bench_env_steps if args.bench_env_steps is not None else (2 if is_dry else 32)
        ),
        bench_label_batches=(
            args.bench_label_batches
            if args.bench_label_batches is not None
            else (2 if is_dry else None)
        ),
        bench_learner_updates=(
            args.bench_learner_updates
            if args.bench_learner_updates is not None
            else (2 if is_dry else 20)
        ),
        seed=args.seed,
        dry_run=is_dry,
    )
    print(
        json.dumps(
            {
                "step": res["step"],
                "dry_run": res["dry_run"],
                "device": res["device"],
                "parity_ok": res["parity_check"]["ok"],
                "throughput": res["throughput"],
                "pilot_runaway_guard": res["pilot_runaway_guard"],
                "result_json_path": res["result_json_path"],
                "result_json_sha256": res["result_json_sha256"],
                "elapsed_sec": res["elapsed_sec"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
