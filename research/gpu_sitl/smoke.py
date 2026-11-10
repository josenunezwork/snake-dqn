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
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from research.gpu_sim_spike import ego_device, r2_cache
from research.gpu_sim_spike.v8_device import v8_decide as gpu_v8_decide
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from research.gpu_sitl.open_stop_gpu import (
    DEFAULT_CUDA_ARCH_LIST,
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_V8_LAMBDA,
    REQUIRED_CUDA_ARCHS,
    FastDeviceVectorRuntime,
    _maybe_cpu_fast_flood,
    _ttl_by_owner,
    batched_open_stop_h32,
    ensure_cuda_initialized,
    fast_v8_decide,
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
    _ENV_TENSOR_FIELDS,
    _VECTOR_RUNTIME_ENV_FIELDS,
    PROBE32_HORIZON,
    PROBE32_SCORING_SUFFIXES,
    PROBE32_WORLD_COUNT,
    _evaluate_scoring_suffixes_on_snapshots,
    build_watch6_ct_device_bundle,
    extract_hero_root_observations,
    harvest_probe32_bank,
    load_probe32_bank,
    sha256_file,
    slice_single_row_cpu_state,
    write_atomic_with_sha256,
)
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    PROBE32_NAMESPACE,
    SCREEN_NAMESPACE,
    SELECTIVITY_NAMESPACE,
    SHAKEOUT_PROBE_NAMESPACE,
    SHAKEOUT_SMOKE_NAMESPACE,
    SHAKEOUT_TRAIN_NAMESPACE,
    TRAIN_NAMESPACE,
    assert_allowed_training_seeds,
    audit_b1b_seeds,
    audit_shakeout_seeds,
    shakeout_smoke_seeds,
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
    run_sitl_training,
    sample_ordinary_minibatch_indices,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint
from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

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

# Target ceiling for projected A100 Step B1b-Smoke wall time (leaves >= 380 s headroom inside
# the ~1,080 s remaining after ~200 s Colab session setup under the 22.5 min = 1,350 s timeout).
SMOKE_A100_PROJECTION_LIMIT_SEC = 700.0

# ESTIMATED L4 -> A100 wall-time multipliers (marked explicitly as estimates, never exact truth):
# - Host/CPU phases (`init_and_extensions`, `build_mature_extension_worlds`, `write_outputs`):
#   1.00 (ESTIMATE) — Colab L4 and A100 runtimes both expose 12 Intel Xeon vCPUs at comparable
#   single-core clock/IPC (`AGENTS.md` "Right tool first").
# - GPU-resident phases (`CTDeviceSim`, `flood_cuda.static_capped`, `vector_geometry`, `Ego2sNet`):
#   0.55 (ESTIMATE, ~1.82x speedup on A100 vs L4) — A100-SXM4-40GB provides 1,555 GB/s HBM2e vs
#   L4 300 GB/s GDDR6 (5.18x bandwidth; 0.19x time if purely bandwidth-bound) and 108 vs 58 SMs
#   (1.86x SM count; 0.54x time on per-(row,dir) flood/vector blocks), tempered by non-TF32 FP32
#   ALU parity and fixed host Python/launch floor.
# - Mixed CPU+GPU `parity_4_open_stop_h32`: 0.85 (ESTIMATE: ~70% single-vCPU `cert_ref` at 1.00 +
#   ~30% GPU `cert_cand` at ~0.50).
ESTIMATED_L4_TO_A100_CPU_FACTOR = 1.00
ESTIMATED_L4_TO_A100_GPU_FACTOR = 0.55
ESTIMATED_L4_TO_A100_PARITY4_FACTOR = 0.85
ESTIMATED_H100_TO_A100_GPU_FACTOR = 1.35


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
        ensure_cuda_initialized(dev)
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _log_smoke_phase(
    event: str,
    phase: str,
    t_process_start: float,
    *,
    phase_start: float | None = None,
    detail: str = "",
) -> float:
    now = time.perf_counter()
    elapsed = now - t_process_start
    msg = f"[gpu_sitl.smoke] {event} phase={phase} elapsed_s={elapsed:.2f}"
    if phase_start is not None:
        msg += f" phase_s={now - phase_start:.2f}"
    if detail:
        msg += f" {detail}"
    print(msg, flush=True)
    return now


@torch.inference_mode()
def run_mature_cohort_row_subset_parity(
    b8_path: Path | str,
    opponent_path: Path | str,
    *,
    device: str | torch.device = "cuda",
    slice_indices: Sequence[int] = DEFAULT_MATURE_SLICE_INDICES,
    branch_horizon: int = OPEN_STOP_BRANCH_HORIZON,
    dry_run: bool = False,
    pre_built_extension_worlds: tuple[list[Any], Any, list[dict[str, Any]]] | None = None,
    t_process_start: float | None = None,
    on_phase_complete: Callable[[str, float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compare device native kernels against pure-Torch CPU paths on mature cohort row subsets.

    Fails closed (raises :class:`RuntimeError`) on any boolean or action mismatch.
    """
    t0 = time.perf_counter()
    proc_t0 = t0 if t_process_start is None else float(t_process_start)
    dev = ensure_cuda_initialized(device)
    if not dry_run and dev.type != "cuda":
        raise RuntimeError(f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}")
    native_info = verify_cuda_native_extensions(dev)

    selected_specs = [MATURE_EXTENSION_SPECS[int(i)] for i in slice_indices]
    n_roots = len(selected_specs)
    if n_roots < 2:
        raise ValueError("slice_indices must select at least 2 mature cohort roots")

    if pre_built_extension_worlds is not None:
        worlds, pol_tpl, build_meta = pre_built_extension_worlds
        if len(worlds) != n_roots or len(build_meta) != n_roots:
            raise ValueError(
                f"pre_built_extension_worlds length ({len(worlds)}, {len(build_meta)}) "
                f"must match slice_indices length ({n_roots})"
            )
    else:
        t_ext = _log_smoke_phase("PHASE_START", "parity_build_extension_worlds", proc_t0)
        worlds, pol_tpl, build_meta = build_extension_worlds(
            Path(b8_path),
            Path(opponent_path),
            specs=selected_specs,
            cpu_featurizer_backend="python",
            ego_raster_backend="numpy",
        )
        _log_smoke_phase(
            "PHASE_END",
            "parity_build_extension_worlds",
            proc_t0,
            phase_start=t_ext,
            detail=f"roots={n_roots}",
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
    t_p1 = _log_smoke_phase("PHASE_START", "parity_1_v8_flood", proc_t0)
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
        v8_fast_cand = fast_v8_decide(
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
    fast_v8_feat_max_abs = float(
        (v8_ref.feature_planes() - v8_fast_cand.feature_planes().cpu()).abs().max().item()
    )
    fast_v8_parity_ok = bool(
        torch.equal(v8_ref.action, v8_fast_cand.action.cpu())
        and torch.equal(v8_ref.outcome, v8_fast_cand.outcome.cpu())
        and torch.equal(v8_ref.counts, v8_fast_cand.counts.cpu())
        and torch.equal(v8_ref.area, v8_fast_cand.area.cpu())
        and torch.equal(v8_ref.cap, v8_fast_cand.cap.cpu())
        and torch.equal(v8_ref.landing, v8_fast_cand.landing.cpu())
        and torch.equal(v8_ref.spacious, v8_fast_cand.spacious.cpu())
        and torch.equal(v8_ref.landing_ok, v8_fast_cand.landing_ok.cpu())
        and torch.equal(v8_ref.head_risky, v8_fast_cand.head_risky.cpu())
        and fast_v8_feat_max_abs <= 1e-6
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
        and fast_v8_parity_ok
    )
    if not v8_parity_ok:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in flood_cuda.flood / v8_decide on row subsets: "
            f"action={v8_action_match}, outcome={v8_outcome_match}, counts={v8_counts_match}, "
            f"area={v8_area_match}, cap={v8_cap_match}, landing={v8_landing_match}, "
            f"spacious={v8_spacious_match}, landing_ok={v8_landing_ok_match}, "
            f"head_risky={v8_head_risky_match}, feat_max_abs={v8_feat_max_abs}, "
            f"fast_v8_parity_ok={fast_v8_parity_ok} (fast_feat_max_abs={fast_v8_feat_max_abs})"
        )
    t_p1_end = _log_smoke_phase("PHASE_END", "parity_1_v8_flood", proc_t0, phase_start=t_p1)
    p1_elapsed = max(0.0, t_p1_end - t_p1)
    if on_phase_complete is not None:
        on_phase_complete(
            "parity_1_v8_flood",
            p1_elapsed,
            {"ok": v8_parity_ok, "fast_v8_decide_exact_match": fast_v8_parity_ok},
        )

    # 2. Parity Check: flood_cuda.components via ego_device._component_sizes on row subsets
    t_p2 = _log_smoke_phase("PHASE_START", "parity_2_components", proc_t0)
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
    t_p2_end = _log_smoke_phase("PHASE_END", "parity_2_components", proc_t0, phase_start=t_p2)
    p2_elapsed = max(0.0, t_p2_end - t_p2)
    if on_phase_complete is not None:
        on_phase_complete("parity_2_components", p2_elapsed, {"ok": comp_match})

    # 3. Parity Check: vector_geometry.exact_features_rows and FastDeviceVectorRuntime row subsets
    t_p3 = _log_smoke_phase("PHASE_START", "parity_3_vector_runtime", proc_t0)
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
    t_p3_end = _log_smoke_phase("PHASE_END", "parity_3_vector_runtime", proc_t0, phase_start=t_p3)
    p3_elapsed = max(0.0, t_p3_end - t_p3)
    if on_phase_complete is not None:
        on_phase_complete(
            "parity_3_vector_runtime",
            p3_elapsed,
            {
                "ok": vec_parity_ok,
                "features_rows_max_abs_diff": vec_max_abs,
                "partial_prepare_sel_state_max_abs_diff": prep_state_max_abs,
            },
        )

    # 4. Parity Check: batched_open_stop_h32 boolean certificates on the mature cohort slice.
    # Subphases 1-3 above already verify flood_cuda.flood, flood_cuda.components, and
    # vector_geometry directly against pure-Torch cpu_fast_flood=False; here cpu_fast_flood=True
    # produces bit-identical CPU integer areas/components while avoiding ~355 s of single-vCPU
    # 160-step Python/Torch 2D padding wavefront loops across 48 x 32 steps.
    t_p4 = _log_smoke_phase("PHASE_START", "parity_4_open_stop_h32", proc_t0)
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
        cpu_fast_flood=True,
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
    t_p4_end = _log_smoke_phase("PHASE_END", "parity_4_open_stop_h32", proc_t0, phase_start=t_p4)
    p4_elapsed = max(0.0, t_p4_end - t_p4)
    n_rescue_in_slice = int(cert_ref["rescue_mask"].sum().item())
    n_ordinary_in_slice = int(cert_ref["ordinary_mask"].sum().item())
    if on_phase_complete is not None:
        on_phase_complete(
            "parity_4_open_stop_h32",
            p4_elapsed,
            {
                "ok": open_stop_parity_ok,
                "num_rescue_roots_in_slice": n_rescue_in_slice,
                "num_ordinary_roots_in_slice": n_ordinary_in_slice,
            },
        )

    return {
        "ok": True,
        "device": str(dev),
        "dry_run": bool(dry_run),
        "native_extensions": native_info,
        "declared_torch_cuda_arch_list": native_info.get("torch_cuda_arch_list"),
        "effective_build_cuda_arch": native_info.get("effective_build_cuda_arch"),
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
            "fast_v8_decide_exact_match": fast_v8_parity_ok,
            "fast_v8_feature_planes_max_abs_diff": fast_v8_feat_max_abs,
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
            "num_rescue_roots_in_slice": n_rescue_in_slice,
            "num_ordinary_roots_in_slice": n_ordinary_in_slice,
        },
        "subphase_timings_sec": {
            "parity_1_v8_flood_sec": float(p1_elapsed),
            "parity_2_components_sec": float(p2_elapsed),
            "parity_3_vector_runtime_sec": float(p3_elapsed),
            "parity_4_open_stop_h32_sec": float(p4_elapsed),
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

    if measured_online_rescue_episode_rate is not None and int(measured_rescue_episode_count) > 0:
        raw_ep_rate = float(measured_online_rescue_episode_rate)
        if int(measured_rescue_episode_count) >= MIN_OBSERVED_RESCUES_FOR_UNCLAMPED_RATE:
            effective_episode_rate = raw_ep_rate
        else:
            effective_episode_rate = min(float(PRIOR_FALLBACK_UPPER_BOUND), raw_ep_rate)
    elif measured_online_rescue_episode_rate is not None and int(k_rescues) > 0:
        effective_episode_rate = min(float(PRIOR_FALLBACK_LOWER_BOUND), effective_rescue_rate)
    else:
        effective_episode_rate = effective_rescue_rate

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
    mature_warmup_steps_override: int | None = None,
    probe_sample_roots_override: int | None = None,
    seed: int = 20261109,
    dry_run: bool = False,
    shakeout: bool = False,
) -> dict[str, Any]:
    """Run Step B1b-Smoke (`short-canary <= 25 min`, cap `3.0 A100 u`) and write JSON + SHA256."""
    t_start = time.perf_counter()
    t_init = _log_smoke_phase("PHASE_START", "init_and_extensions", t_start)
    is_shakeout = bool(shakeout)
    is_non_prod = bool(dry_run or is_shakeout)
    dev = ensure_cuda_initialized(device)
    effective_label_batches = (
        int(bench_label_batches)
        if bench_label_batches is not None
        else (8 if is_non_prod else int(DEFAULT_CUDA_BENCH_LABEL_BATCHES))
    )
    if not is_non_prod:
        if dev.type != "cuda":
            raise RuntimeError(f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}")
        if int(num_envs) != REGISTERED_ENVS or int(branch_horizon) != OPEN_STOP_BRANCH_HORIZON:
            raise ValueError(
                f"Non-dry-run Step B1b-Smoke requires E={REGISTERED_ENVS} and "
                f"H={OPEN_STOP_BRANCH_HORIZON}, got E={num_envs}, H={branch_horizon}"
            )
        if mature_warmup_steps_override is not None and int(mature_warmup_steps_override) != int(
            DEFAULT_CUDA_MATURE_WARMUP_STEPS
        ):
            raise ValueError(
                f"Non-dry-run Step B1b-Smoke requires mature_warmup_steps="
                f"{DEFAULT_CUDA_MATURE_WARMUP_STEPS}, got {mature_warmup_steps_override}"
            )
        if probe_sample_roots_override is not None and int(probe_sample_roots_override) != 64:
            raise ValueError(
                f"Non-dry-run Step B1b-Smoke requires probe_sample_roots=64, "
                f"got {probe_sample_roots_override}"
            )

    if is_shakeout:
        seed_audit = audit_shakeout_seeds(
            smoke_envs=int(num_envs),
            probe_worlds=2,
            train_envs=int(num_envs),
        )
        if not bool(seed_audit["ok"]):
            raise RuntimeError(f"Step B1b shake-out seed audit failed: {seed_audit}")
    else:
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
    if not is_non_prod:
        if b8_sha != B8_S3_SHA256:
            raise ValueError(f"Canonical b8-s3 SHA256 mismatch at {b8_path}: {b8_sha}")
        if opp_sha != FRP3_S12_SHA256:
            raise ValueError(f"Canonical frp3-s12 SHA256 mismatch at {opp_path}: {opp_sha}")

    out_dir = Path(output_dir)
    result_file = out_dir / (
        "b1b_smoke_shakeout_subresult.json" if is_shakeout else "smoke_result.json"
    )
    if Path(f"{result_file}.complete").is_file():
        raise FileExistsError(f"Refusing to overwrite completed smoke output at {result_file}")
    out_dir.mkdir(parents=True, exist_ok=True)
    active_train_namespace = SHAKEOUT_SMOKE_NAMESPACE if is_shakeout else TRAIN_NAMESPACE
    completed_phases_log: list[dict[str, Any]] = []
    phase_seq = 0

    def _record_phase_partial(
        phase_name: str,
        phase_elapsed_sec: float,
        summary: dict[str, Any] | None = None,
    ) -> None:
        nonlocal phase_seq
        phase_seq += 1
        elapsed_total = max(0.0, time.perf_counter() - t_start)
        entry = {
            "phase_index": int(phase_seq),
            "phase": str(phase_name),
            "phase_elapsed_sec": float(phase_elapsed_sec),
            "elapsed_since_start_sec": float(elapsed_total),
            "summary": dict(summary or {}),
        }
        completed_phases_log.append(entry)
        if is_shakeout:
            return
        partial_path = out_dir / f"smoke_partial_{phase_seq:02d}_{phase_name}.json"
        if partial_path.exists() or Path(f"{partial_path}.complete").exists():
            raise FileExistsError(
                f"Refusing to rewrite completed partial smoke file at {partial_path}"
            )
        partial_payload: dict[str, Any] = {
            "schema": "t34-gpu-sitl-b1b-smoke-partial/v1",
            "step": "Step B1b-Smoke",
            "status": "PARTIAL_IN_PROGRESS",
            "partial": True,
            "ok": False,
            "pass": False,
            "dry_run": bool(is_non_prod),
            "not_evidence": True,
            "non_registration_evidence": True,
            "evidence_class": "PARTIAL_SMOKE_DIAGNOSTIC_NOT_PASS",
            "device": str(dev),
            "train_namespace": active_train_namespace,
            "preflight_namespace": PREFLIGHT_NAMESPACE,
            "b8_checkpoint_sha256": b8_sha,
            "opponent_checkpoint_sha256": opp_sha,
            "phase_index": int(phase_seq),
            "completed_phase": str(phase_name),
            "phase_elapsed_sec": float(phase_elapsed_sec),
            "elapsed_since_start_sec": float(elapsed_total),
            "phase_summary": dict(summary or {}),
            "completed_phases": list(completed_phases_log),
        }
        write_atomic_with_sha256(
            partial_path,
            lambda tmp, p=partial_payload: tmp.write_text(
                json.dumps(p, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            ),
            write_complete_marker=True,
        )

    if dev.type == "cuda":
        ensure_cuda_initialized(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    native_extensions = verify_cuda_native_extensions(dev)
    init_elapsed = max(0.0, time.perf_counter() - t_init)
    _log_smoke_phase(
        "PHASE_END",
        "init_and_extensions",
        t_start,
        phase_start=t_init,
        detail=(
            f"device={dev} declared_arch={native_extensions.get('torch_cuda_arch_list')} "
            f"effective_arch={native_extensions.get('effective_build_cuda_arch')}"
        ),
    )
    _record_phase_partial(
        "init_and_extensions",
        init_elapsed,
        {
            "device": str(dev),
            "declared_torch_cuda_arch_list": native_extensions.get("torch_cuda_arch_list"),
            "effective_build_cuda_arch": native_extensions.get("effective_build_cuda_arch"),
        },
    )

    # Build the mature extension worlds once and reuse across parity check and mature splice.
    t_ext_build = _log_smoke_phase("PHASE_START", "build_mature_extension_worlds", t_start)
    parity_specs = [MATURE_EXTENSION_SPECS[int(i)] for i in parity_slice_indices]
    if len(parity_specs) < 2:
        raise ValueError("parity_slice_indices must select at least 2 mature cohort roots")
    ext_worlds_cpu, ext_pol_tpl, ext_build_meta = build_extension_worlds(
        b8_path,
        opp_path,
        specs=parity_specs,
        cpu_featurizer_backend="python",
        ego_raster_backend="numpy",
    )
    ext_build_elapsed = max(0.0, time.perf_counter() - t_ext_build)
    _log_smoke_phase(
        "PHASE_END",
        "build_mature_extension_worlds",
        t_start,
        phase_start=t_ext_build,
        detail=f"roots={len(ext_worlds_cpu)}",
    )
    _record_phase_partial(
        "build_mature_extension_worlds",
        ext_build_elapsed,
        {"roots": len(ext_worlds_cpu), "slice_indices": [int(i) for i in parity_slice_indices]},
    )

    # (a) CUDA-vs-CPU parity check on row subsets against pure-Torch paths
    t_par = _log_smoke_phase("PHASE_START", "parity_check", t_start)
    parity_report = run_mature_cohort_row_subset_parity(
        b8_path,
        opp_path,
        device=dev,
        slice_indices=parity_slice_indices,
        branch_horizon=int(branch_horizon),
        dry_run=is_non_prod,
        pre_built_extension_worlds=(ext_worlds_cpu, ext_pol_tpl, ext_build_meta),
        t_process_start=t_start,
        on_phase_complete=_record_phase_partial,
    )
    parity_total_elapsed = max(0.0, time.perf_counter() - t_par)
    _log_smoke_phase(
        "PHASE_END",
        "parity_check",
        t_start,
        phase_start=t_par,
        detail=f"ok={parity_report.get('ok')}",
    )
    _record_phase_partial(
        "parity_check",
        parity_total_elapsed,
        {
            "ok": bool(parity_report.get("ok")),
            "subphase_timings_sec": dict(parity_report.get("subphase_timings_sec") or {}),
        },
    )
    if not is_non_prod and not bool(parity_report.get("ok")):
        raise RuntimeError(f"Step B1b-Smoke CUDA-vs-CPU parity check failed: {parity_report}")

    # (b) Throughput and peak VRAM benchmark at E = num_envs (256 on A100; dev seeds on shakeout)
    t_bundle = _log_smoke_phase("PHASE_START", "bundle_and_bench_warmup", t_start)
    world_seeds = shakeout_smoke_seeds(int(num_envs)) if is_shakeout else train_seeds(int(num_envs))
    assert_allowed_training_seeds(
        namespace=active_train_namespace,
        seeds=world_seeds,
        context="run_step_b1b_smoke.world_seeds",
    )
    sim, policy = build_watch6_ct_device_bundle(
        world_seeds,
        b8_path,
        opp_path,
        device=dev,
        hero_respawn=True,
        policy_template=ext_pol_tpl,
        cpu_featurizer_backend="python",
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
    bundle_warmup_elapsed = max(0.0, time.perf_counter() - t_bundle)
    _log_smoke_phase(
        "PHASE_END",
        "bundle_and_bench_warmup",
        t_start,
        phase_start=t_bundle,
        detail=f"envs={num_envs}",
    )
    _record_phase_partial(
        "bundle_and_bench_warmup",
        bundle_warmup_elapsed,
        {"num_envs": int(num_envs), "branch_horizon": int(branch_horizon)},
    )

    # Advance bare b8-s3 on train_seeds(num_envs) into mature mid-game traffic (600 frames on
    # CUDA) and splice mature cohort snapshots (L in [210, 823], L >= 160) BEFORE bench_1_env_steps
    # so both env and label throughput benchmarks measure mature mid-game traffic, while online
    # rescue-rate statistics exclude spliced PREFLIGHT_NAMESPACE rows.
    t_warm0 = _log_smoke_phase("PHASE_START", "mature_warmup_and_splice", t_start)
    mature_warmup_steps = (
        int(mature_warmup_steps_override)
        if mature_warmup_steps_override is not None
        else (0 if dry_run or dev.type != "cuda" else int(DEFAULT_CUDA_MATURE_WARMUP_STEPS))
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
    splice_sim: CTDeviceSim | None = None
    splice_pol: Any = None
    label_world_seeds = [int(s) for s in world_seeds]
    if n_mature_splice > 0:
        splice_worlds = ext_worlds_cpu[:n_mature_splice]
        for s_i in range(n_mature_splice):
            label_world_seeds[s_i] = int(splice_worlds[s_i].seed)
        splice_sim, splice_pol = snapshot_worlds_to_device(
            splice_worlds,
            ext_pol_tpl,
            sim_cls=CTDeviceSim,
            device=dev,
            vector_prepared=False,
        )

    def _apply_mature_splice() -> None:
        if n_mature_splice <= 0 or splice_sim is None or splice_pol is None:
            return
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
            sim.hero_respawn = True
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

    _apply_mature_splice()
    _sync_device(dev)
    mature_warmup_elapsed = max(0.0, time.perf_counter() - t_warm0)
    _log_smoke_phase(
        "PHASE_END",
        "mature_warmup_and_splice",
        t_start,
        phase_start=t_warm0,
        detail=f"warmup_steps={mature_warmup_steps} spliced={n_mature_splice}",
    )
    _record_phase_partial(
        "mature_warmup_and_splice",
        mature_warmup_elapsed,
        {
            "mature_warmup_steps": int(mature_warmup_steps),
            "mature_spliced_worlds": int(n_mature_splice),
        },
    )

    # Benchmark 1: env steps/s at E = num_envs on mature mid-game + spliced L >= 160 traffic
    _log_smoke_phase("PHASE_START", "bench_1_env_steps", t_start, detail=f"steps={bench_env_steps}")
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
    _log_smoke_phase(
        "PHASE_END",
        "bench_1_env_steps",
        t_start,
        phase_start=t_env0,
        detail=f"env_steps_per_sec={env_steps_per_sec:.2f}",
    )
    _record_phase_partial(
        "bench_1_env_steps",
        env_elapsed,
        {
            "bench_env_steps": int(bench_env_steps),
            "env_steps_per_sec": float(env_steps_per_sec),
            "hero_decisions_per_sec": float(hero_decisions_per_sec),
        },
    )

    # Re-apply mature splice before Benchmark 2 so label batches also start with alive L >= 160 rows
    _apply_mature_splice()
    _sync_device(dev)

    # Benchmark 2: 6-branch H=branch_horizon labels/s AND online rescue-root rate on
    # train_seeds(num_envs) under bare b8-s3, plus per-step buffer insertion cost.
    t_b2_phase = _log_smoke_phase(
        "PHASE_START",
        "bench_2_label_batches_and_rescue_rate",
        t_start,
        detail=f"batches={effective_label_batches}",
    )
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
    window_alive_decisions = 0
    window_mature_length_ge_160_decisions = 0
    window_length_sum = 0
    window_min_hero_length = 0
    window_max_hero_length = 0
    in_rescue_run_by_env = [False] * int(num_envs)
    last_label_res: dict[str, Any] | None = None
    online_env_start = n_mature_splice if n_mature_splice < int(num_envs) else 0

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
            world_seeds=label_world_seeds,
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
        # and accumulate online rescue-root and distinct-episode rescue counts strictly over
        # non-spliced active_train_namespace worlds.
        t_buf_step = time.perf_counter()
        obs_b = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
        loc_b = obs_b["local_u8"].detach().cpu()
        glb_b = obs_b["global_u8"].detach().cpu()
        sca_b = obs_b["scalars_f32"].detach().cpu()
        leg_b = obs_b["legal_mask"].detach().cpu()
        safe_b = last_label_res["safe_mask"].detach().cpu()
        unsafe_b = last_label_res["unsafe_mask"].detach().cpu()
        alive_list = obs_b["hero_alive"].detach().cpu().tolist()
        len_list = obs_b["hero_length"].detach().cpu().tolist()
        rescue_list = last_label_res["rescue_mask"].detach().cpu().tolist()
        ordinary_list = last_label_res["ordinary_mask"].detach().cpu().tolist()
        bare_list = last_label_res["bare_actions"].detach().cpu().tolist()
        astar_list = last_label_res["a_star_actions"].detach().cpu().tolist()

        for e_idx in range(int(num_envs)):
            if not alive_list[e_idx]:
                in_rescue_run_by_env[e_idx] = False
                continue
            h_len = int(len_list[e_idx])
            if window_alive_decisions == 0 or h_len < window_min_hero_length:
                window_min_hero_length = h_len
            if h_len > window_max_hero_length:
                window_max_hero_length = h_len
            window_length_sum += h_len
            if h_len >= 160:
                window_mature_length_ge_160_decisions += 1
            window_alive_decisions += 1
            if e_idx < online_env_start:
                continue
            online_alive_hero_decisions += 1
            if rescue_list[e_idx]:
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
                    bare_action=int(bare_list[e_idx]),
                    a_star_action=int(astar_list[e_idx]),
                    world_seed=int(world_seeds[e_idx]),
                    decision_frame=int(decs_now[e_idx]),
                    hard_neg_tier=0,
                )
            else:
                in_rescue_run_by_env[e_idx] = False
                if ordinary_list[e_idx]:
                    ordinary_buf.add_single(
                        local_u8=loc_b[e_idx],
                        global_u8=glb_b[e_idx],
                        scalars_f32=sca_b[e_idx],
                        legal_mask=leg_b[e_idx],
                        safe_mask=safe_b[e_idx],
                        unsafe_mask=unsafe_b[e_idx],
                        bare_action=int(bare_list[e_idx]),
                        a_star_action=int(astar_list[e_idx]),
                        world_seed=int(world_seeds[e_idx]),
                        decision_frame=int(decs_now[e_idx]),
                        hard_neg_tier=0,
                    )
        buf_insert_total_elapsed += max(0.0, time.perf_counter() - t_buf_step)

        # Step bare b8-s3 trajectory forward 1 frame for next label batch; reuse
        # last_label_res["bare_actions"] already evaluated on the prepared root sim.
        with torch.inference_mode():
            bare_step_act = last_label_res["bare_actions"]
            sim.step(policy.actions(sim, bare_step_act))
            policy.after_step(sim)
            done_hero_list = sim._last_done[:, 0].detach().cpu().tolist()
            for e_idx, is_done in enumerate(done_hero_list):
                if is_done:
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
    window_mean_hero_length = float(window_length_sum) / float(max(1, window_alive_decisions))
    window_mature_length_ge_160_fraction = float(window_mature_length_ge_160_decisions) / float(
        max(1, window_alive_decisions)
    )
    b2_phase_elapsed = max(0.0, time.perf_counter() - t_b2_phase)
    _log_smoke_phase(
        "PHASE_END",
        "bench_2_label_batches_and_rescue_rate",
        t_start,
        phase_start=t_b2_phase,
        detail=(
            f"label_batches_per_sec={label_batches_per_sec:.2f} "
            f"rescues={online_rescue_frame_count}/{online_alive_hero_decisions}"
        ),
    )
    _record_phase_partial(
        "bench_2_label_batches_and_rescue_rate",
        b2_phase_elapsed,
        {
            "bench_label_batches": int(effective_label_batches),
            "label_benchmark_sec": float(lbl_elapsed),
            "buffer_insertion_sec": float(buf_insert_total_elapsed),
            "label_batches_per_sec": float(label_batches_per_sec),
            "six_branch_h32_root_labels_per_sec": float(root_labels_per_sec),
            "measured_alive_hero_decisions": int(online_alive_hero_decisions),
            "measured_rescue_frame_count": int(online_rescue_frame_count),
            "measured_rescue_episode_count": int(online_rescue_episode_count),
        },
    )

    # Benchmark probe32 4-suffix scoring cost per candidate root using CPU 1-row snapshot slicing
    t_b3_phase = _log_smoke_phase("PHASE_START", "bench_3_probe32_scoring_suffixes", t_start)
    with torch.inference_mode():
        sim.prepare()
        policy.vector_runtime.prepare(sim)
        policy.vector_prepared = True
    default_probe_sample = 1 if is_shakeout else (2 if dry_run else 64)
    n_probe_sample = min(
        int(num_envs),
        max(
            1,
            (
                int(probe_sample_roots_override)
                if probe_sample_roots_override is not None
                else default_probe_sample
            ),
        ),
    )
    obs_p = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
    assert last_label_res is not None
    p_dec_list = obs_p["decision_frame"][:n_probe_sample].detach().cpu().tolist()
    p_len_list = obs_p["hero_length"][:n_probe_sample].detach().cpu().tolist()
    p_loc_cpu = obs_p["local_u8"][:n_probe_sample].detach().cpu()
    p_glb_cpu = obs_p["global_u8"][:n_probe_sample].detach().cpu()
    p_sca_cpu = obs_p["scalars_f32"][:n_probe_sample].detach().cpu()
    p_leg_cpu = obs_p["legal_mask"][:n_probe_sample].detach().cpu()
    p_safe_cpu = last_label_res["safe_mask"][:n_probe_sample].detach().cpu()
    p_bare_list = last_label_res["bare_actions"][:n_probe_sample].detach().cpu().tolist()
    p_astar_list = last_label_res["a_star_actions"][:n_probe_sample].detach().cpu().tolist()
    p_rootq_cpu = last_label_res["root_q"][:n_probe_sample].detach().cpu()
    sample_snaps: list[dict[str, Any]] = []
    for r_i in range(n_probe_sample):
        sample_snaps.append(
            {
                "world_seed": int(label_world_seeds[r_i]),
                "world_index": int(r_i),
                "decision_frame": int(p_dec_list[r_i]),
                "hero_length": int(p_len_list[r_i]),
                "local_u8": p_loc_cpu[r_i],
                "global_u8": p_glb_cpu[r_i],
                "scalars_f32": p_sca_cpu[r_i],
                "legal_mask": p_leg_cpu[r_i],
                "safe_mask_s0": p_safe_cpu[r_i],
                "bare_action": int(p_bare_list[r_i]),
                "a_star_action": int(p_astar_list[r_i]),
                "root_q_b8": p_rootq_cpu[r_i],
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
    b3_phase_elapsed = max(0.0, time.perf_counter() - t_b3_phase)
    _log_smoke_phase(
        "PHASE_END",
        "bench_3_probe32_scoring_suffixes",
        t_start,
        phase_start=t_b3_phase,
        detail=f"sec_per_root={probe32_scoring_suffix_sec_per_root:.4f}",
    )
    _record_phase_partial(
        "bench_3_probe32_scoring_suffixes",
        b3_phase_elapsed,
        {
            "probe_sample_roots": int(n_probe_sample),
            "probe_scoring_benchmark_sec": float(probe_suf_elapsed),
            "probe32_scoring_suffix_sec_per_root": float(probe32_scoring_suffix_sec_per_root),
        },
    )

    # Benchmark 3: Learner updates/s (128 rescue + 128 ordinary with hard-negative mining).
    # Reuse obs_p extracted on the unstepped prepared sim above instead of re-running hero obs.
    t_b4_phase = _log_smoke_phase(
        "PHASE_START",
        "bench_4_learner_updates",
        t_start,
        detail=f"updates={bench_learner_updates}",
    )
    obs_fill = obs_p
    loc_fill = obs_fill["local_u8"].detach().cpu()
    glb_fill = obs_fill["global_u8"].detach().cpu()
    sca_fill = obs_fill["scalars_f32"].detach().cpu()
    leg_fill = obs_fill["legal_mask"].detach().cpu()

    initial_student_state = {
        k: v.detach().cpu().clone() for k, v in student_net.state_dict().items()
    }
    n_fill = max(res_batch_sz, ord_batch_sz, int(num_envs))
    for i in range(n_fill):
        idx = i % int(num_envs)
        leg_row = leg_fill[idx].clone()
        legal_acts = [int(a) for a in leg_row.nonzero(as_tuple=False).squeeze(1).tolist()]
        if len(legal_acts) >= 2:
            a_unsafe_pick, a_star_pick = legal_acts[0], legal_acts[1]
        else:
            a_star_pick = legal_acts[0] if legal_acts else 1
            a_unsafe_pick = (a_star_pick + 1) % 3
            leg_row[a_star_pick] = True
            leg_row[a_unsafe_pick] = True
        safe_synth = torch.zeros(6, dtype=torch.bool)
        safe_synth[a_star_pick] = True
        unsafe_synth = leg_row.clone()
        unsafe_synth[a_star_pick] = False
        seed_fill = int(world_seeds[max(online_env_start, idx) % int(num_envs)])
        rescue_buf.add_single(
            local_u8=loc_fill[idx],
            global_u8=glb_fill[idx],
            scalars_f32=sca_fill[idx],
            legal_mask=leg_row,
            safe_mask=safe_synth,
            unsafe_mask=unsafe_synth,
            bare_action=a_unsafe_pick,
            a_star_action=a_star_pick,
            world_seed=seed_fill,
            decision_frame=int(i),
            hard_neg_tier=0,
        )
        ordinary_buf.add_single(
            local_u8=loc_fill[idx],
            global_u8=glb_fill[idx],
            scalars_f32=sca_fill[idx],
            legal_mask=leg_row,
            safe_mask=safe_synth,
            unsafe_mask=unsafe_synth,
            bare_action=a_star_pick,
            a_star_action=a_star_pick,
            world_seed=seed_fill,
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
        if not bool(torch.isfinite(loss_dict["total_loss"]).item()):
            raise RuntimeError("Non-finite synthetic learner loss in smoke benchmark")
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, float(DEFAULT_MAX_GRAD_NORM))
        optimizer.step()
    _sync_device(dev)
    upd_elapsed = max(1e-9, time.perf_counter() - t_upd0)
    updates_per_sec = float(bench_learner_updates) / upd_elapsed
    b4_phase_elapsed = max(0.0, time.perf_counter() - t_b4_phase)
    _log_smoke_phase(
        "PHASE_END",
        "bench_4_learner_updates",
        t_start,
        phase_start=t_b4_phase,
        detail=f"updates_per_sec={updates_per_sec:.2f}",
    )
    _record_phase_partial(
        "bench_4_learner_updates",
        b4_phase_elapsed,
        {
            "bench_learner_updates": int(bench_learner_updates),
            "learner_benchmark_sec": float(upd_elapsed),
            "updates_per_sec": float(updates_per_sec),
        },
    )

    # Restore student_net weights to initial b8-s3 after synthetic benchmark updates
    student_net.load_state_dict(initial_student_state)
    freeze_ego2s_value_head(student_net)
    t_out_phase = _log_smoke_phase("PHASE_START", "write_outputs", t_start)
    if dev.type == "cuda":
        ensure_cuda_initialized(dev)
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
        "schema": (
            "t34-gpu-sitl-b1b-smoke-shakeout-subresult/v1"
            if is_shakeout
            else "t34-gpu-sitl-b1b-smoke-result/v1"
        ),
        "step": "Step B1b-Smoke-Shakeout-Subphase" if is_shakeout else "Step B1b-Smoke",
        "status": "PASS",
        "ok": True,
        "partial": False,
        "shakeout": is_shakeout,
        "not_evidence": bool(is_shakeout),
        "evidence_class": (
            "SHAKEOUT_NOT_EVIDENCE"
            if is_shakeout
            else ("DRY_RUN_NON_EVIDENCE" if dry_run else "REGISTERED_STEP_B1B_SMOKE_EVIDENCE")
        ),
        "dry_run": bool(is_non_prod),
        "non_registration_evidence": bool(is_non_prod),
        "device": str(dev),
        "native_extensions": native_extensions,
        "declared_torch_cuda_arch_list": native_extensions.get("torch_cuda_arch_list"),
        "effective_build_cuda_arch": native_extensions.get("effective_build_cuda_arch"),
        "train_namespace": active_train_namespace,
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
            "probe_sample_roots": int(n_probe_sample),
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
            "init_and_extensions_sec": float(init_elapsed),
            "mature_extension_build_sec": float(ext_build_elapsed),
            "parity_check_sec": float(parity_report.get("elapsed_sec", 0.0)),
            "bundle_and_bench_warmup_sec": float(bundle_warmup_elapsed),
            "env_benchmark_sec": float(env_elapsed),
            "mature_warmup_and_splice_sec": float(mature_warmup_elapsed),
            "label_benchmark_sec": float(lbl_elapsed),
            "buffer_insertion_sec": float(buf_insert_total_elapsed),
            "probe_scoring_benchmark_sec": float(probe_suf_elapsed),
            "learner_benchmark_sec": float(upd_elapsed),
        },
        "completed_phases": completed_phases_log,
        "pilot_runaway_guard": runaway_guard,
        "elapsed_sec": time.perf_counter() - t_start,
    }

    result_file = out_dir / (
        "b1b_smoke_shakeout_subresult.json" if is_shakeout else "smoke_result.json"
    )
    result_sha = write_atomic_with_sha256(
        result_file,
        lambda tmp: tmp.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
        write_complete_marker=True,
    )
    result_payload["result_json_path"] = str(result_file)
    result_payload["result_json_sha256"] = result_sha

    if not is_shakeout:
        # Write family-canary export-probe artifacts for the pilot's required_exports relative paths
        # if they do not already exist, so the colab_cli runtime export scanner records full
        # producer export coverage in the smoke canary receipt (`canary.py:322`).
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

    write_outputs_elapsed = max(0.0, time.perf_counter() - t_out_phase)
    result_payload["phase_timings_sec"]["write_outputs_sec"] = float(write_outputs_elapsed)
    _log_smoke_phase(
        "PHASE_END",
        "write_outputs",
        t_start,
        phase_start=t_out_phase,
        detail=f"total_elapsed_s={result_payload['elapsed_sec']:.2f}",
    )
    return result_payload


def compute_smoke_wall_time_projection(
    smoke_subresult: dict[str, Any],
    *,
    cold_init_elapsed_sec: float | None = None,
    target_a100_limit_sec: float = SMOKE_A100_PROJECTION_LIMIT_SEC,
) -> dict[str, Any]:
    """Project full registered Step B1b-Smoke wall time per phase from a shake-out subresult.

    For each phase ``i``:
    - ``same_hw_projected_sec = measured_sec * scale_ratio``
    - ``a100_projected_sec_estimate = same_hw_projected_sec * a100_hw_factor_estimate``

    The ``L4 -> A100`` conversion multiplier is an explicit hardware **ESTIMATE** (``0.55`` for
    GPU-resident phases from 108 vs 58 SMs and 1,555 vs 300 GB/s HBM2e/GDDR6 tempered by host
    launch overhead; ``0.85`` for mixed CPU+GPU ``parity_4_open_stop_h32``; ``1.00`` for host CPU
    compile/build phases on 12-vCPU Colab VMs).
    """
    tp = dict(smoke_subresult.get("throughput") or {})
    timings = dict(smoke_subresult.get("phase_timings_sec") or {})
    parity = dict(smoke_subresult.get("parity_check") or {})
    sub_par = dict(parity.get("subphase_timings_sec") or {})

    n_envs = max(1, int(tp.get("num_envs", REGISTERED_ENVS)))
    b_hor = max(1, int(tp.get("branch_horizon", OPEN_STOP_BRANCH_HORIZON)))
    n_par_roots = max(1, int(parity.get("num_roots", len(DEFAULT_MATURE_SLICE_INDICES))))
    n_warm_steps = int(tp.get("mature_warmup_steps", 0))
    n_env_steps = max(1, int(tp.get("bench_env_steps", 32)))
    n_lbl_batches = max(1, int(tp.get("bench_label_batches", DEFAULT_CUDA_BENCH_LABEL_BATCHES)))
    n_probe_roots = max(1, int(tp.get("probe_sample_roots", 64)))
    n_upd = max(1, int(tp.get("bench_learner_updates", 20)))

    env_scale = float(REGISTERED_ENVS) / float(n_envs)
    hor_scale = float(OPEN_STOP_BRANCH_HORIZON) / float(b_hor)
    shape_scale = env_scale * hor_scale
    par_roots_scale = float(len(DEFAULT_MATURE_SLICE_INDICES)) / float(n_par_roots)

    src_dev = str(smoke_subresult.get("device", "cpu"))
    eff_arch = str(smoke_subresult.get("effective_build_cuda_arch") or "")
    if eff_arch == "8.0":
        cpu_factor = 1.00
        gpu_factor = 1.00
        par4_factor = 1.00
        factor_source = "direct_a100_sm80_measurement (factor=1.00)"
    elif eff_arch == "9.0":
        cpu_factor = 1.00
        gpu_factor = float(ESTIMATED_H100_TO_A100_GPU_FACTOR)
        par4_factor = 1.10
        factor_source = "ESTIMATE: H100 sm_90 -> A100 sm_80 (gpu=1.35x, parity_4=1.10x, cpu=1.00x)"
    else:
        cpu_factor = float(ESTIMATED_L4_TO_A100_CPU_FACTOR)
        gpu_factor = float(ESTIMATED_L4_TO_A100_GPU_FACTOR)
        par4_factor = float(ESTIMATED_L4_TO_A100_PARITY4_FACTOR)
        factor_source = (
            "ESTIMATE: L4 sm_89 -> A100 sm_80 derived from 108 vs 58 SMs (1.86x), "
            "1,555 vs 300 GB/s HBM2e/GDDR6 (5.18x), and 12-vCPU parity "
            f"(gpu={gpu_factor:.2f}x, parity_4={par4_factor:.2f}x, cpu={cpu_factor:.2f}x)"
        )

    init_meas = float(timings.get("init_and_extensions_sec", 0.0))
    if cold_init_elapsed_sec is not None:
        init_meas = max(init_meas, float(cold_init_elapsed_sec))

    p1_meas = float(sub_par.get("parity_1_v8_flood_sec", 0.0))
    p2_meas = float(sub_par.get("parity_2_components_sec", 0.0))
    p3_meas = float(sub_par.get("parity_3_vector_runtime_sec", 0.0))
    p4_meas = float(sub_par.get("parity_4_open_stop_h32_sec", 0.0))
    par_total_meas = float(timings.get("parity_check_sec", p1_meas + p2_meas + p3_meas + p4_meas))
    par_setup_meas = max(0.0, par_total_meas - (p1_meas + p2_meas + p3_meas + p4_meas))

    warm_scale = (
        (float(DEFAULT_CUDA_MATURE_WARMUP_STEPS) / float(n_warm_steps)) * env_scale
        if n_warm_steps > 0
        else float(DEFAULT_CUDA_MATURE_WARMUP_STEPS) / float(n_env_steps) * env_scale
    )
    warm_meas = (
        float(timings.get("mature_warmup_and_splice_sec", 0.0))
        if n_warm_steps > 0
        else float(timings.get("env_benchmark_sec", 0.0))
    )

    b2_meas = float(timings.get("label_benchmark_sec", 0.0)) + float(
        timings.get("buffer_insertion_sec", 0.0)
    )
    write_meas = float(timings.get("write_outputs_sec", 0.5))

    phase_specs: list[tuple[str, float, float, float]] = [
        ("init_and_extensions", init_meas, 1.0, cpu_factor),
        (
            "build_mature_extension_worlds",
            float(timings.get("mature_extension_build_sec", 0.0)),
            par_roots_scale,
            cpu_factor,
        ),
        ("parity_setup_and_snapshots", par_setup_meas, par_roots_scale, cpu_factor),
        ("parity_1_v8_flood", p1_meas, par_roots_scale, gpu_factor),
        ("parity_2_components", p2_meas, par_roots_scale, gpu_factor),
        ("parity_3_vector_runtime", p3_meas, par_roots_scale, gpu_factor),
        ("parity_4_open_stop_h32", p4_meas, par_roots_scale * hor_scale, par4_factor),
        (
            "bundle_and_bench_warmup",
            float(timings.get("bundle_and_bench_warmup_sec", 0.0)),
            shape_scale,
            gpu_factor,
        ),
        ("mature_warmup_and_splice", warm_meas, warm_scale, gpu_factor),
        (
            "bench_1_env_steps",
            float(timings.get("env_benchmark_sec", 0.0)),
            (32.0 / float(n_env_steps)) * env_scale,
            gpu_factor,
        ),
        (
            "bench_2_label_batches_and_rescue_rate",
            b2_meas,
            (float(DEFAULT_CUDA_BENCH_LABEL_BATCHES) / float(n_lbl_batches)) * shape_scale,
            gpu_factor,
        ),
        (
            "bench_3_probe32_scoring_suffixes",
            float(timings.get("probe_scoring_benchmark_sec", 0.0)),
            (64.0 / float(n_probe_roots)) * hor_scale,
            gpu_factor,
        ),
        (
            "bench_4_learner_updates",
            float(timings.get("learner_benchmark_sec", 0.0)),
            20.0 / float(n_upd),
            gpu_factor,
        ),
        ("write_outputs", write_meas, 1.0, cpu_factor),
    ]

    per_phase: list[dict[str, Any]] = []
    same_hw_total = 0.0
    a100_total_est = 0.0
    for p_name, meas_s, s_ratio, hw_fac in phase_specs:
        same_hw_s = float(meas_s) * float(s_ratio)
        a100_s = same_hw_s * float(hw_fac)
        same_hw_total += same_hw_s
        a100_total_est += a100_s
        per_phase.append(
            {
                "phase": p_name,
                "measured_sec": float(meas_s),
                "scale_ratio": float(s_ratio),
                "same_hw_projected_sec": float(same_hw_s),
                "a100_hw_factor_estimate": float(hw_fac),
                "a100_projected_sec_estimate": float(a100_s),
                "is_estimate": True,
            }
        )
        print(
            f"[shakeout.projection] phase={p_name} measured_s={meas_s:.2f} "
            f"scale_ratio={s_ratio:.2f}x same_hw_projected_s={same_hw_s:.2f} "
            f"a100_hw_factor_est={hw_fac:.2f} (ESTIMATE) "
            f"a100_projected_s={a100_s:.2f} (ESTIMATE)",
            flush=True,
        )

    within_limit = bool(a100_total_est <= float(target_a100_limit_sec))
    print(
        f"[shakeout.projection] SUMMARY source_device={src_dev} source_arch={eff_arch or 'cpu'} "
        f"same_hw_total_projected_s={same_hw_total:.2f} "
        f"a100_total_projected_s={a100_total_est:.2f} (ESTIMATE) "
        f"limit_s={float(target_a100_limit_sec):.2f} "
        f"status={'PASS' if within_limit else 'FAIL'}",
        flush=True,
    )
    return {
        "schema": "t34-gpu-sitl-smoke-wall-projection/v1",
        "is_estimate": True,
        "source_device": src_dev,
        "source_effective_cuda_arch": eff_arch or None,
        "l4_to_a100_factor_derivation": factor_source,
        "target_a100_limit_sec": float(target_a100_limit_sec),
        "same_hw_total_projected_sec": float(same_hw_total),
        "a100_total_projected_sec_estimate": float(a100_total_est),
        "within_a100_limit": within_limit,
        "phases": per_phase,
    }


def _describe_resume_determinism_mismatch(
    resumed_result: dict[str, Any],
    uninterrupted_result: dict[str, Any],
) -> str:
    """Return a concise diagnostic pinpointing the first differing update index and tensor name."""
    parts: list[str] = []
    hist_r = list(resumed_result.get("update_history") or [])
    hist_u = list(uninterrupted_result.get("update_history") or [])
    first_update_diff: str | None = None
    for idx in range(min(len(hist_r), len(hist_u))):
        row_r = hist_r[idx]
        row_u = hist_u[idx]
        if row_r != row_u:
            upd_num = row_r.get("update", idx + 1)
            diff_keys: list[str] = []
            for key in sorted(set(row_r.keys()) | set(row_u.keys())):
                if row_r.get(key) != row_u.get(key):
                    diff_keys.append(f"{key}: resumed={row_r.get(key)!r} vs unint={row_u.get(key)!r}")
            shown = "; ".join(diff_keys[:4])
            first_update_diff = f"first_differing_update_index={idx} (update={upd_num}; {shown})"
            break
    if first_update_diff is None and len(hist_r) != len(hist_u):
        first_update_diff = (
            f"first_differing_update_index={min(len(hist_r), len(hist_u))} "
            f"(len resumed={len(hist_r)} vs unint={len(hist_u)})"
        )
    parts.append(first_update_diff or "first_differing_update_index=none")

    first_tensor_diff: str | None = None
    path_r = resumed_result.get("final_student_checkpoint_path")
    path_u = uninterrupted_result.get("final_student_checkpoint_path")
    if path_r and path_u and Path(str(path_r)).is_file() and Path(str(path_u)).is_file():
        blob_r = torch.load(Path(str(path_r)), map_location="cpu", weights_only=False)
        blob_u = torch.load(Path(str(path_u)), map_location="cpu", weights_only=False)
        sd_r = blob_r.get("state_dict", blob_r) if isinstance(blob_r, dict) else {}
        sd_u = blob_u.get("state_dict", blob_u) if isinstance(blob_u, dict) else {}
        all_names = list(sd_r.keys()) + [k for k in sd_u.keys() if k not in sd_r]
        for t_name in all_names:
            if t_name not in sd_r or t_name not in sd_u:
                first_tensor_diff = f"first_differing_tensor={t_name} (missing in one state_dict)"
                break
            t_r = sd_r[t_name]
            t_u = sd_u[t_name]
            if isinstance(t_r, torch.Tensor) and isinstance(t_u, torch.Tensor):
                if t_r.shape != t_u.shape or t_r.dtype != t_u.dtype or not torch.equal(t_r, t_u):
                    max_abs = (
                        float((t_r.detach().float() - t_u.detach().float()).abs().max().item())
                        if t_r.shape == t_u.shape and t_r.numel() > 0
                        else float("nan")
                    )
                    first_tensor_diff = (
                        f"first_differing_tensor={t_name} "
                        f"(shape={tuple(t_r.shape)}, max_abs_diff={max_abs:.6e})"
                    )
                    break
            elif t_r != t_u:
                first_tensor_diff = f"first_differing_tensor={t_name} (non-tensor mismatch)"
                break
        if (
            first_tensor_diff is None
            and resumed_result.get("final_student_checkpoint_sha256")
            != uninterrupted_result.get("final_student_checkpoint_sha256")
        ):
            first_tensor_diff = (
                "first_differing_tensor=none_in_state_dict (checkpoint metadata/serialization differs)"
            )
    parts.append(first_tensor_diff or "first_differing_tensor=none")
    return ", ".join(parts)


def run_cuda_shakeout(
    output_dir: Path | str,
    *,
    b8_checkpoint: Path | str | None = None,
    opponent_checkpoint: Path | str | None = None,
    device: str | torch.device = "cuda",
    num_envs: int | None = None,
    branch_horizon: int | None = None,
    parity_slice_indices: Sequence[int] | None = None,
    mature_warmup_steps: int | None = None,
    bench_env_steps: int | None = None,
    bench_label_batches: int | None = None,
    probe_sample_roots: int | None = None,
    bench_learner_updates: int | None = None,
    probe_worlds: int = 2,
    probe_horizon: int = 4,
    train_total_updates: int = 2,
    train_stop_after_updates: int = 1,
    seed: int = 20261110,
    require_cuda: bool = False,
    enforce_projection_gate: bool | None = None,
    target_a100_limit_sec: float = SMOKE_A100_PROJECTION_LIMIT_SEC,
) -> dict[str, Any]:
    """Run the dev-only L4/CUDA shake-out across all GPU SITL phases at predictive smoke scale.

    Exercises on real CUDA (at full ``E = 256, H = 32`` smoke batch shape with scaled iteration
    counts so phase timings linearly predict ``Step B1b-Smoke``; and at tiny size on CPU/MPS in
    local unit tests):
    1. CUDA initialization and caching-allocator statistics before any kernel launch
       (`torch.cuda.reset_peak_memory_stats`, `max_memory_allocated`, `max_memory_reserved`,
       `synchronize`, `get_rng_state_all`).
    2. `TORCH_CUDA_ARCH_LIST` verification (`8.0` A100, `8.9` L4, `9.0` H100) + native extension
       compile/load and functional kernel verification (`flood_cuda`, `vector_geometry`).
    3. Dev-only seed isolation audit (never reading or scoring
       `t34-gpu-sitl-selectivity-20261109-v1`, `t34-gpu-sitl-screen-20261109-v1`,
       `t34-gpu-sitl-probe32-20261109-v1`, or `t34-gpu-sitl-train-20261109-v1`).
    4. Full `run_step_b1b_smoke` code path at predictive scale + per-phase A100 wall-time
       projection (fails closed if projected A100 smoke wall time exceeds `700.0 s`).
    5. Tiny dev-seed probe-bank harvest and namespace-isolation refusal check.
    6. Tiny dev-seed `run_sitl_training` including tripwire evaluation, checkpoint save, and
       verified-ledger resume from that checkpoint.

    Continues past a failing phase where safe and reports every phase's pass/fail with traceback
    in `shakeout_result.json` (`NOT EVIDENCE`).
    """
    t_start = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(f"{out_dir / 'shakeout_result.json'}.complete").is_file():
        raise FileExistsError(f"Refusing to overwrite completed shakeout_result.json in {out_dir}")

    if b8_checkpoint is None or opponent_checkpoint is None:
        default_b8, default_opp = resolve_product_checkpoints()
        b8_checkpoint = b8_checkpoint or default_b8
        opponent_checkpoint = opponent_checkpoint or default_opp
    b8_path = Path(b8_checkpoint)
    opp_path = Path(opponent_checkpoint)

    phase_reports: list[dict[str, Any]] = []

    def _run_phase(phase_name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        print(f"[shakeout] START {phase_name}", flush=True)
        t_p0 = time.perf_counter()
        try:
            detail = fn()
            el = time.perf_counter() - t_p0
            rec = {
                "phase": phase_name,
                "ok": True,
                "status": "PASS",
                "elapsed_sec": el,
                "detail": detail,
                "error": None,
                "traceback": None,
            }
            print(f"[shakeout] PASS {phase_name} ({el:.2f}s)", flush=True)
            phase_reports.append(rec)
            return rec
        except Exception as exc:
            el = time.perf_counter() - t_p0
            tb = traceback.format_exc()
            rec = {
                "phase": phase_name,
                "ok": False,
                "status": "FAIL",
                "elapsed_sec": el,
                "detail": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
            }
            print(
                f"[shakeout] FAIL {phase_name} ({el:.2f}s): {type(exc).__name__}: {exc}\n{tb}",
                flush=True,
            )
            phase_reports.append(rec)
            return rec

    resolved_dev_holder: dict[str, torch.device] = {"dev": torch.device("cpu")}
    ext_info_holder: dict[str, Any] = {"info": {}, "cold_init_elapsed_sec": 0.0}
    projection_holder: dict[str, dict[str, Any] | None] = {"projection": None}

    def _effective_shakeout_knobs() -> dict[str, Any]:
        dev_now = resolved_dev_holder["dev"]
        is_cuda_dev = dev_now.type == "cuda"
        e_val = int(num_envs) if num_envs is not None else (REGISTERED_ENVS if is_cuda_dev else 4)
        h_val = (
            int(branch_horizon)
            if branch_horizon is not None
            else (OPEN_STOP_BRANCH_HORIZON if is_cuda_dev else 2)
        )
        p_slice = (
            tuple(int(i) for i in parity_slice_indices)
            if parity_slice_indices is not None
            else (DEFAULT_MATURE_SLICE_INDICES if is_cuda_dev else (0, 8))
        )
        warm_val = (
            int(mature_warmup_steps)
            if mature_warmup_steps is not None
            else (60 if is_cuda_dev else 2)
        )
        env_steps_val = (
            int(bench_env_steps) if bench_env_steps is not None else (8 if is_cuda_dev else 1)
        )
        lbl_batches_val = (
            int(bench_label_batches)
            if bench_label_batches is not None
            else (8 if is_cuda_dev else 1)
        )
        probe_roots_val = (
            int(probe_sample_roots) if probe_sample_roots is not None else (8 if is_cuda_dev else 1)
        )
        upd_val = (
            int(bench_learner_updates)
            if bench_learner_updates is not None
            else (5 if is_cuda_dev else 2)
        )
        gate_val = (
            bool(enforce_projection_gate)
            if enforce_projection_gate is not None
            else bool(is_cuda_dev)
        )
        return {
            "num_envs": e_val,
            "branch_horizon": h_val,
            "parity_slice_indices": p_slice,
            "mature_warmup_steps": warm_val,
            "bench_env_steps": env_steps_val,
            "bench_label_batches": lbl_batches_val,
            "probe_sample_roots": probe_roots_val,
            "bench_learner_updates": upd_val,
            "train_num_envs": min(e_val, 16),
            "train_branch_horizon": min(h_val, 8),
            "enforce_projection_gate": gate_val,
        }

    def _phase_1_cuda_init() -> dict[str, Any]:
        dev = ensure_cuda_initialized(device)
        resolved_dev_holder["dev"] = dev
        if require_cuda and dev.type != "cuda":
            raise RuntimeError(
                f"CUDA shake-out with require_cuda=True requires device='cuda', got {dev}"
            )
        if dev.type == "cuda":
            if not torch.cuda.is_initialized():
                raise RuntimeError(
                    "torch.cuda.is_initialized() is False after ensure_cuda_initialized"
                )
            torch.cuda.reset_peak_memory_stats(dev)
            alloc_bytes = int(torch.cuda.max_memory_allocated(dev))
            reserved_bytes = int(torch.cuda.max_memory_reserved(dev))
            torch.cuda.synchronize(dev)
            rng_states = torch.cuda.get_rng_state_all()
            cap = torch.cuda.get_device_capability(dev)
            dev_name = str(torch.cuda.get_device_name(dev))
            return {
                "device": str(dev),
                "cuda_is_initialized": True,
                "device_name": dev_name,
                "device_capability": f"{cap[0]}.{cap[1]}",
                "peak_allocated_bytes_after_reset": alloc_bytes,
                "peak_reserved_bytes_after_reset": reserved_bytes,
                "cuda_rng_state_count": len(rng_states),
            }
        return {
            "device": str(dev),
            "cuda_is_initialized": False,
            "device_name": dev.type,
            "device_capability": "non-cuda",
        }

    def _phase_2_native_extensions() -> dict[str, Any]:
        dev = resolved_dev_holder["dev"]
        raw_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
        if not raw_arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_CUDA_ARCH_LIST
            raw_arch = DEFAULT_CUDA_ARCH_LIST
        tokens = {
            tok.split("+")[0].strip()
            for tok in raw_arch.replace(",", ";").replace(" ", ";").split(";")
            if tok.strip()
        }
        missing_archs = [arch for arch in REQUIRED_CUDA_ARCHS if arch not in tokens]
        if missing_archs:
            raise RuntimeError(
                f"TORCH_CUDA_ARCH_LIST={raw_arch!r} is missing required architectures "
                f"{missing_archs} (must include 8.0 A100, 8.9 L4, 9.0 H100)"
            )
        t_ext0 = time.perf_counter()
        ext_info = verify_cuda_native_extensions(dev)
        ext_elapsed = max(0.0, time.perf_counter() - t_ext0)
        p1_elapsed = float(phase_reports[0]["elapsed_sec"]) if phase_reports else 0.0
        ext_info_holder["info"] = ext_info
        ext_info_holder["cold_init_elapsed_sec"] = p1_elapsed + ext_elapsed
        if dev.type == "cuda":
            return {
                "torch_cuda_arch_list": raw_arch,
                "declared_torch_cuda_arch_list": ext_info.get("torch_cuda_arch_list"),
                "effective_build_cuda_arch": ext_info.get("effective_build_cuda_arch"),
                "scoped_torch_extensions_dir": ext_info.get("scoped_torch_extensions_dir"),
                "required_archs_present": list(REQUIRED_CUDA_ARCHS),
                "cuda_native_extensions_verified": True,
                "extension_info": ext_info,
            }
        return {
            "torch_cuda_arch_list": raw_arch,
            "declared_torch_cuda_arch_list": ext_info.get("torch_cuda_arch_list"),
            "effective_build_cuda_arch": ext_info.get("effective_build_cuda_arch"),
            "required_archs_present": list(REQUIRED_CUDA_ARCHS),
            "cuda_native_extensions_verified": False,
            "extension_info": ext_info,
            "note": f"Non-CUDA device {dev}; verified TORCH_CUDA_ARCH_LIST covers 8.0, 8.9, 9.0",
        }

    def _phase_3_seed_isolation() -> dict[str, Any]:
        knobs = _effective_shakeout_knobs()
        audit = audit_shakeout_seeds(
            smoke_envs=int(knobs["num_envs"]),
            probe_worlds=int(probe_worlds),
            train_envs=int(knobs["num_envs"]),
        )
        if not bool(audit["ok"]):
            raise RuntimeError(f"Shake-out seed isolation audit failed: {audit}")
        forbidden_namespaces = {
            SELECTIVITY_NAMESPACE,
            SCREEN_NAMESPACE,
            PROBE32_NAMESPACE,
            TRAIN_NAMESPACE,
        }
        active_namespaces = {
            SHAKEOUT_SMOKE_NAMESPACE,
            SHAKEOUT_PROBE_NAMESPACE,
            SHAKEOUT_TRAIN_NAMESPACE,
        }
        if active_namespaces & forbidden_namespaces:
            raise RuntimeError("Shake-out namespaces overlap forbidden registered namespaces")
        return audit

    with tempfile.TemporaryDirectory(prefix="gpu_sitl_shakeout_work_") as tmp_work_str:
        work_dir = Path(tmp_work_str)
        harvested_probe_path_holder: dict[str, Path | None] = {"path": None}

        def _phase_4_b1b_smoke_reduced() -> dict[str, Any]:
            dev = resolved_dev_holder["dev"]
            knobs = _effective_shakeout_knobs()
            smoke_sub_dir = work_dir / "smoke_subphase"
            smoke_res = run_step_b1b_smoke(
                smoke_sub_dir,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev,
                num_envs=int(knobs["num_envs"]),
                branch_horizon=int(knobs["branch_horizon"]),
                parity_slice_indices=knobs["parity_slice_indices"],
                bench_env_steps=int(knobs["bench_env_steps"]),
                bench_label_batches=int(knobs["bench_label_batches"]),
                bench_learner_updates=int(knobs["bench_learner_updates"]),
                mature_warmup_steps_override=int(knobs["mature_warmup_steps"]),
                probe_sample_roots_override=int(knobs["probe_sample_roots"]),
                seed=int(seed),
                dry_run=True,
                shakeout=True,
            )
            if not bool(smoke_res["parity_check"]["ok"]):
                raise RuntimeError("Reduced smoke row-subset parity check did not pass")
            tp = smoke_res["throughput"]
            if int(tp["mature_spliced_worlds"]) < 1:
                raise RuntimeError("Reduced smoke did not splice any mature cohort worlds")
            if int(tp["window_max_hero_length"]) < 80:
                raise RuntimeError(
                    f"Reduced smoke mature splice did not exercise hero length >= 80: "
                    f"window_max_hero_length={tp['window_max_hero_length']}"
                )
            if int(tp["window_mature_length_ge_160_decisions"]) < 1:
                raise RuntimeError(
                    "Reduced smoke mature splice did not exercise flood cap 160 (L >= 160)"
                )
            if smoke_res.get("schema") != "t34-gpu-sitl-b1b-smoke-shakeout-subresult/v1":
                raise RuntimeError(f"Unexpected sub-smoke schema: {smoke_res.get('schema')}")
            if smoke_res.get("not_evidence") is not True:
                raise RuntimeError("Sub-smoke result missing not_evidence=True")
            if (smoke_sub_dir / "smoke_result.json").exists():
                raise RuntimeError("Shake-out must never write smoke_result.json")

            projection = compute_smoke_wall_time_projection(
                smoke_res,
                cold_init_elapsed_sec=float(ext_info_holder.get("cold_init_elapsed_sec", 0.0)),
                target_a100_limit_sec=float(target_a100_limit_sec),
            )
            projection_holder["projection"] = projection
            if bool(knobs["enforce_projection_gate"]) and not bool(projection["within_a100_limit"]):
                raise RuntimeError(
                    f"Shake-out projected A100 smoke wall time "
                    f"{projection['a100_total_projected_sec_estimate']:.2f}s (ESTIMATE) "
                    f"exceeds limit {projection['target_a100_limit_sec']:.2f}s"
                )

            return {
                "schema": smoke_res["schema"],
                "train_namespace": smoke_res["train_namespace"],
                "parity_ok": bool(smoke_res["parity_check"]["ok"]),
                "mature_spliced_worlds": int(tp["mature_spliced_worlds"]),
                "window_max_hero_length": int(tp["window_max_hero_length"]),
                "window_mature_length_ge_160_decisions": int(
                    tp["window_mature_length_ge_160_decisions"]
                ),
                "env_steps_per_sec": float(tp["env_steps_per_sec"]),
                "label_batches_per_sec": float(tp["label_batches_per_sec"]),
                "updates_per_sec": float(tp["updates_per_sec"]),
                "peak_vram_allocated_mib": float(tp["peak_vram_allocated_mib"]),
                "runaway_guard_timeout_minutes_2x": int(
                    smoke_res["pilot_runaway_guard"]["runaway_guard_timeout_minutes_2x"]
                ),
                "smoke_wall_time_projection": projection,
                "subresult_json_sha256": smoke_res["result_json_sha256"],
            }

        def _phase_5_tiny_probe_harvest() -> dict[str, Any]:
            dev = resolved_dev_holder["dev"]
            knobs = _effective_shakeout_knobs()
            probe_sub_dir = work_dir / "probe_subphase"
            probe_pt = probe_sub_dir / "shakeout_probe_bank.pt"
            meta = harvest_probe32_bank(
                probe_pt,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev,
                num_worlds=int(probe_worlds),
                horizon=int(probe_horizon),
                branch_horizon=int(knobs["train_branch_horizon"]),
                ordinary_per_world=3,
                harvest_batch_roots=max(2, int(knobs["train_num_envs"])),
                dry_run=True,
                shakeout=True,
            )
            loaded = load_probe32_bank(
                probe_pt,
                expected_namespace=SHAKEOUT_PROBE_NAMESPACE,
            )
            if loaded["meta"].get("namespace") != SHAKEOUT_PROBE_NAMESPACE:
                raise RuntimeError(
                    f"Tiny probe harvest namespace mismatch: {loaded['meta'].get('namespace')}"
                )
            if loaded["meta"].get("not_evidence") is not True:
                raise RuntimeError("Tiny probe harvest missing not_evidence=True")
            # Verify default registered loader refuses the shake-out probe bank
            refused_by_default_loader = False
            try:
                load_probe32_bank(probe_pt)
            except (RuntimeError, ValueError):
                refused_by_default_loader = True
            if not refused_by_default_loader:
                raise RuntimeError(
                    "load_probe32_bank with default PROBE32_NAMESPACE failed to refuse "
                    "shake-out bank"
                )
            harvested_probe_path_holder["path"] = probe_pt
            return {
                "schema": meta["schema"],
                "namespace": meta["namespace"],
                "not_evidence": bool(meta["not_evidence"]),
                "num_worlds": int(meta["num_worlds"]),
                "horizon": int(meta["horizon"]),
                "decisive_rescue_roots_count": int(meta["decisive_rescue_roots_count"]),
                "ordinary_roots_count": int(meta["ordinary_roots_count"]),
                "refused_by_registered_probe32_loader": refused_by_default_loader,
                "sha256": str(loaded["verified_sha256"]),
            }

        def _phase_6_tiny_sitl_training_and_resume() -> dict[str, Any]:
            dev = resolved_dev_holder["dev"]
            knobs = _effective_shakeout_knobs()
            t_envs = int(knobs["train_num_envs"])
            t_bhor = int(knobs["train_branch_horizon"])
            probe_pt = harvested_probe_path_holder["path"]
            if probe_pt is None or not probe_pt.exists():
                probe_fallback_dir = work_dir / "probe_fallback"
                probe_pt = probe_fallback_dir / "shakeout_probe_bank.pt"
                harvest_probe32_bank(
                    probe_pt,
                    b8_checkpoint=b8_path,
                    opponent_checkpoint=opp_path,
                    device=dev,
                    num_worlds=int(probe_worlds),
                    horizon=max(4, int(probe_horizon)),
                    branch_horizon=t_bhor,
                    ordinary_per_world=3,
                    harvest_batch_roots=max(2, t_envs),
                    dry_run=True,
                    shakeout=True,
                )

            train_sub_dir = work_dir / "train_subphase"
            total_u = max(2, int(train_total_updates))
            stop_u = max(1, min(total_u - 1, int(train_stop_after_updates)))

            part1 = run_sitl_training(
                train_sub_dir,
                probe_bank_path=probe_pt,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev,
                num_envs=t_envs,
                total_updates=total_u,
                warmup_rescue_roots=1,
                min_unique_rescue_roots=total_u,
                rescue_batch_size=2,
                ordinary_batch_size=4,
                ordinary_uniform_half=2,
                ordinary_hard_neg_half=2,
                branch_horizon=t_bhor,
                tripwire_interval=1,
                decision_window_size=8,
                checkpoint_every_updates=1,
                stop_after_updates=stop_u,
                max_collection_steps=12,
                seed=int(seed),
                dry_run=True,
                shakeout=True,
            )
            if part1.get("status") != "PAUSED_FOR_RESUME":
                raise RuntimeError(
                    f"Expected PAUSED_FOR_RESUME after stop_after_updates={stop_u}, got {part1}"
                )
            ledger_path = Path(part1["verified_ledger_path"])
            if not ledger_path.exists():
                raise RuntimeError(f"Missing verified ledger after pause: {ledger_path}")

            part2 = run_sitl_training(
                train_sub_dir,
                probe_bank_path=probe_pt,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                ledger_path=ledger_path,
                device=dev,
                num_envs=t_envs,
                total_updates=total_u,
                warmup_rescue_roots=1,
                min_unique_rescue_roots=total_u,
                rescue_batch_size=2,
                ordinary_batch_size=4,
                ordinary_uniform_half=2,
                ordinary_hard_neg_half=2,
                branch_horizon=t_bhor,
                tripwire_interval=1,
                decision_window_size=8,
                checkpoint_every_updates=1,
                stop_after_updates=None,
                max_collection_steps=16,
                seed=int(seed),
                dry_run=True,
                shakeout=True,
                resume=True,
            )
            if not part2.get("resumed_from_checkpoint"):
                raise RuntimeError("Resumed training run did not load verified checkpoint")
            if int(part2["counts"]["updates_completed"]) != total_u:
                raise RuntimeError(
                    f"Expected {total_u} updates completed after resume, got "
                    f"{part2['counts']['updates_completed']}"
                )
            if len(part2["probe_bank_tripwires"]["history"]) < 1:
                raise RuntimeError("No probe-bank tripwire evaluations recorded during shake-out")
            fc = part2["unconditional_fault_checks"]
            if (
                int(fc["non_finite_loss_count"]) != 0
                or int(fc["non_finite_q_count"]) != 0
                or int(fc["non_finite_grad_count"]) != 0
                or int(fc["illegal_action_count"]) != 0
            ):
                raise RuntimeError(f"Unconditional fault check failed during shake-out: {fc}")
            if part2.get("schema") != "t34-gpu-sitl-shakeout-train-subresult/v1":
                raise RuntimeError(f"Unexpected train subresult schema: {part2.get('schema')}")
            if part2.get("not_evidence") is not True:
                raise RuntimeError("Train subresult missing not_evidence=True")
            if (train_sub_dir / "sitl_pilot_result.json").exists():
                raise RuntimeError("Shake-out training must never write sitl_pilot_result.json")

            uninterrupted = run_sitl_training(
                work_dir / "train_uninterrupted",
                probe_bank_path=probe_pt,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev,
                num_envs=t_envs,
                total_updates=total_u,
                warmup_rescue_roots=1,
                min_unique_rescue_roots=total_u,
                rescue_batch_size=2,
                ordinary_batch_size=4,
                ordinary_uniform_half=2,
                ordinary_hard_neg_half=2,
                branch_horizon=t_bhor,
                tripwire_interval=1,
                decision_window_size=8,
                checkpoint_every_updates=1,
                stop_after_updates=None,
                max_collection_steps=16,
                seed=int(seed),
                dry_run=True,
                shakeout=True,
                resume=False,
            )
            if (
                part2["final_student_checkpoint_sha256"]
                != uninterrupted["final_student_checkpoint_sha256"]
                or part2["update_history"] != uninterrupted["update_history"]
            ):
                diag = _describe_resume_determinism_mismatch(part2, uninterrupted)
                print(
                    f"[shakeout] phase_6 resume determinism mismatch: {diag}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Shake-out resume determinism check failed: resumed != uninterrupted ({diag})"
                )
            return {
                "schema": part2["schema"],
                "namespace": part2["namespace"],
                "probe_namespace": part2["probe_namespace"],
                "not_evidence": bool(part2["not_evidence"]),
                "paused_at_updates": int(part1["updates_completed"]),
                "resumed_from_checkpoint": str(part2["resumed_from_checkpoint"]),
                "final_updates_completed": int(part2["counts"]["updates_completed"]),
                "tripwire_evaluations": len(part2["probe_bank_tripwires"]["history"]),
                "deterministic_resume_verified": True,
                "subresult_json_sha256": part2["result_json_sha256"],
            }

        _run_phase("phase_1_cuda_init_and_allocator_stats", _phase_1_cuda_init)
        _run_phase("phase_2_native_extensions_and_arch_list", _phase_2_native_extensions)
        _run_phase("phase_3_seed_isolation_audit", _phase_3_seed_isolation)
        _run_phase("phase_4_b1b_smoke_reduced", _phase_4_b1b_smoke_reduced)
        _run_phase("phase_5_tiny_probe_harvest", _phase_5_tiny_probe_harvest)
        _run_phase("phase_6_tiny_sitl_training_and_resume", _phase_6_tiny_sitl_training_and_resume)

    all_ok = bool(phase_reports) and all(bool(p["ok"]) for p in phase_reports)
    failed_phases = [str(p["phase"]) for p in phase_reports if not bool(p["ok"])]
    elapsed_sec = time.perf_counter() - t_start

    shakeout_payload: dict[str, Any] = {
        "schema": "t34-gpu-sitl-cuda-shakeout-result/v1",
        "step": "Step B1b-CUDA-Shakeout",
        "mode": "shakeout",
        "shakeout": True,
        "dry_run": True,
        "not_evidence": True,
        "non_registration_evidence": True,
        "evidence_class": "SHAKEOUT_NOT_EVIDENCE",
        "evidence_banner": "NOT EVIDENCE — DEV-ONLY L4 CUDA SHAKE-OUT",
        "ok": all_ok,
        "verdict": "SHAKEOUT_PASS_NOT_EVIDENCE" if all_ok else "SHAKEOUT_FAIL_NOT_EVIDENCE",
        "device": str(resolved_dev_holder["dev"]),
        "native_extensions": ext_info_holder["info"],
        "declared_torch_cuda_arch_list": ext_info_holder["info"].get("torch_cuda_arch_list"),
        "effective_build_cuda_arch": ext_info_holder["info"].get("effective_build_cuda_arch"),
        "namespaces": {
            "smoke": SHAKEOUT_SMOKE_NAMESPACE,
            "probe": SHAKEOUT_PROBE_NAMESPACE,
            "train": SHAKEOUT_TRAIN_NAMESPACE,
        },
        "forbidden_namespaces_never_read_or_scored": [
            SELECTIVITY_NAMESPACE,
            SCREEN_NAMESPACE,
            PROBE32_NAMESPACE,
            TRAIN_NAMESPACE,
        ],
        "phases": phase_reports,
        "failed_phases": failed_phases,
        "smoke_wall_time_projection": projection_holder["projection"],
        "elapsed_sec": elapsed_sec,
    }

    result_file = out_dir / "shakeout_result.json"
    result_sha = write_atomic_with_sha256(
        result_file,
        lambda tmp: tmp.write_text(
            json.dumps(shakeout_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
        write_complete_marker=True,
    )
    shakeout_payload["result_json_path"] = str(result_file)
    shakeout_payload["result_json_sha256"] = result_sha
    print(
        f"[shakeout] DONE verdict={shakeout_payload['verdict']} "
        f"failed_phases={failed_phases} elapsed={elapsed_sec:.2f}s",
        flush=True,
    )
    return shakeout_payload


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
    parser.add_argument("--mature-warmup-steps", type=int, default=None)
    parser.add_argument("--bench-env-steps", type=int, default=None)
    parser.add_argument("--bench-label-batches", type=int, default=None)
    parser.add_argument("--probe-sample-roots", type=int, default=None)
    parser.add_argument("--bench-learner-updates", type=int, default=None)
    parser.add_argument(
        "--enforce-projection-gate",
        action="store_true",
        default=None,
        help="Fail shake-out if projected A100 smoke wall time exceeds --target-a100-limit-sec.",
    )
    parser.add_argument(
        "--target-a100-limit-sec",
        type=float,
        default=SMOKE_A100_PROJECTION_LIMIT_SEC,
    )
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny CPU dry-run configuration (marks dry_run=true).",
    )
    parser.add_argument(
        "--shakeout",
        action="store_true",
        help=(
            "Run the dev-only CUDA shake-out across all phases at predictive scale "
            "(writes shakeout_result.json, NOT EVIDENCE)."
        ),
    )
    args = parser.parse_args()
    if bool(args.shakeout):
        shake_res = run_cuda_shakeout(
            args.output_dir,
            b8_checkpoint=args.b8_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            device=args.device,
            num_envs=args.num_envs,
            branch_horizon=args.branch_horizon,
            mature_warmup_steps=args.mature_warmup_steps,
            bench_env_steps=args.bench_env_steps,
            bench_label_batches=args.bench_label_batches,
            probe_sample_roots=args.probe_sample_roots,
            bench_learner_updates=args.bench_learner_updates,
            seed=args.seed,
            require_cuda=str(args.device).startswith("cuda"),
            enforce_projection_gate=args.enforce_projection_gate,
            target_a100_limit_sec=float(args.target_a100_limit_sec),
        )
        print(
            json.dumps(
                {
                    "step": shake_res["step"],
                    "mode": shake_res["mode"],
                    "not_evidence": shake_res["not_evidence"],
                    "evidence_class": shake_res["evidence_class"],
                    "ok": shake_res["ok"],
                    "verdict": shake_res["verdict"],
                    "device": shake_res["device"],
                    "failed_phases": shake_res["failed_phases"],
                    "smoke_wall_time_projection": shake_res["smoke_wall_time_projection"],
                    "result_json_path": shake_res["result_json_path"],
                    "result_json_sha256": shake_res["result_json_sha256"],
                    "elapsed_sec": shake_res["elapsed_sec"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if bool(shake_res["ok"]) else 1

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
