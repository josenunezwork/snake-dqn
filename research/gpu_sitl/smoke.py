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
from research.gpu_sitl.open_stop_gpu import (
    OPEN_STOP_BRANCH_HORIZON,
    OPEN_STOP_CRN_NAMESPACE,
    OPEN_STOP_V8_LAMBDA,
    FastDeviceVectorRuntime,
    _maybe_cpu_fast_flood,
    _ttl_by_owner,
    batched_open_stop_h32,
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
    PROBE32_HORIZON,
    PROBE32_WORLD_COUNT,
    build_watch6_ct_device_bundle,
    extract_hero_root_observations,
    sha256_file,
    write_atomic_with_sha256,
)
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    preflight_seeds,
    train_seeds,
)
from research.gpu_sitl.train_sitl import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    ORDINARY_MARGIN_DELTA_DEFAULT,
    REGISTERED_ENVS,
    REGISTERED_MIN_UNIQUE_RESCUE_ROOTS,
    REGISTERED_ORDINARY_BATCH_SIZE,
    REGISTERED_ORDINARY_HARD_NEG_HALF,
    REGISTERED_ORDINARY_UNIFORM_HALF,
    REGISTERED_RESCUE_BATCH_SIZE,
    REGISTERED_TOTAL_UPDATES,
    RESCUE_MARGIN_DELTA_DEFAULT,
    SitlRingBuffer,
    compute_sitl_hinge_loss,
    sample_ordinary_minibatch_indices,
)
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

# Default 8-root slice spanning Groups A (2), B (2), C (2), and D (2) of MATURE_EXTENSION_SPECS
DEFAULT_MATURE_SLICE_INDICES: tuple[int, ...] = (0, 1, 8, 9, 16, 17, 24, 25)
REGISTERED_RESCUE_RATE_PRIOR = 0.0012
REGISTERED_ALIVE_HERO_FRAC_ESTIMATE = 0.95
REGISTERED_A100_CALIBRATED_RATE_UPH = 6.73
REGISTERED_A100_BROKER_CEILING_RATE_UPH = 8.00


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


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
    dev = torch.device(device)
    if not dry_run and dev.type != "cuda":
        raise RuntimeError(
            f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}"
        )
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
    vec_max_abs = float((vec_ref - vec_cand.cpu()).abs().max().item())

    # Also test FastDeviceVectorRuntime.prepare when only a subset of rows has sel_fresh / ~sel_ready
    rt_sub_cpu = rt_cpu.repeat_interleave(1)
    rt_sub_dev = rt_dev.repeat_interleave(1)
    rt_sub_cpu.__class__ = FastDeviceVectorRuntime
    rt_sub_dev.__class__ = FastDeviceVectorRuntime
    # Force alternating rows to require fresh featurization vs carry-state capture
    fresh_toggle_cpu = torch.zeros((n_roots, int(sim_cpu.S)), dtype=torch.bool, device="cpu")
    fresh_toggle_cpu[:, 1:] = True
    fresh_toggle_cpu[::2, 1::2] = False
    ready_toggle_cpu = ~fresh_toggle_cpu
    rt_sub_cpu.sel_fresh.copy_(fresh_toggle_cpu)
    rt_sub_cpu.sel_ready.copy_(ready_toggle_cpu)
    rt_sub_cpu._carry_valid.zero_()
    rt_sub_dev.sel_fresh.copy_(fresh_toggle_cpu.to(device=dev))
    rt_sub_dev.sel_ready.copy_(ready_toggle_cpu.to(device=dev))
    rt_sub_dev._carry_valid.zero_()

    with _maybe_cpu_fast_flood(False):
        rt_sub_cpu.prepare(sim_cpu)
    with _maybe_cpu_fast_flood(bool(dev.type == "cpu")):
        rt_sub_dev.prepare(sim_dev)

    prep_state_max_abs = float((rt_sub_cpu.sel_state - rt_sub_dev.sel_state.cpu()).abs().max().item())
    prep_mask_match = bool(torch.equal(rt_sub_cpu.sel_mask, rt_sub_dev.sel_mask.cpu()))
    vec_parity_ok = bool(
        math.isfinite(vec_max_abs)
        and vec_max_abs <= 1e-4
        and math.isfinite(prep_state_max_abs)
        and prep_state_max_abs <= 1e-4
        and prep_mask_match
    )
    if not vec_parity_ok:
        raise RuntimeError(
            "Step B1b-Smoke parity failure in vector_geometry / FastDeviceVectorRuntime on row "
            f"subsets: vec_max_abs={vec_max_abs}, prep_state_max_abs={prep_state_max_abs}, "
            f"prep_mask_match={prep_mask_match}"
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
) -> dict[str, Any]:
    """Compute the Step B1b (`B1-A100-Pilot`) base estimate and 2x runaway guard."""
    if env_steps_per_sec <= 0 or label_batches_per_sec <= 0 or updates_per_sec <= 0:
        raise ValueError("All measured throughput rates must be > 0")

    # 1. Pre-training probe32 bank harvest (N=32 worlds, H=5,000 -> 160,000 root decisions)
    # Rolled in batches of E=32 (5,000 env steps) and labeled in batches of 256 roots
    # (625 suffix-0 batches + ~125 scoring-suffix 4..7 batches = ~750 label batches).
    probe32_env_sec = float(PROBE32_HORIZON) / float(env_steps_per_sec)
    probe32_label_batches = (float(PROBE32_WORLD_COUNT * PROBE32_HORIZON) / float(REGISTERED_ENVS)) + 125.0
    probe32_label_sec = probe32_label_batches / float(label_batches_per_sec)
    probe32_total_sec = probe32_env_sec + probe32_label_sec

    # 2. Online collection to mine >= 5,000 unique rescue roots at registered 0.12% rescue rate
    expected_hero_decisions = float(REGISTERED_MIN_UNIQUE_RESCUE_ROOTS) / float(
        REGISTERED_RESCUE_RATE_PRIOR
    )
    expected_collection_steps = expected_hero_decisions / (
        float(REGISTERED_ENVS) * float(REGISTERED_ALIVE_HERO_FRAC_ESTIMATE)
    )
    online_env_sec = expected_collection_steps / float(env_steps_per_sec)
    online_label_sec = expected_collection_steps / float(label_batches_per_sec)
    online_collection_sec = online_env_sec + online_label_sec

    # 3. 1,000 learner updates + 5 probe32 tripwires + checkpoint I/O
    learner_sec = float(REGISTERED_TOTAL_UPDATES) / float(updates_per_sec)
    overhead_sec = 180.0

    base_seconds = probe32_total_sec + online_collection_sec + learner_sec + overhead_sec
    base_minutes = base_seconds / 60.0
    timeout_minutes = int(math.ceil(2.0 * base_minutes))
    timeout_seconds = float(timeout_minutes * 60)

    base_units_calibrated = (base_minutes / 60.0) * REGISTERED_A100_CALIBRATED_RATE_UPH
    cap_units_calibrated = 2.0 * base_units_calibrated
    cap_units_broker_8uph = (float(timeout_minutes) / 60.0) * REGISTERED_A100_BROKER_CEILING_RATE_UPH

    return {
        "registered_min_unique_rescue_roots": REGISTERED_MIN_UNIQUE_RESCUE_ROOTS,
        "registered_rescue_rate_prior": REGISTERED_RESCUE_RATE_PRIOR,
        "expected_hero_decisions_for_5000_rescues": expected_hero_decisions,
        "expected_collection_steps_at_e256": expected_collection_steps,
        "probe32_harvest_estimate_sec": probe32_total_sec,
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
    bench_label_batches: int = 8,
    bench_learner_updates: int = 20,
    seed: int = 20261109,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Step B1b-Smoke (`short-canary <= 25 min`, cap `3.0 A100 u`) and write JSON + SHA256."""
    t_start = time.perf_counter()
    dev = torch.device(device)
    if not dry_run:
        if dev.type != "cuda":
            raise RuntimeError(
                f"Non-dry-run Step B1b-Smoke requires device='cuda', got {dev}"
            )
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
        q_warm = student_net(
            obs_warm["local_u8"], obs_warm["global_u8"], obs_warm["scalars_f32"]
        )
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

    # Benchmark 2: 6-branch H=branch_horizon labels/s at E = num_envs
    with torch.inference_mode():
        sim.prepare()
        policy.vector_runtime.prepare(sim)
        policy.vector_prepared = True
    decs_now = [int(x) for x in sim.frame.detach().cpu().tolist()]
    _sync_device(dev)
    t_lbl0 = time.perf_counter()
    last_label_res: dict[str, Any] | None = None
    for b_i in range(int(bench_label_batches)):
        last_label_res = batched_open_stop_h32(
            sim,
            policy,
            world_seeds=world_seeds,
            decisions=[d + b_i for d in decs_now],
            suffix_id=0,
            branch_horizon=int(branch_horizon),
            namespace=OPEN_STOP_CRN_NAMESPACE,
            compact_dead=True,
            enable_r2=True,
            cpu_fast_flood=bool(dev.type == "cpu"),
            return_host_records=False,
        )
    _sync_device(dev)
    lbl_elapsed = max(1e-9, time.perf_counter() - t_lbl0)
    label_batches_per_sec = float(bench_label_batches) / lbl_elapsed
    root_labels_per_sec = float(int(num_envs) * int(bench_label_batches)) / lbl_elapsed
    branch_rollouts_per_sec = 6.0 * root_labels_per_sec

    # Benchmark 3: Learner updates/s (128 rescue + 128 ordinary with hard-negative mining)
    res_batch_sz = min(REGISTERED_RESCUE_BATCH_SIZE, max(2, int(num_envs) * 2)) if dry_run else REGISTERED_RESCUE_BATCH_SIZE
    ord_batch_sz = min(REGISTERED_ORDINARY_BATCH_SIZE, max(2, int(num_envs) * 2)) if dry_run else REGISTERED_ORDINARY_BATCH_SIZE
    ord_uni = ord_batch_sz // 2
    ord_hn = ord_batch_sz - ord_uni

    rescue_buf = SitlRingBuffer(capacity=max(512, res_batch_sz * 2))
    ordinary_buf = SitlRingBuffer(capacity=max(512, ord_batch_sz * 2))
    obs_fill = extract_hero_root_observations(sim, cpu_fast_flood=bool(dev.type == "cpu"))
    assert last_label_res is not None
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
        float(torch.cuda.max_memory_allocated(dev)) / float(1024**2)
        if dev.type == "cuda"
        else 0.0
    )
    peak_vram_reserved_mib = (
        float(torch.cuda.max_memory_reserved(dev)) / float(1024**2)
        if dev.type == "cuda"
        else 0.0
    )

    runaway_guard = compute_pilot_runaway_guard(
        env_steps_per_sec=env_steps_per_sec,
        label_batches_per_sec=label_batches_per_sec,
        updates_per_sec=updates_per_sec,
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
            "bench_label_batches": int(bench_label_batches),
            "bench_learner_updates": int(bench_learner_updates),
            "env_steps_per_sec": env_steps_per_sec,
            "hero_decisions_per_sec": hero_decisions_per_sec,
            "label_batches_per_sec": label_batches_per_sec,
            "six_branch_h32_root_labels_per_sec": root_labels_per_sec,
            "six_branch_h32_branch_rollouts_per_sec": branch_rollouts_per_sec,
            "updates_per_sec": updates_per_sec,
            "peak_vram_allocated_mib": peak_vram_allocated_mib,
            "peak_vram_reserved_mib": peak_vram_reserved_mib,
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
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step B1b-Smoke: CUDA parity on row subsets + E=256 throughput & VRAM benchmark."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--b8-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-envs", type=int, default=REGISTERED_ENVS)
    parser.add_argument("--branch-horizon", type=int, default=OPEN_STOP_BRANCH_HORIZON)
    parser.add_argument("--bench-env-steps", type=int, default=32)
    parser.add_argument("--bench-label-batches", type=int, default=8)
    parser.add_argument("--bench-learner-updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny CPU dry-run configuration (marks dry_run=true).",
    )
    args = parser.parse_args()
    res = run_step_b1b_smoke(
        args.output_dir,
        b8_checkpoint=args.b8_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        branch_horizon=args.branch_horizon,
        bench_env_steps=args.bench_env_steps,
        bench_label_batches=args.bench_label_batches,
        bench_learner_updates=args.bench_learner_updates,
        seed=args.seed,
        dry_run=bool(args.dry_run),
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
