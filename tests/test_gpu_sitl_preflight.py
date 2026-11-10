"""Unit tests for Step B1a (``M0-GPU-SITL-Preflight``) in ``research/gpu_sitl/``."""

from __future__ import annotations

import numpy as np

from research.bl_served_search import ceiling
from research.devicesim_v2_parity.harness import prepare_cpu_policy_phase
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sitl.open_stop_gpu import (
    MARGIN_DELTA_DEFAULT,
    batched_open_stop_h32,
    evaluate_terminal_openness_device,
    margin_loss_gradient_check,
    openness_row_to_dict,
)
from research.gpu_sitl.preflight import (
    LONG_HERO_BOX_HEIGHT,
    advance_to_next_live_hero_seam,
    build_watch6_captured_worlds,
    evaluate_cpu_open_stop_all_actions,
    resolve_product_checkpoints,
    snapshot_worlds_to_device,
)
from research.gpu_sitl.seed_check import (
    PREFLIGHT_NAMESPACE,
    audit_preflight_seeds,
    preflight_seeds,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.t3_rollout_labels_20261015.bank_engine import prepared_q
from research.watch_replay.open_stop import (
    OPEN_STOP_CRN_NAMESPACE,
    evaluate_terminal_openness,
)
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_scenarios import inject_serpentines


def test_seed_audit_disjointness() -> None:
    """Verify that gpu-sitl-preflight-20261109-v1 collides with no registered seeds."""
    report = audit_preflight_seeds(index_horizon=500, scan_code_roots=[])
    assert report["namespace"] == PREFLIGHT_NAMESPACE
    assert report["ok"] is True
    assert report["internal_256_unique"] is True
    assert report["allocated_unique"] is True
    assert report["collisions_allocated"] == {}
    assert report["collisions_prefix_256"] == {}
    assert len(report["natural_world_seeds"]) == 4
    assert len(report["long_hero_world_seeds"]) == 4


def test_terminal_openness_device_parity_short_and_long_hero() -> None:
    """Verify evaluate_terminal_openness_device matches CPU evaluate_terminal_openness."""
    b8_path, frp3_path = resolve_product_checkpoints()
    seeds = preflight_seeds((0, 4))
    worlds, dev_policy_tpl = build_watch6_captured_worlds(
        seeds,
        b8_path,
        frp3_path,
        long_hero_profiles=[
            [3, 3, 3, 3, 3, 3],
            [840, 64, 48, 32, 24, 16],
        ],
    )
    for w in worlds:
        prepare_cpu_policy_phase(w.sim)

    for sim_cls in (CTDeviceSim, DeviceSim):
        dev_sim, _ = snapshot_worlds_to_device(
            worlds,
            dev_policy_tpl,
            sim_cls=sim_cls,
            device="cpu",
            vector_prepared=False,
        )
        dev_open = evaluate_terminal_openness_device(dev_sim)
        for e_idx, w in enumerate(worlds):
            cpu_info = evaluate_terminal_openness(w.sim)
            gpu_info = openness_row_to_dict(dev_open, e_idx)
            assert gpu_info["length"] == cpu_info["length"]
            assert gpu_info["advisory_any"] == cpu_info["advisory_any"]
            assert gpu_info["v8_need"] == cpu_info["v8_need"]
            assert gpu_info["v8_best_safe_area"] == cpu_info["v8_best_safe_area"]
            assert gpu_info["max_pocket_r"] == cpu_info["max_pocket_r"]
            assert gpu_info["un_enclosed"] == cpu_info["un_enclosed"]
        assert int(dev_open["length"][1].item()) >= 800

    # Dead-hero row returns closed openness.
    dead_sim = dev_sim.clone()
    dead_sim.alive[0, 0] = False
    dead_open = evaluate_terminal_openness_device(dead_sim)
    dead_dict = openness_row_to_dict(dead_open, 0)
    assert dead_dict["length"] == 0
    assert dead_dict["advisory_any"] is False
    assert dead_dict["un_enclosed"] is False


def test_batched_open_stop_compacted_vs_uncompacted_and_cpu() -> None:
    """Verify compacted == uncompacted on device and parity with CPU open_stop_batch."""
    b8_path, frp3_path = resolve_product_checkpoints()
    seeds = preflight_seeds((0, 4))
    worlds, dev_policy_tpl = build_watch6_captured_worlds(
        seeds,
        b8_path,
        frp3_path,
    )
    # Inject a long hero (L=800) into world 1 so one root has L >= 800 and fatal right-turn actions.
    inject_serpentines([worlds[1].sim], [800, 64, 48, 32, 24, 16], box_height=LONG_HERO_BOX_HEIGHT)
    worlds[1].sim._rebuild_env_grids(0, food=True)
    worlds[1].runtime.bind(worlds[1].sim, np.array([[False, True, True, True, True, True]]))

    for w in worlds:
        advance_to_next_live_hero_seam(w)

    dev_sim, dev_policy = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device="cpu",
        vector_prepared=False,
    )
    decisions = [int(w.sim.frame[0]) for w in worlds]
    horizon = 6

    res_compacted = batched_open_stop_h32(
        dev_sim,
        dev_policy,
        world_seeds=seeds,
        decisions=decisions,
        branch_horizon=horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        cpu_fast_flood=True,
    )
    res_uncompacted = batched_open_stop_h32(
        dev_sim,
        dev_policy,
        world_seeds=seeds,
        decisions=decisions,
        branch_horizon=horizon,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=False,
        cpu_fast_flood=False,
    )

    assert np.array_equal(res_compacted["certified_matrix"], res_uncompacted["certified_matrix"])
    assert np.array_equal(
        res_compacted["raw_survives_matrix"], res_uncompacted["raw_survives_matrix"]
    )

    ego_cfg = EgoRasterConfig(version="bv8")
    for e_idx, w in enumerate(worlds):
        r_comp = res_compacted["roots"][e_idx]
        r_uncomp = res_uncompacted["roots"][e_idx]
        assert r_comp["chosen"] == r_uncomp["chosen"]
        assert r_comp["all_outcomes"] == r_uncomp["all_outcomes"]

        w.runtime.prepare(w.sim)
        w.sim._policy_phase_prepared = True
        obs = build_ego_raster(
            w.sim, ego_cfg, backend="numba", rows=np.array([[0, 0]], dtype=np.int64)
        )
        q = np.asarray(prepared_q(w, obs), dtype=np.float32)
        advisory = np.asarray(w.sim.get_advisory_action_mask()[0, 0], dtype=bool)
        legal, _ = ceiling._legal(q, advisory)
        cpu_res = evaluate_cpu_open_stop_all_actions(
            w,
            decisions[e_idx],
            q,
            legal,
            branch_horizon=horizon,
            namespace=OPEN_STOP_CRN_NAMESPACE,
        )
        assert r_comp["chosen"] == cpu_res["chosen"]
        for a in range(6):
            assert r_comp["all_outcomes"][a]["certified"] == cpu_res["all_outcomes"][a]["certified"]
            assert (
                r_comp["all_outcomes"][a]["raw_survives"]
                == cpu_res["all_outcomes"][a]["raw_survives"]
            )


def test_margin_loss_gradient_check_b8_s3() -> None:
    """Verify finite gradients on all Ego2sNet parameters under the registration margin loss."""
    b8_path, _ = resolve_product_checkpoints()
    report = margin_loss_gradient_check(
        str(b8_path),
        delta=MARGIN_DELTA_DEFAULT,
        device="cpu",
    )
    assert report["loss_finite"] is True
    assert report["loss_value"] > 0.0
    assert report["all_grads_present"] is True
    assert report["all_grads_finite"] is True
    assert report["total_param_tensors"] == 26
    assert report["total_scalar_params"] == 1_638_263
    assert report["non_value_zero_param_tensors"] == 0
    assert report["value_head_zero_param_tensors"] == 2
    assert report["value_head_scalar_params"] == 257
