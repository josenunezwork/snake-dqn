"""Unit tests for Step B1a (``M0-GPU-SITL-Preflight``) in ``research/gpu_sitl/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.bl_served_search import ceiling
from research.devicesim_v2_parity.harness import prepare_cpu_policy_phase
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sitl.open_stop_gpu import (
    MARGIN_DELTA_DEFAULT,
    RESCUE_MARGIN_DELTA_DEFAULT,
    batched_open_stop_h32,
    evaluate_terminal_openness_device,
    freeze_ego2s_value_head,
    margin_loss_gradient_check,
    openness_row_to_dict,
)
from research.gpu_sitl.preflight import (
    LONG_HERO_BOX_HEIGHT,
    MATURE_EXTENSION_SPECS,
    advance_to_next_live_hero_seam,
    build_extension_worlds,
    build_watch6_captured_worlds,
    evaluate_cpu_open_stop_all_actions,
    resolve_product_checkpoints,
    snapshot_worlds_to_device,
)
from research.gpu_sitl.seed_check import (
    EXTENSION_MATURE_WORLD_INDICES,
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
    assert len(report["extension_mature_world_seeds"]) == len(EXTENSION_MATURE_WORLD_INDICES)
    assert len(report["all_allocated_seeds"]) == 40


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
    """Verify finite non-zero gradients on all trainable non-value Ego2sNet parameters."""
    from src.model.ego2s_network import Ego2sNet

    net = Ego2sNet()
    freeze_info = freeze_ego2s_value_head(net)
    assert freeze_info["frozen_param_tensors"] == 2
    assert freeze_info["frozen_scalar_params"] == 257
    assert net.value.weight.requires_grad is False
    assert net.value.bias.requires_grad is False

    b8_path, _ = resolve_product_checkpoints()
    report = margin_loss_gradient_check(
        str(b8_path),
        delta=MARGIN_DELTA_DEFAULT,
        device="cpu",
        freeze_value_head=False,
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


def test_surviving_but_enclosed_branch_device_vs_cpu() -> None:
    """Verify surviving-but-enclosed branches (both v8_open=False and pocket_open=False) vs CPU."""
    b8_path, frp3_path = resolve_product_checkpoints()
    specs = (MATURE_EXTENSION_SPECS[0], MATURE_EXTENSION_SPECS[9])
    worlds, dev_policy_tpl, meta_rows = build_extension_worlds(
        b8_path,
        frp3_path,
        specs=specs,
        device="cpu",
    )
    dev_sim, dev_pol = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device="cpu",
        vector_prepared=False,
    )
    decisions = [int(m["decision_frame"]) for m in meta_rows]
    world_seeds = [int(m["world_seed"]) for m in meta_rows]

    gpu_res = batched_open_stop_h32(
        dev_sim,
        dev_pol,
        world_seeds=world_seeds,
        decisions=decisions,
        branch_horizon=32,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=True,
        return_host_records=True,
    )

    ego_cfg = EgoRasterConfig(version="bv8")
    enclosed_count = 0
    v8_false_count = 0
    pocket_false_count = 0
    for e_idx, w in enumerate(worlds):
        w.runtime.prepare(w.sim)
        w.sim._policy_phase_prepared = True
        obs = build_ego_raster(
            w.sim,
            ego_cfg,
            backend="numba",
            rows=np.array([[0, 0]], dtype=np.int64),
        )
        q = np.asarray(prepared_q(w, obs), dtype=np.float32)
        adv = np.asarray(w.sim.get_advisory_action_mask()[0, 0], dtype=bool)
        legal, _ = ceiling._legal(q, adv)
        cpu_res = evaluate_cpu_open_stop_all_actions(
            w,
            decisions[e_idx],
            q,
            legal,
            branch_horizon=32,
            namespace=OPEN_STOP_CRN_NAMESPACE,
            matched_rng=True,
        )
        g_root = gpu_res["roots"][e_idx]
        assert g_root["chosen"] == cpu_res["chosen"]
        assert g_root["bare"] == cpu_res["bare"]
        assert g_root["chosen"] != g_root["bare"]
        for a in range(6):
            c_out = cpu_res["all_outcomes"][a]
            g_out = g_root["all_outcomes"][a]
            assert g_out["certified"] == c_out["certified"]
            assert g_out["raw_survives"] == c_out["raw_survives"]
            assert g_out["terminal_openness"] == c_out["terminal_openness"]
            if c_out["raw_survives"] and not c_out["terminal_openness"]["un_enclosed"]:
                enclosed_count += 1
                assert g_out["certified"] is False
                if not c_out["terminal_openness"]["v8_open"]:
                    v8_false_count += 1
                if not c_out["terminal_openness"]["pocket_open"]:
                    pocket_false_count += 1

    assert enclosed_count >= 4
    assert v8_false_count >= 1
    assert pocket_false_count >= 1


def test_cpu_fast_flood_false_parity_h32_mature_root() -> None:
    """Verify cpu_fast_flood=False parity at H=32 on a mature root (L >= 300) vs CPU."""
    b8_path, frp3_path = resolve_product_checkpoints()
    specs = (MATURE_EXTENSION_SPECS[8],)
    worlds, dev_policy_tpl, meta_rows = build_extension_worlds(
        b8_path,
        frp3_path,
        specs=specs,
        device="cpu",
    )
    dev_sim, dev_pol = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device="cpu",
        vector_prepared=False,
    )
    decisions = [int(m["decision_frame"]) for m in meta_rows]
    world_seeds = [int(m["world_seed"]) for m in meta_rows]

    gpu_wavefront = batched_open_stop_h32(
        dev_sim,
        dev_pol,
        world_seeds=world_seeds,
        decisions=decisions,
        branch_horizon=32,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=False,
        return_host_records=True,
    )
    gpu_fast_flood = batched_open_stop_h32(
        dev_sim,
        dev_pol,
        world_seeds=world_seeds,
        decisions=decisions,
        branch_horizon=32,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        compact_dead=True,
        enable_r2=True,
        cpu_fast_flood=True,
        return_host_records=True,
    )
    assert np.array_equal(gpu_wavefront["certified_matrix"], gpu_fast_flood["certified_matrix"])
    assert np.array_equal(
        gpu_wavefront["raw_survives_matrix"], gpu_fast_flood["raw_survives_matrix"]
    )

    ego_cfg = EgoRasterConfig(version="bv8")
    w = worlds[0]
    w.runtime.prepare(w.sim)
    w.sim._policy_phase_prepared = True
    obs = build_ego_raster(
        w.sim,
        ego_cfg,
        backend="numba",
        rows=np.array([[0, 0]], dtype=np.int64),
    )
    q = np.asarray(prepared_q(w, obs), dtype=np.float32)
    adv = np.asarray(w.sim.get_advisory_action_mask()[0, 0], dtype=bool)
    legal, _ = ceiling._legal(q, adv)
    cpu_res = evaluate_cpu_open_stop_all_actions(
        w,
        decisions[0],
        q,
        legal,
        branch_horizon=32,
        namespace=OPEN_STOP_CRN_NAMESPACE,
        matched_rng=True,
    )
    g_root = gpu_wavefront["roots"][0]
    f_root = gpu_fast_flood["roots"][0]
    assert int(w.sim.length[0, 0]) >= 300
    assert g_root["chosen"] == cpu_res["chosen"]
    for a in range(6):
        c_out = cpu_res["all_outcomes"][a]
        g_out = g_root["all_outcomes"][a]
        f_out = f_root["all_outcomes"][a]
        assert g_out["certified"] == c_out["certified"]
        assert g_out["raw_survives"] == c_out["raw_survives"]
        assert g_out["terminal_openness"] == f_out["terminal_openness"]
        assert g_out["terminal_openness"] == c_out["terminal_openness"]


def test_registered_rescue_hinge_gradient_check_frozen_value_head() -> None:
    """Verify registered rescue hinge (delta=0.3002) with frozen value.* on rescue roots."""
    import torch

    b8_path, frp3_path = resolve_product_checkpoints()
    specs = (MATURE_EXTENSION_SPECS[0], MATURE_EXTENSION_SPECS[8])
    worlds, dev_policy_tpl, _ = build_extension_worlds(
        b8_path,
        frp3_path,
        specs=specs,
        device="cpu",
    )
    dev_sim, _ = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device="cpu",
        vector_prepared=False,
    )
    legal_mask = torch.tensor(
        [
            [True, True, False, True, True, False],
            [True, True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    safe_mask = torch.tensor(
        [
            [True, False, False, True, False, False],
            [True, False, False, True, False, False],
        ],
        dtype=torch.bool,
    )
    unsafe_mask = legal_mask & (~safe_mask)
    a_star = torch.tensor([0, 0], dtype=torch.long)

    report = margin_loss_gradient_check(
        str(b8_path),
        sim=dev_sim,
        delta=RESCUE_MARGIN_DELTA_DEFAULT,
        device="cpu",
        freeze_value_head=True,
        legal_mask=legal_mask,
        safe_mask=safe_mask,
        unsafe_mask=unsafe_mask,
        a_star=a_star,
    )
    assert report["delta"] == RESCUE_MARGIN_DELTA_DEFAULT
    assert report["value_head_frozen"] is True
    assert report["freeze_summary"]["frozen_param_tensors"] == 2
    assert report["freeze_summary"]["frozen_scalar_params"] == 257
    assert report["freeze_summary"]["trainable_param_tensors"] == 24
    assert report["freeze_summary"]["trainable_scalar_params"] == 1_638_006
    assert report["all_grads_present"] is True
    assert report["all_grads_finite"] is True
    assert report["all_non_value_grads_finite_and_nonzero"] is True
    assert report["non_value_zero_param_tensors"] == 0


def test_review_preflight_findings_pf1_pf2_pf4_pf5_pf6_pf7_pf8(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Regression coverage for confirmed preflight.py findings PF1, PF2, PF4, PF5, PF6, PF7, PF8."""
    import struct
    import sys
    from dataclasses import replace

    import pytest

    import research.gpu_sitl.preflight as preflight_mod
    from src.core.game_config import get_config, initialize_config

    # PF6: empty-denominator agreement rates must default to 0.0, never 1.0.
    empty_summary = preflight_mod.summarize_agreement([], [])
    assert empty_summary["agreement_rate"] == 0.0
    assert empty_summary["cpu_certified_safe_agreement_rate"] == 0.0
    assert empty_summary["exact_int_surviving_branches"]["all_three_rate"] == 0.0
    assert empty_summary["roots_chosen_agreement_rate"] == 0.0

    # PF7: ensure_watch6_v2_game_config must restore min_boost_length, boost_length_cost_frames,
    # and arena_type even when mechanics_version/num_snakes/input_size already match.
    preflight_mod.ensure_watch6_v2_game_config()
    cfg_before = get_config()
    initialize_config(
        replace(
            cfg_before,
            game=replace(
                cfg_before.game,
                min_boost_length=9,
                boost_length_cost_frames=7,
                arena_type="circular",
            ),
        )
    )
    preflight_mod.ensure_watch6_v2_game_config()
    cfg_after = get_config()
    assert cfg_after.game.min_boost_length == 5
    assert cfg_after.game.boost_length_cost_frames == 3
    assert cfg_after.game.arena_type == "rectangular"

    # PF2 + PF4 + PF8: freeze-cohort, shard bounds, overlapping-shard refusal, CLI --delta,
    # deterministic gzip mtime=0, and create-only sidecars (.sha256 + .complete).
    two_specs = (MATURE_EXTENSION_SPECS[0], MATURE_EXTENSION_SPECS[9])
    monkeypatch.setattr(preflight_mod, "MATURE_EXTENSION_SPECS", two_specs)
    monkeypatch.setattr(
        preflight_mod,
        "EXTENSION_MATURE_WORLD_INDICES",
        tuple(int(s["world_index"]) for s in two_specs),
    )
    monkeypatch.setattr(preflight_mod, "EXT_MIN_SURVIVING_ENCLOSED", 2)
    monkeypatch.setattr(preflight_mod, "EXT_MIN_NEAR_THRESHOLD_SURVIVING", 1)
    monkeypatch.setattr(preflight_mod, "EXT_MIN_RESCUE_ROOTS", 2)
    monkeypatch.setattr(preflight_mod, "EXT_MATCHED_RNG_CERT_AGREE_MIN", 12)
    monkeypatch.setattr(preflight_mod, "EXT_CROSS_RNG_CERT_AGREE_MIN", 10)
    monkeypatch.setattr(preflight_mod, "EXT_CROSS_RNG_CHOSEN_AGREE_MIN", 2)
    monkeypatch.setattr(
        preflight_mod,
        "audit_preflight_seeds",
        lambda: audit_preflight_seeds(index_horizon=64, scan_code_roots=[]),
    )

    ext_dir = tmp_path / "ext_out"
    freeze_res = preflight_mod.freeze_mature_cohort_cpu(out_dir=ext_dir, branch_horizon=32)
    cohort_file = ext_dir / preflight_mod.FROZEN_COHORT_FILENAME
    assert cohort_file.is_file()
    assert Path(f"{cohort_file}.sha256").is_file()
    assert Path(f"{cohort_file}.complete").is_file()
    assert freeze_res["cohort_sha256"] in Path(f"{cohort_file}.complete").read_text()

    # PF2: re-running freeze_mature_cohort_cpu in same out_dir must raise FileExistsError
    with pytest.raises(FileExistsError):
        preflight_mod.freeze_mature_cohort_cpu(out_dir=ext_dir, branch_horizon=32)

    # PF8: invalid shard_start / shard_end bounds must raise ValueError
    with pytest.raises(ValueError):
        preflight_mod.run_extension_shard(
            out_dir=ext_dir, shard_start=1, shard_end=1, branch_horizon=32
        )
    with pytest.raises(ValueError):
        preflight_mod.run_extension_shard(
            out_dir=ext_dir, shard_start=0, shard_end=99, branch_horizon=32
        )

    s0 = preflight_mod.run_extension_shard(
        out_dir=ext_dir, shard_start=0, shard_end=1, branch_horizon=32
    )
    preflight_mod.run_extension_shard(
        out_dir=ext_dir, shard_start=1, shard_end=2, branch_horizon=32
    )

    # PF4: duplicate/overlapping shard file in _shards must be refused by finalize
    dup_shard = ext_dir / "_shards" / "shard_00_02.json"
    dup_shard.write_text(Path(s0["shard_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shard|contiguous|Expected"):
        preflight_mod.finalize_extension_preflight(out_dir=ext_dir)
    dup_shard.unlink()

    # PF4: main() --mode ext-finalize must forward --delta and write create-only deterministic gzip
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--mode",
            "ext-finalize",
            "--out-dir",
            str(ext_dir),
            "--delta",
            "0.1234",
        ],
    )
    rc_ext = preflight_mod.main()
    assert rc_ext == 0
    ext_sum_path = ext_dir / "gpu_sitl_preflight_ext_summary_20261109.json"
    ext_gz_path = ext_dir / "gpu_sitl_preflight_ext_certificates_20261109.json.gz"
    assert Path(f"{ext_sum_path}.sha256").is_file()
    assert Path(f"{ext_sum_path}.complete").is_file()
    assert Path(f"{ext_gz_path}.sha256").is_file()
    assert Path(f"{ext_gz_path}.complete").is_file()
    # Gzip header bytes 4..8 store 32-bit little-endian mtime; deterministic gzip sets mtime=0
    assert struct.unpack("<I", ext_gz_path.read_bytes()[4:8])[0] == 0
    import json

    ext_sum = json.loads(ext_sum_path.read_text(encoding="utf-8"))
    assert abs(float(ext_sum["gradient_check"]["delta"]) - 0.1234) < 1e-6

    with pytest.raises(FileExistsError):
        preflight_mod.finalize_extension_preflight(out_dir=ext_dir)

    # PF1 + PF5: run_preflight must not call inject_serpentines with [3,3,3,3,3,3] on natural4,
    # must enforce long_hero min(hero_length) >= 800, and main() must return non-zero on FAIL.
    seen_serp_lengths: list[list[int]] = []
    orig_inject = preflight_mod.inject_serpentines

    def _spy_inject(sims: Any, lengths: Any, **kwargs: Any) -> Any:
        seen_serp_lengths.append([int(x) for x in lengths])
        return orig_inject(sims, lengths, **kwargs)

    monkeypatch.setattr(preflight_mod, "inject_serpentines", _spy_inject)
    b1a_dir = tmp_path / "b1a_pass"
    b1a_sum = preflight_mod.run_preflight(
        out_dir=b1a_dir,
        natural_ticks=1,
        long_hero_ticks=1,
        branch_horizon=2,
        device="cpu",
    )
    assert [3, 3, 3, 3, 3, 3] not in seen_serp_lengths
    assert b1a_sum["verdict"] == "PASS"
    assert b1a_sum["integrity_pass"] is True
    assert set(b1a_sum["bar_checks"]) == {
        "natural_agreement_ge_99pct",
        "combined_agreement_ge_99pct",
        "all_gradients_finite",
    }
    assert b1a_sum["integrity_checks"]["long_hero_min_length_ge_800"] is True
    assert b1a_sum["integrity_checks"]["wavefront_24certs_exact_agree"] is True
    b1a_gz = b1a_dir / "gpu_sitl_preflight_certificates_20261109.json.gz"
    assert Path(f"{b1a_gz}.complete").is_file()
    assert struct.unpack("<I", b1a_gz.read_bytes()[4:8])[0] == 0

    # P3: if long-hero length drops below 800 while registered bars (>=99.0% + finite non-zero
    # gradients) pass, verdict stays PASS (rc == 0) and integrity_pass reports False.
    monkeypatch.setattr(
        preflight_mod,
        "LONG_HERO_LENGTH_PROFILES",
        (
            [120, 64, 48, 32, 24, 16],
            [840, 72, 56, 40, 28, 20],
            [900, 80, 64, 48, 32, 24],
            [960, 96, 72, 56, 40, 28],
        ),
    )
    b1a_diag_dir = tmp_path / "b1a_integrity_diag"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--mode",
            "b1a",
            "--out-dir",
            str(b1a_diag_dir),
            "--natural-ticks",
            "1",
            "--long-hero-ticks",
            "1",
            "--branch-horizon",
            "1",
        ],
    )
    rc_diag = preflight_mod.main()
    assert rc_diag == 0
    diag_sum = json.loads(
        (b1a_diag_dir / "gpu_sitl_preflight_summary_20261109.json").read_text(encoding="utf-8")
    )
    assert diag_sum["verdict"] == "PASS"
    assert diag_sum["integrity_pass"] is False
    assert diag_sum["integrity_checks"]["long_hero_min_length_ge_800"] is False

    # PF5: when a registered Step B1a bar fails (e.g. non-finite/zero trainable grad), verdict is
    # FAIL and main() returns 1.
    orig_grad_check = preflight_mod.margin_loss_gradient_check

    def _failing_grad_check(*args: Any, **kwargs: Any) -> dict[str, Any]:
        rep = dict(orig_grad_check(*args, **kwargs))
        rep["all_non_value_grads_finite_and_nonzero"] = False
        return rep

    monkeypatch.setattr(preflight_mod, "margin_loss_gradient_check", _failing_grad_check)
    b1a_fail_dir = tmp_path / "b1a_fail"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--mode",
            "b1a",
            "--out-dir",
            str(b1a_fail_dir),
            "--natural-ticks",
            "1",
            "--long-hero-ticks",
            "1",
            "--branch-horizon",
            "1",
        ],
    )
    rc_fail = preflight_mod.main()
    assert rc_fail == 1
    fail_sum = json.loads(
        (b1a_fail_dir / "gpu_sitl_preflight_summary_20261109.json").read_text(encoding="utf-8")
    )
    assert fail_sum["verdict"] == "FAIL"
    assert fail_sum["bar_checks"]["all_gradients_finite"] is False


def test_opus_p3_and_p4_verdicts_use_only_registered_bars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """P3 & P4: B1a and B1a-ext verdicts gate only on registered bars; extras are integrity."""
    import json
    import sys

    import research.gpu_sitl.preflight as preflight_mod

    # P3: Wavefront 23/24 mismatch with >=99.0% natural & combined agreement + finite non-zero grads
    # must yield verdict == "PASS" (exit 0) and integrity_pass == False.
    orig_batched_os = preflight_mod.batched_open_stop_h32

    def _wavefront_one_flip(*args: Any, **kwargs: Any) -> dict[str, Any]:
        out = orig_batched_os(*args, **kwargs)
        if kwargs.get("cpu_fast_flood") is False and out.get("roots"):
            roots_copy = [dict(r) for r in out["roots"]]
            r0 = dict(roots_copy[0])
            outcomes0 = {int(k): dict(v) for k, v in r0["all_outcomes"].items()}
            outcomes0[0]["certified"] = not bool(outcomes0[0]["certified"])
            r0["all_outcomes"] = outcomes0
            roots_copy[0] = r0
            out = {**out, "roots": roots_copy}
        return out

    monkeypatch.setattr(preflight_mod, "batched_open_stop_h32", _wavefront_one_flip)
    p3_dir = tmp_path / "p3_wavefront_23_of_24"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--mode",
            "b1a",
            "--out-dir",
            str(p3_dir),
            "--natural-ticks",
            "1",
            "--long-hero-ticks",
            "1",
            "--branch-horizon",
            "1",
        ],
    )
    rc_p3 = preflight_mod.main()
    assert rc_p3 == 0
    p3_sum = json.loads(
        (p3_dir / "gpu_sitl_preflight_summary_20261109.json").read_text(encoding="utf-8")
    )
    assert p3_sum["verdict"] == "PASS"
    assert p3_sum["bar_checks"] == {
        "natural_agreement_ge_99pct": True,
        "combined_agreement_ge_99pct": True,
        "all_gradients_finite": True,
    }
    assert p3_sum["integrity_pass"] is False
    assert p3_sum["integrity_checks"]["wavefront_24certs_exact_agree"] is False
    assert (
        p3_sum["throughput"]["pure_torch_wavefront_4roots_24certs_sample"]["agreed_certificates"]
        == 23
    )
    monkeypatch.setattr(preflight_mod, "batched_open_stop_h32", orig_batched_os)

    # P4: B1a-ext finalize where each per-field exact-int rate >= 0.98 (189/192 = 0.984375)
    # while joint all_three_rate < 0.98 (183/192 = 0.953125) must PASS verdict and report
    # joint all_three_rate under integrity_checks (integrity_pass == False).
    one_spec = preflight_mod.MATURE_EXTENSION_SPECS[:1]
    monkeypatch.setattr(preflight_mod, "MATURE_EXTENSION_SPECS", one_spec)
    monkeypatch.setattr(
        preflight_mod,
        "EXTENSION_MATURE_WORLD_INDICES",
        tuple(int(s["world_index"]) for s in one_spec),
    )
    monkeypatch.setattr(preflight_mod, "EXT_MIN_SURVIVING_ENCLOSED", 0)
    monkeypatch.setattr(preflight_mod, "EXT_MIN_NEAR_THRESHOLD_SURVIVING", 0)
    monkeypatch.setattr(preflight_mod, "EXT_MIN_RESCUE_ROOTS", 0)

    orig_eval_comp = preflight_mod.evaluate_cpu_cohort_reference

    def _fast_comp(*a: Any, **kw: Any) -> tuple[Any, dict[str, Any], float]:
        roots_c, comp_c, wall_c = orig_eval_comp(*a, **kw)
        comp_c = {**comp_c, "composition_bars_met": True}
        return roots_c, comp_c, wall_c

    monkeypatch.setattr(preflight_mod, "evaluate_cpu_cohort_reference", _fast_comp)
    ext_dir = tmp_path / "p4_ext"
    preflight_mod.freeze_mature_cohort_cpu(out_dir=ext_dir, branch_horizon=1)
    preflight_mod.run_extension_shard(out_dir=ext_dir, shard_start=0, shard_end=1, branch_horizon=1)

    orig_sum_agree = preflight_mod.summarize_agreement

    def _fake_ext_summary(
        cert_rows: list[dict[str, Any]], root_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        base = dict(orig_sum_agree(cert_rows, root_rows))
        per_field = 189.0 / 192.0  # 0.984375 >= 0.98
        joint = 183.0 / 192.0  # 0.953125 < 0.98
        int_block = {
            "denominator": 192,
            "length_rate": per_field,
            "v8_best_safe_area_rate": per_field,
            "max_pocket_r_rate": per_field,
            "all_three_rate": joint,
        }
        base["certificates_total"] = 192
        base["agreed_certificates"] = 192
        base["roots_total"] = 32
        base["roots_chosen_agreed"] = 32
        base["exact_int_surviving_branches"] = dict(int_block)
        base["exact_int_all_certificates"] = dict(int_block)
        return base

    monkeypatch.setattr(preflight_mod, "summarize_agreement", _fake_ext_summary)
    p4_sum = preflight_mod.finalize_extension_preflight(out_dir=ext_dir)
    assert p4_sum["verdict"] == "PASS"
    assert p4_sum["bar_checks"]["part1_matched_rng_exact_int_ge_98pct"] is True
    assert "seed_audit_ok" not in p4_sum["bar_checks"]
    assert p4_sum["integrity_pass"] is False
    assert p4_sum["integrity_checks"]["part1_matched_rng_surviving_all_three_ge_98pct"] is False
    assert p4_sum["integrity_checks"]["part1_matched_rng_all_certs_all_three_ge_98pct"] is False
