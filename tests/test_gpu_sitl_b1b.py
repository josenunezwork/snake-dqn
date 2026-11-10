"""Step B1b tests: review fixes (Findings 1 & 2), seed registration/audit, probe-bank
harvester, online SITL training loop, and smoke entrypoint."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from research.gpu_sitl.open_stop_gpu import (
    batched_open_stop_h32,
    margin_loss_gradient_check,
    select_a_star_nb_on_device,
)
from research.gpu_sitl.preflight import (
    build_watch6_captured_worlds,
    resolve_product_checkpoints,
    snapshot_worlds_to_device,
)
from research.gpu_sitl.probe_bank import (
    evaluate_probe_bank_tripwires,
    harvest_probe32_bank,
    load_probe32_bank,
    verify_file_sha256,
)
from research.gpu_sitl.seed_check import (
    CONTAMINATED_FORBIDDEN_DATA_NAMESPACES,
    PREFLIGHT_NAMESPACE,
    PROBE32_NAMESPACE,
    SCREEN_NAMESPACE,
    SELECTIVITY_NAMESPACE,
    T34_PREFLIGHT_NAMESPACE,
    TRAIN_NAMESPACE,
    audit_b1b_seeds,
    preflight_seeds,
    probe32_seeds,
    screen_seeds,
    selectivity_seeds,
    train_seeds,
)
from research.gpu_sitl.train_sitl import (
    VERDICT_INPUTS_INSUFFICIENT,
    SitlRingBuffer,
    compute_sitl_hinge_loss,
    evaluate_training_set_aliasing_diagnostic,
    run_sitl_training,
    sample_ordinary_minibatch_indices,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.watch_replay.open_stop import evaluation_order_nb


def _host_a_star_from_evaluation_order_nb(
    q_row: np.ndarray,
    legal_row: np.ndarray,
    safe_row: np.ndarray,
    bare_act: int,
) -> int:
    legal_indices = np.flatnonzero(legal_row).tolist()
    order = evaluation_order_nb(legal_indices, int(bare_act), q_row)
    for act in order:
        if bool(safe_row[int(act)]):
            return int(act)
    return -1


def test_finding1_device_a_star_matches_host_evaluation_order_nb_boost_vs_nonboost() -> None:
    """Device ``a_star_actions`` must follow ``evaluation_order_nb``: safe non-boost actions
    ``0..2`` before higher-Q safe boost actions ``3..5`` when ``bare`` is unsafe."""
    # Root 0: bare=3 (boost, unsafe); action 1 (non-boost, Q=0.2) and action 4 (boost, Q=0.9)
    # are both certified safe. evaluation_order_nb must pick action 1 (safe non-boost), NOT 4!
    # Root 1: bare=0 (unsafe); actions 1 (Q=0.1), 2 (Q=0.4), 5 (Q=1.5) are safe -> picks 2.
    # Root 2: bare=1 (unsafe); only boost actions 3 (Q=0.3) and 5 (Q=0.8) are safe -> picks 5.
    # Root 3: bare=4 (boost, SAFE, Q=1.2) and action 0 (non-boost, SAFE, Q=0.5) -> picks 4.
    # Root 4: bare=2 (unsafe); no safe actions -> returns -1.
    # Root 5: bare=0 (unsafe); tie between safe non-boost 1 and 2 (Q=0.5), boost 3 (Q=2.0) -> 1.
    root_q = torch.tensor(
        [
            [-0.5, 0.2, -0.1, 1.8, 0.9, 0.1],
            [2.0, 0.1, 0.4, -0.2, 0.3, 1.5],
            [-0.3, 1.1, -0.2, 0.3, -0.4, 0.8],
            [0.5, 0.2, 0.1, 0.3, 1.2, 0.4],
            [0.9, 0.4, 1.1, 0.2, 0.1, 0.0],
            [1.0, 0.5, 0.5, 2.0, 1.9, 1.8],
        ],
        dtype=torch.float32,
    )
    legal_mask = torch.ones((6, 6), dtype=torch.bool)
    safe_mask = torch.tensor(
        [
            [False, True, False, False, True, False],
            [False, True, True, False, False, True],
            [False, False, False, True, False, True],
            [True, False, False, False, True, False],
            [False, False, False, False, False, False],
            [False, True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    bare_actions = torch.tensor([3, 0, 1, 4, 2, 0], dtype=torch.long)

    device_a_star = select_a_star_nb_on_device(root_q, safe_mask, bare_actions)
    expected = [
        _host_a_star_from_evaluation_order_nb(
            root_q[r].numpy(),
            legal_mask[r].numpy(),
            safe_mask[r].numpy(),
            int(bare_actions[r].item()),
        )
        for r in range(root_q.shape[0])
    ]
    assert expected == [1, 2, 5, 4, -1, 1]
    assert device_a_star.tolist() == expected

    # Randomized stress check over 256 roots ensuring every root with safe non-boost +
    # higher-Q safe boost matches host evaluation_order_nb.
    rng = np.random.default_rng(20261109)
    q_rand = torch.from_numpy(rng.normal(size=(256, 6)).astype(np.float32))
    # Force boost columns 3..5 to have higher Q than 0..2 on the first 128 rows
    q_rand[:128, 3:] += 5.0
    legal_rand = torch.ones((256, 6), dtype=torch.bool)
    bare_rand = torch.argmax(q_rand, dim=1)
    safe_rand = torch.from_numpy(rng.random(size=(256, 6)) > 0.4)
    # On rows 0..63, make bare unsafe, at least one non-boost safe, and at least one boost safe
    for r in range(64):
        safe_rand[r, int(bare_rand[r].item())] = False
        safe_rand[r, r % 3] = True
        other_boost = 3 + ((int(bare_rand[r].item()) - 3 + 1) % 3)
        safe_rand[r, other_boost] = True

    dev_out = select_a_star_nb_on_device(q_rand, safe_rand, bare_rand).tolist()
    host_out = [
        _host_a_star_from_evaluation_order_nb(
            q_rand[r].numpy(),
            legal_rand[r].numpy(),
            safe_rand[r].numpy(),
            int(bare_rand[r].item()),
        )
        for r in range(256)
    ]
    assert dev_out == host_out


def test_findings_1_and_2_in_batched_open_stop_and_margin_loss_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Findings 1 & 2 end-to-end inside ``batched_open_stop_h32`` and
    ``margin_loss_gradient_check`` fallback."""
    from research.gpu_sim_spike.device_policy import DevicePolicy

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")
    device = torch.device("cpu")
    seeds = preflight_seeds((0, 1))
    worlds, dev_policy_tpl = build_watch6_captured_worlds(seeds, b8_path, opp_path)
    ct_sim, policy = snapshot_worlds_to_device(
        worlds,
        dev_policy_tpl,
        sim_cls=CTDeviceSim,
        device="cpu",
        vector_prepared=False,
    )
    decisions = [int(w.sim.frame[0]) for w in worlds]

    # Patch DevicePolicy.evaluate on the root call (sim.E == 2) so root 0 has:
    # - bare = 3 (boost)
    # - Q[4] (boost) = 9.0 > Q[0..2] (non-boost = 1.0, 2.0, 1.5)
    # and we verify that device a_star_actions equals host roots[r]["chosen"].
    orig_evaluate = DevicePolicy.evaluate

    def _patched_evaluate(self_pol, sim_arg):
        q, adv, act = orig_evaluate(self_pol, sim_arg)
        if int(sim_arg.E) == 2:
            q = q.clone()
            adv = torch.ones_like(adv, dtype=torch.bool)
            # Make action 3 have highest Q (bare=3), action 4 second highest (boost=9.0),
            # and non-boost actions 0,1,2 lower (1.0, 2.0, 1.5).
            q[0] = torch.tensor([1.0, 2.0, 1.5, 12.0, 9.0, 8.0], dtype=q.dtype, device=q.device)
            act = q.masked_fill(~adv, -torch.inf).argmax(dim=1)
        return q, adv, act

    monkeypatch.setattr(DevicePolicy, "evaluate", _patched_evaluate)
    import research.gpu_sitl.open_stop_gpu as open_stop_gpu_mod

    orig_openness = open_stop_gpu_mod.evaluate_terminal_openness_device

    def _patched_openness(sim_arg):
        out = dict(orig_openness(sim_arg))
        un_enc = out["un_enclosed"].clone()
        if int(un_enc.numel()) >= 5:
            # Force branch 0..5 of root 0: branch 3 (bare) un_enclosed=False;
            # branch 1 (non-boost Q=2.0) and branch 4 (boost Q=9.0) un_enclosed=True.
            un_enc[0] = False
            un_enc[1] = True
            un_enc[2] = False
            un_enc[3] = False
            un_enc[4] = True
        out["un_enclosed"] = un_enc
        return out

    monkeypatch.setattr(open_stop_gpu_mod, "evaluate_terminal_openness_device", _patched_openness)
    res = batched_open_stop_h32(
        ct_sim,
        policy,
        world_seeds=seeds,
        decisions=decisions,
        suffix_id=0,
        branch_horizon=1,
        compact_dead=False,
        enable_r2=True,
        return_host_records=True,
    )

    # Finding 2: re_rank_target_actions dropped; ordinary_mask == bare_is_safe
    assert "re_rank_target_actions" not in res
    safe_mask = res["safe_mask"]
    bare_actions = res["bare_actions"].clamp(0, 5)
    expected_bare_is_safe = safe_mask[torch.arange(len(seeds)), bare_actions]
    assert torch.equal(res["ordinary_mask"], expected_bare_is_safe)

    # Verify root 0 had bare=3 unsafe, safe non-boost 1 (Q=2.0) and safe boost 4 (Q=9.0),
    # and both device a_star_actions[0] and host evaluation_order_nb chose 1 (NOT 4).
    assert int(res["bare_actions"][0].item()) == 3
    assert bool(safe_mask[0, 3].item()) is False
    assert bool(safe_mask[0, 1].item()) is True
    assert bool(safe_mask[0, 4].item()) is True
    assert int(res["roots"][0]["chosen"]) == 1
    assert int(res["a_star_actions"][0].item()) == 1

    # Finding 1: device a_star_actions matches host evaluation_order_nb on all safe roots
    for r, root_rec in enumerate(res["roots"]):
        if bool(safe_mask[r].any().item()):
            assert int(res["a_star_actions"][r].item()) == int(root_rec["chosen"])

    # Also test margin_loss_gradient_check fallback when a_star is None and both safe non-boost
    # (lower Q) and safe boost (higher Q) exist.
    tac = torch.randint(0, 256, (2, 7, 31, 31), dtype=torch.uint8)
    strat = torch.randint(0, 256, (2, 4, 37, 37), dtype=torch.uint8)
    scalars = torch.zeros((2, 60), dtype=torch.float32)
    legal_m = torch.ones((2, 6), dtype=torch.bool)
    from src.model.ego2s_network import load_ego2s_checkpoint

    raw_ckpt = torch.load(b8_path, map_location="cpu", weights_only=False)
    net = load_ego2s_checkpoint(b8_path, device=device)
    with torch.no_grad():
        net.adv.bias.copy_(torch.tensor([1.0, 2.0, 1.5, 50.0, 40.0, 30.0], dtype=torch.float32))
        q0 = net(tac.float() / 255.0, strat.float() / 255.0, scalars)
    assert int(q0[0].argmax().item()) == 3
    assert float(q0[0, 4].item()) > float(q0[0, 1].item())

    mod_ckpt_path = tmp_path / "b8_boost_biased.pth"
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        raw_ckpt = dict(raw_ckpt)
        raw_ckpt["model_state_dict"] = net.state_dict()
    elif isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        raw_ckpt = dict(raw_ckpt)
        raw_ckpt["state_dict"] = net.state_dict()
    else:
        raw_ckpt = net.state_dict()
    torch.save(raw_ckpt, mod_ckpt_path)

    # bare=3 is unsafe; non-boost 1 (lower Q) and boost 4 (higher Q) are both safe.
    safe_m = torch.tensor(
        [
            [False, True, False, False, True, False],
            [True, False, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    grad_res = margin_loss_gradient_check(
        str(mod_ckpt_path),
        sim=ct_sim,
        a_star=None,
        safe_mask=safe_m,
        legal_mask=legal_m,
        device=device,
    )
    assert grad_res["selected_a_star"] == [1, 0]


def test_b1b_seed_registration_and_disjointness_audit() -> None:
    """Verify all Step B1b namespaces (train, probe32, selectivity, screen, preflight)
    are mutually disjoint and have zero collisions with every repository seed source."""
    report = audit_b1b_seeds(index_horizon=512, scan_code_roots=[])
    assert report["ok"] is True
    assert report["status"] == "CLEAR"
    assert report["union_allocated_unique"] is True
    assert report["cross_collisions_allocated"] == {}
    assert report["cross_collisions_prefix"] == {}
    assert report["contaminated_data_files_opened"] == 0
    assert report["contaminated_namespaces_checked_algebraically"] == list(
        CONTAMINATED_FORBIDDEN_DATA_NAMESPACES
    )

    ns_info = report["namespaces"]
    for ns, expected_count in (
        (PREFLIGHT_NAMESPACE, 40),
        (T34_PREFLIGHT_NAMESPACE, 4),
        (TRAIN_NAMESPACE, 256),
        (PROBE32_NAMESPACE, 32),
        (SELECTIVITY_NAMESPACE, 48),
        (SCREEN_NAMESPACE, 48),
    ):
        assert ns in ns_info
        assert ns_info[ns]["ok"] is True
        assert ns_info[ns]["allocated_count"] == expected_count
        assert len(ns_info[ns]["allocated_seeds"]) == expected_count
        assert ns_info[ns]["repo_collisions_allocated"] == {}
        assert ns_info[ns]["repo_collisions_prefix"] == {}

    assert len(train_seeds()) == 256
    assert len(probe32_seeds()) == 32
    assert len(selectivity_seeds()) == 48
    assert len(screen_seeds()) == 48


def test_probe32_harvester_and_tripwire_evaluator(tmp_path: Path) -> None:
    """Verify dry-run probe32 bank harvest, SHA256 verification, and tripwire evaluation."""
    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # Non-dry-run with reduced parameters must fail closed
    with pytest.raises(ValueError, match="Non-dry-run probe32 harvest"):
        harvest_probe32_bank(
            tmp_path / "bad_probe.pt",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            num_worlds=2,
            horizon=4,
            branch_horizon=2,
            ordinary_per_world=2,
            dry_run=False,
        )

    out_pt = tmp_path / "probe32_dry.pt"
    meta = harvest_probe32_bank(
        out_pt,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=5,
        branch_horizon=2,
        ordinary_per_world=4,
        harvest_batch_roots=4,
        dry_run=True,
    )
    assert meta["dry_run"] is True
    assert meta["evidence_class"] == "DRY_RUN_NON_EVIDENCE"
    assert meta["namespace"] == PROBE32_NAMESPACE
    assert meta["decisive_rescue_roots_count"] >= 1
    assert meta["ordinary_roots_count"] >= 4
    assert Path(f"{out_pt}.sha256").is_file()
    assert Path(f"{out_pt}.complete").is_file()
    assert verify_file_sha256(out_pt) == meta["probe_bank_sha256"]

    bank = load_probe32_bank(out_pt)
    assert bank["verified_sha256"] == meta["probe_bank_sha256"]

    from src.model.ego2s_network import load_ego2s_checkpoint

    student = load_ego2s_checkpoint(b8_path, device="cpu")
    # At init (student == b8-s3), ordinary disagreement in every decile is 0.0 <= 5.0%
    tw_200 = evaluate_probe_bank_tripwires(
        student,
        bank,
        update_step=200,
        check_decisive_rescue_bar=False,
        device="cpu",
    )
    assert tw_200["max_decile_ordinary_disagreement_rate"] == 0.0
    assert tw_200["decile_tripwire_pass"] is True
    assert tw_200["boost_tripwire_pass"] is True
    assert tw_200["all_q_finite"] is True
    assert tw_200["illegal_action_count"] == 0
    assert tw_200["all_pass"] is True

    # At update 1000 (check_decisive_rescue_bar=True), bare b8-s3 has 0% rescue agreement < 70%
    tw_1000 = evaluate_probe_bank_tripwires(
        student,
        bank,
        update_step=1000,
        check_decisive_rescue_bar=True,
        device="cpu",
    )
    assert tw_1000["decisive_rescue_agreement_rate"] == 0.0
    assert tw_1000["decisive_rescue_bar_pass"] is False
    assert tw_1000["all_pass"] is False


def test_sitl_hinge_loss_hard_neg_sampler_and_aliasing_diagnostic() -> None:
    """Verify the pure hinge sum loss, hard-negative priority sampler, and
    STOP_EGO2S_BV8_INPUTS_INSUFFICIENT aliasing diagnostic."""
    b8_path, _ = resolve_product_checkpoints()
    if not Path(b8_path).exists():
        pytest.skip("b8-s3 checkpoint not present")
    from research.gpu_sitl.open_stop_gpu import freeze_ego2s_value_head
    from src.model.ego2s_network import load_ego2s_checkpoint

    device = torch.device("cpu")
    net = load_ego2s_checkpoint(b8_path, device=device)
    freeze_ego2s_value_head(net)

    res_buf = SitlRingBuffer(capacity=16)
    ord_buf = SitlRingBuffer(capacity=16)

    for i in range(8):
        loc = torch.full((7, 31, 31), i * 10, dtype=torch.uint8)
        glb = torch.full((4, 37, 37), i * 5, dtype=torch.uint8)
        sca = torch.zeros((60,), dtype=torch.float32)
        leg = torch.ones((6,), dtype=torch.bool)
        with torch.no_grad():
            q_i = net(loc.unsqueeze(0), glb.unsqueeze(0), sca.unsqueeze(0)).squeeze(0)
        bare_a = int(q_i.argmax().item())
        a_star = (bare_a + 1) % 3
        safe_res = torch.zeros((6,), dtype=torch.bool)
        safe_res[a_star] = True
        unsafe_res = leg & (~safe_res)
        res_buf.add_single(
            local_u8=loc,
            global_u8=glb,
            scalars_f32=sca,
            legal_mask=leg,
            safe_mask=safe_res,
            unsafe_mask=unsafe_res,
            bare_action=bare_a,
            a_star_action=a_star,
            world_seed=100 + i,
            decision_frame=i,
        )

        safe_ord = torch.zeros((6,), dtype=torch.bool)
        safe_ord[bare_a] = True
        safe_ord[(bare_a + 1) % 6] = True
        ord_buf.add_single(
            local_u8=loc,
            global_u8=glb,
            scalars_f32=sca,
            legal_mask=leg,
            safe_mask=safe_ord,
            unsafe_mask=leg & (~safe_ord),
            bare_action=bare_a,
            a_star_action=bare_a,
            world_seed=200 + i,
            decision_frame=i,
        )

    # At b8-s3 init: student picks bare_a on all rescue roots (0% rescue agreement < 80%)
    # and picks bare_a on all ordinary states (0% ordinary flips <= 2.0%) ->
    # triggers STOP_EGO2S_BV8_INPUTS_INSUFFICIENT
    diag = evaluate_training_set_aliasing_diagnostic(
        net,
        res_buf,
        ord_buf,
        device=device,
    )
    assert diag["train_rescue_safe_agreement_rate"] == 0.0
    assert diag["train_ordinary_flip_rate"] == 0.0
    assert diag["stop_ego2s_bv8_inputs_insufficient"] is True
    assert diag["verdict"] == VERDICT_INPUTS_INSUFFICIENT

    # Test pure hinge loss + backward pass with value.* frozen
    rng = np.random.default_rng(42)
    ord_idx, hn_stats = sample_ordinary_minibatch_indices(
        ord_buf,
        net,
        rng,
        uniform_count=4,
        hard_neg_count=4,
        device=device,
    )
    assert len(ord_idx) == 8
    assert hn_stats["uniform_sampled"] == 4

    res_batch = res_buf.gather_batch(range(4), device)
    ord_batch = ord_buf.gather_batch(ord_idx[:4], device)
    net.train()
    net.zero_grad(set_to_none=True)
    loss_dict = compute_sitl_hinge_loss(net, res_batch, ord_batch)
    assert bool(torch.isfinite(loss_dict["total_loss"]).item()) is True
    assert float(loss_dict["loss_rescue"].item()) > 0.3002
    assert abs(float(loss_dict["loss_ordinary"].item()) - 0.0632) < 0.0632 + 1.0
    loss_dict["total_loss"].backward()
    assert net.value.weight.grad is None
    assert net.value.bias.grad is None
    for name, p in net.named_parameters():
        if not name.startswith("value."):
            assert p.grad is not None
            assert bool(torch.isfinite(p.grad).all().item()) is True


def test_sitl_training_dry_run_and_resume_determinism(tmp_path: Path) -> None:
    """Run dry-run SITL training uninterrupted vs paused+resumed and verify bit-identical."""
    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    probe_file = tmp_path / "shared_probe32_dry.pt"
    harvest_probe32_bank(
        probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        dry_run=True,
    )

    common_kwargs = dict(
        probe_bank_path=probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=4,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=2,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        decision_window_size=10,
        checkpoint_every_updates=2,
        seed=20261109,
        dry_run=True,
    )

    dir_uninterrupted = tmp_path / "run_uninterrupted"
    res_full = run_sitl_training(dir_uninterrupted, **common_kwargs)
    assert res_full["dry_run"] is True
    assert res_full["evidence_class"] == "DRY_RUN_NON_EVIDENCE"
    assert res_full["counts"]["updates_completed"] == 4
    assert res_full["counts"]["unique_mined_rescue_roots"] >= 2
    assert len(res_full["probe_bank_tripwires"]["history"]) == 2
    assert Path(res_full["result_json_path"]).is_file()
    assert verify_file_sha256(res_full["result_json_path"]) == res_full["result_json_sha256"]

    # Run 2 updates, pause, then resume to 4 updates and verify bit-identical result
    dir_resumed = tmp_path / "run_resumed"
    res_part1 = run_sitl_training(dir_resumed, stop_after_updates=2, **common_kwargs)
    assert res_part1["status"] == "PAUSED_FOR_RESUME"
    assert res_part1["updates_completed"] == 2

    res_part2 = run_sitl_training(dir_resumed, resume=True, **common_kwargs)
    assert res_part2["resumed_from_checkpoint"] is not None
    assert res_part2["counts"] == res_full["counts"]
    assert (
        res_part2["final_student_checkpoint_sha256"] == res_full["final_student_checkpoint_sha256"]
    )
    assert res_part2["update_history"] == res_full["update_history"]
    assert res_part2["probe_bank_tripwires"] == res_full["probe_bank_tripwires"]
    assert (
        res_part2["training_set_aliasing_diagnostic"]
        == res_full["training_set_aliasing_diagnostic"]
    )
    assert res_part2["verdict"] == res_full["verdict"]


def test_step_b1b_smoke_parity_and_prepare_packets(tmp_path: Path) -> None:
    """Verify Step B1b-Smoke row-subset parity, runaway guard, and prepare.py packets."""
    from research.gpu_sitl.prepare import prepare_b1b_packet
    from research.gpu_sitl.smoke import run_step_b1b_smoke

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # Non-dry-run smoke on CPU must fail closed
    with pytest.raises(RuntimeError, match="requires device='cuda'"):
        run_step_b1b_smoke(
            tmp_path / "bad_smoke",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            dry_run=False,
        )

    smoke_dir = tmp_path / "smoke_dry"
    smoke_res = run_step_b1b_smoke(
        smoke_dir,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=(0, 8, 16, 24),
        bench_env_steps=2,
        bench_label_batches=2,
        bench_learner_updates=2,
        dry_run=True,
    )
    assert smoke_res["dry_run"] is True
    assert smoke_res["parity_check"]["ok"] is True
    assert smoke_res["parity_check"]["rebuilt_from_repo_specs_only"] is True
    assert smoke_res["parity_check"]["flood_cuda_v8_decide_row_subset"]["ok"] is True
    assert smoke_res["parity_check"]["flood_cuda_components_row_subset"]["ok"] is True
    vec_par = smoke_res["parity_check"]["vector_geometry_row_subset"]
    assert vec_par["ok"] is True
    assert vec_par["pre_prepare_fresh_rows"] > 0
    assert vec_par["pre_prepare_cap_rows"] > 0
    assert vec_par["pre_prepare_already_ready_rows"] > 0
    assert vec_par["fresh_rows"] > 0
    assert vec_par["cap_rows"] > 0
    assert smoke_res["parity_check"]["batched_open_stop_h32_mature_slice"]["ok"] is True
    assert smoke_res["throughput"]["env_steps_per_sec"] > 0.0
    assert smoke_res["throughput"]["six_branch_h32_root_labels_per_sec"] > 0.0
    assert smoke_res["throughput"]["updates_per_sec"] > 0.0
    assert smoke_res["throughput"]["measured_online_rescue_rate"] >= 0.0
    assert smoke_res["throughput"]["measured_online_rescue_episode_rate"] >= 0.0
    assert smoke_res["throughput"]["buffer_insertion_sec_per_step"] >= 0.0
    assert smoke_res["throughput"]["probe32_scoring_suffix_sec_per_root"] > 0.0
    rg = smoke_res["pilot_runaway_guard"]
    assert rg["probe32_harvest_estimate_sec"] > 0.0
    assert rg["online_buffer_insertion_estimate_sec"] >= 0.0
    assert rg["runaway_guard_max_collection_steps_2x"] > 0
    assert rg["runaway_guard_timeout_minutes_2x"] >= 2 * int(rg["base_estimate_minutes"])
    assert verify_file_sha256(smoke_res["result_json_path"]) == smoke_res["result_json_sha256"]

    # Prepare smoke packet (H100-first with A100 fallback, short-canary <= 25 min)
    pkt_smoke = tmp_path / "pkt_smoke"
    info_smoke = prepare_b1b_packet(
        pkt_smoke,
        commit="origin/main",
        mode="smoke",
        output_dir=tmp_path / "out_smoke",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        attempt="test-01",
    )
    manifest_smoke = json.loads(Path(info_smoke["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_smoke["timeout_minutes"] == 22.5
    assert manifest_smoke["requirements"]["gpu"] == "H100"
    assert manifest_smoke["requirements"]["allowed_gpus"] == ["H100"]
    assert manifest_smoke["requirements"]["prefer"] == ["H100", "A100"]
    assert manifest_smoke["canary"] == {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/gpu_sitl/job.sh",
    }
    assert manifest_smoke["env"]["SNAKE_COLAB_CANARY"] == "1"
    assert manifest_smoke["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/gpu_sitl"
    assert manifest_smoke["env"]["MAX_JOBS"] == "4"
    assert manifest_smoke["technical_envelope"]["registered_a100_unit_cap"] == 3.0
    assert manifest_smoke["budget"]["unit_cap"] == 4.5
    assert manifest_smoke["resume_exemption"]["kind"] == "short-canary"
    assert "resume" not in manifest_smoke
    assert manifest_smoke["required_exports"] == ["smoke_result.json"]
    for req_exp in manifest_smoke["required_exports"]:
        assert (smoke_dir / req_exp).is_file()
        assert (smoke_dir / f"{req_exp}.sha256").is_file()
        assert (smoke_dir / f"{req_exp}.complete").is_file()
    for expected_src in (
        "research/gpu_sitl/open_stop_gpu.py",
        "research/gpu_sitl/seed_check.py",
        "research/gpu_sitl/probe_bank.py",
        "research/gpu_sitl/train_sitl.py",
        "research/gpu_sitl/smoke.py",
        "research/gpu_sitl/job.sh",
        "research/gpu_sitl/requirements.txt",
    ):
        assert expected_src in manifest_smoke["source_paths"]

    # Prepare pilot packet from smoke_result.json (resume block, no resume_exemption)
    pkt_pilot = tmp_path / "pkt_pilot"
    info_pilot = prepare_b1b_packet(
        pkt_pilot,
        commit="origin/main",
        mode="pilot",
        output_dir=tmp_path / "out_pilot",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        smoke_result_json=smoke_res["result_json_path"],
        attempt="test-01",
    )
    manifest_pilot = json.loads(Path(info_pilot["manifest_path"]).read_text(encoding="utf-8"))
    assert "resume_exemption" not in manifest_pilot
    assert manifest_pilot["requirements"]["gpu"] == "H100"
    assert manifest_pilot["requirements"]["allowed_gpus"] == ["H100"]
    assert manifest_pilot["requirements"]["prefer"] == ["H100", "A100"]
    assert manifest_pilot["env"]["MAX_JOBS"] == "4"
    assert manifest_pilot["resume"] == {
        "unit_kind": "update",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    assert manifest_pilot["required_exports"] == [
        "probe32_bank.pt",
        "probe32_bank.pt.meta.json",
        "b8_sitl_student_final.pth",
        "sitl_pilot_result.json",
    ]
    for req_exp in manifest_pilot["required_exports"]:
        assert (smoke_dir / req_exp).is_file()
        assert (smoke_dir / f"{req_exp}.sha256").is_file()
        assert (smoke_dir / f"{req_exp}.complete").is_file()
    assert manifest_pilot["timeout_minutes"] == rg["runaway_guard_timeout_minutes_2x"]


def test_review_answer7_ordinary_striding_no_clone_and_refuse_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Review Answer 7(b) and Findings 2, 3, 8:
    - uniform ordinary striding across the full horizon when horizon > max_s0_buffer_per_world;
    - 1-row CPU state slicing without full-batch GPU clones;
    - refusal to overwrite an existing probe bank;
    - intra-harvest checkpointing and resume."""
    from research.gpu_sitl.probe_bank import (
        reconstruct_batch_from_cpu_row_states,
        select_ordinary_buffer_slot,
        slice_single_row_cpu_state,
    )

    # 1. Verify select_ordinary_buffer_slot over horizon=5000 with max_buffer_per_world=252
    horizon = 5000
    oversample_cap = 252
    sim_buf: list[int] = []
    for t in range(horizon):
        slot = select_ordinary_buffer_slot(
            t,
            len(sim_buf),
            horizon=horizon,
            max_buffer_per_world=oversample_cap,
            existing_frames=sim_buf,
        )
        if slot is None:
            continue
        if slot == len(sim_buf):
            sim_buf.append(t)
        else:
            sim_buf[slot] = t

    assert len(sim_buf) == oversample_cap
    # Every decile of [0, 5000) must have 25 or 26 sampled frames (uniform coverage)
    for d in range(10):
        lo, hi = d * 500, (d + 1) * 500
        count_in_decile = sum(1 for frame_idx in sim_buf if lo <= frame_idx < hi)
        assert 24 <= count_in_decile <= 26, f"Decile {d} had {count_in_decile} frames"

    # 2. Verify 1-row CPU slicing and reconstruction match original batch
    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")
    from research.gpu_sitl.probe_bank import build_watch6_ct_device_bundle

    seeds = probe32_seeds(2)
    sim, policy = build_watch6_ct_device_bundle(seeds, b8_path, opp_path, device="cpu")
    sim.prepare()
    policy.vector_runtime.prepare(sim)
    policy.vector_prepared = True

    row0_state = slice_single_row_cpu_state(sim, policy, 0)
    row1_state = slice_single_row_cpu_state(sim, policy, 1)
    assert int(row0_state["sim_tensors"]["length"].shape[0]) == 1
    rec_sim, rec_pol = reconstruct_batch_from_cpu_row_states(
        [row0_state, row1_state],
        sim,
        policy,
        torch.device("cpu"),
    )
    assert torch.equal(rec_sim.length, sim.length)
    assert torch.equal(rec_sim.bodies, sim.bodies)
    assert torch.equal(rec_pol.vector_runtime.prev_dist, policy.vector_runtime.prev_dist)

    # 3. Verify harvest_probe32_bank refuses to overwrite existing probe bank (Finding 8)
    probe_out = tmp_path / "probe_no_overwrite.pt"
    harvest_probe32_bank(
        probe_out,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        dry_run=True,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing probe32 bank"):
        harvest_probe32_bank(
            probe_out,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            num_worlds=2,
            horizon=4,
            branch_horizon=2,
            ordinary_per_world=3,
            harvest_batch_roots=4,
            dry_run=True,
        )

    # 4. Verify intra-harvest checkpointing and resume produces identical bank
    probe_resumed = tmp_path / "probe_resumed.pt"
    harvest_probe32_bank(
        probe_resumed,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=1,
        checkpoint_every_seconds=0.0,
        dry_run=True,
    )
    ckpt_pt = Path(f"{probe_resumed}.ckpt.pt")
    assert ckpt_pt.is_file()
    assert verify_file_sha256(ckpt_pt)

    probe_from_ckpt = tmp_path / "probe_from_ckpt.pt"
    meta_from_ckpt = harvest_probe32_bank(
        probe_from_ckpt,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=1,
        resume_ckpt_path=ckpt_pt,
        dry_run=True,
    )
    assert meta_from_ckpt["decisive_rescue_roots_count"] >= 1
    assert meta_from_ckpt["ordinary_roots_count"] >= 3


def test_review_answer7_hard_neg_sampling_without_replacement() -> None:
    """Verify Review Answer 7(c) and Finding 5:
    When only 1..5 hard negatives exist in D_ordinary^train, sample_ordinary_minibatch_indices
    draws each hard negative at most once (without replacement) and fills the remainder of the
    64-state hard-negative half uniformly without replacement."""
    b8_path, _ = resolve_product_checkpoints()
    if not Path(b8_path).exists():
        pytest.skip("b8-s3 checkpoint not present")
    from research.gpu_sitl.open_stop_gpu import freeze_ego2s_value_head
    from src.model.ego2s_network import load_ego2s_checkpoint

    device = torch.device("cpu")
    net = load_ego2s_checkpoint(b8_path, device=device)
    freeze_ego2s_value_head(net)

    for num_hard_neg in (1, 3, 5):
        ord_buf = SitlRingBuffer(capacity=256)
        for i in range(128):
            loc = torch.full((7, 31, 31), (i * 7) % 256, dtype=torch.uint8)
            glb = torch.full((4, 37, 37), (i * 3) % 256, dtype=torch.uint8)
            sca = torch.zeros((60,), dtype=torch.float32)
            leg = torch.ones((6,), dtype=torch.bool)
            with torch.no_grad():
                q_i = net(loc.unsqueeze(0), glb.unsqueeze(0), sca.unsqueeze(0)).squeeze(0)
            student_act = int(q_i.argmax().item())
            if i < num_hard_neg:
                # Make bare_action != student_act so this row is a true hard negative
                bare_a = (student_act + 1) % 6
                safe_m = torch.zeros((6,), dtype=torch.bool)
                safe_m[bare_a] = True
                # Alternate Tier-2 (student_act unsafe) and Tier-1 (student_act safe)
                if i % 2 == 1:
                    safe_m[student_act] = True
            else:
                bare_a = student_act
                safe_m = torch.ones((6,), dtype=torch.bool)
            ord_buf.add_single(
                local_u8=loc,
                global_u8=glb,
                scalars_f32=sca,
                legal_mask=leg,
                safe_mask=safe_m,
                unsafe_mask=leg & (~safe_m),
                bare_action=bare_a,
                a_star_action=bare_a,
                world_seed=1000 + i,
                decision_frame=i,
            )

        rng = np.random.default_rng(100 + num_hard_neg)
        ord_idx, stats = sample_ordinary_minibatch_indices(
            ord_buf,
            net,
            rng,
            uniform_count=64,
            hard_neg_count=64,
            device=device,
        )
        assert len(ord_idx) == 128
        hn_half = ord_idx[64:].tolist()
        assert len(hn_half) == 64
        # Zero duplicate indices inside the hard-negative half!
        assert len(set(hn_half)) == 64
        # Every hard negative 0..num_hard_neg-1 was included once
        for hn_i in range(num_hard_neg):
            assert hn_half.count(hn_i) == 1
        assert stats["hard_neg_unsafe_sampled"] + stats["hard_neg_safe_sampled"] == num_hard_neg
        assert stats["hard_neg_uniform_fallback"] == 64 - num_hard_neg


def test_review_answer7_verify_cuda_native_extensions_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Review Answer 7(e) and Finding 4:
    verify_cuda_native_extensions('cuda') fails closed when env vars are missing/wrong or
    when flood_cuda._load() or vector_geometry._load_exact_extension() returns None."""
    from research.gpu_sim_spike import event_cuda, flood_cuda, vector_geometry
    from research.gpu_sitl.open_stop_gpu import verify_cuda_native_extensions

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev=None: (8, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(event_cuda, "_EXTENSION", object(), raising=False)

    # 1. Invalid GPU_SIM_VECTOR_GEOMETRY
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "auto")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    with pytest.raises(RuntimeError, match="GPU_SIM_VECTOR_GEOMETRY"):
        verify_cuda_native_extensions("cuda")

    # 2. Invalid GPU_SIM_VECTOR_FLOOD
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "auto")
    with pytest.raises(RuntimeError, match="GPU_SIM_VECTOR_FLOOD"):
        verify_cuda_native_extensions("cuda")

    # 3. flood_cuda._load() returns None
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    monkeypatch.setattr(flood_cuda, "_load", lambda: None)
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", lambda: object())
    with pytest.raises(RuntimeError, match="flood_cuda=None"):
        verify_cuda_native_extensions("cuda")

    # 4. vector_geometry._load_exact_extension() returns None
    monkeypatch.setattr(flood_cuda, "_load", lambda: object())
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", lambda: None)
    with pytest.raises(RuntimeError, match="vector_geometry=None"):
        verify_cuda_native_extensions("cuda")

    # 5. Both native extensions load -> passes
    monkeypatch.setattr(flood_cuda, "_load", lambda: object())
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", lambda: object())
    res = verify_cuda_native_extensions("cuda")
    assert res["cuda_active"] is True
    assert res["flood_cuda_loaded"] is True
    assert res["vector_geometry_loaded"] is True


def test_review_answer7_verdict_precedence_non_finite_q_and_runaway_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Review Answer 7(f) and Findings 6, 7, 8, 9:
    - seed-disjointness assertion on probe_bank load;
    - non-finite Q halts before backward() and yields STOP_TRIPWIRE_BREACH (taking precedence
      over STOP_EGO2S_BV8_INPUTS_INSUFFICIENT);
    - runaway guard stops cleanly with STOP_RUNAWAY_GUARD_REACHED;
    - FastDeviceVectorRuntime is preserved across compact_sim_and_policy."""
    import research.gpu_sitl.train_sitl as train_sitl_mod
    from research.gpu_sitl.open_stop_gpu import FastDeviceVectorRuntime, compact_sim_and_policy
    from research.gpu_sitl.probe_bank import build_watch6_ct_device_bundle, write_atomic_with_sha256
    from research.gpu_sitl.train_sitl import (
        VERDICT_RUNAWAY_GUARD_STOP,
        VERDICT_TRIPWIRE_STOP,
    )

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # 1. FastDeviceVectorRuntime preservation across compact_sim_and_policy
    sim, pol = build_watch6_ct_device_bundle(train_seeds(4), b8_path, opp_path, device="cpu")
    assert isinstance(pol.vector_runtime, FastDeviceVectorRuntime)
    pol_clone = pol.repeat_interleave(1)
    compact_sim_and_policy(sim, pol_clone, torch.tensor([0, 2], dtype=torch.long))
    assert isinstance(pol_clone.vector_runtime, FastDeviceVectorRuntime)

    # 2.Harvest a valid dry-run probe bank
    probe_file = tmp_path / "probe_valid.pt"
    harvest_probe32_bank(
        probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        dry_run=True,
    )

    # 3. Seed contamination check: inject a train_seeds(4) seed into probe_bank and verify refusal
    bad_probe_file = tmp_path / "probe_contaminated.pt"
    bad_payload = torch.load(probe_file, map_location="cpu", weights_only=False)
    bad_payload["meta"]["world_seeds"] = [int(train_seeds(4)[0])]
    write_atomic_with_sha256(
        bad_probe_file,
        lambda tmp: torch.save(bad_payload, tmp),
        write_complete_marker=True,
    )
    with pytest.raises(ValueError, match="Seed contamination between probe32 bank and training"):
        run_sitl_training(
            tmp_path / "out_contaminated",
            probe_bank_path=bad_probe_file,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            num_envs=4,
            total_updates=1,
            warmup_rescue_roots=1,
            min_unique_rescue_roots=1,
            dry_run=True,
        )

    # 4. Runaway guard stop: max_collection_steps=1 when min_unique_rescue_roots=999
    res_guard = run_sitl_training(
        tmp_path / "out_guard",
        probe_bank_path=probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=2,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=999,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        max_collection_steps=1,
        dry_run=True,
    )
    assert res_guard["runaway_guard_reached"] is True
    assert res_guard["verdict"] == VERDICT_RUNAWAY_GUARD_STOP

    # 5. Non-finite Q in compute_sitl_hinge_loss halts before backward() and yields
    # STOP_TRIPWIRE_BREACH (taking precedence over STOP_EGO2S_BV8_INPUTS_INSUFFICIENT)
    orig_hinge = train_sitl_mod.compute_sitl_hinge_loss
    backward_called = {"count": 0}

    def _nan_q_hinge(*args, **kwargs):
        out = dict(orig_hinge(*args, **kwargs))
        out["q_all_finite"] = torch.tensor(False)
        return out

    monkeypatch.setattr(train_sitl_mod, "compute_sitl_hinge_loss", _nan_q_hinge)
    res_nan_q = run_sitl_training(
        tmp_path / "out_nan_q",
        probe_bank_path=probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=2,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=1,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        dry_run=True,
    )
    assert backward_called["count"] == 0
    assert res_nan_q["counts"]["updates_completed"] == 0
    assert res_nan_q["unconditional_fault_checks"]["non_finite_q_count"] == 1
    assert res_nan_q["probe_bank_tripwires"]["tripwire_breached"] is True
    # Even though aliasing diagnostic at 0 updates triggers stop_ego2s_bv8_inputs_insufficient,
    # STOP_TRIPWIRE_BREACH must take precedence!
    assert (
        res_nan_q["training_set_aliasing_diagnostic"]["stop_ego2s_bv8_inputs_insufficient"] is True
    )
    assert res_nan_q["verdict"] == VERDICT_TRIPWIRE_STOP


def test_cross_session_kill_and_resume_via_prepare_and_verified_ledger(
    tmp_path: Path,
) -> None:
    """Verify Prompt item 3 & 4:
    - prepare_b1b_packet refuses a commit that is not an ancestor of origin/main;
    - killing a dry-run mid-way (with SIGTERM in a subprocess) and staging its outputs into
      a retry packet via prepare_b1b_packet(..., prior_output_dir=...) resumes from the
      already harvested probe32_bank.pt (same SHA256) and latest checkpoint via
      inputs/verified-ledger.json to a bit-identical final result."""
    import os
    import signal
    import subprocess
    import sys
    import time

    from research.gpu_sitl.prepare import prepare_b1b_packet

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # 1. Verify prepare_b1b_packet refuses an unmerged commit (simulated via git commit-tree)
    # Create a detached commit object that is not on origin/main using git commit-tree
    head_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    unmerged_sha = subprocess.check_output(
        ["git", "commit-tree", head_tree, "-p", head_sha, "-m", "ephemeral-unmerged-test"],
        text=True,
    ).strip()
    with pytest.raises(ValueError, match="not an ancestor of origin/main"):
        prepare_b1b_packet(
            tmp_path / "pkt_unmerged",
            commit=unmerged_sha,
            mode="smoke",
            output_dir=tmp_path / "out_unmerged",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
        )

    # 2. Run uninterrupted dry-run in dir_uninterrupted (harvests probe32_bank.pt + 4 updates)
    common_kwargs = dict(
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=4,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=2,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        decision_window_size=10,
        checkpoint_every_updates=2,
        seed=20261109,
        dry_run=True,
    )
    dir_uninterrupted = tmp_path / "uninterrupted"
    res_full = run_sitl_training(dir_uninterrupted, **common_kwargs)
    full_probe_sha = verify_file_sha256(dir_uninterrupted / "probe32_bank.pt")

    # 3. Run attempt-01 in a subprocess and kill it with SIGTERM once u0002 checkpoint is written
    dir_attempt1 = tmp_path / "attempt1_output"
    cmd = [
        sys.executable,
        "-c",
        (
            "import time, signal, sys; "
            "from pathlib import Path; "
            "from research.gpu_sitl.train_sitl import run_sitl_training; "
            f"run_sitl_training({str(dir_attempt1)!r}, stop_after_updates=2, "
            f"b8_checkpoint={str(b8_path)!r}, opponent_checkpoint={str(opp_path)!r}, "
            "device='cpu', num_envs=4, total_updates=4, warmup_rescue_roots=1, "
            "min_unique_rescue_roots=2, rescue_batch_size=2, ordinary_batch_size=4, "
            "ordinary_uniform_half=2, ordinary_hard_neg_half=2, branch_horizon=2, "
            "tripwire_interval=2, decision_window_size=10, checkpoint_every_updates=2, "
            "seed=20261109, dry_run=True); "
            "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(143)); "
            "time.sleep(30)"
        ),
    ]
    proc = subprocess.Popen(cmd)
    ckpt_u2_complete = dir_attempt1 / "checkpoints" / "sitl_state_u0002.pt.complete"
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if (
            ckpt_u2_complete.is_file()
            and (dir_attempt1 / "verified-ledger.json.complete").is_file()
        ):
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert ckpt_u2_complete.is_file()
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=15)
    assert not (dir_attempt1 / "sitl_pilot_result.json").exists()

    # 4. Prepare retry packet (attempt-02) from attempt-01's output directory
    pkt_attempt2 = tmp_path / "pkt_attempt2"
    dir_attempt2 = tmp_path / "attempt2_output"
    info_retry = prepare_b1b_packet(
        pkt_attempt2,
        commit="origin/main",
        mode="pilot",
        output_dir=dir_attempt2,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        base_seconds=600.0,
        prior_output_dir=dir_attempt1,
        attempt="attempt-02",
    )
    manifest_retry = json.loads(Path(info_retry["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_retry["resume"]["ledger_path"] == "inputs/verified-ledger.json"
    staged_ledger_path = pkt_attempt2 / "inputs" / "verified-ledger.json"
    assert staged_ledger_path.is_file()
    staged_ledger = json.loads(staged_ledger_path.read_text(encoding="utf-8"))
    assert staged_ledger["probe32_bank"]["sha256"] == full_probe_sha
    assert staged_ledger["resume_checkpoint"]["relpath"] == "sitl_state_u0002.pt"

    # 5. Resume in fresh output directory dir_attempt2 using staged inputs and verified-ledger.json
    res_attempt2 = run_sitl_training(
        dir_attempt2,
        input_dir=pkt_attempt2 / "inputs",
        ledger_path=staged_ledger_path,
        **common_kwargs,
    )
    # Verify probe32_bank.pt was reused with identical SHA256 and final result is bit-identical
    assert verify_file_sha256(dir_attempt2 / "probe32_bank.pt") == full_probe_sha
    assert res_attempt2["resumed_from_checkpoint"] is not None
    assert (
        res_attempt2["final_student_checkpoint_sha256"]
        == res_full["final_student_checkpoint_sha256"]
    )
    assert res_attempt2["counts"] == res_full["counts"]
    assert res_attempt2["update_history"] == res_full["update_history"]
    assert res_attempt2["probe_bank_tripwires"] == res_full["probe_bank_tripwires"]
    assert (
        res_attempt2["training_set_aliasing_diagnostic"]
        == res_full["training_set_aliasing_diagnostic"]
    )
    assert res_attempt2["verdict"] == res_full["verdict"]


def test_b1_m1_ordinary_roots_dead_unsafe_worlds_suffix47_and_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Findings B1 and M1:
    1. Worlds with many dead or unsafe frames (e.g. only 220 suffix-0-safe ordinary frames out
       of 5,000) still fill >= 210 candidates and span the full 5,000-frame horizon by actual
       decision_frame deterministically.
    2. harvest_probe32_bank scores ordinary candidates on suffixes 4..7, prefers candidates safe
       under suffixes 0 and 4..7 (n_fail_ordinary == 0), backfills by ascending n_fail_ordinary
       when a world has fewer than ordinary_per_world fully certified candidates instead of
       raising, and records per-world certified counts and decile counts.
    3. harvest_probe32_bank raises RuntimeError only when a world has fewer than
       ordinary_per_world suffix-0-safe ordinary frames in total."""
    import research.gpu_sitl.probe_bank as probe_bank_mod
    from research.gpu_sitl.probe_bank import select_ordinary_buffer_slot

    # 1. Test select_ordinary_buffer_slot on a world where 95.6% of frames are dead/unsafe:
    # only 220 suffix-0-safe ordinary frames scattered across [0, 5000).
    horizon = 5000
    oversample_cap = 252
    alive_safe_frames = [
        min(horizon - 1, int(round(i * (horizon - 1) / 219.0))) for i in range(220)
    ]
    sim_buf_sparse: list[int] = []
    for f in alive_safe_frames:
        slot = select_ordinary_buffer_slot(
            f,
            len(sim_buf_sparse),
            horizon=horizon,
            max_buffer_per_world=oversample_cap,
            existing_frames=sim_buf_sparse,
        )
        assert slot is not None
        if slot == len(sim_buf_sparse):
            sim_buf_sparse.append(f)
        else:
            sim_buf_sparse[slot] = f
    assert len(sim_buf_sparse) == 220 >= 210
    assert min(sim_buf_sparse) == 0
    assert max(sim_buf_sparse) == 4999

    # Also test a world with 600 suffix-0-safe ordinary frames clustered unevenly (300 in [0, 600)
    # and 300 spread across [600, 5000)): buffer fills to 252 and spans all 10 horizon deciles!
    uneven_frames = list(range(0, 600, 2)) + [
        600 + int(round(i * 4399 / 299.0)) for i in range(300)
    ]
    sim_buf_uneven: list[int] = []
    for f in uneven_frames:
        slot = select_ordinary_buffer_slot(
            f,
            len(sim_buf_uneven),
            horizon=horizon,
            max_buffer_per_world=oversample_cap,
            existing_frames=sim_buf_uneven,
        )
        if slot is None:
            continue
        if slot == len(sim_buf_uneven):
            sim_buf_uneven.append(f)
        else:
            sim_buf_uneven[slot] = f
    assert len(sim_buf_uneven) == oversample_cap
    for d in range(10):
        lo, hi = d * 500, (d + 1) * 500
        cnt_d = sum(1 for f in sim_buf_uneven if lo <= f < hi)
        assert cnt_d >= 15, f"Decile {d} had only {cnt_d} frames under uneven mortality"

    # 2. Test suffix 4..7 scoring and ascending-n_fail_ordinary backfill in harvest_probe32_bank
    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    orig_eval_suf = probe_bank_mod._evaluate_scoring_suffixes_on_snapshots

    def _mock_eval_suf(snapshots, **kwargs):
        out = orig_eval_suf(snapshots, **kwargs)
        # For ordinary candidates (where bare_action is safe under s0), assign controlled
        # n_fail_ordinary values so world 0 has only 1 fully certified (n_fail=0), 1 with n_fail=1,
        # 1 with n_fail=2, and 1 with n_fail=3, while ordinary_per_world=3 requires backfilling
        # n_fail=0, 1, 2 (and excluding n_fail=3!).
        by_ws: dict[int, int] = {}
        for rec in out:
            bare_a = int(rec["bare_action"])
            if bool(rec["safe_mask_s0"][bare_a].item()):
                ws = int(rec["world_seed"])
                idx_in_w = by_ws.get(ws, 0)
                by_ws[ws] = idx_in_w + 1
                rec["n_fail_ordinary"] = min(4, idx_in_w)
        return out

    monkeypatch.setattr(probe_bank_mod, "_evaluate_scoring_suffixes_on_snapshots", _mock_eval_suf)
    probe_backfill_path = tmp_path / "probe_backfill.pt"
    meta_bf = harvest_probe32_bank(
        probe_backfill_path,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        dry_run=True,
    )
    assert meta_bf["ordinary_roots_count"] == 6
    assert meta_bf["ordinary_fully_certified_count"] == 2  # 1 per world
    assert meta_bf["ordinary_backfilled_count"] == 4  # 2 per world (n_fail=1 and n_fail=2)
    for ws_str, w_cert in meta_bf["per_world_ordinary_certification"].items():
        assert w_cert["selected_count"] == 3
        assert w_cert["selected_fully_certified_count"] == 1
        assert w_cert["selected_backfilled_count"] == 2
        assert w_cert["selected_n_fail_histogram"]["0"] == 1
        assert w_cert["selected_n_fail_histogram"]["1"] == 1
        assert w_cert["selected_n_fail_histogram"]["2"] == 1
        assert w_cert["selected_n_fail_histogram"]["3"] == 0
        assert sum(w_cert["decile_counts"].values()) == 3

    # 3. Verify that when a world has fewer than ordinary_per_world (210) suffix-0-safe ordinary
    # frames in total, non-dry-run harvest_probe32_bank raises RuntimeError.
    monkeypatch.setattr(probe_bank_mod, "PROBE32_WORLD_COUNT", 1)
    monkeypatch.setattr(probe_bank_mod, "PROBE32_HORIZON", 2)
    monkeypatch.setattr(probe_bank_mod, "OPEN_STOP_BRANCH_HORIZON", 2)
    with pytest.raises(RuntimeError, match="suffix-0-safe ordinary roots"):
        harvest_probe32_bank(
            tmp_path / "probe_too_few_s0_ordinary.pt",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            num_worlds=1,
            horizon=2,
            branch_horizon=2,
            ordinary_per_world=210,
            harvest_batch_roots=2,
            dry_run=False,
        )


def test_b2_smoke_rescue_rate_ci95_and_under_20_prior_fallback() -> None:
    """Verify Finding B2:
    - wilson_rate_ci95 computes valid 95% confidence bounds;
    - when fewer than 20 rescue frames are observed, compute_pilot_runaway_guard falls back to
      bounds [0.0006, 0.0024] around the registered prior 0.0012 (or 0.0012 on 0 events) and
      writes the exact rescue_rate_guard_rule into the result;
    - when >= 20 rescue frames are observed, compute_pilot_runaway_guard uses the observed rate."""
    from research.gpu_sitl.smoke import (
        PRIOR_FALLBACK_LOWER_BOUND,
        PRIOR_FALLBACK_UPPER_BOUND,
        REGISTERED_RESCUE_RATE_PRIOR,
        compute_pilot_runaway_guard,
        wilson_rate_ci95,
    )

    ci_0 = wilson_rate_ci95(0, 16000)
    assert ci_0[0] == 0.0 and 0.0 < ci_0[1] < 0.001

    # Case 1: 0 rescues observed (< 20) -> registered prior 0.0012
    g0 = compute_pilot_runaway_guard(
        env_steps_per_sec=100.0,
        label_batches_per_sec=10.0,
        updates_per_sec=10.0,
        measured_online_rescue_rate=0.0,
        measured_alive_hero_decisions=16000,
        measured_rescue_frame_count=0,
    )
    assert g0["rescue_rate_used_for_guard"] == pytest.approx(REGISTERED_RESCUE_RATE_PRIOR)
    assert g0["rescue_rate_guard_rule"] == "prior_fallback_under_20_rescues_zero_observed"

    # Case 2: 2 rescues in 16,000 decisions (rate 0.000125 < 0.0006, count 2 < 20) -> lower bound
    g_low = compute_pilot_runaway_guard(
        env_steps_per_sec=100.0,
        label_batches_per_sec=10.0,
        updates_per_sec=10.0,
        measured_online_rescue_rate=2.0 / 16000.0,
        measured_alive_hero_decisions=16000,
        measured_rescue_frame_count=2,
    )
    assert g_low["rescue_rate_used_for_guard"] == pytest.approx(PRIOR_FALLBACK_LOWER_BOUND)
    assert g_low["rescue_rate_guard_rule"] == "prior_lower_bound_clamped_under_20_rescues"

    # Case 3: 15 rescues in 2,000 decisions (rate 0.0075 > 0.0024, count 15 < 20) -> upper bound
    g_high = compute_pilot_runaway_guard(
        env_steps_per_sec=100.0,
        label_batches_per_sec=10.0,
        updates_per_sec=10.0,
        measured_online_rescue_rate=15.0 / 2000.0,
        measured_alive_hero_decisions=2000,
        measured_rescue_frame_count=15,
    )
    assert g_high["rescue_rate_used_for_guard"] == pytest.approx(PRIOR_FALLBACK_UPPER_BOUND)
    assert g_high["rescue_rate_guard_rule"] == "prior_upper_bound_clamped_under_20_rescues"

    # Case 4: 25 rescues in 16,000 decisions (count 25 >= 20) -> unclamped observed rate
    g_obs = compute_pilot_runaway_guard(
        env_steps_per_sec=100.0,
        label_batches_per_sec=10.0,
        updates_per_sec=10.0,
        measured_online_rescue_rate=25.0 / 16000.0,
        measured_alive_hero_decisions=16000,
        measured_rescue_frame_count=25,
    )
    assert g_obs["rescue_rate_used_for_guard"] == pytest.approx(25.0 / 16000.0)
    assert g_obs["rescue_rate_guard_rule"] == "observed_rate_ge_20_rescues"
    assert len(g_obs["measured_rescue_rate_ci95_wilson"]) == 2
    assert (
        g_obs["measured_rescue_rate_ci95_wilson"][0]
        < (25.0 / 16000.0)
        < g_obs["measured_rescue_rate_ci95_wilson"][1]
    )


def test_b3_canary_check_accepts_pilot_manifest_against_smoke_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Finding B3:
    Build both smoke and pilot manifests via prepare_b1b_packet and run the real
    research.colab_cli.canary.check (and supervisor.validate) against a synthetic smoke receipt
    of the exact shape supervisor.py writes."""
    import subprocess

    from research.colab_cli import canary as colab_canary
    from research.colab_cli import supervisor as colab_supervisor
    from research.gpu_sitl.prepare import ROOT, digest, prepare_b1b_packet
    from research.gpu_sitl.smoke import run_step_b1b_smoke

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # Allow HEAD on the worktree branch before merge to origin/main so `git show HEAD:job.sh`
    # resolves the real committed entrypoint in canary.check and supervisor.validate.
    orig_subproc_run = subprocess.run

    def _allow_head_ancestor(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_subproc_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _allow_head_ancestor)

    smoke_rt_root = tmp_path / "smoke_rt_root"
    smoke_out_dir = smoke_rt_root / "output"
    smoke_res = run_step_b1b_smoke(
        smoke_out_dir,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=(0, 8),
        bench_env_steps=2,
        bench_label_batches=2,
        bench_learner_updates=2,
        dry_run=True,
    )
    assert smoke_res["throughput"]["window_max_hero_length"] >= 160
    assert smoke_res["throughput"]["window_mature_length_ge_160_decisions"] > 0

    pkt_smoke = tmp_path / "pkt_smoke_canary"
    info_smoke = prepare_b1b_packet(
        pkt_smoke,
        commit="HEAD",
        mode="smoke",
        output_dir=smoke_out_dir,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        attempt="canary-01",
    )
    smoke_manifest = json.loads(Path(info_smoke["manifest_path"]).read_text(encoding="utf-8"))
    # Real canary.check on the smoke manifest itself must pass
    colab_canary.check(smoke_manifest, ROOT)

    # Construct the exact receipt directory layout that research/colab_cli/supervisor.py writes
    # when the smoke canary completes on A100 (or H100).
    for hw_kind in ("A100", "H100"):
        smoke_rcpt_dir = tmp_path / f"smoke_receipt_{hw_kind}"
        (smoke_rcpt_dir / "exports").mkdir(parents=True, exist_ok=True)
        for p in smoke_out_dir.iterdir():
            if p.is_file():
                (smoke_rcpt_dir / "exports" / p.name).write_bytes(p.read_bytes())

        smoke_manifest_for_hw = dict(smoke_manifest)
        smoke_manifest_for_hw["output_dir"] = str(smoke_rcpt_dir)
        smoke_manifest_for_hw["requirements"] = {
            "gpu": hw_kind,
            "allowed_gpus": [hw_kind],
            "min_vcpu": 2,
            "est_minutes": smoke_manifest["requirements"]["est_minutes"],
        }
        saved_smoke_manifest_path = smoke_rcpt_dir / "manifest.json"
        saved_smoke_manifest_path.write_text(
            json.dumps(smoke_manifest_for_hw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (smoke_rcpt_dir / "hardware.json").write_text(
            json.dumps(
                {"hardware": {"kind": hw_kind, "name": f"NVIDIA {hw_kind}"}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (smoke_rcpt_dir / "upload-measurement.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": 1.5,
                    "transfer_seconds": 0.9,
                    "source_delivery_seconds": 0.4,
                    "staging_completed": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        from research.colab_cli import runtime as colab_runtime

        exports_list = colab_runtime._exports(
            smoke_rt_root,
            int(smoke_manifest_for_hw["max_export_bytes"]),
            {},
        )
        gen_id = f"gen-canary-{hw_kind.lower()}-01"
        transfers_list = [
            {
                "direction": "upload",
                "remote": f"/content/snake-cli/{gen_id}/inputs/{row['dest']}",
                "status": "verified",
                "size_bytes": row["size"],
                "sha256": row["sha256"],
            }
            for row in colab_canary.input_inventory(smoke_manifest_for_hw)
        ]
        receipt_payload: dict[str, Any] = {
            "schema": "colab-cli-receipt/v1",
            "job_id": smoke_manifest_for_hw["job_id"],
            "commit": smoke_manifest_for_hw["commit"],
            "generation": gen_id,
            "manifest_sha256": digest(saved_smoke_manifest_path),
            "status": "PASS",
            "stop_verified": True,
            "error": None,
            "cleanup_error": None,
            "failure_class": None,
            "hardware": {"kind": hw_kind},
            "exports": exports_list,
            "transfers": transfers_list,
            "terminal": {
                "job_id": smoke_manifest_for_hw["job_id"],
                "generation": gen_id,
                "returncode": 0,
                "status": "completed",
                "export_scanner": {
                    "name": "runtime._exports",
                    "inventory": exports_list,
                },
                "canary_observations": [
                    {
                        "cuda_decisions": 16384,
                        "device": "cuda:0",
                        "package": "research/gpu_sitl",
                        "caller": "research/gpu_sitl/smoke.py:run_step_b1b_smoke",
                        "operation": "torch.argmax",
                        "shape": [256, 6],
                        "root": f"/content/snake-cli/{gen_id}",
                    }
                ],
            },
        }
        receipt_payload["canary_evidence"] = colab_canary.evidence(
            smoke_manifest_for_hw,
            receipt_payload,
            smoke_rcpt_dir,
            ROOT,
        )
        rcpt_file = smoke_rcpt_dir / "receipt.json"
        rcpt_file.write_text(
            json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Now prepare the pilot manifest bound to this smoke canary receipt
        pkt_pilot = tmp_path / f"pkt_pilot_{hw_kind}"
        pilot_out_dir = tmp_path / f"pilot_out_{hw_kind}"
        info_pilot = prepare_b1b_packet(
            pkt_pilot,
            commit="HEAD",
            mode="pilot",
            output_dir=pilot_out_dir,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            smoke_result_json=smoke_res["result_json_path"],
            canary_receipt=rcpt_file,
            attempt=f"pilot-{hw_kind.lower()}-01",
        )
        pilot_manifest = json.loads(Path(info_pilot["manifest_path"]).read_text(encoding="utf-8"))
        assert pilot_manifest["requirements"]["gpu"] == hw_kind
        assert pilot_manifest["requirements"]["allowed_gpus"] == [hw_kind]
        assert "prefer" not in pilot_manifest["requirements"]

        # Verify both real colab_canary.check and colab_supervisor.validate accept pilot_manifest!
        colab_canary.check(pilot_manifest, ROOT)
        colab_supervisor.validate(pilot_manifest, ROOT)


def test_m2_learner_update_pacing_by_unique_rescues_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Finding M2:
    Learner updates are paced solely by required unique rescue roots (no step-count fallback,
    no burst of updates after warmup, and no collection step after the last update). Assert the
    exact collection step at which each learner update fires."""
    import research.gpu_sitl.train_sitl as train_sitl_mod

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    probe_file = tmp_path / "probe_m2.pt"
    harvest_probe32_bank(
        probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=4,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        dry_run=True,
    )

    # Configure a controlled labeler that yields 1 unique rescue root per collection step
    # (on env 0) and ordinary roots on envs 1..3.
    # With warmup_rescue_roots=2, min_unique_rescue_roots=8, total_updates=4:
    # required_unique_rescues_for_update is:
    #   u=1 -> 2 (reached at collection_steps == 2)
    #   u=2 -> 4 (reached at collection_steps == 4)
    #   u=3 -> 6 (reached at collection_steps == 6)
    #   u=4 -> 8 (reached at collection_steps == 8)
    step_counter = {"k": 0}

    def _controlled_labeler(sim_eval, pol_eval, *, world_seeds, decisions, **kwargs):
        step_counter["k"] += 1
        n = len(world_seeds)
        safe_mask = torch.ones((n, 6), dtype=torch.bool)
        unsafe_mask = torch.zeros((n, 6), dtype=torch.bool)
        rescue_mask = torch.zeros((n,), dtype=torch.bool)
        ordinary_mask = torch.ones((n,), dtype=torch.bool)
        bare_actions = torch.ones((n,), dtype=torch.long)
        a_star_actions = torch.ones((n,), dtype=torch.long)
        # Env 0 is a rescue root on every step; envs 1..n-1 are ordinary roots
        safe_mask[0] = torch.tensor([False, True, False, False, False, False])
        unsafe_mask[0] = torch.tensor([True, False, True, True, True, True])
        rescue_mask[0] = True
        ordinary_mask[0] = False
        bare_actions[0] = 0
        a_star_actions[0] = 1
        return {
            "safe_mask": safe_mask,
            "unsafe_mask": unsafe_mask,
            "rescue_mask": rescue_mask,
            "ordinary_mask": ordinary_mask,
            "bare_actions": bare_actions,
            "a_star_actions": a_star_actions,
            "root_q": torch.zeros((n, 6), dtype=torch.float32),
        }

    monkeypatch.setattr(train_sitl_mod, "batched_open_stop_h32", _controlled_labeler)
    res_paced = run_sitl_training(
        tmp_path / "out_paced",
        probe_bank_path=probe_file,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=4,
        warmup_rescue_roots=2,
        min_unique_rescue_roots=8,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        max_collection_steps=20,
        dry_run=True,
    )
    schedule = res_paced["update_pacing_schedule"]
    assert len(schedule) == 4
    assert [u["update"] for u in schedule] == [1, 2, 3, 4]
    assert [u["required_unique_rescue_roots"] for u in schedule] == [2, 4, 6, 8]
    assert [u["unique_mined_rescue_roots"] for u in schedule] == [2, 4, 6, 8]
    assert [u["collection_steps"] for u in schedule] == [2, 4, 6, 8]
    # Zero collection steps after the final update (collection_steps == 8)
    assert res_paced["counts"]["collection_steps"] == 8


def test_m3_verdict_precedence_all_four_probe_conditions_and_missing_key_raises() -> None:
    """Verify Finding M3:
    1. compute_step_b1b_pilot_verdict uses the exact keys written by evaluate_probe_bank_tripwires
       and raises KeyError if any required tripwire key is missing.
    2. At the final update (updates_completed == total_updates), when
       stop_ego2s_bv8_inputs_insufficient is True, failing any of the four probe fault/drift
       conditions (decile_tripwire_pass, boost_tripwire_pass, all_q_finite, illegal_action_count)
       gives STOP_TRIPWIRE_BREACH rather than STOP_EGO2S_BV8_INPUTS_INSUFFICIENT."""
    from research.gpu_sitl.train_sitl import (
        REQUIRED_PROBE_TRIPWIRE_KEYS,
        VERDICT_INPUTS_INSUFFICIENT,
        VERDICT_PROCEED_TO_GATE_B1,
        VERDICT_TRIPWIRE_STOP,
        compute_step_b1b_pilot_verdict,
    )

    base_tw = {
        "all_pass": False,
        "decile_tripwire_pass": True,
        "boost_tripwire_pass": True,
        "decisive_rescue_bar_pass": False,
        "all_q_finite": True,
        "illegal_action_count": 0,
    }

    # 1. Missing any required tripwire key must raise KeyError (never default to passing)
    for key_to_drop in REQUIRED_PROBE_TRIPWIRE_KEYS:
        broken_tw = dict(base_tw)
        broken_tw.pop(key_to_drop)
        with pytest.raises(KeyError):
            compute_step_b1b_pilot_verdict(
                runaway_guard_reached=False,
                updates_completed=1000,
                total_updates=1000,
                completed_windows=[{"pass": True}],
                delivered_boost_band_pass=True,
                non_finite_loss_count=0,
                non_finite_q_count=0,
                non_finite_grad_count=0,
                illegal_action_count=0,
                tripwire_history=[broken_tw],
                tripwire_breached=True,
                stop_ego2s_bv8_inputs_insufficient=True,
            )

    # 2. Each of the 4 probe fault/drift conditions failing at final update with
    # stop_ego2s_bv8_inputs_insufficient=True must yield STOP_TRIPWIRE_BREACH:
    for failing_override in (
        {"decile_tripwire_pass": False},
        {"boost_tripwire_pass": False},
        {"all_q_finite": False},
        {"illegal_action_count": 1},
    ):
        tw_fail = {**base_tw, **failing_override}
        verdict = compute_step_b1b_pilot_verdict(
            runaway_guard_reached=False,
            updates_completed=1000,
            total_updates=1000,
            completed_windows=[{"pass": True}],
            delivered_boost_band_pass=True,
            non_finite_loss_count=0,
            non_finite_q_count=0,
            non_finite_grad_count=0,
            illegal_action_count=0,
            tripwire_history=[tw_fail],
            tripwire_breached=True,
            stop_ego2s_bv8_inputs_insufficient=True,
        )
        assert verdict == VERDICT_TRIPWIRE_STOP, f"Failed for override {failing_override}"

    # 3. When all 4 fault/drift checks pass at update 1000 and only decisive_rescue_bar_pass
    # fails alongside stop_ego2s_bv8_inputs_insufficient=True -> STOP_EGO2S_BV8_INPUTS_INSUFFICIENT
    verdict_aliased = compute_step_b1b_pilot_verdict(
        runaway_guard_reached=False,
        updates_completed=1000,
        total_updates=1000,
        completed_windows=[{"pass": True}],
        delivered_boost_band_pass=True,
        non_finite_loss_count=0,
        non_finite_q_count=0,
        non_finite_grad_count=0,
        illegal_action_count=0,
        tripwire_history=[base_tw],
        tripwire_breached=True,
        stop_ego2s_bv8_inputs_insufficient=True,
    )
    assert verdict_aliased == VERDICT_INPUTS_INSUFFICIENT

    # 4. When all checks pass and stop_ego2s_bv8_inputs_insufficient=False -> PROCEED
    tw_all_pass = {**base_tw, "decisive_rescue_bar_pass": True, "all_pass": True}
    verdict_pass = compute_step_b1b_pilot_verdict(
        runaway_guard_reached=False,
        updates_completed=1000,
        total_updates=1000,
        completed_windows=[{"pass": True}],
        delivered_boost_band_pass=True,
        non_finite_loss_count=0,
        non_finite_q_count=0,
        non_finite_grad_count=0,
        illegal_action_count=0,
        tripwire_history=[tw_all_pass],
        tripwire_breached=False,
        stop_ego2s_bv8_inputs_insufficient=False,
    )
    assert verdict_pass == VERDICT_PROCEED_TO_GATE_B1


def test_m1_all_registration_quotes_are_verbatim_with_exact_line_numbers() -> None:
    """Verify Finding m1:
    Every string in code or result JSON that claims to quote the registration is a byte-for-byte
    verbatim substring of docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md
    at the exact claimed 1-based line number."""
    from research.gpu_sitl.prepare import REGISTRATION_DOC, ROOT
    from research.gpu_sitl.train_sitl import (
        REGISTRATION_VERBATIM_QUOTES,
        RESCUE_QUOTA_REGISTRATION_QUOTES,
    )

    reg_file = ROOT / REGISTRATION_DOC
    reg_lines = reg_file.read_text(encoding="utf-8").splitlines()
    assert len(REGISTRATION_VERBATIM_QUOTES) >= 3
    for entry in REGISTRATION_VERBATIM_QUOTES:
        line_no = int(entry["line"])
        quote_str = str(entry["quote"])
        assert 1 <= line_no <= len(reg_lines), f"Invalid line number {line_no}"
        actual_line = reg_lines[line_no - 1]
        assert quote_str in actual_line, (
            f"Quote for {entry['section']} (line {line_no}) is not a verbatim substring of "
            f"registration line {line_no}:\n  quote:  {quote_str!r}\n  actual: {actual_line!r}"
        )
    assert RESCUE_QUOTA_REGISTRATION_QUOTES in reg_lines[143 - 1]
    assert "5,000 gradient updates" not in RESCUE_QUOTA_REGISTRATION_QUOTES
    assert "after 5,000 updates" not in RESCUE_QUOTA_REGISTRATION_QUOTES
    assert "`1,000` balanced learner updates" in RESCUE_QUOTA_REGISTRATION_QUOTES


def test_b1b_smoke_device_placement_and_flood_cap_regression() -> None:
    """Regression test for the A100 Step B1b-Smoke mixed-device failure in
    snapshot_worlds_to_device -> rt_dev._features_rows and the flood_cuda max_steps cap."""
    from research.gpu_sim_spike import r2_cache, v8_device
    from research.gpu_sim_spike.vector_device import FREE_SPACE_BFS_CAP
    from research.gpu_sitl.open_stop_gpu import (
        FastDeviceVectorRuntime,
        assert_sim_and_policy_on_device,
        compact_sim_and_policy,
        concat_sims_and_policies,
        move_device_policy_to_device,
        normalize_device,
    )
    from research.gpu_sitl.preflight import (
        MATURE_EXTENSION_SPECS,
        build_extension_worlds,
        resolve_product_checkpoints,
        snapshot_worlds_to_device,
    )

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    assert normalize_device("cpu") == torch.device("cpu")
    assert normalize_device("cuda") == torch.device("cuda", 0)
    assert normalize_device("mps") == torch.device("mps", 0)

    worlds, pol_tpl, _ = build_extension_worlds(
        Path(b8_path),
        Path(opp_path),
        specs=MATURE_EXTENSION_SPECS[:2],
    )
    assert pol_tpl.device == torch.device("cpu")

    sim_cpu, pol_cpu = snapshot_worlds_to_device(
        worlds,
        pol_tpl,
        device="cpu",
        vector_prepared=True,
    )
    assert_sim_and_policy_on_device(sim_cpu, pol_cpu, "cpu")
    r2_cache.enable(sim_cpu)

    # Verify FastDeviceVectorRuntime._free_space_for_rows passes max_steps == FREE_SPACE_BFS_CAP
    # (160) so snakes with length >= 80 never trip flood_cuda.cu's wanted > max_cap watchdog.
    recorded_flood_calls: list[tuple[int, int]] = []
    orig_flood = v8_device._flood

    def _spy_flood(start, blocked, cap, release=None, *, max_steps=None):
        max_wanted = int(cap.max().item()) if cap.numel() > 0 else 0
        eff_max_steps = int(max_steps) if max_steps is not None else int(v8_device._AREA_MAX)
        recorded_flood_calls.append((max_wanted, eff_max_steps))
        assert (
            max_wanted <= eff_max_steps
        ), f"flood_cuda.cu watchdog violation: wanted={max_wanted} > max_steps={eff_max_steps}"
        return orig_flood(start, blocked, cap, release, max_steps=max_steps)

    v8_device._flood = _spy_flood
    try:
        rt_sub = pol_cpu.vector_runtime.repeat_interleave(1)
        rt_sub.__class__ = FastDeviceVectorRuntime
        rows = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
        _ = rt_sub._free_space_for_rows(sim_cpu, rows)
    finally:
        v8_device._flood = orig_flood

    assert len(recorded_flood_calls) >= 1
    for max_wanted, eff_max_steps in recorded_flood_calls:
        assert eff_max_steps == FREE_SPACE_BFS_CAP
        assert max_wanted == FREE_SPACE_BFS_CAP
        assert max_wanted <= eff_max_steps

    # Verify concat_sims_and_policies and compact_sim_and_policy enforce device alignment
    sim_cat, pol_cat = concat_sims_and_policies([sim_cpu, sim_cpu], [pol_cpu, pol_cpu])
    assert_sim_and_policy_on_device(sim_cat, pol_cat, "cpu")
    compact_sim_and_policy(sim_cat, pol_cat, torch.tensor([0, 2], dtype=torch.long))
    assert_sim_and_policy_on_device(sim_cat, pol_cat, "cpu")
    assert int(sim_cat.E) == 2
    assert move_device_policy_to_device(pol_cat, "cpu", e_count=4).vector_runtime.E == 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS accelerator not available")
def test_mps_accelerator_device_placement_shakeout(tmp_path: Path) -> None:
    """End-to-end MPS accelerator shake-out for snapshot_worlds_to_device, _features_rows,
    FastDeviceVectorRuntime.observe_step/prepare, batched_open_stop_h32, probe reconstruction,
    tripwires, learner update, and checkpoint save/restore."""
    from contextlib import contextmanager

    from research.gpu_sim_spike import ego_device, r2_cache, v8_device
    from research.gpu_sitl.open_stop_gpu import (
        _cpu_queue_flood,
        assert_sim_and_policy_on_device,
        compact_sim_and_policy,
        concat_sims_and_policies,
        freeze_ego2s_value_head,
        margin_loss_gradient_check,
        normalize_device,
    )
    from research.gpu_sitl.preflight import (
        MATURE_EXTENSION_SPECS,
        build_extension_worlds,
        resolve_product_checkpoints,
        snapshot_worlds_to_device,
    )
    from research.gpu_sitl.probe_bank import (
        build_watch6_ct_device_bundle,
        extract_hero_root_observations,
        reconstruct_batch_from_cpu_row_states,
        slice_single_row_cpu_state,
    )
    from research.gpu_sitl.seed_check import train_seeds
    from research.gpu_sitl.smoke import run_cuda_shakeout, run_mature_cohort_row_subset_parity
    from research.gpu_sitl.train_sitl import (
        SitlRingBuffer,
        _capture_sim_and_runtime_state,
        _restore_sim_and_runtime_state,
        compute_sitl_hinge_loss,
        evaluate_training_set_aliasing_diagnostic,
        sample_ordinary_minibatch_indices,
    )
    from src.model.ego2s_network import load_ego2s_checkpoint

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    @contextmanager
    def _mps_op_adapter():
        orig_zeros, orig_full, orig_empty = torch.zeros, torch.full, torch.empty
        orig_tensor, orig_as_tensor, orig_arange = torch.tensor, torch.as_tensor, torch.arange
        orig_to, orig_double = torch.Tensor.to, torch.Tensor.double
        orig_scatter_reduce_ = torch.Tensor.scatter_reduce_
        orig_assert_async = torch._assert_async
        orig_v8_flood = v8_device._flood
        orig_ego_comp = ego_device._component_sizes

        def _is_mps(dev):
            return dev is not None and torch.device(dev).type == "mps"

        def _wrap_creation(fn):
            def wrapped(*args, **kwargs):
                if _is_mps(kwargs.get("device")) and kwargs.get("dtype") == torch.float64:
                    kwargs = dict(kwargs, dtype=torch.float32)
                return fn(*args, **kwargs)

            return wrapped

        def wrapped_as_tensor(data, dtype=None, device=None):
            if _is_mps(device):
                if dtype == torch.float64 or (
                    dtype is None
                    and (
                        (isinstance(data, np.ndarray) and data.dtype == np.float64)
                        or (isinstance(data, torch.Tensor) and data.dtype == torch.float64)
                    )
                ):
                    dtype = torch.float32
            return orig_as_tensor(data, dtype=dtype, device=device)

        def wrapped_to(self, *args, **kwargs):
            new_args = list(args)
            target_mps = (
                (self.device.type == "mps")
                or any(isinstance(a, (str, torch.device)) and _is_mps(a) for a in new_args)
                or _is_mps(kwargs.get("device"))
            )
            if target_mps:
                new_args = [torch.float32 if a == torch.float64 else a for a in new_args]
                if kwargs.get("dtype") == torch.float64 or (
                    self.dtype == torch.float64
                    and "dtype" not in kwargs
                    and not any(isinstance(a, torch.dtype) for a in new_args)
                ):
                    kwargs = dict(kwargs, dtype=torch.float32)
            return orig_to(self, *new_args, **kwargs)

        def wrapped_double(self, *args, **kwargs):
            if self.device.type == "mps":
                return orig_to(self, dtype=torch.float32)
            return orig_double(self, *args, **kwargs)

        def wrapped_scatter_reduce_(self, dim, index, src, reduce, *, include_self=True):
            if self.device.type == "mps" and self.dtype == torch.int64:
                if index.device != self.device or src.device != self.device:
                    raise RuntimeError(
                        f"Mixed devices in scatter_reduce_: "
                        f"{self.device}, {index.device}, {src.device}"
                    )
                tmp = orig_to(self, dtype=torch.int32).clone()
                orig_scatter_reduce_(
                    tmp,
                    dim,
                    index,
                    orig_to(src, dtype=torch.int32),
                    reduce,
                    include_self=include_self,
                )
                self.copy_(orig_to(tmp, dtype=torch.int64))
                return self
            return orig_scatter_reduce_(self, dim, index, src, reduce, include_self=include_self)

        def wrapped_assert_async(tensor, msg):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "mps":
                if not bool(tensor.all().item()):
                    raise AssertionError(msg)
                return
            return orig_assert_async(tensor, msg)

        def wrapped_v8_flood(start, blocked, cap, release=None, *, max_steps=None):
            if start.device.type == "mps":
                if blocked.device != start.device or cap.device != start.device:
                    raise RuntimeError(
                        f"Mixed devices in _flood: {start.device}, {blocked.device}, {cap.device}"
                    )
                if release is not None and release.device != start.device:
                    raise RuntimeError(
                        f"Mixed devices in _flood release: {start.device}, {release.device}"
                    )
                return _cpu_queue_flood(
                    start.cpu(),
                    blocked.cpu(),
                    cap.cpu(),
                    release.cpu() if release is not None else None,
                    max_steps=max_steps,
                ).to(device=start.device, dtype=torch.long)
            return orig_v8_flood(start, blocked, cap, release, max_steps=max_steps)

        def wrapped_ego_comp(sim, rows, ttl):
            if sim.device.type == "mps":
                if rows.device != sim.device or ttl.device != sim.device:
                    raise RuntimeError(
                        f"Mixed devices in _component_sizes: "
                        f"{sim.device}, {rows.device}, {ttl.device}"
                    )
                n_rows, h, w = rows.shape[0], int(sim.grid_h), int(sim.grid_w)
                env, snake = rows[:, 0], rows[:, 1]
                cardinal = ego_device._cardinal(sim)
                heads, direction = sim.heads()[env, snake], sim.direction[env, snake]
                rel = torch.arange(3, device=sim.device)[None, :] - 1
                move = cardinal[(direction[:, None] + rel) % 4]
                start = heads[:, None, :] + move
                v_max = h * w
                blocked = ~(ttl.amax(dim=1) <= 1)
                blocked_rows = blocked[env][:, None].expand(-1, 3, -1, -1)
                cap = torch.full((n_rows, 3), v_max, dtype=torch.long, device=sim.device)
                sizes = wrapped_v8_flood(start, blocked_rows, cap, max_steps=v_max)
                return torch.where(sim.alive[env, snake, None], sizes, torch.zeros_like(sizes))
            return orig_ego_comp(sim, rows, ttl)

        torch.zeros = _wrap_creation(orig_zeros)
        torch.full = _wrap_creation(orig_full)
        torch.empty = _wrap_creation(orig_empty)
        torch.tensor = _wrap_creation(orig_tensor)
        torch.arange = _wrap_creation(orig_arange)
        torch.as_tensor = wrapped_as_tensor
        torch.Tensor.to = wrapped_to
        torch.Tensor.double = wrapped_double
        torch.Tensor.scatter_reduce_ = wrapped_scatter_reduce_
        torch._assert_async = wrapped_assert_async
        v8_device._flood = wrapped_v8_flood
        ego_device._component_sizes = wrapped_ego_comp
        try:
            yield
        finally:
            torch.zeros, torch.full, torch.empty = orig_zeros, orig_full, orig_empty
            torch.tensor, torch.arange, torch.as_tensor = orig_tensor, orig_arange, orig_as_tensor
            torch.Tensor.to, torch.Tensor.double = orig_to, orig_double
            torch.Tensor.scatter_reduce_ = orig_scatter_reduce_
            torch._assert_async = orig_assert_async
            v8_device._flood = orig_v8_flood
            ego_device._component_sizes = orig_ego_comp

    dev = normalize_device("mps")
    assert dev == torch.device("mps", 0)

    with _mps_op_adapter():
        # 1. Verify that the pre-fix state (CPU rt_dev with MPS sim_dev) would have raised
        # RuntimeError on starts = heads[..., None, :] + delta, and that the fixed
        # snapshot_worlds_to_device places both sim_dev and pol_dev on mps:0 and succeeds.
        worlds, pol_tpl, _ = build_extension_worlds(
            Path(b8_path),
            Path(opp_path),
            specs=MATURE_EXTENSION_SPECS[:2],
        )
        sim_mps, pol_mps = snapshot_worlds_to_device(
            worlds,
            pol_tpl,
            device="mps",
            vector_prepared=True,
        )
        assert_sim_and_policy_on_device(sim_mps, pol_mps, dev)
        r2_cache.enable(sim_mps)

        rows_mps = torch.tensor([[0, 1], [0, 0], [1, 3]], dtype=torch.long, device=dev)
        # Confirm that calling CPU pol_tpl.vector_runtime._features_rows with MPS sim_mps
        # reproduces the exact A100 mixed-device RuntimeError:
        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            pol_tpl.vector_runtime._features_rows(sim_mps, rows_mps, update_memory=False)

        # Whereas the fixed pol_mps.vector_runtime._features_rows runs cleanly on mps:0:
        feats_mps = pol_mps.vector_runtime._features_rows(sim_mps, rows_mps, update_memory=False)
        assert feats_mps.device == dev
        assert feats_mps.shape == (3, 61)

        # 2. Run the full mature cohort row-subset parity path from smoke.py on MPS:
        parity_rep = run_mature_cohort_row_subset_parity(
            b8_path,
            opp_path,
            device="mps",
            slice_indices=(0, 1),
            branch_horizon=2,
            dry_run=True,
        )
        assert parity_rep["ok"] is True

        # 3. Exercise bundle construction, splicing, reconstruction, checkpoint state,
        # ring buffer sampling, hinge loss, and aliasing diagnostic on MPS:
        sim_b, pol_b = build_watch6_ct_device_bundle(
            train_seeds(2),
            b8_path,
            opp_path,
            device="mps",
            hero_respawn=True,
        )
        assert_sim_and_policy_on_device(sim_b, pol_b, dev)
        sim_cat, pol_cat = concat_sims_and_policies([sim_mps, sim_b], [pol_mps, pol_b])
        assert_sim_and_policy_on_device(sim_cat, pol_cat, dev)
        compact_sim_and_policy(sim_cat, pol_cat, torch.tensor([0, 2], device=dev))
        assert_sim_and_policy_on_device(sim_cat, pol_cat, dev)

        row0 = slice_single_row_cpu_state(sim_cat, pol_cat, 0)
        row1 = slice_single_row_cpu_state(sim_cat, pol_cat, 1)
        rec_sim, rec_pol = reconstruct_batch_from_cpu_row_states(
            [row0, row1],
            sim_cat,
            pol_cat,
            "mps",
        )
        assert_sim_and_policy_on_device(rec_sim, rec_pol, dev)

        captured_state = _capture_sim_and_runtime_state(rec_sim, rec_pol)
        _restore_sim_and_runtime_state(rec_sim, rec_pol, captured_state)
        assert_sim_and_policy_on_device(rec_sim, rec_pol, dev)

        obs_mps = extract_hero_root_observations(rec_sim, cpu_fast_flood=False)
        assert obs_mps["local_u8"].device == dev
        student_mps = load_ego2s_checkpoint(str(b8_path), device=dev)
        freeze_ego2s_value_head(student_mps)
        res_buf = SitlRingBuffer(capacity=8)
        ord_buf = SitlRingBuffer(capacity=8)
        for i in range(4):
            res_buf.add_single(
                local_u8=obs_mps["local_u8"][0],
                global_u8=obs_mps["global_u8"][0],
                scalars_f32=obs_mps["scalars_f32"][0],
                legal_mask=obs_mps["legal_mask"][0],
                safe_mask=torch.tensor([False, True, False, False, False, False], dtype=torch.bool),
                unsafe_mask=torch.tensor(
                    [True, False, True, False, False, False], dtype=torch.bool
                ),
                bare_action=0,
                a_star_action=1,
                world_seed=100 + i,
                decision_frame=i,
            )
            ord_buf.add_single(
                local_u8=obs_mps["local_u8"][1],
                global_u8=obs_mps["global_u8"][1],
                scalars_f32=obs_mps["scalars_f32"][1],
                legal_mask=obs_mps["legal_mask"][1],
                safe_mask=torch.tensor([True, True, False, False, False, False], dtype=torch.bool),
                unsafe_mask=torch.tensor(
                    [False, False, True, False, False, False], dtype=torch.bool
                ),
                bare_action=0,
                a_star_action=0,
                world_seed=200 + i,
                decision_frame=i,
            )
        ord_idx, _ = sample_ordinary_minibatch_indices(
            ord_buf,
            student_mps,
            np.random.default_rng(7),
            uniform_count=2,
            hard_neg_count=2,
            device=dev,
        )
        res_batch = res_buf.gather_batch([0, 1], dev)
        ord_batch = ord_buf.gather_batch(ord_idx[:2], dev)
        loss_d = compute_sitl_hinge_loss(student_mps, res_batch, ord_batch)
        assert loss_d["total_loss"].device == dev
        alias_d = evaluate_training_set_aliasing_diagnostic(
            student_mps,
            res_buf,
            ord_buf,
            device=dev,
        )
        assert "verdict" in alias_d

        grad_check = margin_loss_gradient_check(str(b8_path), sim=rec_sim, device="mps")
        assert grad_check["all_non_value_grads_finite_and_nonzero"] is True

        # 4. Run full 6-phase run_cuda_shakeout on MPS (mps:0) under the adapter:
        shake_mps = run_cuda_shakeout(
            tmp_path / "mps_shakeout_out",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="mps",
        )
        assert shake_mps["ok"] is True
        assert shake_mps["verdict"] == "SHAKEOUT_PASS_NOT_EVIDENCE"
        assert shake_mps["device"] == "mps:0"
        assert shake_mps["not_evidence"] is True


def test_cuda_init_ordering_before_reset_peak_memory_stats_and_all_cuda_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the root cause of the A100 `RuntimeError: Invalid device argument` crash:
    When `torch.device('cuda', 0)` has an explicit index `0`, `torch/cuda/memory.py:376`
    (`reset_peak_memory_stats`) calls `torch._C._cuda_resetPeakMemoryStats(0)` directly without
    calling `torch.cuda._lazy_init()`. Verify `ensure_cuda_initialized` / `normalize_device`
    initializes CUDA (`torch.cuda.init()` + `torch.cuda.set_device(dev)`) before any
    `reset_peak_memory_stats`, `max_memory_allocated`, `max_memory_reserved`, `synchronize`,
    `get_rng_state_all`, `set_rng_state_all`, or `manual_seed_all` call."""
    from research.gpu_sitl.open_stop_gpu import ensure_cuda_initialized, normalize_device
    from research.gpu_sitl.smoke import _sync_device

    state = {"initialized": False, "current_device": -1, "calls": []}

    def _fake_is_available() -> bool:
        return True

    def _fake_is_initialized() -> bool:
        return bool(state["initialized"])

    def _fake_init() -> None:
        state["initialized"] = True
        state["calls"].append("init")

    def _fake_set_device(dev: Any) -> None:
        if not state["initialized"]:
            raise RuntimeError("Invalid device argument: CUDA not initialized before set_device")
        d = torch.device(dev)
        if d.index is None or int(d.index) != 0:
            raise RuntimeError(f"Invalid device argument: {d}")
        state["current_device"] = int(d.index)
        state["calls"].append(f"set_device:{d}")

    def _require_init_and_dev0(op_name: str, dev: Any = None) -> int:
        if not state["initialized"]:
            raise RuntimeError(
                f"RuntimeError: Invalid device argument in {op_name} (CUDA caching allocator "
                "not initialized before explicit device=cuda:0 call)"
            )
        if dev is not None:
            d = torch.device(dev)
            if d.type != "cuda" or d.index != 0:
                raise RuntimeError(f"Invalid device argument in {op_name}: {d}")
        state["calls"].append(op_name)
        return 0

    monkeypatch.setattr(torch.cuda, "is_available", _fake_is_available)
    monkeypatch.setattr(torch.cuda, "is_initialized", _fake_is_initialized)
    monkeypatch.setattr(torch.cuda, "init", _fake_init)
    monkeypatch.setattr(torch.cuda, "set_device", _fake_set_device)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda dev=None: _require_init_and_dev0("reset_peak_memory_stats", dev),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda dev=None: _require_init_and_dev0("max_memory_allocated", dev),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda dev=None: _require_init_and_dev0("max_memory_reserved", dev),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda dev=None: _require_init_and_dev0("synchronize", dev),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [_require_init_and_dev0("get_rng_state_all")],
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda s: _require_init_and_dev0(f"manual_seed_all:{s}"),
    )

    # 1. Confirm that calling reset_peak_memory_stats(torch.device("cuda", 0)) before
    # ensure_cuda_initialized raises the exact RuntimeError: Invalid device argument:
    with pytest.raises(RuntimeError, match="Invalid device argument"):
        torch.cuda.reset_peak_memory_stats(torch.device("cuda", 0))

    # 2. Calling normalize_device("cuda") / ensure_cuda_initialized("cuda") initializes CUDA
    # first and sets device to cuda:0:
    dev = normalize_device("cuda")
    assert dev == torch.device("cuda", 0)
    assert state["initialized"] is True
    assert state["current_device"] == 0
    assert state["calls"][:2] == ["init", "set_device:cuda:0"]

    # 3. Now reset_peak_memory_stats, max_memory_allocated, max_memory_reserved, _sync_device,
    # and get_rng_state_all all succeed on dev:
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.max_memory_allocated(dev)
    torch.cuda.max_memory_reserved(dev)
    _sync_device(dev)
    torch.cuda.get_rng_state_all()

    # 4. Also verify that if _sync_device(torch.device("cuda", 0)) is called on a fresh
    # uninitialized state, it initializes CUDA before calling torch.cuda.synchronize:
    state["initialized"] = False
    state["calls"].clear()
    _sync_device(torch.device("cuda", 0))
    assert state["initialized"] is True
    assert state["calls"] == ["init", "set_device:cuda:0", "synchronize"]

    # 5. Verify ensure_cuda_initialized("cuda:0") is idempotent when already initialized:
    ensure_cuda_initialized("cuda:0")
    assert state["initialized"] is True


def test_torch_cuda_arch_list_includes_l4_89_a100_80_h100_90(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TORCH_CUDA_ARCH_LIST covers L4 (8.9), A100 (8.0), and H100 (9.0) in both
    verify_cuda_native_extensions and research/gpu_sitl/job.sh."""
    from research.gpu_sitl.open_stop_gpu import (
        DEFAULT_CUDA_ARCH_LIST,
        REQUIRED_CUDA_ARCHS,
        verify_cuda_native_extensions,
    )
    from research.gpu_sitl.prepare import ROOT

    assert DEFAULT_CUDA_ARCH_LIST == "8.0;8.9;9.0"
    assert set(REQUIRED_CUDA_ARCHS) == {"8.0", "8.9", "9.0"}

    job_sh_text = (ROOT / "research" / "gpu_sitl" / "job.sh").read_text(encoding="utf-8")
    assert 'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0}"' in job_sh_text
    assert 'elif [ "$MODE" = "shakeout" ]; then' in job_sh_text

    # Verify verify_cuda_native_extensions rejects a TORCH_CUDA_ARCH_LIST missing 8.9 (L4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;9.0")
    with pytest.raises(RuntimeError, match="missing required GPU architectures"):
        verify_cuda_native_extensions("cuda")


def test_cuda_shakeout_end_to_end_and_continues_past_failing_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify Requirement B:
    1. `run_cuda_shakeout` runs all 6 phases end-to-end using dev-only seeds, exercises the
       mature cohort splice with length >= 80 snakes (flood cap 160), harvests a tiny dev
       probe bank, runs tiny SITL training with tripwire evaluation + checkpoint save + resume,
       and writes `shakeout_result.json` labelled `NOT EVIDENCE` (never `smoke_result.json`).
    2. When a mid-run phase fails, `run_cuda_shakeout` records the failure + traceback in
       `shakeout_result.json`, continues to execute subsequent phases, and `smoke.main()`
       returns non-zero (`1`)."""
    import research.gpu_sitl.smoke as smoke_mod
    from research.gpu_sitl.preflight import resolve_product_checkpoints
    from research.gpu_sitl.seed_check import (
        PROBE32_NAMESPACE,
        SCREEN_NAMESPACE,
        SELECTIVITY_NAMESPACE,
        SHAKEOUT_PROBE_NAMESPACE,
        SHAKEOUT_SMOKE_NAMESPACE,
        SHAKEOUT_TRAIN_NAMESPACE,
        TRAIN_NAMESPACE,
    )
    from research.gpu_sitl.smoke import run_cuda_shakeout

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    out_pass = tmp_path / "shakeout_pass"
    res = run_cuda_shakeout(
        out_pass,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
    )
    assert res["ok"] is True
    assert res["verdict"] == "SHAKEOUT_PASS_NOT_EVIDENCE"
    assert res["schema"] == "t34-gpu-sitl-cuda-shakeout-result/v1"
    assert res["not_evidence"] is True
    assert res["evidence_class"] == "SHAKEOUT_NOT_EVIDENCE"
    assert res["failed_phases"] == []
    assert len(res["phases"]) == 6
    assert all(p["ok"] is True and p["status"] == "PASS" for p in res["phases"])

    # Verify mature cohort splice exercised length >= 80 and flood cap 160 in Phase 4
    p4_detail = res["phases"][3]["detail"]
    assert p4_detail["mature_spliced_worlds"] >= 1
    assert p4_detail["window_max_hero_length"] >= 80
    assert p4_detail["window_mature_length_ge_160_decisions"] >= 1

    # Verify Phase 5 harvested dev probe bank and default registered loader refused it
    p5_detail = res["phases"][4]["detail"]
    assert p5_detail["namespace"] == SHAKEOUT_PROBE_NAMESPACE
    assert p5_detail["not_evidence"] is True
    assert p5_detail["refused_by_registered_probe32_loader"] is True

    # Verify Phase 6 paused, saved checkpoint, resumed, and evaluated tripwires on dev namespaces
    p6_detail = res["phases"][5]["detail"]
    assert p6_detail["namespace"] == SHAKEOUT_TRAIN_NAMESPACE
    assert p6_detail["probe_namespace"] == SHAKEOUT_PROBE_NAMESPACE
    assert p6_detail["not_evidence"] is True
    assert p6_detail["paused_at_updates"] == 1
    assert p6_detail["final_updates_completed"] == 2
    assert p6_detail["tripwire_evaluations"] >= 1

    # Verify only shakeout_result.json (+ sidecars) exists in out_pass; never smoke_result.json
    assert (out_pass / "shakeout_result.json").is_file()
    assert (out_pass / "shakeout_result.json.sha256").is_file()
    assert (out_pass / "shakeout_result.json.complete").is_file()
    assert not (out_pass / "smoke_result.json").exists()
    assert set(res["namespaces"].values()) == {
        SHAKEOUT_SMOKE_NAMESPACE,
        SHAKEOUT_PROBE_NAMESPACE,
        SHAKEOUT_TRAIN_NAMESPACE,
    }
    assert set(res["namespaces"].values()).isdisjoint(
        {SELECTIVITY_NAMESPACE, SCREEN_NAMESPACE, PROBE32_NAMESPACE, TRAIN_NAMESPACE}
    )

    # 2. Inject a failure in Phase 4 and verify Phases 5 and 6 still execute and pass,
    # traceback is recorded, and CLI main() returns exit code 1.
    def _boom_smoke(*args, **kwargs):
        raise RuntimeError("injected_phase_4_cuda_failure")

    monkeypatch.setattr(smoke_mod, "run_step_b1b_smoke", _boom_smoke)
    out_fail = tmp_path / "shakeout_partial_fail"
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke.py",
            "--shakeout",
            "--device",
            "cpu",
            "--output-dir",
            str(out_fail),
            "--b8-checkpoint",
            str(b8_path),
            "--opponent-checkpoint",
            str(opp_path),
        ],
    )
    rc = smoke_mod.main()
    assert rc == 1
    captured_out = capsys.readouterr().out
    assert "[shakeout] FAIL phase_4_b1b_smoke_reduced" in captured_out
    assert "Traceback (most recent call last):" in captured_out
    assert "RuntimeError: injected_phase_4_cuda_failure" in captured_out
    fail_json = json.loads((out_fail / "shakeout_result.json").read_text(encoding="utf-8"))
    assert fail_json["ok"] is False
    assert fail_json["verdict"] == "SHAKEOUT_FAIL_NOT_EVIDENCE"
    assert fail_json["failed_phases"] == ["phase_4_b1b_smoke_reduced"]
    by_phase = {p["phase"]: p for p in fail_json["phases"]}
    assert by_phase["phase_4_b1b_smoke_reduced"]["ok"] is False
    assert "injected_phase_4_cuda_failure" in by_phase["phase_4_b1b_smoke_reduced"]["traceback"]
    # Phases 5 and 6 still ran after Phase 4 failed and both passed!
    assert by_phase["phase_5_tiny_probe_harvest"]["ok"] is True
    assert by_phase["phase_6_tiny_sitl_training_and_resume"]["ok"] is True


def test_prepare_shakeout_manifest_and_pilot_refuses_shakeout_result(
    tmp_path: Path,
) -> None:
    """Verify Requirement A & B:
    1. `prepare_b1b_packet(..., mode='shakeout')` produces an L4 manifest (`gpu='L4'`,
       `rate_units_per_hour=1.54`, `resume_exemption` `short-canary`, `est_minutes=15.0`,
       `timeout_minutes=30.0 <= 40.0`, `unit_cap=0.77 <= 1.0`) accepted by `colab_canary.check`,
       `colab_supervisor.validate`, and `dispatch.require_resume_policy`.
    2. `prepare_b1b_packet(..., mode='pilot')` refuses a shake-out result JSON (both under its
       `shakeout_result.json` filename and even if copied/renamed to `smoke_result.json`)."""
    import shutil

    from research.colab_cli import canary as colab_canary
    from research.colab_cli import supervisor as colab_supervisor
    from research.gpu_sitl.preflight import resolve_product_checkpoints
    from research.gpu_sitl.prepare import ROOT, prepare_b1b_packet
    from tools.ops.dispatch import require_resume_policy

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    pkt_shake = tmp_path / "pkt_shakeout"
    out_shake = tmp_path / "out_shakeout"
    info_shake = prepare_b1b_packet(
        pkt_shake,
        commit="HEAD",
        mode="shakeout",
        output_dir=out_shake,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        attempt="shakeout-01",
    )
    manifest = json.loads(Path(info_shake["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["requirements"]["gpu"] == "L4"
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["requirements"]["est_minutes"] == pytest.approx(15.0)
    assert "prefer" not in manifest["requirements"]
    assert manifest["budget"]["rate_units_per_hour"] == pytest.approx(1.54)
    assert float(manifest["timeout_minutes"]) == pytest.approx(30.0)
    assert float(manifest["timeout_minutes"]) <= 40.0
    assert float(manifest["budget"]["unit_cap"]) == pytest.approx(0.77)
    assert float(manifest["budget"]["unit_cap"]) <= 1.0
    assert manifest["technical_envelope"]["base_seconds"] == pytest.approx(900.0)
    assert manifest["technical_envelope"]["effective_seconds"] == pytest.approx(1800.0)
    assert manifest["technical_envelope"]["registered_a100_unit_cap"] == pytest.approx(0.77)
    assert manifest["resume_exemption"]["kind"] == "short-canary"
    assert "resume" not in manifest
    assert manifest["env"]["GPU_SITL_MODE"] == "shakeout"
    assert manifest["not_evidence"] is True
    assert manifest["evidence_class"] == "SHAKEOUT_NOT_EVIDENCE"
    assert manifest["required_exports"] == [
        "shakeout_result.json",
    ]

    # Verify colab_canary.check, colab_supervisor.validate, and dispatch.require_resume_policy
    # accept the manifest
    manifest_for_val = dict(manifest)
    require_resume_policy(manifest_for_val)
    colab_canary.check(manifest_for_val, ROOT)
    colab_supervisor.validate(manifest_for_val, ROOT)

    # Create a shakeout_result.json (even containing a fake pilot_runaway_guard block) and verify
    # prepare_b1b_packet(mode="pilot") refuses it both by filename and by schema / not_evidence tag.
    fake_shakeout_json = tmp_path / "shakeout_result.json"
    fake_shakeout_payload = {
        "schema": "t34-gpu-sitl-cuda-shakeout-result/v1",
        "mode": "shakeout",
        "shakeout": True,
        "not_evidence": True,
        "evidence_class": "SHAKEOUT_NOT_EVIDENCE",
        "pilot_runaway_guard": {
            "base_estimate_seconds": 600.0,
            "base_estimate_minutes": 10.0,
            "runaway_guard_timeout_minutes_2x": 20,
            "runaway_guard_timeout_seconds_2x": 1200.0,
            "runaway_guard_a100_units_2x_at_8p00_broker": 2.67,
            "runaway_guard_max_collection_steps_2x": 1000,
        },
    }
    fake_shakeout_json.write_text(
        json.dumps(fake_shakeout_payload, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Refusing shake-out result"):
        prepare_b1b_packet(
            tmp_path / "pkt_pilot_refuse_name",
            commit="origin/main",
            mode="pilot",
            output_dir=tmp_path / "out_pilot_refuse_name",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            smoke_result_json=fake_shakeout_json,
        )

    renamed_as_smoke = tmp_path / "smoke_result.json"
    shutil.copyfile(fake_shakeout_json, renamed_as_smoke)
    with pytest.raises(ValueError, match="Refusing non-smoke or NOT EVIDENCE result"):
        prepare_b1b_packet(
            tmp_path / "pkt_pilot_refuse_schema",
            commit="origin/main",
            mode="pilot",
            output_dir=tmp_path / "out_pilot_refuse_schema",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            smoke_result_json=renamed_as_smoke,
        )


def test_required_exports_accepted_by_supervisor_validate_export_inventory_and_runtime_scanner(
    tmp_path: Path,
) -> None:
    """Offline test (Requirement 3) feeding each mode's (`shakeout`, `smoke`, `pilot`) manifest
    `required_exports` and both:
    (a) a fake runtime inventory shaped exactly as `research/colab_cli/runtime.py:_exports`
        produces (payload relative paths only, `{"path", "sha256", "size"}`, no `.sha256` or
        `.complete` entries), and
    (b) a real `runtime._exports` scan over the mode's actual job outputs (including stateful
        multi-checkpoint `cache` polling for `pilot`),
    directly into the supervisor's real required-export check function
    `research.colab_cli.schema.validate_export_inventory` (called by
    `research/colab_cli/supervisor.py:2032`)."""
    from research.colab_cli.runtime import _exports as runtime_export_scanner
    from research.colab_cli.schema import SchemaError, validate_export_inventory
    from research.gpu_sitl.preflight import resolve_product_checkpoints
    from research.gpu_sitl.prepare import (
        PILOT_REQUIRED_EXPORTS,
        SHAKEOUT_REQUIRED_EXPORTS,
        SMOKE_REQUIRED_EXPORTS,
        prepare_b1b_packet,
    )
    from research.gpu_sitl.smoke import run_cuda_shakeout, run_step_b1b_smoke

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # 1. Build real manifests for all three modes (`shakeout`, `smoke`, `pilot`)
    info_shake = prepare_b1b_packet(
        tmp_path / "pkt_shakeout",
        commit="origin/main",
        mode="shakeout",
        output_dir=tmp_path / "out_shakeout_mf",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        attempt="shake-01",
    )
    manifest_shake = json.loads(Path(info_shake["manifest_path"]).read_text(encoding="utf-8"))
    assert tuple(manifest_shake["required_exports"]) == SHAKEOUT_REQUIRED_EXPORTS

    info_smoke = prepare_b1b_packet(
        tmp_path / "pkt_smoke",
        commit="origin/main",
        mode="smoke",
        output_dir=tmp_path / "out_smoke_mf",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        attempt="smoke-01",
    )
    manifest_smoke = json.loads(Path(info_smoke["manifest_path"]).read_text(encoding="utf-8"))
    assert tuple(manifest_smoke["required_exports"]) == SMOKE_REQUIRED_EXPORTS

    info_pilot = prepare_b1b_packet(
        tmp_path / "pkt_pilot",
        commit="origin/main",
        mode="pilot",
        output_dir=tmp_path / "out_pilot_mf",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        base_seconds=600.0,
        attempt="pilot-01",
    )
    manifest_pilot = json.loads(Path(info_pilot["manifest_path"]).read_text(encoding="utf-8"))
    assert tuple(manifest_pilot["required_exports"]) == PILOT_REQUIRED_EXPORTS

    # 2. Feed a fake runtime inventory shaped exactly as `runtime.py:_exports` produces
    # (payload relative paths only, never `.sha256` or `.complete` sidecars) through the
    # supervisor's real `validate_export_inventory` function for each mode:
    fake_inventories_by_mode: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "shakeout": (
            manifest_shake,
            [
                {
                    "path": "shakeout_result.json",
                    "sha256": "a" * 64,
                    "size": 4096,
                }
            ],
        ),
        "smoke": (
            manifest_smoke,
            [
                {"path": "b8_sitl_student_final.pth", "sha256": "b" * 64, "size": 6_600_000},
                {"path": "probe32_bank.pt", "sha256": "c" * 64, "size": 52_000},
                {"path": "probe32_bank.pt.meta.json", "sha256": "d" * 64, "size": 2_400},
                {"path": "sitl_pilot_result.json", "sha256": "e" * 64, "size": 1_800},
                {"path": "smoke_result.json", "sha256": "f" * 64, "size": 5_200},
            ],
        ),
        "pilot": (
            manifest_pilot,
            [
                {"path": "b8_sitl_student_final.pth", "sha256": "1" * 64, "size": 6_600_000},
                {
                    "path": "checkpoints/sitl_state_final.meta.json",
                    "sha256": "2" * 64,
                    "size": 1_200,
                },
                {
                    "path": "checkpoints/sitl_state_final.pt",
                    "sha256": "3" * 64,
                    "size": 22_000_000,
                },
                {"path": "probe32_bank.pt", "sha256": "4" * 64, "size": 85_000_000},
                {"path": "probe32_bank.pt.meta.json", "sha256": "5" * 64, "size": 3_400},
                {"path": "sitl_pilot_result.json", "sha256": "6" * 64, "size": 18_000},
                {"path": "verified-ledger.json", "sha256": "7" * 64, "size": 2_100},
            ],
        ),
    }

    for mode_name, (mf, fake_inv) in fake_inventories_by_mode.items():
        # Must not contain any .sha256 or .complete entries in required_exports or inventory
        for req_p in mf["required_exports"]:
            assert not req_p.endswith(
                (".sha256", ".complete")
            ), f"{mode_name} required_exports must not contain sidecar {req_p!r}"
        # Real supervisor required-export check function must accept:
        validate_export_inventory(mf, fake_inv)

        # Verify that if sidecars were listed in required_exports (the 2fe6b243 bug),
        # validate_export_inventory raises SchemaError ("required exports absent ..."):
        first_payload = mf["required_exports"][0]
        buggy_mf = {
            **mf,
            "required_exports": [
                first_payload,
                f"{first_payload}.sha256",
                f"{first_payload}.complete",
            ],
        }
        with pytest.raises(
            SchemaError,
            match="required exports absent from runtime scanner inventory",
        ):
            validate_export_inventory(buggy_mf, fake_inv)

    # 3. Also run the real `runtime._exports` scanner over actual job outputs for all 3 modes
    # (including stateful cache polling across multiple checkpoints in `pilot`) and feed the
    # resulting inventories into `validate_export_inventory`:
    rt_shake_root = tmp_path / "real_shakeout_rt"
    out_shake_real = rt_shake_root / "output"
    run_cuda_shakeout(
        out_shake_real,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
    )
    shake_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    inv_shake = runtime_export_scanner(
        rt_shake_root,
        int(manifest_shake["max_export_bytes"]),
        shake_cache,
    )
    assert [e["path"] for e in inv_shake] == ["shakeout_result.json"]
    validate_export_inventory(manifest_shake, inv_shake)

    rt_smoke_root = tmp_path / "real_smoke_rt"
    out_smoke_real = rt_smoke_root / "output"
    run_step_b1b_smoke(
        out_smoke_real,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=(0, 8),
        bench_env_steps=1,
        bench_label_batches=1,
        bench_learner_updates=2,
        dry_run=True,
    )
    smoke_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    inv_smoke = runtime_export_scanner(
        rt_smoke_root,
        int(manifest_smoke["max_export_bytes"]),
        smoke_cache,
    )
    inv_smoke_paths = {e["path"] for e in inv_smoke}
    assert set(SMOKE_REQUIRED_EXPORTS) <= inv_smoke_paths
    # Smoke also exports the 4 pilot required exports so canary.check(pilot_manifest) passes
    assert set(PILOT_REQUIRED_EXPORTS) <= inv_smoke_paths
    validate_export_inventory(manifest_smoke, inv_smoke)

    # For pilot: simulate periodic runtime._exports polling across multiple checkpoints
    # with the same stateful `pilot_cache` dict to prove no `.complete` file is ever mutated!
    rt_pilot_root = tmp_path / "real_pilot_rt"
    out_pilot_real = rt_pilot_root / "output"
    pilot_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    run_sitl_training(
        out_pilot_real,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=4,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=2,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        decision_window_size=10,
        checkpoint_every_updates=1,
        seed=20261109,
        stop_after_updates=2,
        dry_run=True,
    )
    # Poll 1 after pause at update 2
    inv_pilot_mid = runtime_export_scanner(
        rt_pilot_root,
        int(manifest_pilot["max_export_bytes"]),
        pilot_cache,
    )
    assert len(inv_pilot_mid) >= 1

    # Stage into a fresh attempt-2 output dir via prepare_b1b_packet and complete updates 3..4
    pkt_pilot_retry = tmp_path / "pkt_pilot_retry"
    rt_retry_root = tmp_path / "real_pilot_retry_rt"
    out_pilot_retry = rt_retry_root / "output"
    prepare_b1b_packet(
        pkt_pilot_retry,
        commit="origin/main",
        mode="pilot",
        output_dir=out_pilot_retry,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        base_seconds=600.0,
        prior_output_dir=out_pilot_real,
        attempt="pilot-02",
    )
    retry_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    run_sitl_training(
        out_pilot_retry,
        input_dir=pkt_pilot_retry / "inputs",
        ledger_path=pkt_pilot_retry / "inputs" / "verified-ledger.json",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=4,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=2,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        decision_window_size=10,
        checkpoint_every_updates=1,
        seed=20261109,
        dry_run=True,
    )
    inv_pilot_final = runtime_export_scanner(
        rt_retry_root,
        int(manifest_pilot["max_export_bytes"]),
        retry_cache,
    )
    # Second poll on same retry_cache must also succeed (immutability check)
    inv_pilot_final_2 = runtime_export_scanner(
        rt_retry_root,
        int(manifest_pilot["max_export_bytes"]),
        retry_cache,
    )
    assert inv_pilot_final == inv_pilot_final_2
    validate_export_inventory(manifest_pilot, inv_pilot_final)
    total_exported_bytes = sum(int(e["size"]) for e in inv_pilot_final)
    assert total_exported_bytes < int(manifest_pilot["max_export_bytes"])


def test_b1b_smoke_reuses_mature_extension_worlds_state_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 1: Verify that building the mature extension worlds once in run_step_b1b_smoke
    and reusing them after run_mature_cohort_row_subset_parity produces a splice sim/policy
    state that is tensor-equal to a fresh build_extension_worlds call."""
    import research.gpu_sitl.smoke as smoke_mod
    from research.gpu_sitl.open_stop_gpu import (
        _ENV_TENSOR_FIELDS,
        _VECTOR_RUNTIME_ENV_FIELDS,
    )
    from research.gpu_sitl.preflight import (
        MATURE_EXTENSION_SPECS,
        build_extension_worlds,
        snapshot_worlds_to_device,
    )
    from research.gpu_sitl.smoke import (
        CTDeviceSim,
        run_mature_cohort_row_subset_parity,
        run_step_b1b_smoke,
    )

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    slice_indices = (0, 8)
    specs = [MATURE_EXTENSION_SPECS[i] for i in slice_indices]

    # 1. Build fresh reference worlds and snapshot with vector_prepared=False (splice path)
    fresh_worlds, fresh_pol_tpl, _ = build_extension_worlds(
        b8_path,
        opp_path,
        specs=specs,
    )
    fresh_splice_sim, fresh_splice_pol = snapshot_worlds_to_device(
        fresh_worlds,
        fresh_pol_tpl,
        sim_cls=CTDeviceSim,
        device=torch.device("cpu"),
        vector_prepared=False,
    )

    # 2. Build shared worlds once, pass through run_mature_cohort_row_subset_parity
    # (which uses vector_prepared=True internally), then snapshot with vector_prepared=False
    reused_worlds, reused_pol_tpl, reused_meta = build_extension_worlds(
        b8_path,
        opp_path,
        specs=specs,
    )
    parity_res = run_mature_cohort_row_subset_parity(
        b8_path,
        opp_path,
        device="cpu",
        slice_indices=slice_indices,
        branch_horizon=2,
        dry_run=True,
        pre_built_extension_worlds=(reused_worlds, reused_pol_tpl, reused_meta),
    )
    assert parity_res["ok"] is True

    reused_splice_sim, reused_splice_pol = snapshot_worlds_to_device(
        reused_worlds,
        reused_pol_tpl,
        sim_cls=CTDeviceSim,
        device=torch.device("cpu"),
        vector_prepared=False,
    )

    # 3. Verify every sim tensor and vector_runtime tensor is bitwise tensor-equal
    for field_name in _ENV_TENSOR_FIELDS:
        t_fresh = getattr(fresh_splice_sim, field_name, None)
        t_reused = getattr(reused_splice_sim, field_name, None)
        if isinstance(t_fresh, torch.Tensor) and isinstance(t_reused, torch.Tensor):
            assert torch.equal(t_fresh, t_reused), f"Splice sim tensor mismatch on {field_name}"

    for vr_field in _VECTOR_RUNTIME_ENV_FIELDS:
        v_fresh = getattr(fresh_splice_pol.vector_runtime, vr_field, None)
        v_reused = getattr(reused_splice_pol.vector_runtime, vr_field, None)
        if isinstance(v_fresh, torch.Tensor) and isinstance(v_reused, torch.Tensor):
            assert torch.equal(
                v_fresh, v_reused
            ), f"Splice vector_runtime tensor mismatch on {vr_field}"

    # 4. Verify run_step_b1b_smoke calls build_extension_worlds exactly once
    build_calls: list[int] = []
    orig_build = smoke_mod.build_extension_worlds

    def _counting_build(*args: Any, **kwargs: Any) -> Any:
        build_calls.append(1)
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(smoke_mod, "build_extension_worlds", _counting_build)
    smoke_res = run_step_b1b_smoke(
        tmp_path / "smoke_single_build",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=slice_indices,
        bench_env_steps=1,
        bench_label_batches=1,
        bench_learner_updates=2,
        dry_run=True,
    )
    assert len(build_calls) == 1
    assert smoke_res["parity_check"]["ok"] is True


def test_verify_cuda_native_extensions_scopes_arch_and_keys_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 2: Verify verify_cuda_native_extensions keeps validation of the declared/default
    8.0;8.9;9.0 arch list, scopes TORCH_CUDA_ARCH_LIST to the active device's compute
    capability from torch.cuda.get_device_capability during cpp_extension.load, keys
    TORCH_EXTENSIONS_DIR by arch so scoped builds cannot collide with stale multi-arch
    builds, records declared + effective arch, and fails loudly on unexpected capability."""
    import os

    import research.gpu_sitl.open_stop_gpu as open_stop_gpu_mod
    from research.gpu_sim_spike import event_cuda, flood_cuda, vector_geometry
    from research.gpu_sitl.open_stop_gpu import verify_cuda_native_extensions

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0")
    base_ext_dir = tmp_path / "torch_ext_cache"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(base_ext_dir))

    seen_load_env: list[tuple[str, str, str]] = []

    def _fake_flood_load() -> object:
        seen_load_env.append(
            (
                "flood",
                os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
                os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            )
        )
        return object()

    def _fake_vec_load() -> object:
        seen_load_env.append(
            (
                "vector",
                os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
                os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            )
        )
        return object()

    def _fake_event_load() -> object:
        seen_load_env.append(
            (
                "event",
                os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
                os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            )
        )
        return object()

    monkeypatch.setattr(flood_cuda, "_load", _fake_flood_load)
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", _fake_vec_load)
    monkeypatch.setattr(open_stop_gpu_mod, "_load_event_cuda_extension", _fake_event_load)
    monkeypatch.setattr(event_cuda, "_EXTENSION", None, raising=False)

    # 1. Capability (8, 0) -> A100: scopes build to "8.0" and gpu_sitl_sm_80
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: (8, 0))
    res_80 = verify_cuda_native_extensions("cuda")
    assert res_80["declared_torch_cuda_arch_list"] == "8.0;8.9;9.0"
    assert res_80["torch_cuda_arch_list"] == "8.0;8.9;9.0"
    assert res_80["effective_build_cuda_arch"] == "8.0"
    assert res_80["device_capability"] == [8, 0]
    assert res_80["scoped_torch_extensions_dir"].endswith("gpu_sitl_sm_80")
    assert res_80["event_cuda_loaded"] is True
    # During load, TORCH_CUDA_ARCH_LIST was scoped to "8.0"
    assert len(seen_load_env) == 3
    for _kind, arch_during_load, ext_dir_during_load in seen_load_env:
        assert arch_during_load == "8.0"
        assert ext_dir_during_load.endswith("gpu_sitl_sm_80")
    # After verify_cuda_native_extensions returns, declared TORCH_CUDA_ARCH_LIST is preserved
    assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.0;8.9;9.0"
    assert os.environ.get("TORCH_EXTENSIONS_DIR", "").endswith("gpu_sitl_sm_80")

    # 2. Capability (8, 9) -> L4: scopes build to "8.9" and a distinct cache dir gpu_sitl_sm_89
    seen_load_env.clear()
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(base_ext_dir))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: (8, 9))
    res_89 = verify_cuda_native_extensions("cuda")
    assert res_89["effective_build_cuda_arch"] == "8.9"
    assert res_89["scoped_torch_extensions_dir"].endswith("gpu_sitl_sm_89")
    assert res_89["scoped_torch_extensions_dir"] != res_80["scoped_torch_extensions_dir"]
    for _kind, arch_during_load, ext_dir_during_load in seen_load_env:
        assert arch_during_load == "8.9"
        assert ext_dir_during_load.endswith("gpu_sitl_sm_89")

    # 3. Capability (9, 0) -> H100: scopes build to "9.0" and gpu_sitl_sm_90
    seen_load_env.clear()
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(base_ext_dir))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: (9, 0))
    res_90 = verify_cuda_native_extensions("cuda")
    assert res_90["effective_build_cuda_arch"] == "9.0"
    assert res_90["scoped_torch_extensions_dir"].endswith("gpu_sitl_sm_90")

    # 4. Unexpected capabilities (e.g. T4 7.5 or A10G 8.6) must fail loudly
    for bad_cap in ((7, 5), (8, 6), (10, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None, c=bad_cap: c)
        with pytest.raises(RuntimeError, match="Unsupported CUDA compute capability"):
            verify_cuda_native_extensions("cuda")

    # 5. Missing arch from declared TORCH_CUDA_ARCH_LIST still fails loudly before build
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: (8, 0))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.9")
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST='8.0;8.9' is missing"):
        verify_cuda_native_extensions("cuda")


def test_stage_prior_attempt_artifacts_orders_by_tick_and_updates_steps_before_mtime(
    tmp_path: Path,
) -> None:
    """Task 3 (Review Finding 2): In prepare._stage_prior_attempt_artifacts, choose the latest
    probe tick checkpoint by parsed tick and the latest training checkpoint by
    (updates_completed, collection_steps) from its .meta.json, with mtime only as last tie-break."""
    import os

    from research.gpu_sitl.prepare import _stage_prior_attempt_artifacts
    from research.gpu_sitl.probe_bank import verify_file_sha256, write_atomic_with_sha256
    from research.gpu_sitl.seed_check import PROBE32_NAMESPACE, TRAIN_NAMESPACE

    prior_dir = tmp_path / "prior_attempt"
    ckpt_dir = prior_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _write_verified_bytes(path: Path, payload: bytes, mtime: float) -> str:
        sha = write_atomic_with_sha256(
            path,
            lambda tmp, data=payload: Path(tmp).write_bytes(data),
            write_complete_marker=True,
        )
        for p in (path, Path(f"{path}.sha256"), Path(f"{path}.complete")):
            os.utime(p, (mtime, mtime))
        return sha

    def _write_verified_json(path: Path, obj: dict[str, Any], mtime: float) -> str:
        data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
        return _write_verified_bytes(path, data, mtime)

    # 1. Create probe tick checkpoints with mtimes deliberately in REVERSE order:
    # t0001 gets newest mtime (3000.0), t0002 gets 2000.0, t0003 gets oldest mtime (1000.0).
    sha_t1 = _write_verified_bytes(
        ckpt_dir / "probe32_bank.pt.t0001.ckpt.pt", b"probe-tick-1", mtime=3000.0
    )
    sha_t3 = _write_verified_bytes(
        ckpt_dir / "probe32_bank.pt.t0003.ckpt.pt", b"probe-tick-3", mtime=1000.0
    )
    sha_t2 = _write_verified_bytes(
        ckpt_dir / "probe32_bank.pt.t0002.ckpt.pt", b"probe-tick-2", mtime=2000.0
    )
    assert len({sha_t1, sha_t2, sha_t3}) == 3

    # 2. Create training checkpoints where mtimes are deliberately in the WRONG order:
    # - u0100_s000500: updates=100, steps=500, newest mtime=5000.0
    # - u0100_s001205: updates=100, steps=1205, middle mtime=4000.0
    # - u0200_s002100: updates=200, steps=2100, oldest mtime=1500.0
    ckpt_specs = [
        ("sitl_state_u0100_s000500", 100, 500, 5000.0, ["unit_u100_s500"]),
        ("sitl_state_u0100_s001205", 100, 1205, 4000.0, ["unit_u100_s500", "unit_u100_s1205"]),
        (
            "sitl_state_u0200_s002100",
            200,
            2100,
            1500.0,
            ["unit_u100_s500", "unit_u100_s1205", "unit_u200_s2100"],
        ),
    ]
    ckpt_shas: dict[str, str] = {}
    for stem, upd, steps, mtime, unit_ids in ckpt_specs:
        pt_path = ckpt_dir / f"{stem}.pt"
        pt_sha = _write_verified_bytes(pt_path, f"payload-{stem}".encode("utf-8"), mtime=mtime)
        ckpt_shas[stem] = pt_sha
        meta_path = ckpt_dir / f"{stem}.meta.json"
        _write_verified_json(
            meta_path,
            {
                "namespace": TRAIN_NAMESPACE,
                "probe_namespace": PROBE32_NAMESPACE,
                "updates_completed": upd,
                "collection_steps": steps,
                "sha256": pt_sha,
                "world_seeds": [11, 22, 33, 44],
                "verified_units": [{"unit_id": uid, "sha256": f"sha-{uid}"} for uid in unit_ids],
            },
            mtime=mtime,
        )

    # Case A: Stage with all 3 checkpoints present -> must pick t0003 and u0200_s002100
    pkt_inputs_a = tmp_path / "pkt_a" / "inputs"
    inputs_list_a: list[dict[str, str]] = []
    ledger_a: dict[str, Any] = {"verified_units": []}
    _stage_prior_attempt_artifacts(prior_dir, pkt_inputs_a, inputs_list_a, ledger_a)

    assert ledger_a["probe32_harvest_ckpt"]["sha256"] == sha_t3
    assert verify_file_sha256(pkt_inputs_a / "probe32_bank.pt.ckpt.pt") == sha_t3
    assert ledger_a["resume_checkpoint"]["relpath"] == "sitl_state_u0200_s002100.pt"
    assert ledger_a["resume_checkpoint"]["sha256"] == ckpt_shas["sitl_state_u0200_s002100"]
    assert [u["unit_id"] for u in ledger_a["verified_units"]][:3] == [
        "unit_u100_s500",
        "unit_u100_s1205",
        "unit_u200_s2100",
    ]

    # Case B: Remove u0200_s002100 -> must pick u0100_s001205 over u0100_s000500 by collection_steps
    # even though u0100_s000500 has a newer mtime (5000.0 vs 4000.0)
    for suffix in (
        ".pt",
        ".pt.sha256",
        ".pt.complete",
        ".meta.json",
        ".meta.json.sha256",
        ".meta.json.complete",
    ):
        (ckpt_dir / f"sitl_state_u0200_s002100{suffix}").unlink()

    pkt_inputs_b = tmp_path / "pkt_b" / "inputs"
    inputs_list_b: list[dict[str, str]] = []
    ledger_b: dict[str, Any] = {"verified_units": []}
    _stage_prior_attempt_artifacts(prior_dir, pkt_inputs_b, inputs_list_b, ledger_b)
    assert ledger_b["resume_checkpoint"]["relpath"] == "sitl_state_u0100_s001205.pt"
    assert ledger_b["resume_checkpoint"]["sha256"] == ckpt_shas["sitl_state_u0100_s001205"]
    assert [u["unit_id"] for u in ledger_b["verified_units"]][:2] == [
        "unit_u100_s500",
        "unit_u100_s1205",
    ]

    # Case C: Tie-break by mtime when (updates_completed, collection_steps) are identical
    tie_stem = "sitl_state_sig_u0100_s001205"
    tie_sha = _write_verified_bytes(
        ckpt_dir / f"{tie_stem}.pt", b"payload-tie-newer-mtime", mtime=9000.0
    )
    _write_verified_json(
        ckpt_dir / f"{tie_stem}.meta.json",
        {
            "namespace": TRAIN_NAMESPACE,
            "probe_namespace": PROBE32_NAMESPACE,
            "updates_completed": 100,
            "collection_steps": 1205,
            "sha256": tie_sha,
            "world_seeds": [11, 22, 33, 44],
            "verified_units": [{"unit_id": "unit_tie", "sha256": "sha-tie"}],
        },
        mtime=9000.0,
    )
    pkt_inputs_c = tmp_path / "pkt_c" / "inputs"
    inputs_list_c: list[dict[str, str]] = []
    ledger_c: dict[str, Any] = {"verified_units": []}
    _stage_prior_attempt_artifacts(prior_dir, pkt_inputs_c, inputs_list_c, ledger_c)
    assert ledger_c["resume_checkpoint"]["relpath"] == f"{tie_stem}.pt"
    assert ledger_c["resume_checkpoint"]["sha256"] == tie_sha


def test_b1b_smoke_emits_per_phase_wall_time_stamps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Task 5: Verify run_step_b1b_smoke logs per-phase PHASE_START and PHASE_END lines
    with elapsed_s since process start and phase_s duration."""
    from research.gpu_sitl.smoke import run_step_b1b_smoke

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    run_step_b1b_smoke(
        tmp_path / "smoke_phase_logs",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=(0, 8),
        bench_env_steps=1,
        bench_label_batches=1,
        bench_learner_updates=2,
        dry_run=True,
    )
    captured = capsys.readouterr().out
    phase_lines = [
        line for line in captured.splitlines() if line.startswith("[gpu_sitl.smoke] PHASE_")
    ]
    started_phases = [
        line.split("phase=")[1].split()[0]
        for line in phase_lines
        if "PHASE_START" in line and "elapsed_s=" in line
    ]
    ended_phases = [
        line.split("phase=")[1].split()[0]
        for line in phase_lines
        if "PHASE_END" in line and "elapsed_s=" in line and "phase_s=" in line
    ]
    expected_phases = {
        "init_and_extensions",
        "build_mature_extension_worlds",
        "parity_check",
        "parity_1_v8_flood",
        "parity_2_components",
        "parity_3_vector_runtime",
        "parity_4_open_stop_h32",
        "bundle_and_bench_warmup",
        "bench_1_env_steps",
        "mature_warmup_and_splice",
        "bench_2_label_batches_and_rescue_rate",
        "bench_3_probe32_scoring_suffixes",
        "bench_4_learner_updates",
        "write_outputs",
    }
    assert expected_phases <= set(started_phases)
    assert expected_phases <= set(ended_phases)


def test_review_search_findings_se1_to_se9(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression coverage for confirmed search/seed_check review findings SE1-SE7, SE9."""
    import copy as copy_mod

    import research.gpu_sitl.open_stop_gpu as osg
    import research.gpu_sitl.seed_check as sc
    from research.gpu_sitl.probe_bank import build_watch6_ct_device_bundle

    b8_path, opp_path = resolve_product_checkpoints()
    seeds2 = preflight_seeds((0, 1))
    sim2, pol2 = build_watch6_ct_device_bundle(seeds2, b8_path, opp_path, device="cpu")

    # SE1: move_device_policy_to_device must not deepcopy networks already on target_dev.
    deepcopy_calls = 0
    orig_deepcopy = copy_mod.deepcopy

    def _counting_deepcopy(obj: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal deepcopy_calls
        deepcopy_calls += 1
        return orig_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(osg.copy, "deepcopy", _counting_deepcopy)
    pol_chk = pol2.repeat_interleave(1)
    pol_chk.device = torch.device("meta")  # mismatched wrapper attr while params live on cpu
    osg.move_device_policy_to_device(pol_chk, "cpu")
    assert pol_chk.device == torch.device("cpu")
    assert deepcopy_calls == 0
    monkeypatch.setattr(osg.copy, "deepcopy", orig_deepcopy)

    # SE2: batched_open_stop_h32 must fail closed when policy E != sim.E (not modulo-wrap).
    bad_pol = pol2.repeat_interleave(1)
    osg.move_device_policy_to_device(bad_pol, "cpu", e_count=1)
    assert int(bad_pol.vector_runtime.controlled.shape[0]) == 1
    with pytest.raises(RuntimeError, match="controlled E=1 != sim.E=2"):
        batched_open_stop_h32(
            sim2,
            bad_pol,
            world_seeds=seeds2,
            decisions=[0, 0],
            branch_horizon=1,
        )

    # SE3: compact_sim_and_policy and concat_sims_and_policies must not emit inference tensors,
    # must reconcile heterogeneous _prepared/vector_prepared, and tolerate vector_runtime=None.
    sim_a, pol_a = build_watch6_ct_device_bundle([seeds2[0]], b8_path, opp_path, device="cpu")
    sim_b, pol_b = build_watch6_ct_device_bundle([seeds2[1]], b8_path, opp_path, device="cpu")
    sim_a.prepare()
    pol_a.vector_runtime.prepare(sim_a)
    pol_a.vector_prepared = True
    assert sim_a._prepared is True and sim_b._prepared is False
    cat_sim, cat_pol = osg.concat_sims_and_policies([sim_a, sim_b], [pol_a, pol_b])
    assert not cat_sim.alive.is_inference()
    assert cat_sim._prepared is True
    assert cat_pol.vector_prepared is True
    osg.compact_sim_and_policy(cat_sim, cat_pol, torch.tensor([1], dtype=torch.long))
    assert not cat_sim.alive.is_inference()
    _, _, hero_act = cat_pol.evaluate(cat_sim)
    cat_sim.step(cat_pol.actions(cat_sim, hero_act))

    pol_none_a = pol_a.repeat_interleave(1)
    pol_none_b = pol_b.repeat_interleave(1)
    pol_none_a.vector_runtime = None
    pol_none_b.vector_runtime = None
    cat_sim_nr, cat_pol_nr = osg.concat_sims_and_policies([sim_a, sim_b], [pol_none_a, pol_none_b])
    assert int(cat_sim_nr.E) == 2 and cat_pol_nr.vector_runtime is None

    # SE4 + SE9: default scan_code_roots scans research/gpu_sitl (not docs), checks heldout refs,
    # propagates index_horizon >= 256, and audit_shakeout_seeds guards preflight namespaces.
    b1b_rep = audit_b1b_seeds(index_horizon=600)
    assert b1b_rep["ok"] is True
    assert b1b_rep["heldout_code_reference_hits"] == {}
    assert b1b_rep["namespaces"][PROBE32_NAMESPACE]["prefix_horizon"] == 600
    shake_rep = sc.audit_shakeout_seeds(smoke_envs=2, probe_worlds=2, train_envs=2)
    assert PREFLIGHT_NAMESPACE in shake_rep["forbidden_registered_namespaces_untouched"]
    assert T34_PREFLIGHT_NAMESPACE in shake_rep["forbidden_registered_namespaces_untouched"]

    fake_code_dir = tmp_path / "fake_code"
    fake_code_dir.mkdir()
    (fake_code_dir / "leak.py").write_text(f"LEAK = {train_seeds(1)[0]}\n", encoding="utf-8")
    leak_rep = audit_b1b_seeds(scan_code_roots=[fake_code_dir])
    assert leak_rep["ok"] is False
    assert "fake_code/leak.py" in leak_rep["literal_code_hits"]

    with pytest.raises(ValueError, match="Forbidden|held-out|contaminated"):
        sc.assert_allowed_training_seeds(
            namespace=SELECTIVITY_NAMESPACE,
            seeds=[12345],
            context="unit-test",
        )

    # SE5: verify_cuda_native_extensions normalizes Hopper '9.0a' arch token.
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0a")
    ext_cpu = osg.verify_cuda_native_extensions("cpu")
    assert ext_cpu["arch_list_covers_a100_l4_h100"] is True

    # SE6: batched_open_stop_h32 skips terminal openness kernel when 0 branches survive.
    dead_sim = sim2.clone()
    dead_sim.alive[:, 0] = False
    openness_calls = 0
    orig_eval_open = osg.evaluate_terminal_openness_device

    def _counting_open(s: Any) -> dict[str, torch.Tensor]:
        nonlocal openness_calls
        openness_calls += 1
        return orig_eval_open(s)

    monkeypatch.setattr(osg, "evaluate_terminal_openness_device", _counting_open)
    dead_res = batched_open_stop_h32(
        dead_sim,
        pol2,
        world_seeds=seeds2,
        decisions=[0, 0],
        branch_horizon=2,
        compact_dead=False,
    )
    assert not bool(dead_res["raw_survives"].any().item())
    assert openness_calls == 0
    monkeypatch.setattr(osg, "evaluate_terminal_openness_device", orig_eval_open)

    # SE7: margin_loss_gradient_check prepares unprepared sim and skips dead hero rows.
    unprep_sim, _ = build_watch6_ct_device_bundle(seeds2, b8_path, opp_path, device="cpu")
    assert unprep_sim._prepared is False
    unprep_sim.alive[0, 0] = False
    grad_chk = margin_loss_gradient_check(str(b8_path), sim=unprep_sim, device="cpu")
    assert grad_chk["all_non_value_grads_finite_and_nonzero"] is True
    assert 0 not in grad_chk["selected_root_indices"]


def test_review_train_findings_t1_t2_t3_t5_t6_t7_t8_t10(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression coverage for confirmed train/probe_bank findings T1-T3, T5-T8, T10."""
    import research.gpu_sitl.train_sitl as ts
    from research.gpu_sitl.probe_bank import (
        build_watch6_ct_device_bundle,
        write_atomic_with_sha256,
    )
    from src.model.ego2s_network import load_ego2s_checkpoint

    b8_path, opp_path = resolve_product_checkpoints()

    # T1a: write_atomic_with_sha256 must refuse overwriting .complete-marked files,
    # and verify_file_sha256 must validate the digest inside <path>.complete.
    immutable_f = tmp_path / "immutable.json"
    write_atomic_with_sha256(
        immutable_f,
        lambda tmp: tmp.write_text('{"a": 1}\n', encoding="utf-8"),
        write_complete_marker=True,
    )
    with pytest.raises(FileExistsError):
        write_atomic_with_sha256(
            immutable_f,
            lambda tmp: tmp.write_text('{"a": 2}\n', encoding="utf-8"),
            write_complete_marker=False,
        )
    Path(f"{immutable_f}.complete").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete|mismatch"):
        verify_file_sha256(immutable_f, require_complete=True)

    # T2a: CLI _parse_args supports pause/stop/shakeout flags.
    parsed = ts._parse_args(
        [
            "--output-dir",
            str(tmp_path / "cli_out"),
            "--stop-after-updates",
            "2",
            "--stop-after-collection-steps",
            "10",
            "--pause-after-seconds",
            "300",
            "--shakeout",
        ]
    )
    assert parsed.stop_after_updates == 2
    assert parsed.stop_after_collection_steps == 10
    assert parsed.pause_after_seconds == 300.0
    assert parsed.shakeout is True

    # T3a: lazy _last_done tensor and int/tensor fresh_rows restored; pending_rows preserved across
    # wall-clock harvest checkpoints so checkpoint_every_seconds=0.0 matches uninterrupted harvest.
    seeds2 = preflight_seeds((0, 1))
    sim_src, pol_src = build_watch6_ct_device_bundle(seeds2, b8_path, opp_path, device="cpu")
    sim_src.prepare()
    pol_src.vector_runtime.prepare(sim_src)
    pol_src.vector_prepared = True
    _, _, h_act = pol_src.evaluate(sim_src)
    sim_src.step(pol_src.actions(sim_src, h_act))
    pol_src.after_step(sim_src)
    assert isinstance(sim_src._last_done, torch.Tensor)
    captured_state = ts._capture_sim_and_runtime_state(sim_src, pol_src)

    sim_dst, pol_dst = build_watch6_ct_device_bundle(seeds2, b8_path, opp_path, device="cpu")
    sim_dst._last_done = None
    pol_dst.vector_runtime.fresh_rows = 0
    pol_dst.vector_runtime.carried_rows = 0
    ts._restore_sim_and_runtime_state(sim_dst, pol_dst, captured_state)
    assert isinstance(sim_dst._last_done, torch.Tensor)
    assert torch.equal(sim_dst._last_done, sim_src._last_done)
    assert isinstance(pol_dst.vector_runtime.fresh_rows, torch.Tensor)
    assert int(pol_dst.vector_runtime.fresh_rows) == int(pol_src.vector_runtime.fresh_rows)

    pb_unint = tmp_path / "pb_unint.pt"
    pb_ckpt0 = tmp_path / "pb_ckpt0.pt"
    harvest_probe32_bank(
        pb_unint,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=5,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        checkpoint_every_seconds=9999.0,
        dry_run=True,
    )
    harvest_probe32_bank(
        pb_ckpt0,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_worlds=2,
        horizon=5,
        branch_horizon=2,
        ordinary_per_world=3,
        harvest_batch_roots=4,
        checkpoint_every_seconds=0.0,
        dry_run=True,
    )
    b_u = load_probe32_bank(pb_unint)
    b_c = load_probe32_bank(pb_ckpt0)
    assert torch.equal(b_u["ordinary"]["decision_frame"], b_c["ordinary"]["decision_frame"])

    # T1b + T2b: multi-unit pause/resume keeps verified-ledger.json.complete valid,
    # final state checkpoint has unmutated completed_windows, and re-entry is idempotent.
    common = dict(
        probe_bank_path=pb_unint,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        total_updates=2,
        warmup_rescue_roots=1,
        min_unique_rescue_roots=2,
        rescue_batch_size=2,
        ordinary_batch_size=4,
        ordinary_uniform_half=2,
        ordinary_hard_neg_half=2,
        branch_horizon=2,
        tripwire_interval=2,
        decision_window_size=10,
        checkpoint_every_updates=1,
        seed=20261109,
        dry_run=True,
    )
    multi_dir = tmp_path / "multi_unit"
    p1 = run_sitl_training(multi_dir, stop_after_updates=1, **common)
    assert p1["status"] == "PAUSED_FOR_RESUME"
    ledger1_sha = verify_file_sha256(multi_dir / "verified-ledger.json", require_complete=True)
    p2 = run_sitl_training(multi_dir, resume=True, **common)
    assert p2["counts"]["updates_completed"] == 2
    # Original verified-ledger.json + .complete stays digest-intact!
    assert (
        verify_file_sha256(multi_dir / "verified-ledger.json", require_complete=True) == ledger1_sha
    )
    final_ckpt_obj = torch.load(
        p2["final_state_checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert all(not bool(w.get("is_partial_final")) for w in final_ckpt_obj["completed_windows"])

    # Re-entering completed run with resume=True returns identical summary without rewriting files
    res_sha_before = p2["result_json_sha256"]
    p2_again = run_sitl_training(multi_dir, resume=True, **common)
    assert p2_again["result_json_sha256"] == res_sha_before
    assert len(p2_again["unconditional_fault_checks"]["decision_windows"]) == len(
        p2["unconditional_fault_checks"]["decision_windows"]
    )
    with pytest.raises(FileExistsError):
        run_sitl_training(multi_dir, resume=False, **common)

    # T5: <100k decision collapse must trigger STOP_TRIPWIRE_BREACH
    verdict_collapse = ts.compute_step_b1b_pilot_verdict(
        runaway_guard_reached=False,
        updates_completed=1000,
        total_updates=1000,
        completed_windows=[],
        delivered_boost_band_pass=True,
        non_finite_loss_count=0,
        non_finite_q_count=0,
        non_finite_grad_count=0,
        illegal_action_count=0,
        tripwire_history=[
            {
                "all_pass": True,
                "decile_tripwire_pass": True,
                "boost_tripwire_pass": True,
                "decisive_rescue_bar_pass": True,
                "all_q_finite": True,
                "illegal_action_count": 0,
            }
        ],
        tripwire_breached=False,
        stop_ego2s_bv8_inputs_insufficient=False,
        overall_max_single_action_share=0.95,
    )
    assert verdict_collapse == ts.VERDICT_TRIPWIRE_STOP

    # T6: evaluate_probe_bank_tripwires with b8_boost_share == 0 and student_boost_share > 0
    # must serialize cleanly under allow_nan=False.
    net = load_ego2s_checkpoint(str(b8_path), device="cpu")
    with torch.no_grad():
        net.adv.bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 50.0, 40.0, 30.0]))
    zero_boost_bank = {
        "meta": dict(b_u["meta"]),
        "ordinary": {
            **b_u["ordinary"],
            "bare_action": torch.zeros_like(b_u["ordinary"]["bare_action"]),
            "legal_mask": torch.ones_like(b_u["ordinary"]["legal_mask"], dtype=torch.bool),
        },
        "rescue": b_u["rescue"],
    }
    tw_zero = evaluate_probe_bank_tripwires(net, zero_boost_bank, update_step=200, device="cpu")
    assert tw_zero["boost_tripwire_pass"] is False
    json.dumps(tw_zero, allow_nan=False)

    # T7 + T10: non-dry-run rejects tampered checkpoint SHA256; hinge loss rejects a_star == -1.
    tampered_b8 = tmp_path / "tampered_b8.pth"
    tampered_b8.write_bytes(Path(b8_path).read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="SHA256|sha256"):
        harvest_probe32_bank(
            tmp_path / "bad_sha_pb.pt",
            b8_checkpoint=tampered_b8,
            opponent_checkpoint=opp_path,
            dry_run=False,
        )
    with pytest.raises(RuntimeError, match="a_star_action"):
        compute_sitl_hinge_loss(
            net,
            rescue_batch={
                "local_u8": torch.zeros((1, 7, 31, 31), dtype=torch.uint8),
                "global_u8": torch.zeros((1, 4, 37, 37), dtype=torch.uint8),
                "scalars_f32": torch.zeros((1, 60), dtype=torch.float32),
                "legal_mask": torch.ones((1, 6), dtype=torch.bool),
                "unsafe_mask": torch.tensor([[True, False, False, False, False, False]]),
                "a_star_action": torch.tensor([-1], dtype=torch.long),
            },
            ordinary_batch={
                "local_u8": torch.zeros((1, 7, 31, 31), dtype=torch.uint8),
                "global_u8": torch.zeros((1, 4, 37, 37), dtype=torch.uint8),
                "scalars_f32": torch.zeros((1, 60), dtype=torch.float32),
                "legal_mask": torch.ones((1, 6), dtype=torch.bool),
                "bare_action": torch.tensor([1], dtype=torch.long),
            },
        )


def test_review_smoke_findings_sm1_sm2_sm3_sm4_sm5_sm6_sm7_sm10_sm11_sm12(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression coverage for confirmed smoke/prepare/job.sh findings SM1-SM7, SM10-SM12."""
    import os
    import subprocess

    from research.gpu_sitl.prepare import (
        ROOT,
        _stage_prior_attempt_artifacts,
        prepare_b1b_packet,
        runtime_source_paths,
    )
    from research.gpu_sitl.smoke import run_cuda_shakeout, run_step_b1b_smoke

    b8_path, opp_path = resolve_product_checkpoints()

    # SM1: host-only prepare.py must not be shipped in runtime_source_paths.
    src_paths = runtime_source_paths("origin/main")
    assert "research/gpu_sitl/prepare.py" not in src_paths

    # SM2a: job.sh must fail closed when MODE=pilot and verified-ledger.json is missing.
    fake_cli_root = tmp_path / "fake_cli_root"
    fake_in_dir = tmp_path / "fake_in_dir"
    (fake_cli_root / "output").mkdir(parents=True)
    fake_in_dir.mkdir(parents=True)
    job_proc = subprocess.run(
        ["bash", str(ROOT / "research" / "gpu_sitl" / "job.sh")],
        env={
            **os.environ,
            "COLAB_CLI_ROOT": str(fake_cli_root),
            "COLAB_INPUT_DIR": str(fake_in_dir),
            "GPU_SITL_SOURCE_COMMIT": "deadbeef",
            "GPU_SITL_MODE": "pilot",
        },
        capture_output=True,
        text=True,
    )
    assert job_proc.returncode == 2
    assert "verified-ledger.json" in job_proc.stderr

    # SM3 + SM11 + SM12: run_step_b1b_smoke must exclude spliced PREFLIGHT_NAMESPACE rows from
    # online_alive_hero_decisions (2 non-spliced envs * 2 label batches <= 4 decisions),
    # pass splice world seeds for spliced CRN rows, and refuse overwriting a completed smoke dir.
    seen_seed_lists: list[list[int]] = []
    import research.gpu_sitl.smoke as smoke_mod

    orig_batched_os = smoke_mod.batched_open_stop_h32

    def _spy_batched_os(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if "world_seeds" in kwargs:
            seen_seed_lists.append([int(x) for x in kwargs["world_seeds"]])
        return orig_batched_os(*args, **kwargs)

    monkeypatch.setattr(smoke_mod, "batched_open_stop_h32", _spy_batched_os)

    smoke_dir = tmp_path / "smoke_sm"
    smoke_res = run_step_b1b_smoke(
        smoke_dir,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
        num_envs=4,
        branch_horizon=2,
        parity_slice_indices=(0, 8),
        bench_env_steps=2,
        bench_label_batches=2,
        bench_learner_updates=2,
        dry_run=True,
    )
    # 4 envs, 2 spliced mature worlds -> only 2 train_seeds worlds contribute to online decisions
    assert smoke_res["throughput"]["mature_spliced_worlds"] == 2
    assert 1 <= smoke_res["throughput"]["measured_alive_hero_decisions"] <= 4
    # Label batches (after warmup) must use spliced mature world seeds on rows 0..1
    train4 = train_seeds(4)
    assert any(sl[:2] != train4[:2] and sl[2:] == train4[2:] for sl in seen_seed_lists)

    # SM12: re-running smoke in a completed directory must raise FileExistsError immediately
    with pytest.raises(FileExistsError):
        run_step_b1b_smoke(
            smoke_dir,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            num_envs=4,
            branch_horizon=2,
            parity_slice_indices=(0, 8),
            bench_env_steps=1,
            bench_label_batches=1,
            bench_learner_updates=1,
            dry_run=True,
        )

    # SM5: _stage_prior_attempt_artifacts must refuse staging smoke canary_export_probe artifacts!
    with pytest.raises(ValueError, match="canary_export_probe|non-evidence|probe"):
        _stage_prior_attempt_artifacts(
            smoke_dir,
            tmp_path / "bad_stage_inputs",
            [],
            {"verified_units": []},
        )

    # SM4: prepare_b1b_packet(mode="pilot") must verify smoke_result.json .complete + parity_check.ok
    # and reject dry_run smoke when allow_non_evidence_smoke=False.
    with pytest.raises(ValueError, match="non-evidence|dry_run|evidence"):
        prepare_b1b_packet(
            tmp_path / "pkt_reject_dry_smoke",
            commit="origin/main",
            mode="pilot",
            output_dir=tmp_path / "out_reject_dry",
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            smoke_result_json=smoke_res["result_json_path"],
            allow_non_evidence_smoke=False,
        )

    # SM6: both smoke_result_json and base_seconds paths set GPU_SITL_MAX_WALL_SECONDS < timeout * 60
    info_bs = prepare_b1b_packet(
        tmp_path / "pkt_bs",
        commit="origin/main",
        mode="pilot",
        output_dir=tmp_path / "out_bs",
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        base_seconds=600.0,
    )
    mf_bs = json.loads(Path(info_bs["manifest_path"]).read_text(encoding="utf-8"))
    max_wall_bs = float(mf_bs["env"]["GPU_SITL_MAX_WALL_SECONDS"])
    assert 0.0 < max_wall_bs < float(mf_bs["timeout_minutes"]) * 60.0

    # SM10: run_cuda_shakeout phase 6 verifies uninterrupted == resumed determinism
    shake_dir = tmp_path / "shake_sm10"
    shake_res = run_cuda_shakeout(
        shake_dir,
        b8_checkpoint=b8_path,
        opponent_checkpoint=opp_path,
        device="cpu",
    )
    assert shake_res["ok"] is True
    p6 = [p for p in shake_res["phases"] if "phase_6" in p["phase"]][0]
    assert p6["detail"]["deterministic_resume_verified"] is True


@contextlib.contextmanager
def _mps_readonly_spike_compat() -> Any:
    """Adapt float64 and int64 scatter_reduce_ calls inside read-only gpu_sim_spike on MPS."""
    orig_zeros, orig_full, orig_empty = torch.zeros, torch.full, torch.empty
    orig_tensor, orig_as_tensor, orig_arange = torch.tensor, torch.as_tensor, torch.arange
    orig_to, orig_double = torch.Tensor.to, torch.Tensor.double
    orig_scatter_reduce_ = torch.Tensor.scatter_reduce_
    orig_assert_async = torch._assert_async

    def _is_mps(dev: Any) -> bool:
        return dev is not None and torch.device(dev).type == "mps"

    def _wrap_creation(fn: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _is_mps(kwargs.get("device")) and kwargs.get("dtype") == torch.float64:
                kwargs = dict(kwargs, dtype=torch.float32)
            return fn(*args, **kwargs)

        return wrapped

    def wrapped_as_tensor(data: Any, dtype: Any = None, device: Any = None) -> torch.Tensor:
        if _is_mps(device):
            if dtype == torch.float64 or (
                dtype is None
                and (
                    (isinstance(data, np.ndarray) and data.dtype == np.float64)
                    or (isinstance(data, torch.Tensor) and data.dtype == torch.float64)
                )
            ):
                dtype = torch.float32
        return orig_as_tensor(data, dtype=dtype, device=device)

    def wrapped_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        new_args = list(args)
        target_mps = (
            (self.device.type == "mps")
            or any(isinstance(a, (str, torch.device)) and _is_mps(a) for a in new_args)
            or _is_mps(kwargs.get("device"))
        )
        if target_mps:
            new_args = [torch.float32 if a == torch.float64 else a for a in new_args]
            if kwargs.get("dtype") == torch.float64 or (
                self.dtype == torch.float64
                and "dtype" not in kwargs
                and not any(isinstance(a, torch.dtype) for a in new_args)
            ):
                kwargs = dict(kwargs, dtype=torch.float32)
        return orig_to(self, *new_args, **kwargs)

    def wrapped_double(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.device.type == "mps":
            return orig_to(self, dtype=torch.float32)
        return orig_double(self, *args, **kwargs)

    def wrapped_scatter_reduce_(
        self: torch.Tensor,
        dim: int,
        index: torch.Tensor,
        src: torch.Tensor,
        reduce: str,
        *,
        include_self: bool = True,
    ) -> torch.Tensor:
        if self.device.type == "mps" and self.dtype == torch.int64:
            tmp = orig_to(self, dtype=torch.int32).clone()
            orig_scatter_reduce_(
                tmp, dim, index, orig_to(src, dtype=torch.int32), reduce, include_self=include_self
            )
            self.copy_(orig_to(tmp, dtype=torch.int64))
            return self
        return orig_scatter_reduce_(self, dim, index, src, reduce, include_self=include_self)

    def wrapped_assert_async(tensor: torch.Tensor, msg: str) -> None:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mps":
            if not bool(tensor.all().item()):
                raise AssertionError(msg)
            return
        orig_assert_async(tensor, msg)

    torch.zeros = _wrap_creation(orig_zeros)  # type: ignore[assignment]
    torch.full = _wrap_creation(orig_full)  # type: ignore[assignment]
    torch.empty = _wrap_creation(orig_empty)  # type: ignore[assignment]
    torch.tensor = _wrap_creation(orig_tensor)  # type: ignore[assignment]
    torch.arange = _wrap_creation(orig_arange)  # type: ignore[assignment]
    torch.as_tensor = wrapped_as_tensor  # type: ignore[assignment]
    torch.Tensor.to = wrapped_to  # type: ignore[assignment]
    torch.Tensor.double = wrapped_double  # type: ignore[assignment]
    torch.Tensor.scatter_reduce_ = wrapped_scatter_reduce_  # type: ignore[assignment]
    torch._assert_async = wrapped_assert_async  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.zeros, torch.full, torch.empty = orig_zeros, orig_full, orig_empty  # type: ignore[assignment]
        torch.tensor, torch.as_tensor, torch.arange = orig_tensor, orig_as_tensor, orig_arange  # type: ignore[assignment]
        torch.Tensor.to, torch.Tensor.double = orig_to, orig_double  # type: ignore[assignment]
        torch.Tensor.scatter_reduce_ = orig_scatter_reduce_  # type: ignore[assignment]
        torch._assert_async = orig_assert_async  # type: ignore[assignment]


def test_ct_device_sim_step_and_reset_bit_identical_cpu_and_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove optimized CTDeviceSim step + reset_fresh_worlds are 100% bit-identical to the
    pre-optimization reference implementation across all 28 _ENV_TENSOR_FIELDS on CPU and MPS,
    and default move_backend='indexed' on CUDA."""
    from research.gpu_sim_spike import r2_cache
    from research.gpu_sim_spike.device_sim import PHI_LENGTH_DIVISOR, DeviceSim
    from research.gpu_sitl.probe_bank import build_watch6_ct_device_bundle
    from research.sandbox_a_resident.ct_mechanics import _CARDINAL, CTDeviceSim
    from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS

    class _ReferenceCTDeviceSim(CTDeviceSim):
        """Pre-optimization CTDeviceSim methods for before/after bit-identity comparison."""

        def _refill_food_rows(self, active_mask: torch.Tensor) -> None:
            DeviceSim._refill_food_rows(self, active_mask)

        @torch.inference_mode()
        def reset_fresh_worlds(self) -> None:
            pending = ~self.alive & ~self.terminal_slots & (self.respawn_timer <= 0)
            if not bool(pending.any()):
                return
            any_placed = False
            for sidx in range(self.S):
                slot_need = pending[:, sidx]
                if not bool(slot_need.any()):
                    continue
                rows = slot_need.nonzero(as_tuple=False).flatten()
                r2_cache.bump(self)
                sub = object.__new__(DeviceSim)
                sub.__dict__ = self.__dict__.copy()
                for name in _ENV_TENSOR_FIELDS:
                    value = getattr(self, name, None)
                    if isinstance(value, torch.Tensor):
                        setattr(sub, name, value.index_select(0, rows))
                sub.E = int(rows.numel())
                sub.exact_mechanics = self.exact_mechanics
                r2_cache.inherit(sub, self)
                occupied = sub.occupancy()
                unplaced = torch.ones(sub.E, dtype=torch.bool, device=self.device)
                for _ in range(100):
                    if not bool(unplaced.any()):
                        break
                    local_idx = unplaced.nonzero(as_tuple=False).flatten()
                    global_idx = rows.index_select(0, local_idx)
                    cand = self._draw_for_rows(global_idx)
                    cx, cy = cand[:, 0], cand[:, 1]
                    in_bounds = (cx >= 0) & (cx < self.grid_w) & (cy >= 0) & (cy < self.grid_h)
                    safe_x = cx.clamp(0, self.grid_w - 1)
                    safe_y = cy.clamp(0, self.grid_h - 1)
                    free = in_bounds & (occupied[local_idx, safe_y, safe_x] < 0)
                    if bool(free.any()):
                        win_global = global_idx[free]
                        win_local = local_idx[free]
                        win_cand = cand[free]
                        self.bodies[win_global, sidx, 0] = win_cand.to(self.bodies.dtype)
                        self.head_ptr[win_global, sidx] = 0
                        self.seg_count[win_global, sidx] = 1
                        self.length[win_global, sidx] = 1
                        self.direction[win_global, sidx] = 1
                        self.alive[win_global, sidx] = True
                        self.respawn_timer[win_global, sidx] = 0
                        self.boost_frames[win_global, sidx] = 0
                        self.frames_since_food[win_global, sidx] = 0
                        self._reward_prev_length[win_global, sidx] = 1
                        unplaced[win_local] = False
                        any_placed = True
            if any_placed:
                r2_cache.bump(self)

        @torch.inference_mode()
        def step(self, actions: torch.Tensor) -> None:
            actions = actions.long().clamp(0, 5)
            r2_cache.bump(self)
            self.prepare()
            acted, prev_length = self.alive.clone(), self._reward_prev_length.clone()
            new_dir = (
                self.direction
                + torch.where(
                    actions.remainder(3) == 0, -1, torch.where(actions.remainder(3) == 2, 1, 0)
                )
            ) % 4
            self.direction = torch.where(self.alive, new_dir, self.direction)
            cardinal = (
                r2_cache.const(_CARDINAL, torch.long, self.device)
                if r2_cache.enabled(self)
                else torch.tensor(_CARDINAL, dtype=torch.long, device=self.device)
            )
            delta = cardinal[self.direction]
            head0 = self.heads()
            move1 = self.alive
            head1 = head0 + delta
            self._push(head1, move1)
            boost = self.alive & (actions >= 3) & (self.length >= int(self.cfg.min_boost_length))
            head2 = head1 + delta
            self._push(head2, boost)
            self._boosted_this_step = boost
            self.boost_frames = torch.where(boost, self.boost_frames + 1, self.boost_frames)
            burn = (
                boost
                & (self.boost_frames >= int(self.cfg.boost_length_cost_frames))
                & (self.length > 1)
            )
            self.boost_frames = torch.where(
                burn, torch.zeros_like(self.boost_frames), self.boost_frames
            )
            tail_ring = (self.head_ptr - (self.seg_count - 1).clamp_min(0)) % self.cap
            e = torch.arange(self.E, device=self.device)[:, None]
            s = torch.arange(self.S, device=self.device)[None, :]
            paid_tail = self.bodies[e, s, tail_ring]
            self.length -= burn.long()
            extra = burn & (self.seg_count > self.length)
            self.seg_count -= extra.long()
            if self.v2:
                for slot in range(self.S):
                    self._append_food(
                        paid_tail[:, slot],
                        extra[:, slot] & self._in_arena(paid_tail[:, slot]),
                        True,
                    )
            traversed = torch.stack((head1, head2), dim=2)
            traversed_valid = torch.stack((move1, boost), dim=2)
            ate = torch.zeros_like(self.alive)
            ate_corpse = torch.zeros_like(self.alive)
            eq_any = (traversed[:, :, :, None, :] == self.food_cells[:, None, None, :, :]).all(
                dim=-1
            )
            if bool(
                (eq_any & self.food_present[:, None, None, :] & traversed_valid[..., None]).any()
            ):
                food_slots = torch.arange(self.food_capacity, device=self.device)[None, :]
                t_idx = torch.arange(2, device=self.device)[None, :]
                for slot in range(self.S):
                    eq_s = (traversed[:, slot, :, None, :] == self.food_cells[:, None, :, :]).all(
                        dim=-1
                    )
                    available = (
                        eq_s & self.food_present[:, None, :] & traversed_valid[:, slot, :, None]
                    )
                    by_t = available.any(dim=2)
                    first_t = by_t.long().argmax(dim=1)
                    tmask = t_idx == first_t[:, None]
                    candidates = (available & tmask[:, :, None]).any(dim=1)
                    has = candidates.any(dim=1) & self.alive[:, slot]
                    if not bool(has.any()):
                        continue
                    chosen = candidates.long().argmax(dim=1)
                    select = has[:, None] & (food_slots == chosen[:, None])
                    is_corpse = (select & self.corpse_food).any(dim=1)
                    ate[:, slot] = has
                    ate_corpse[:, slot] = is_corpse
                    self.length[:, slot] += has.long()
                    self.food_present &= ~select
                    self.corpse_food &= self.food_present
                    self._compact_food()
                    r2_cache.bump(self)
                    self._refill_food_rows(has)
                    r2_cache.bump(self)
            in_arena = ((~traversed_valid) | self._in_arena(traversed)).all(dim=2)
            body, self_hit, head_pair, body_pair = self._collision_flags_reference(
                head0, traversed, traversed_valid
            )
            death, death_rank, dead, kills, kill_mass = self._resolve_collisions_reference(
                in_arena, self_hit, head_pair, body_pair
            )
            died = dead
            if bool(died.any().item()):
                snake_order = death_rank.argsort(dim=1, stable=True)
                body = body.gather(1, snake_order[:, :, None, None].expand(-1, -1, self.cap, 2))
                died_ordered = died.gather(1, snake_order)
                seg_ordered = self.seg_count.gather(1, snake_order)
                offsets = torch.arange(self.cap, device=self.device)[None, None, :]
                corpse_mask = died_ordered[..., None] & (offsets < seg_ordered[..., None])
                if not self.v2:
                    corpse_mask &= offsets.remainder(2) == 0
                self._append_many(
                    body.reshape(self.E, self.S * self.cap, 2),
                    corpse_mask.reshape(self.E, self.S * self.cap),
                    self.v2,
                )
            phi_prev = prev_length.float() / PHI_LENGTH_DIVISOR
            phi_new = torch.where(
                died, torch.zeros_like(phi_prev), self.length.float() / PHI_LENGTH_DIVISOR
            )
            self._last_reward = (
                float(self.cfg.gamma) * phi_new
                - phi_prev
                + torch.where(
                    died,
                    torch.full_like(phi_new, float(self.cfg.death_value)),
                    torch.zeros_like(phi_new),
                )
                + kill_mass.float() * float(self.cfg.kill_scale)
            )
            self.frames_since_food = torch.where(
                died,
                self.frames_since_food,
                torch.where(
                    ate,
                    torch.zeros_like(self.frames_since_food),
                    self.frames_since_food + self.alive.long(),
                ),
            )
            self._reward_prev_length = torch.where(died, self._reward_prev_length, self.length)
            self.alive &= ~died
            self.respawn_timer = torch.where(
                died,
                torch.full_like(self.respawn_timer, int(self.cfg.frame_rate)),
                self.respawn_timer,
            )
            self._last_done, self._last_death_cause, self._last_kills, self._last_food_ate = (
                died,
                death,
                kills,
                ate,
            )
            self._last_ambient_food_ate = ate & ~ate_corpse
            self._last_corpse_food_ate = ate_corpse
            self._last_transition_valid = acted
            self._prepared = False
            r2_cache.bump(self)

    b8_path, opp_path = resolve_product_checkpoints()
    if not Path(b8_path).exists() or not Path(opp_path).exists():
        pytest.skip("b8-s3 or opponent checkpoint not present")

    # Verify CTDeviceSim.__init__ and CTDeviceSim.from_reference default move_backend="indexed" on CUDA
    from research.sandbox_a_resident.training_sim import TrainingDeviceSim
    from src.simd_env.batch_sim import BatchSimConfig

    seen_init_kwargs: list[dict[str, Any]] = []
    orig_tds_init = TrainingDeviceSim.__init__

    def _spy_tds_init(self: Any, *args: Any, **kwargs: Any) -> None:
        seen_init_kwargs.append(dict(kwargs))
        cpu_kwargs = dict(kwargs, device="cpu")
        orig_tds_init(self, *args, **cpu_kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(TrainingDeviceSim, "__init__", _spy_tds_init)
        cuda_init_sim = CTDeviceSim(BatchSimConfig(num_envs=2, num_snakes=6), device="cuda")
        assert seen_init_kwargs[-1].get("move_backend") == "indexed"
        assert cuda_init_sim.move_backend == "indexed"

        orig_from_ref = TrainingDeviceSim.from_reference

        def _fake_cuda_from_ref(cls_arg: Any, sim_arg: Any, **kw: Any) -> Any:
            res_sim = orig_from_ref(sim_arg, device="cpu", repeat=int(kw.get("repeat", 1)))
            res_sim.capacity_overflow = torch.zeros_like(res_sim.alive)
            res_sim.device = torch.device("cuda:0")
            return res_sim

        mp.setattr(TrainingDeviceSim, "from_reference", classmethod(_fake_cuda_from_ref))
        from src.simd_env.grid_sim import GridBatchSim

        cuda_ref_sim = CTDeviceSim.from_reference(
            GridBatchSim(BatchSimConfig(num_envs=1, num_snakes=6)),
            device="cuda",
        )
        assert cuda_ref_sim.move_backend == "indexed"

    seeds = preflight_seeds(tuple(range(16)))
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev_name in devices:
        ctx = _mps_readonly_spike_compat() if dev_name == "mps" else contextlib.nullcontext()
        with ctx:
            dev = torch.device("mps", 0) if dev_name == "mps" else torch.device("cpu")
            sim_fast, _ = build_watch6_ct_device_bundle(seeds, b8_path, opp_path, device=dev)
            # Test reset_fresh_worlds bit-identity first
            sim_fast.alive[::2, :] = False
            sim_fast.respawn_timer[::2, :] = 0
            sim_ref_reset = sim_fast.clone()
            sim_ref_reset.__class__ = _ReferenceCTDeviceSim
            sim_fast.reset_fresh_worlds()
            sim_ref_reset.reset_fresh_worlds()
            for field_name in _ENV_TENSOR_FIELDS:
                assert torch.equal(
                    getattr(sim_fast, field_name), getattr(sim_ref_reset, field_name)
                ), f"[{dev_name}] reset_fresh_worlds mismatch on {field_name}"

            # Exercise boost burn paid_tail corpse drop, food eat/refill, deaths, and respawns
            sim_fast.length[:, :] = 8
            sim_fast._reward_prev_length[:, :] = 8
            sim_fast.boost_frames[:, :] = int(sim_fast.cfg.boost_length_cost_frames) - 1
            sim_fast.alive[:, 4] = False
            sim_fast.respawn_timer[:, 4] = 2

            sim_ref = sim_fast.clone()
            sim_ref.__class__ = _ReferenceCTDeviceSim
            r2_cache.enable(sim_fast)
            r2_cache.enable(sim_ref)

            gen = torch.Generator(device="cpu").manual_seed(20261110)
            for step_i in range(24):
                act = torch.randint(0, 6, (16, 6), generator=gen, dtype=torch.long).to(dev)
                sim_fast.step(act)
                sim_ref.step(act)
                for field_name in _ENV_TENSOR_FIELDS:
                    assert torch.equal(
                        getattr(sim_fast, field_name), getattr(sim_ref, field_name)
                    ), f"[{dev_name}] step {step_i} mismatch on {field_name}"


def test_fast_v8_decide_and_fast_device_policy_bit_identical_cpu_and_mps() -> None:
    """Prove fast_v8_decide and FastDevicePolicy match v8_decide/DevicePolicy bit-for-bit."""
    from dataclasses import fields as dc_fields

    from research.gpu_sim_spike import r2_cache, v8_device
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sitl.open_stop_gpu import (
        FastDevicePolicy,
        _maybe_cpu_fast_flood,
        fast_v8_decide,
    )
    from research.gpu_sitl.preflight import (
        MATURE_EXTENSION_SPECS,
        build_extension_worlds,
        snapshot_worlds_to_device,
    )
    from research.gpu_sitl.smoke import DEFAULT_MATURE_SLICE_INDICES

    b8_path, opp_path = resolve_product_checkpoints()
    specs = [MATURE_EXTENSION_SPECS[i] for i in DEFAULT_MATURE_SLICE_INDICES[:4]]
    worlds, pol_tpl, _ = build_extension_worlds(
        Path(b8_path),
        Path(opp_path),
        specs=specs,
        cpu_featurizer_backend="python",
        ego_raster_backend="numpy",
    )

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev_name in devices:
        ctx = _mps_readonly_spike_compat() if dev_name == "mps" else _maybe_cpu_fast_flood(True)
        with ctx:
            dev = torch.device("mps", 0) if dev_name == "mps" else torch.device("cpu")
            sim, pol_fast = snapshot_worlds_to_device(
                worlds * 2, pol_tpl, sim_cls=CTDeviceSim, device=dev, vector_prepared=False
            )
            # Mark one opponent dead to prove dead-slot parity as well
            sim.alive[1, 3] = False
            r2_cache.enable(sim)
            assert type(pol_fast) is FastDevicePolicy

            pol_ref = pol_fast.repeat_interleave(1)
            pol_ref.__class__ = DevicePolicy

            e = int(sim.E)
            hero_rows = torch.stack(
                (
                    torch.arange(e, dtype=torch.long, device=dev),
                    torch.zeros(e, dtype=torch.long, device=dev),
                ),
                dim=1,
            )
            opp_rows = torch.stack(
                (
                    torch.arange(e, dtype=torch.long, device=dev).repeat(5),
                    torch.arange(1, 6, dtype=torch.long, device=dev).repeat_interleave(e),
                ),
                dim=1,
            )
            gen = torch.Generator(device="cpu").manual_seed(20261110)
            q_hero = torch.randn((e, 6), generator=gen, dtype=torch.float32).to(dev)
            q_opp = torch.randn((e * 5, 6), generator=gen, dtype=torch.float32).to(dev)
            adv_all = sim.advisory_mask()
            mask_hero = adv_all[:, 0]
            mask_opp = adv_all[opp_rows[:, 0], opp_rows[:, 1]]

            for feat_flag in (True, False):
                ref_h = v8_device.v8_decide(
                    sim, hero_rows, q_hero, mask_hero, 8.0, features=feat_flag
                )
                cand_h = fast_v8_decide(
                    sim,
                    hero_rows,
                    q_hero,
                    mask_hero,
                    8.0,
                    features=feat_flag,
                    slot_layout="hero",
                )
                for f in dc_fields(ref_h):
                    assert torch.equal(
                        getattr(ref_h, f.name), getattr(cand_h, f.name)
                    ), f"[{dev_name} feat={feat_flag}] hero field mismatch: {f.name}"
                if feat_flag:
                    assert torch.equal(ref_h.feature_planes(), cand_h.feature_planes())

                ref_o = v8_device.v8_decide(sim, opp_rows, q_opp, mask_opp, 8.0, features=feat_flag)
                cand_o = fast_v8_decide(
                    sim,
                    opp_rows,
                    q_opp,
                    mask_opp,
                    8.0,
                    features=feat_flag,
                    slot_layout="opp_slot_major",
                )
                for f in dc_fields(ref_o):
                    assert torch.equal(
                        getattr(ref_o, f.name), getattr(cand_o, f.name)
                    ), f"[{dev_name} feat={feat_flag}] opp field mismatch: {f.name}"

            # Multi-step rollout comparing FastDevicePolicy vs DevicePolicy bit-for-bit
            sim_ref = sim.clone()
            r2_cache.enable(sim_ref)
            for step_idx in range(4):
                sim.prepare()
                sim_ref.prepare()
                q_f, leg_f, bare_f = pol_fast.evaluate(sim)
                q_r, leg_r, bare_r = pol_ref.evaluate(sim_ref)
                assert torch.equal(q_f, q_r), f"[{dev_name}] step {step_idx} evaluate Q mismatch"
                assert torch.equal(leg_f, leg_r)
                assert torch.equal(bare_f, bare_r)

                act_f = pol_fast.actions(sim, bare_f)
                act_r = pol_ref.actions(sim_ref, bare_r)
                assert torch.equal(act_f, act_r), f"[{dev_name}] step {step_idx} actions mismatch"
                sim.step(act_f)
                sim_ref.step(act_r)
                pol_fast.after_step(sim)
                pol_ref.after_step(sim_ref)


def test_fast_device_vector_runtime_and_open_stop_h32_bit_identical_cpu_and_mps() -> None:
    """Prove FastDeviceVectorRuntime and batched_open_stop_h32 match reference on CPU and MPS."""
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from research.gpu_sitl import open_stop_gpu as osg
    from research.gpu_sitl.open_stop_gpu import (
        FastDeviceVectorRuntime,
        _maybe_cpu_fast_flood,
        batched_open_stop_h32,
        compact_sim_and_policy,
    )
    from research.gpu_sitl.preflight import (
        MATURE_EXTENSION_SPECS,
        build_extension_worlds,
        snapshot_worlds_to_device,
    )
    from research.gpu_sitl.smoke import DEFAULT_MATURE_SLICE_INDICES

    b8_path, opp_path = resolve_product_checkpoints()
    specs = [MATURE_EXTENSION_SPECS[i] for i in DEFAULT_MATURE_SLICE_INDICES[:4]]
    worlds, pol_tpl, _ = build_extension_worlds(
        Path(b8_path),
        Path(opp_path),
        specs=specs,
        cpu_featurizer_backend="python",
        ego_raster_backend="numpy",
    )
    slice_seeds = [int(w.seed) for w in worlds]
    slice_decisions = [int(w.sim.frame[0]) for w in worlds]

    # 1. Prove cpu_fast_flood=True vs cpu_fast_flood=False bit-identity on CPU (parity_4 speedup)
    sim_cpu, pol_cpu = snapshot_worlds_to_device(
        worlds, pol_tpl, sim_cls=CTDeviceSim, device="cpu", vector_prepared=False
    )
    cert_slow = batched_open_stop_h32(
        sim_cpu,
        pol_cpu,
        world_seeds=slice_seeds,
        decisions=slice_decisions,
        branch_horizon=6,
        cpu_fast_flood=False,
        return_host_records=False,
    )
    cert_fast = batched_open_stop_h32(
        sim_cpu,
        pol_cpu,
        world_seeds=slice_seeds,
        decisions=slice_decisions,
        branch_horizon=6,
        cpu_fast_flood=True,
        return_host_records=False,
    )
    for key in (
        "certified",
        "raw_survives",
        "safe_mask",
        "unsafe_mask",
        "rescue_mask",
        "ordinary_mask",
        "bare_actions",
        "a_star_actions",
        "v8_best_safe_area",
        "max_pocket_r",
        "root_q",
    ):
        assert torch.equal(cert_slow[key], cert_fast[key]), f"cpu_fast_flood mismatch on {key}"

    # 2. Prove FastDeviceVectorRuntime + FastDevicePolicy + _compact_sim_and_policy_fast
    # match DeviceVectorRuntime + DevicePolicy + compact_sim_and_policy on CPU and MPS
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev_name in devices:
        ctx = _mps_readonly_spike_compat() if dev_name == "mps" else _maybe_cpu_fast_flood(True)
        with ctx:
            dev = torch.device("mps", 0) if dev_name == "mps" else torch.device("cpu")
            sim_d, pol_d = snapshot_worlds_to_device(
                worlds, pol_tpl, sim_cls=CTDeviceSim, device=dev, vector_prepared=False
            )
            rt_fast = pol_d.vector_runtime
            assert type(rt_fast) is FastDeviceVectorRuntime
            rt_base = rt_fast.repeat_interleave(1)
            rt_base.__class__ = DeviceVectorRuntime
            sim_rt = sim_d.clone()
            sim_rt.prepare()
            rt_fast.prepare(sim_rt)
            rt_base.prepare(sim_rt)
            assert torch.equal(rt_fast.sel_state, rt_base.sel_state)
            assert torch.equal(rt_fast.sel_mask, rt_base.sel_mask)

            res_fast = batched_open_stop_h32(
                sim_d,
                pol_d,
                world_seeds=slice_seeds,
                decisions=slice_decisions,
                branch_horizon=6,
                cpu_fast_flood=True,
                return_host_records=False,
            )

            # Force unoptimized DevicePolicy + original compact_sim_and_policy
            orig_compact = osg._compact_sim_and_policy_fast
            orig_move = osg.move_device_policy_to_device

            def _move_plain(pol_arg: Any, dev_arg: Any, **kw: Any) -> Any:
                out_p = orig_move(pol_arg, dev_arg, **kw)
                out_p.__class__ = DevicePolicy
                return out_p

            try:
                osg._compact_sim_and_policy_fast = compact_sim_and_policy
                osg.move_device_policy_to_device = _move_plain
                pol_plain = pol_d.repeat_interleave(1)
                pol_plain.__class__ = DevicePolicy
                res_plain = batched_open_stop_h32(
                    sim_d,
                    pol_plain,
                    world_seeds=slice_seeds,
                    decisions=slice_decisions,
                    branch_horizon=6,
                    cpu_fast_flood=True,
                    return_host_records=False,
                )
            finally:
                osg._compact_sim_and_policy_fast = orig_compact
                osg.move_device_policy_to_device = orig_move

            for key in (
                "certified",
                "raw_survives",
                "safe_mask",
                "unsafe_mask",
                "rescue_mask",
                "ordinary_mask",
                "bare_actions",
                "a_star_actions",
                "v8_best_safe_area",
                "max_pocket_r",
                "root_q",
            ):
                assert torch.equal(
                    res_fast[key], res_plain[key]
                ), f"[{dev_name}] open_stop_h32 fast vs plain mismatch on {key}"


def test_extension_worlds_and_watch6_bundle_template_reuse_bit_identical() -> None:
    """Prove python/numpy world building and policy_template bundle reuse match numba/disk."""
    from research.gpu_sitl.open_stop_gpu import _ENV_TENSOR_FIELDS, _VECTOR_RUNTIME_ENV_FIELDS
    from research.gpu_sitl.preflight import MATURE_EXTENSION_SPECS, build_extension_worlds
    from research.gpu_sitl.probe_bank import build_watch6_ct_device_bundle

    b8_path, opp_path = resolve_product_checkpoints()
    specs = [MATURE_EXTENSION_SPECS[0], MATURE_EXTENSION_SPECS[26]]
    w_nb, tpl_nb, meta_nb = build_extension_worlds(
        Path(b8_path),
        Path(opp_path),
        specs=specs,
        cpu_featurizer_backend="numba",
        ego_raster_backend="numba",
    )
    w_py, tpl_py, meta_py = build_extension_worlds(
        Path(b8_path),
        Path(opp_path),
        specs=specs,
        cpu_featurizer_backend="python",
        ego_raster_backend="numpy",
    )
    assert meta_nb == meta_py
    for wn, wp in zip(w_nb, w_py):
        assert np.array_equal(wn.sim.bodies, wp.sim.bodies)
        assert np.array_equal(wn.sim.head_ptr, wp.sim.head_ptr)
        assert np.array_equal(wn.sim.seg_count, wp.sim.seg_count)
        assert np.array_equal(wn.sim.length, wp.sim.length)
        assert np.array_equal(wn.sim.alive, wp.sim.alive)
        assert np.array_equal(wn.sim.direction, wp.sim.direction)
        assert np.array_equal(wn.sim.frame, wp.sim.frame)
        assert wn.sim.food_cells == wp.sim.food_cells
        assert wn.sim.corpse_cells == wp.sim.corpse_cells
        np.testing.assert_allclose(wn.runtime._carry_state, wp.runtime._carry_state, atol=1e-6)
        assert np.array_equal(wn.runtime._carry_mask, wp.runtime._carry_mask)
        assert np.array_equal(wn.runtime._carry_valid, wp.runtime._carry_valid)

    seeds = preflight_seeds((0, 1, 2, 3))
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev_name in devices:
        ctx = _mps_readonly_spike_compat() if dev_name == "mps" else contextlib.nullcontext()
        with ctx:
            dev = torch.device("mps", 0) if dev_name == "mps" else torch.device("cpu")
            sim_disk, pol_disk = build_watch6_ct_device_bundle(
                seeds,
                b8_path,
                opp_path,
                device=dev,
                policy_template=None,
                cpu_featurizer_backend="numba",
            )
            sim_tpl, pol_tpl_out = build_watch6_ct_device_bundle(
                seeds, b8_path, opp_path, device=dev, policy_template=tpl_py
            )
            for fname in _ENV_TENSOR_FIELDS:
                assert torch.equal(getattr(sim_disk, fname), getattr(sim_tpl, fname))
            for fname in _VECTOR_RUNTIME_ENV_FIELDS:
                assert torch.equal(
                    getattr(pol_disk.vector_runtime, fname),
                    getattr(pol_tpl_out.vector_runtime, fname),
                )
            for (_, p1), (_, p2) in zip(
                pol_disk.b8_network.named_parameters(), pol_tpl_out.b8_network.named_parameters()
            ):
                assert torch.equal(p1, p2)


def test_verify_cuda_native_extensions_parallel_cold_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify cold CUDA extension compilation dispatches all 4 modules in parallel."""
    import threading
    import time

    from research.gpu_sim_spike import event_cuda, flood_cuda, move_cuda, vector_geometry
    from research.gpu_sitl import open_stop_gpu as osg

    active = 0
    max_active = 0
    calls: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=5.0)

    def _make_loader(tag: str) -> Any:
        def _loader() -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(tag)
            try:
                barrier.wait()
                time.sleep(0.01)
                return f"ext_{tag}"
            finally:
                with lock:
                    active -= 1

        return _loader

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext_cache"))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0")
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    monkeypatch.setattr(osg, "_LAST_SCOPED_BUILD_KEY", None)
    monkeypatch.setattr(osg, "ensure_cuda_initialized", lambda d: torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)
    monkeypatch.setattr(flood_cuda, "_load", _make_loader("flood"))
    monkeypatch.setattr(vector_geometry, "_load_exact_extension", _make_loader("vector"))
    monkeypatch.setattr(osg, "_load_event_cuda_extension", _make_loader("event"))
    monkeypatch.setattr(osg, "_load_move_cuda_extension", _make_loader("move"))

    info = osg.verify_cuda_native_extensions(torch.device("cuda:0"))
    assert info["effective_build_cuda_arch"] == "8.0"
    assert set(calls) == {"flood", "move", "event", "vector"}
    assert max_active == 4

    # Warm second call with same build_key must NOT re-spawn parallel barrier threads
    calls.clear()
    monkeypatch.setattr(flood_cuda, "_load", lambda: (calls.append("flood"), "ext")[1])
    monkeypatch.setattr(
        vector_geometry, "_load_exact_extension", lambda: (calls.append("vector"), "ext")[1]
    )
    monkeypatch.setattr(
        osg, "_load_event_cuda_extension", lambda: (calls.append("event"), "ext")[1]
    )
    monkeypatch.setattr(osg, "_load_move_cuda_extension", lambda: (calls.append("move"), "ext")[1])
    osg.verify_cuda_native_extensions(torch.device("cuda:0"))
    assert calls == ["flood", "vector", "event", "move"]


def test_b1b_smoke_partial_phase_files_timeout_evidence_and_pilot_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify per-phase immutable partial exports survive timeouts and never count as PASS."""
    import shutil

    from research.colab_cli import runtime as colab_runtime
    from research.colab_cli import schema as colab_schema
    from research.gpu_sitl import smoke as smoke_mod
    from research.gpu_sitl.prepare import SMOKE_REQUIRED_EXPORTS, prepare_b1b_packet
    from research.gpu_sitl.probe_bank import verify_file_sha256, write_atomic_with_sha256

    assert SMOKE_REQUIRED_EXPORTS == ("smoke_result.json",)

    b8_path, opp_path = resolve_product_checkpoints()

    # 1. Full dry-run smoke on CPU (and MPS if available): 13 create-only partial files + final PASS
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    cpu_tp: dict[str, Any] | None = None
    for dev_name in devices:
        ctx = _mps_readonly_spike_compat() if dev_name == "mps" else contextlib.nullcontext()
        out_ok = tmp_path / f"smoke_ok_{dev_name}"
        with ctx:
            res = smoke_mod.run_step_b1b_smoke(
                out_ok,
                b8_checkpoint=b8_path,
                opponent_checkpoint=opp_path,
                device=dev_name,
                num_envs=4,
                branch_horizon=2,
                parity_slice_indices=(0, 8),
                bench_env_steps=2,
                bench_label_batches=2,
                bench_learner_updates=2,
                dry_run=True,
            )
        assert res["ok"] is True
        assert res["partial"] is False
        assert res["status"] == "PASS"
        partial_files = sorted((out_ok / "phases").glob("smoke_partial_*.json"))
        assert len(partial_files) == 13
        prev_elapsed = 0.0
        for idx, pf in enumerate(partial_files, start=1):
            verify_file_sha256(pf, require_complete=True)
            pdata = json.loads(pf.read_text(encoding="utf-8"))
            assert pdata["schema"] == "t34-gpu-sitl-b1b-smoke-partial/v1"
            assert pdata["partial"] is True
            assert pdata["ok"] is False
            assert pdata["pass"] is False
            assert pdata["status"] == "PARTIAL_IN_PROGRESS"
            assert pdata["not_evidence"] is True
            assert pdata["phase_index"] == idx
            assert float(pdata["elapsed_since_start_sec"]) >= prev_elapsed
            prev_elapsed = float(pdata["elapsed_since_start_sec"])
        if dev_name == "cpu":
            cpu_tp = res["throughput"]
        elif cpu_tp is not None:
            for key in (
                "measured_alive_hero_decisions",
                "measured_rescue_frame_count",
                "measured_rescue_episode_count",
                "window_min_hero_length",
                "window_max_hero_length",
                "window_mature_length_ge_160_decisions",
            ):
                assert res["throughput"][key] == cpu_tp[key], f"CPU vs MPS mismatch on {key}"

    # 2. Simulated timeout mid-run (during bench_3): leaves immutable partial evidence,
    # never writes smoke_result.json, fails colab_cli required_exports, and is rejected by prepare.
    fake_root = tmp_path / "colab_root"
    fake_out = fake_root / "output"
    scanner_cache: dict[str, Any] = {}

    def _timeout_in_bench_3(*args: Any, **kwargs: Any) -> Any:
        # Scanner snapshot mid-run before timeout
        mid_records = colab_runtime._exports(fake_root, 2 * 1024**3, scanner_cache)
        assert len(mid_records) == 11
        raise TimeoutError("simulated 22.5 min A100 timeout")

    monkeypatch.setattr(smoke_mod, "_evaluate_scoring_suffixes_on_snapshots", _timeout_in_bench_3)
    with pytest.raises(TimeoutError, match="22.5 min"):
        smoke_mod.run_step_b1b_smoke(
            fake_out,
            b8_checkpoint=b8_path,
            opponent_checkpoint=opp_path,
            device="cpu",
            num_envs=4,
            branch_horizon=2,
            parity_slice_indices=(0, 8),
            bench_env_steps=1,
            bench_label_batches=1,
            bench_learner_updates=1,
            dry_run=True,
        )
    assert not (fake_out / "smoke_result.json").exists()
    post_records = colab_runtime._exports(fake_root, 2 * 1024**3, scanner_cache)
    assert len(post_records) == 11
    with pytest.raises(colab_schema.SchemaError, match="smoke_result.json"):
        colab_schema.validate_export_inventory(
            {"required_exports": list(SMOKE_REQUIRED_EXPORTS)}, post_records
        )

    last_partial = sorted((fake_out / "phases").glob("smoke_partial_*.json"))[-1]
    with pytest.raises(ValueError, match="partial"):
        prepare_b1b_packet(
            tmp_path / "pkt_reject_partial_name",
            commit="origin/main",
            mode="pilot",
            smoke_result_json=last_partial,
        )

    disguised = tmp_path / "smoke_result.json"
    write_atomic_with_sha256(
        disguised,
        lambda tmp: shutil.copyfile(last_partial, tmp),
        write_complete_marker=True,
    )
    with pytest.raises(ValueError, match="partial|non-passing"):
        prepare_b1b_packet(
            tmp_path / "pkt_reject_disguised_partial",
            commit="origin/main",
            mode="pilot",
            smoke_result_json=disguised,
        )


def test_shakeout_smoke_wall_time_projection_and_700s_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify shakeout predictive scale in job.sh, per-phase projection math, and >700s failure."""
    from research.gpu_sitl import smoke as smoke_mod
    from research.gpu_sitl.prepare import ROOT

    job_sh_text = (ROOT / "research" / "gpu_sitl" / "job.sh").read_text(encoding="utf-8")
    shake_section = job_sh_text.split('elif [ "$MODE" = "shakeout" ]; then', 1)[1].split(
        'elif [ "$MODE" = "pilot" ]; then', 1
    )[0]
    for expected_flag in (
        "--enforce-projection-gate",
        "--num-envs 256",
        "--branch-horizon 32",
        "--mature-warmup-steps 60",
        "--bench-env-steps 8",
        "--bench-label-batches 8",
        "--probe-sample-roots 8",
        "--bench-learner-updates 5",
    ):
        assert expected_flag in shake_section, f"Missing {expected_flag} in job.sh shakeout"

    # 1. Exact per-phase scale_ratio and L4 -> A100 estimate math at E=256, H=32
    synthetic_l4_sub = {
        "device": "cuda:0",
        "effective_build_cuda_arch": "8.9",
        "parity_check": {
            "num_roots": 8,
            "subphase_timings_sec": {
                "parity_1_v8_flood_sec": 0.4,
                "parity_2_components_sec": 0.2,
                "parity_3_vector_runtime_sec": 0.4,
                "parity_4_open_stop_h32_sec": 24.0,
            },
        },
        "throughput": {
            "num_envs": 256,
            "branch_horizon": 32,
            "mature_warmup_steps": 60,
            "bench_env_steps": 8,
            "bench_label_batches": 8,
            "probe_sample_roots": 8,
            "bench_learner_updates": 5,
        },
        "phase_timings_sec": {
            "init_and_extensions_sec": 0.2,
            "mature_extension_build_sec": 3.0,
            "parity_check_sec": 25.5,
            "bundle_and_bench_warmup_sec": 5.0,
            "mature_warmup_and_splice_sec": 4.0,
            "env_benchmark_sec": 0.6,
            "label_benchmark_sec": 18.0,
            "buffer_insertion_sec": 0.5,
            "probe_scoring_benchmark_sec": 2.5,
            "learner_benchmark_sec": 0.8,
            "write_outputs_sec": 0.5,
        },
    }
    proj_ok = smoke_mod.compute_smoke_wall_time_projection(
        synthetic_l4_sub,
        cold_init_elapsed_sec=70.0,
        target_a100_limit_sec=700.0,
    )
    out_text = capsys.readouterr().out
    assert "[shakeout.projection] SUMMARY" in out_text
    assert "(ESTIMATE)" in out_text
    assert proj_ok["is_estimate"] is True
    assert "ESTIMATE" in proj_ok["l4_to_a100_factor_derivation"]
    assert proj_ok["within_a100_limit"] is True
    assert proj_ok["a100_total_projected_sec_estimate"] < 300.0

    by_p = {row["phase"]: row for row in proj_ok["phases"]}
    assert by_p["init_and_extensions"]["measured_sec"] == pytest.approx(70.0)
    assert by_p["init_and_extensions"]["scale_ratio"] == pytest.approx(1.0)
    assert by_p["mature_warmup_and_splice"]["scale_ratio"] == pytest.approx(10.0)
    assert by_p["bench_1_env_steps"]["scale_ratio"] == pytest.approx(4.0)
    assert by_p["bench_2_label_batches_and_rescue_rate"]["scale_ratio"] == pytest.approx(8.0)
    assert by_p["bench_3_probe32_scoring_suffixes"]["scale_ratio"] == pytest.approx(8.0)
    assert by_p["bench_4_learner_updates"]["scale_ratio"] == pytest.approx(4.0)

    # 2. End-to-end CLI --shakeout --enforce-projection-gate fails closed when A100 proj > 700s
    b8_path, opp_path = resolve_product_checkpoints()
    orig_proj_fn = smoke_mod.compute_smoke_wall_time_projection

    def _slow_projection(sub_res: dict[str, Any], **kw: Any) -> dict[str, Any]:
        inflated = dict(sub_res)
        inflated["phase_timings_sec"] = {
            **dict(sub_res["phase_timings_sec"]),
            "label_benchmark_sec": 200.0,
        }
        return orig_proj_fn(inflated, **kw)

    monkeypatch.setattr(smoke_mod, "compute_smoke_wall_time_projection", _slow_projection)
    out_slow = tmp_path / "shakeout_exceeds_700s"
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke.py",
            "--shakeout",
            "--enforce-projection-gate",
            "--device",
            "cpu",
            "--output-dir",
            str(out_slow),
            "--b8-checkpoint",
            str(b8_path),
            "--opponent-checkpoint",
            str(opp_path),
        ],
    )
    rc = smoke_mod.main()
    assert rc == 1
    saved_shake = json.loads((out_slow / "shakeout_result.json").read_text(encoding="utf-8"))
    assert saved_shake["ok"] is False
    assert saved_shake["verdict"] == "SHAKEOUT_FAIL_NOT_EVIDENCE"
    assert saved_shake["failed_phases"] == ["phase_4_b1b_smoke_reduced"]
    assert saved_shake["smoke_wall_time_projection"]["within_a100_limit"] is False
    assert saved_shake["smoke_wall_time_projection"]["a100_total_projected_sec_estimate"] > 700.0
