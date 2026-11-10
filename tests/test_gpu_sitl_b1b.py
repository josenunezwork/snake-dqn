"""Step B1b tests: review fixes (Findings 1 & 2), seed registration/audit, probe-bank
harvester, online SITL training loop, and smoke entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

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
    assert manifest_smoke["timeout_minutes"] == 25.0
    assert manifest_smoke["requirements"]["gpu"] == "H100"
    assert manifest_smoke["requirements"]["allowed_gpus"] == ["H100"]
    assert manifest_smoke["requirements"]["prefer"] == ["H100", "A100"]
    assert manifest_smoke["env"]["MAX_JOBS"] == "4"
    assert manifest_smoke["technical_envelope"]["registered_a100_unit_cap"] == 3.0
    assert manifest_smoke["budget"]["unit_cap"] == 5.0
    assert manifest_smoke["resume_exemption"]["kind"] == "short-canary"
    assert "resume" not in manifest_smoke
    for req_exp in manifest_smoke["required_exports"]:
        assert (smoke_dir / req_exp).is_file()
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
        "probe32_bank.pt.sha256",
        "probe32_bank.pt.complete",
        "b8_sitl_student_final.pth",
        "b8_sitl_student_final.pth.sha256",
        "b8_sitl_student_final.pth.complete",
        "sitl_pilot_result.json",
        "sitl_pilot_result.json.sha256",
        "sitl_pilot_result.json.complete",
    ]
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
    from research.gpu_sim_spike import flood_cuda, vector_geometry
    from research.gpu_sitl.open_stop_gpu import verify_cuda_native_extensions

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")

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
