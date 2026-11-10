"""Step B1b tests: review fixes (Findings 1 & 2), seed registration/audit, probe-bank
harvester, online SITL training loop, and smoke entrypoint."""

from __future__ import annotations

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
from research.gpu_sitl.seed_check import PREFLIGHT_NAMESPACE, preflight_seeds
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
    # Root 3: bare=4 (boost, SAFE, Q=1.2) and action 0 (non-boost, SAFE, Q=0.5) -> bare is safe so picks 4.
    # Root 4: bare=2 (unsafe); no safe actions -> returns -1.
    # Root 5: bare=0 (unsafe); tie between safe non-boost 1 (Q=0.5) and 2 (Q=0.5), plus boost 3 (Q=2.0) -> picks 1.
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

    # Randomized stress check over 256 roots ensuring every root with safe non-boost + higher-Q safe boost matches host
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

    # Finding 1: device a_star_actions matches host evaluation_order_nb on every root where has_safe is True
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
        net.adv.bias.copy_(
            torch.tensor([1.0, 2.0, 1.5, 50.0, 40.0, 30.0], dtype=torch.float32)
        )
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
