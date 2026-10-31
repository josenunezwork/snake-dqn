"""Unit and offline integration tests for Track-4 Gate 1 privileged factored critic package."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.t4_gate1 import (
    ALLOCATION_INDEX,
    ALLOCATION_NAME,
    MAIN_NAMESPACE,
    STUDY,
    T4_GATE1_THRESHOLDS,
    UNIT_KIND,
    evaluate_gate1_bars,
    replicate_unit_id,
)
from research.t4_gate1.featurizer import (
    ARENA_CHANNELS,
    EGO_CHANNELS,
    SCALAR_DIM,
    CompactSampleBuilder,
    FeaturizedBatch,
    featurize_dataset_batch,
    featurize_live_sim,
)
from research.t4_gate1.harvest import GridBatchSim, compute_per_life_targets, watch6_v2_config
from research.t4_gate1.model import (
    FORAGING_GAMMA,
    HAZARD_BIN_EDGES,
    LengthHorizonBaseline,
    PrivilegedFactoredCritic,
    compose_factored_value,
    discrete_survival_nll,
)
from research.t4_gate1.pipeline import (
    EVAL_STRATA,
    UNIT_SCHEMA,
    load_and_validate_ledger,
    publish_verified_ledger,
)
from research.t4_gate1.prepare import (
    PACKAGE,
    REQUIREMENTS,
    ROOT,
    SCRIPT,
    prepare_inputs,
    t4_gate1_source_paths,
)
from research.t4_gate1.readout import (
    aggregate_gate1_readout,
    build_verified_ledger_outcome_blind,
)
from research.t4_gate1.seed_audit import seed_for, world_allocation
from research.t4_gate1.uploads import (
    BASE_PAYLOADS,
    MAX_ESTIMATED_TRANSFER_SECONDS,
    MAX_FILE_BYTES,
    MAX_TRANSFERS,
    METADATA,
    check_uploads,
    stage_metadata,
)
from tools.ops.package_spec import publish_json


def test_featurizer_rng_invariance_and_live_dataset_parity() -> None:
    """Critic features must be invariant to DeviceSim RNG state and match live vs compact CSR."""
    device = torch.device("cpu")
    cfg = watch6_v2_config(3, max_capacity=256)
    cpu_sim = GridBatchSim(
        cfg,
        seeds=[
            seed_for(MAIN_NAMESPACE, 0),
            seed_for(MAIN_NAMESPACE, 1),
            seed_for(MAIN_NAMESPACE, 2),
        ],
        train_mode=False,
        allow_respawn=True,
    )
    sim = DeviceSim.from_reference(
        cpu_sim,
        device=device,
        hero_respawn=True,
        exact_mechanics=True,
    )
    actions = torch.ones((3, 6), dtype=torch.long, device=device)
    for _ in range(12):
        sim.step(actions)

    h_rem = np.array([5800, 1200, 300], dtype=np.int32)
    live_before = featurize_live_sim(sim, h_rem=h_rem)

    legal_np = sim.advisory_mask()[:, 0].detach().cpu().numpy()
    builder = CompactSampleBuilder()
    builder.append_from_sim(
        sim,
        env_indices=torch.tensor([0, 1, 2], dtype=torch.long, device=device),
        world_ids=np.array([0, 1, 2], dtype=np.int32),
        seeds=np.array([101, 102, 103], dtype=np.int64),
        life_ids=np.array([0, 0, 0], dtype=np.int32),
        ticks=np.array([12, 12, 12], dtype=np.int32),
        h_rems=h_rem,
        b8_actions=np.array([0, 2, 4], dtype=np.int64),
        legal_masks=legal_np,
    )
    ds = builder.build()

    # Mutate DeviceSim RNG state arbitrarily and re-featurize live.
    sim.rng_key.fill_(0x12345678)
    sim.rng_counter.fill_(999999)
    live_after = featurize_live_sim(sim, h_rem=h_rem)

    for attr in ("arena", "ego", "scalars", "hero_xy_dir", "legal_mask", "hero_length", "h_rem"):
        torch.testing.assert_close(
            getattr(live_before, attr), getattr(live_after, attr), rtol=0.0, atol=0.0
        )

    # Mutate b8_action in dataset and verify critic features are completely independent of it.
    fb_orig = featurize_dataset_batch(ds, np.array([0, 1, 2]), device=device)
    ds.b8_action[:] = np.array([5, 5, 5], dtype=np.int64)
    fb_mut = featurize_dataset_batch(ds, np.array([0, 1, 2]), device=device)

    assert fb_orig.arena.shape == (3, ARENA_CHANNELS, 83, 145)
    assert fb_orig.ego.shape == (3, EGO_CHANNELS, 31, 31)
    assert fb_orig.scalars.shape == (3, SCALAR_DIM)

    for attr in ("arena", "ego", "scalars", "hero_xy_dir", "legal_mask", "hero_length", "h_rem"):
        torch.testing.assert_close(
            getattr(fb_orig, attr), getattr(fb_mut, attr), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            getattr(fb_orig, attr), getattr(live_before, attr), rtol=1e-5, atol=1e-5
        )


def test_per_life_target_decomposition_and_respawn_hard_cut() -> None:
    """Targets must decompose survival vs foraging per life and never leak across hero respawn."""
    # Env 0: Life 0 survives steps 0, 1, 2, dies on step 3; dead frame at step 4;
    #        Life 1 respawns and survives steps 5, 6, 7 through session end.
    # Env 1: Life 0 survives steps 0, 1, dies on step 2, and IMMEDIATELY respawns at step 3
    #        (hero_respawn=True with 0 dead frames between lives!).
    acted = np.array(
        [
            [True, True],
            [True, True],
            [True, True],
            [True, True],
            [False, True],
            [True, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    died = np.zeros((8, 2), dtype=bool)
    died[3, 0] = True  # Env 0 dies during step 3
    died[2, 1] = True  # Env 1 dies during step 2, immediate respawn at step 3

    pre_len = np.array(
        [
            [600.0, 700.0],
            [603.0, 704.0],
            [605.0, 710.0],
            [610.0, 10.0],
            [0.0, 11.0],
            [10.0, 14.0],
            [12.0, 15.0],
            [15.0, 19.0],
        ],
        dtype=np.float64,
    )
    post_len = np.array(
        [
            [603.0, 704.0],
            [605.0, 710.0],
            [610.0, 0.0],
            [0.0, 11.0],
            [0.0, 14.0],
            [12.0, 15.0],
            [15.0, 19.0],
            [18.0, 22.0],
        ],
        dtype=np.float64,
    )

    targets = compute_per_life_targets(
        acted=acted,
        died=died,
        pre_length=pre_len,
        post_length=post_len,
    )

    # Env 0, Life 0 (steps 0..2 survive, step 3 dies):
    np.testing.assert_array_equal(targets["tau_alive"][:4, 0], [3, 2, 1, 0])
    assert all(targets["died_in_session"][:4, 0])
    # Standing-mass sum for Life 0 ONLY (never leaks Life 1's steps 5..7!):
    np.testing.assert_allclose(
        targets["return_g"][:4, 0],
        [(603 + 605 + 610) / 1000.0, (605 + 610) / 1000.0, 610 / 1000.0, 0.0],
        rtol=1e-6,
    )
    num_f0 = (3.0 + FORAGING_GAMMA * 2.0 + (FORAGING_GAMMA**2) * 5.0) / 1000.0
    den_f0 = 1.0 + FORAGING_GAMMA + FORAGING_GAMMA**2
    np.testing.assert_allclose(targets["forage_short"][0, 0], num_f0 / den_f0, rtol=1e-5)

    # Env 0, Life 1 (steps 5..7 survive session end):
    assert not any(targets["died_in_session"][5:, 0])
    np.testing.assert_array_equal(targets["tau_alive"][5:, 0], [3, 2, 1])
    np.testing.assert_allclose(
        targets["return_g"][5:, 0],
        [(12 + 15 + 18) / 1000.0, (15 + 18) / 1000.0, 18 / 1000.0],
        rtol=1e-6,
    )

    # Env 1 immediate respawn boundary between step 2 (dies) and step 3 (new life):
    np.testing.assert_array_equal(targets["tau_alive"][:3, 1], [2, 1, 0])
    assert all(targets["died_in_session"][:3, 1])
    assert not any(targets["died_in_session"][3:, 1])
    np.testing.assert_allclose(
        targets["return_g"][:3, 1],
        [(704 + 710) / 1000.0, 710 / 1000.0, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(targets["tau_alive"][3:, 1], [5, 4, 3, 2, 1])


def test_compose_factored_value_closed_form_and_critic_zero_horizon() -> None:
    """Closed-form compose_factored_value must match brute-force tick-by-tick survival integral."""
    torch.manual_seed(7)
    B = 4
    haz_logits = torch.randn(B, 6, 16) * 0.4 - 6.0
    forage_raw = torch.randn(B, 6, 3) * 0.3
    length_t = torch.tensor([620.0, 850.0, 140.0, 1100.0], dtype=torch.float32)
    h_rem = torch.tensor([0.0, 27.0, 300.0, 1450.0], dtype=torch.float32)

    out = compose_factored_value(haz_logits, forage_raw, length_t, h_rem)

    # Exact zero-horizon identity: Q(s, a; H_rem=0) == 0, E[tau | H_rem=0] == 0, S(0) == 1.
    torch.testing.assert_close(out["q_life"][0], torch.zeros(6), rtol=0.0, atol=0.0)
    torch.testing.assert_close(out["e_tau"][0], torch.zeros(6), rtol=0.0, atol=0.0)
    torch.testing.assert_close(out["s_hrem"][0], torch.ones(6), rtol=0.0, atol=0.0)

    # Brute-force tick-by-tick survival + bin-flux check for b=1..3, actions 0 and 4:
    p_bins = torch.sigmoid(haz_logits.double())
    edges = [int(x) for x in HAZARD_BIN_EDGES]
    mids = [0.5 * (edges[k] + edges[k + 1]) for k in range(16)]

    for b in range(1, B):
        H = int(h_rem[b].item())
        L_scaled = float(length_t[b].item()) / 1000.0
        for a in (0, 4):
            gs = float(out["g_short"][b, a].item())
            gl = float(out["g_long"][b, a].item())
            cc = float(out["c_cov"][b, a].item())
            g_bin_vals = [
                gs * (FORAGING_GAMMA ** mids[k]) + gl * (1.0 - (FORAGING_GAMMA ** mids[k]))
                for k in range(16)
            ]

            p_u = np.zeros(H, dtype=np.float64)
            bin_of_u = np.zeros(H, dtype=np.int64)
            for k in range(16):
                lo, hi = edges[k], edges[k + 1]
                if lo >= H:
                    break
                p_u[lo : min(hi, H)] = float(p_bins[b, a, k].item())
                bin_of_u[lo : min(hi, H)] = k

            surv = 1.0
            e_tau_brute = 0.0
            cum_flux = 0.0
            flux_integral = 0.0
            for u in range(H):
                surv = surv * (1.0 - p_u[u])
                e_tau_brute += surv
                cum_flux += g_bin_vals[int(bin_of_u[u])]
                flux_integral += surv * cum_flux
            q_brute = max(0.0, (1.0 + cc) * L_scaled * e_tau_brute + flux_integral)

            assert abs(float(out["s_hrem"][b, a].item()) - surv) < 1e-5
            assert abs(float(out["e_tau"][b, a].item()) - e_tau_brute) < 1e-3
            assert abs(float(out["q_life"][b, a].item()) - q_brute) <= max(
                5e-3, 1e-3 * abs(q_brute)
            )

    # Test PrivilegedFactoredCritic forward + backward and LengthHorizonBaseline zero-horizon rule.
    critic = PrivilegedFactoredCritic()
    batch = FeaturizedBatch(
        arena=torch.randn(3, ARENA_CHANNELS, 83, 145),
        ego=torch.randn(3, EGO_CHANNELS, 31, 31),
        scalars=torch.randn(3, SCALAR_DIM),
        hero_xy_dir=torch.tensor([[40, 40, 0], [50, 30, 1], [70, 50, 2]], dtype=torch.long),
        legal_mask=torch.ones((3, 6), dtype=torch.bool),
        hero_length=torch.tensor([650.0, 820.0, 910.0], dtype=torch.float32),
        h_rem=torch.tensor([0.0, 300.0, 2400.0], dtype=torch.float32),
    )
    pred = critic(batch)
    torch.testing.assert_close(pred["q_life"][0], torch.zeros(6), rtol=0.0, atol=0.0)
    nll = discrete_survival_nll(
        pred["hazard_logits"][:, 0],
        torch.tensor([10.0, 120.0, 500.0]),
        torch.tensor([False, True, False]),
    )
    loss = nll.mean() + pred["q_life"].pow(2).mean()
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in critic.hazard_head.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in critic.foraging_head.parameters()
    )

    rng = np.random.default_rng(42)
    fit_L = rng.uniform(50.0, 1200.0, size=160).astype(np.float32)
    fit_H = rng.uniform(10.0, 6000.0, size=160).astype(np.float32)
    fit_tau = np.minimum(fit_H, rng.exponential(500.0, size=160)).astype(np.float32)
    fit_died = fit_tau < fit_H
    fit_G = ((fit_L / 1000.0) * fit_tau).astype(np.float32)
    baseline = LengthHorizonBaseline.fit(
        fit_length=fit_L,
        fit_h_rem=fit_H,
        fit_tau_alive=fit_tau,
        fit_died=fit_died,
        fit_return_g=fit_G,
        fit_forage_short=np.full(160, 0.005, dtype=np.float32),
        fit_forage_weight=np.full(160, 0.8, dtype=np.float32),
        val_length=fit_L[:32],
        val_h_rem=fit_H[:32],
        val_return_g=fit_G[:32],
    )
    v0 = baseline.predict_v(np.array([100.0, 650.0, 950.0]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(v0, np.zeros(3), rtol=0.0, atol=0.0)


def _make_synthetic_unit(
    rep_idx: int,
    *,
    namespace: str = MAIN_NAMESPACE,
    canary_only: bool = False,
    seed_offset: int = 0,
    ev_scale: float = 0.45,
    rescue_delta: float = 65.0,
) -> dict:
    base = seed_offset + rep_idx * 10
    fit_seeds = [seed_for(namespace, base + i) for i in range(3)]
    val_seeds = [seed_for(namespace, base + 3 + i) for i in range(2)]
    test_seeds = [seed_for(namespace, base + 5 + i) for i in range(4)]

    worlds = []
    roots = []
    for j, s in enumerate(test_seeds):
        wid = rep_idx * 100 + j
        sse_base = 10000.0 + 500.0 * j
        sse_crit = sse_base * (1.0 - ev_scale)
        strata = {}
        for st in EVAL_STRATA:
            strata[st] = {
                "n": 25,
                "sum_y": 500.0,
                "sum_y2": 25000.0,
                "sse_base": sse_base,
                "sse_critic": sse_crit,
                "sae_base": 400.0,
                "sae_critic": 260.0,
                "sum_hazard_nll": 30.0,
            }
        worlds.append({"world_id": wid, "seed": s, "strata": strata})
        roots.append(
            {
                "root_uid": f"r{rep_idx}-d{j}",
                "world_id": wid,
                "seed": s,
                "tick": 900,
                "h_rem": 5100,
                "kind": "death_lookback",
                "lookback": 48,
                "hero_length": 720,
                "legal_mask": [True, True, True, True, True, True],
                "b8_action": 1,
                "base_action": 1,
                "critic_action": 0,
                "oracle_a_star": 0,
                "s_hat": [0.92, 0.12, 0.40, 0.85, 0.15, 0.35],
                "m_hat": [740.0, 90.0, 310.0, 690.0, 110.0, 290.0],
                "s_score": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5],
                "m_score": [
                    rescue_delta + 120.0,
                    120.0,
                    140.0,
                    rescue_delta + 110.0,
                    120.0,
                    135.0,
                ],
                "base_s_hat": 0.70,
                "base_m_hat": 510.0,
            }
        )
        roots.append(
            {
                "root_uid": f"r{rep_idx}-o{j}",
                "world_id": wid,
                "seed": s,
                "tick": 1200,
                "h_rem": 4800,
                "kind": "ordinary",
                "lookback": None,
                "hero_length": 680,
                "legal_mask": [True, True, True, True, True, True],
                "b8_action": 1,
                "base_action": 1,
                "critic_action": 1,
                "oracle_a_star": 1,
                "s_hat": [0.85, 0.96, 0.80, 0.82, 0.90, 0.78],
                "m_hat": [690.0, 705.0, 685.0, 688.0, 698.0, 680.0],
                "s_score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "m_score": [690.0, 704.0, 689.0, 690.0, 700.0, 687.0],
                "base_s_hat": 0.88,
                "base_m_hat": 640.0,
            }
        )

    return {
        "schema": UNIT_SCHEMA,
        "study": STUDY,
        "namespace": namespace,
        "allocation": ALLOCATION_NAME,
        "allocation_index": ALLOCATION_INDEX,
        "unit_kind": UNIT_KIND,
        "unit_id": replicate_unit_id(namespace, rep_idx),
        "replicate_index": rep_idx,
        "canary_only": canary_only,
        "world_disjointness": {
            "fit_world_count": len(fit_seeds),
            "val_world_count": len(val_seeds),
            "test_world_count": len(test_seeds),
            "fit_seeds": fit_seeds,
            "val_seeds": val_seeds,
            "test_seeds": test_seeds,
            "pairwise_overlap_count": 0,
            "critic_frozen_before_test_harvest": True,
            "frozen_critic_sha256": "a" * 64,
            "post_eval_critic_sha256": "a" * 64,
            "frozen_baseline_sha256": "b" * 64,
        },
        "telemetry": {"replicate_wall_sec": 12.3},
        "held_out_test_worlds": worlds,
        "held_out_test_roots": roots,
    }


def test_world_disjointness_ledger_and_readout_kill_go_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readout must enforce stopped receipts, disjoint worlds, blind ledgers, and kill/GO bars."""
    alloc = world_allocation(
        MAIN_NAMESPACE,
        num_replicates=2,
        fit_per_replicate=4,
        val_per_replicate=2,
        test_per_replicate=3,
    )
    seeds = [r["seed"] for r in alloc["rows"]]
    assert len(seeds) == len(set(seeds)) == 18

    foreign_ledger = tmp_path / "foreign-ledger.json"
    publish_verified_ledger(foreign_ledger, {}, namespace="foreign-ns-1031")
    with pytest.raises(ValueError, match="Refusing foreign ledger"):
        load_and_validate_ledger(
            foreign_ledger, expected_namespace=MAIN_NAMESPACE, valid_replicates={0, 1}
        )

    att1 = tmp_path / "attempt1"
    att2 = tmp_path / "attempt2"
    u1_dir = att1 / "exports" / "t4_gate1" / "units"
    u2_dir = att2 / "exports" / "t4_gate1" / "units"
    publish_json(u1_dir / "replicate-0000.json", _make_synthetic_unit(0))
    publish_json(u2_dir / "replicate-0001.json", _make_synthetic_unit(1))

    # Unstopped receipt on Attempt 1 must be refused.
    (att1 / "receipt.json").write_text(
        json.dumps({"job_id": "m1", "status": "FAILED", "stop_verified": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Refusing readout from unstopped runtime"):
        aggregate_gate1_readout([att1], expected_replicates=1)

    (att1 / "receipt.json").write_text(
        json.dumps(
            {"job_id": "m1", "status": "FAILED", "stop_verified": True, "cleanup_error": None}
        ),
        encoding="utf-8",
    )
    (att2 / "receipt.json").write_text(
        json.dumps(
            {"job_id": "m2", "status": "PASS", "stop_verified": True, "cleanup_error": None}
        ),
        encoding="utf-8",
    )

    # Outcome-blind ledger builder never parses unit JSON payloads.
    real_loads = json.loads

    def _forbid_unit_json_loads(s: str | bytes, *args, **kwargs):
        text = s.decode("utf-8") if isinstance(s, bytes) else s
        if UNIT_SCHEMA in text:
            raise AssertionError("Outcome-blind ledger builder must never parse unit JSON payloads")
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(json, "loads", _forbid_unit_json_loads)
    ledger_out = tmp_path / "resume-ledger.json"
    ledger_rep = build_verified_ledger_outcome_blind(
        [att1],
        ledger_out=ledger_out,
        namespace=MAIN_NAMESPACE,
        valid_replicates={0, 1},
        local_preflight=True,
    )
    monkeypatch.setattr(json, "loads", real_loads)
    assert ledger_rep["verified_replicates"] == [0]
    assert load_and_validate_ledger(
        ledger_out, expected_namespace=MAIN_NAMESPACE, valid_replicates={0, 1}
    ) == {0}

    # Full pooled readout across Attempt 1 + Attempt 2 -> GATE1_PASS when EV=0.45 and rescue=+65.
    readout_pass = aggregate_gate1_readout([att1, att2], expected_replicates=2)
    assert readout_pass["decision"]["verdict"] == "GATE1_PASS"
    assert readout_pass["world_disjointness_proof"]["held_out_test_worlds_total"] == 8

    # OPUS_META kill rule: residual EV < +0.25 -> KILL_T4.
    low_ev_metrics = {
        "l_ge_600": {"residual_ev": 0.21, "residual_ev_ci95": [0.16, 0.24]},
        "per_replicate_l_ge_600_residual_ev": [0.20, 0.22],
        "h300_rescue": {
            "death_lookback": {
                "critic_vs_b8_mass_delta": 55.0,
                "critic_vs_b8_mass_delta_ci95": [20.0, 90.0],
            },
            "ordinary": {
                "critic_vs_b8_mass_delta": 2.0,
                "newly_caused_lethal_flip_rate": 0.01,
            },
            "all_mature": {"per_action_doom_auc": 0.88},
        },
    }
    kill_ev = evaluate_gate1_bars(low_ev_metrics, thresholds=T4_GATE1_THRESHOLDS)
    assert kill_ev["verdict"] == "KILL_T4"
    assert "opus_primary_l600_residual_ev_ge_0p25" in kill_ev["failed_bars"]

    # Rescue kill rule: death-root H300 rescue < +40 -> KILL_T4.
    low_rescue_metrics = {
        **low_ev_metrics,
        "l_ge_600": {"residual_ev": 0.36, "residual_ev_ci95": [0.25, 0.46]},
        "per_replicate_l_ge_600_residual_ev": [0.34, 0.38],
        "h300_rescue": {
            **low_ev_metrics["h300_rescue"],
            "death_lookback": {
                "critic_vs_b8_mass_delta": 28.0,
                "critic_vs_b8_mass_delta_ci95": [5.0, 51.0],
            },
        },
    }
    kill_rescue = evaluate_gate1_bars(low_rescue_metrics, thresholds=T4_GATE1_THRESHOLDS)
    assert kill_rescue["verdict"] == "KILL_T4"
    assert "opus_death_root_h300_rescue_mass_ge_40" in kill_rescue["failed_bars"]

    # Cross-replicate seed overlap refusal.
    att_dup = tmp_path / "attempt_dup"
    publish_json(
        att_dup / "exports" / "t4_gate1" / "units" / "replicate-0001.json",
        _make_synthetic_unit(1, seed_offset=-10),
    )
    (att_dup / "receipt.json").write_text(
        json.dumps({"job_id": "m_dup", "status": "PASS", "stop_verified": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cross-replicate world seed overlap"):
        aggregate_gate1_readout([att1, att_dup], expected_replicates=2)


def test_packed_metadata_and_upload_budget_gate(tmp_path: Path) -> None:
    """Packed input-metadata.tar.gz must keep total transfers <= 8 and estimated time <= 60 s."""
    rep = prepare_inputs(tmp_path / "out", queue_subdir="queue-01", scan_filesystem=False)
    assert rep["status"] == "PASS"
    assert rep["metadata_size_bytes"] < 100_000

    unpack_dir = tmp_path / "unpack"
    unpack_dir.mkdir()
    shutil.copyfile(rep["metadata_archive"], unpack_dir / METADATA)
    staged = stage_metadata(unpack_dir, rep["metadata_sha256"])
    assert set(staged["staged_payloads"]) == set(BASE_PAYLOADS)

    # Check if research/t4_gate1 is already committed on HEAD; if running pre-commit,
    # test check_uploads against a lightweight shared git clone with current Gate-1 files committed.
    has_req_in_head = (
        subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{REQUIREMENTS}"],
            cwd=ROOT,
            capture_output=True,
            timeout=10,
        ).returncode
        == 0
    )
    if has_req_in_head:
        target_repo = ROOT
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip()
    else:
        target_repo = tmp_path / "repo_clone"
        subprocess.run(
            ["git", "clone", "--shared", "--quiet", str(ROOT), str(target_repo)],
            check=True,
            timeout=30,
        )
        shutil.copytree(
            ROOT / "research/t4_gate1",
            target_repo / "research/t4_gate1",
            dirs_exist_ok=True,
        )
        subprocess.run(["git", "add", "research/t4_gate1"], cwd=target_repo, check=True, timeout=15)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Argon",
                "-c",
                "user.email=argon@local",
                "commit",
                "-q",
                "-m",
                "test snapshot",
            ],
            cwd=target_repo,
            check=True,
            timeout=15,
        )
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target_repo, text=True, timeout=10
        ).strip()

    manifest_stub = {
        "job_id": "t4-gate1-critic-canary-test",
        "commit": commit_sha,
        "script": SCRIPT,
        "source_paths": t4_gate1_source_paths(ROOT),
        "device": "cuda",
        "timeout_minutes": 24,
        "max_export_bytes": MAX_FILE_BYTES,
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": 12,
        },
        "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in rep["packed_rows"]],
        "env": {
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS,
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": PACKAGE,
            "T4_GATE1_DEVICE": "cuda",
        },
    }
    budget_report = check_uploads(manifest_stub, repo=target_repo)
    assert budget_report["status"] == "PASS"
    assert budget_report["normal_transfer_count"] == 3
    assert budget_report["transfer_count"] == 4
    assert budget_report["transfer_count"] <= MAX_TRANSFERS
    assert budget_report["normal_estimated_transfer_seconds"] <= MAX_ESTIMATED_TRANSFER_SECONDS
    assert budget_report["estimated_transfer_seconds"] <= MAX_ESTIMATED_TRANSFER_SECONDS
