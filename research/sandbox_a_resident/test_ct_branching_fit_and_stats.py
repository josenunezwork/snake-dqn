"""Unit tests for C/T H300 CRN branching, 3-arm distillation, probe AUC, and ordered gates."""

from __future__ import annotations

import numpy as np
import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_branching import (
    MergedScoredRoots,
    ScoredRootChunk,
    merge_scored_chunks,
    partition_roots_into_chunks,
    score_root_chunk,
    select_a_star_and_forbidden,
)
from research.sandbox_a_resident.ct_fit import (
    ArmEvaluationPredictions,
    fit_all_three_arms,
    float64_masked_kl_from_logits,
)
from research.sandbox_a_resident.ct_mechanics import (
    RNG_CONSUMER_INVENTORY,
    CTDeviceSim,
    CTOpponentRuntime,
    derive_branch_keys,
)
from research.sandbox_a_resident.ct_models import verify_zero_residual_init
from research.sandbox_a_resident.ct_snapshots import (
    LosslessBoundedSnapshotStore,
    OmitIndexRootMeta,
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.ct_stats import (
    evaluate_ct_gates,
    holm_step_down,
    validate_ct_support,
    world_grouped_mann_whitney_auc,
)
from research.sandbox_a_resident.ct_task import AnchorArchive, CTControl
from research.sandbox_a_resident.ct_topology import (
    IncrementalSparseTopologyBuilder,
    compute_topology_features,
)
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


def _models() -> tuple[Ego2sNet, dict[str, torch.nn.Module]]:
    torch.manual_seed(29)
    b8 = Ego2sNet(local_channels=7, n_scalars=60).eval()
    ckpts: dict[str, torch.nn.Module] = {
        FRP3_S12_SHA256: torch.nn.Linear(61, 6).eval(),
        CHAMPION_SHA256: torch.nn.Linear(61, 6).eval(),
    }
    return b8, ckpts


def test_crn_across_legal_actions_and_rng_consumer_inventory() -> None:
    ids = {item["id"] for item in RNG_CONSUMER_INVENTORY}
    assert ids == {
        "sim_instep_food_refill",
        "sim_prepare_food_refill",
        "sim_prepare_opponent_respawn",
        "opponent_random_safe",
    }
    assert all(item["includes_action_id"] is False for item in RNG_CONSUMER_INVENTORY)

    k_s0 = derive_branch_keys("root-uid-7", 0)
    k_s0_repeat = derive_branch_keys("root-uid-7", 0)
    k_s1 = derive_branch_keys("root-uid-7", 1)
    k_score = derive_branch_keys("root-uid-7", 4)
    assert k_s0 == k_s0_repeat
    assert k_s0.phase == "selection" and k_score.phase == "scoring"
    assert k_s0.sim_rng_key > 0 and k_s0.opp_rng_key > 0
    assert k_s0.sim_rng_key != k_s1.sim_rng_key
    assert k_s0.opp_rng_key != k_s1.opp_rng_key

    # Verify two different hero actions stepped from the same restored root under suffix k=0
    # see identical step-0 opponent actions (including random_safe mixed roster) under CRN!
    b8, ckpts = _models()
    rosters = training_rosters(6)
    mixed_roster = next(r for r in rosters if r.mix == "mixed")
    cfg1 = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=240,
        game_height=240,
        initial_food=10,
        max_food=18,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg1, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([8888], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    sim.prepare()
    opp = CTOpponentRuntime(
        sim,
        (mixed_roster,),
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[9999],
    )
    snap = capture_world_snapshot(sim, opp, 0, world_id=2, life_id=1, tick=0, max_body_capacity=256)

    cfg2 = BatchSimConfig(**{**cfg1.__dict__, "num_envs": 2})
    bsim = CTDeviceSim(cfg2, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(bsim)
    bopp = CTOpponentRuntime(
        bsim,
        (mixed_roster, mixed_roster),
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[1, 2],
    )
    restore_world_snapshots_into([snap, snap], bsim, bopp)
    bsim.rng_key.copy_(torch.tensor([k_s0.sim_rng_key, k_s0.sim_rng_key], dtype=torch.long))
    bsim.rng_counter.zero_()
    bopp.rekey_branches(torch.tensor([k_s0.opp_rng_key, k_s0.opp_rng_key], dtype=torch.long))

    joint = bopp.actions(bsim, torch.tensor([0, 2], dtype=torch.long))
    assert int(joint[0, 0].item()) == 0 and int(joint[1, 0].item()) == 2
    assert torch.equal(joint[0, 1:], joint[1, 1:])


@torch.inference_mode()
def test_selection_freeze_before_scoring_and_compaction_parity() -> None:
    legal = np.array([[True, True, True, False, False, False]], dtype=bool)
    # Action 0: S=1.0, M=800 -> Q = 1.0 + 0.25*(800/1000) = 1.20 (winner a*)
    # Action 1: S=0.75, M=600 -> gap = 0.25 < 0.50 (NOT forbidden)
    # Action 2: S=0.50, M=900 -> gap = 0.50 >= 0.50 (FORBIDDEN)
    s_sel = np.array([[1.0, 0.75, 0.50, np.nan, np.nan, np.nan]], dtype=np.float64)
    m_sel = np.array([[800.0, 600.0, 900.0, np.nan, np.nan, np.nan]], dtype=np.float64)
    b8_actions = np.array([1], dtype=np.int64)

    q_sel, a_star, forbidden = select_a_star_and_forbidden(
        s_sel=s_sel,
        m_sel=m_sel,
        legal_mask=legal,
        b8_actions=b8_actions,
        forbidden_gap=0.50,
    )
    assert int(a_star[0]) == 0
    np.testing.assert_allclose(q_sel[0, 0], 1.20, atol=1e-9)
    assert forbidden[0].tolist() == [False, False, True, False, False, False]

    # End-to-end score_root_chunk compaction_interval=2 vs compaction_interval=0 bitwise parity.
    b8, ckpts = _models()
    rosters = training_rosters(6)
    cfg = BatchSimConfig(
        num_envs=6,
        num_snakes=6,
        game_width=240,
        game_height=240,
        initial_food=10,
        max_food=18,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.arange(3000, 3006, dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    # Put hero row 0 right against top wall facing North so some candidate actions die on step 0
    # and trigger dead-hero compaction on h >= 2!
    hp0 = int(sim.head_ptr[0, 0].item())
    sim.bodies[0, 0, hp0, 0] = 5
    sim.bodies[0, 0, hp0, 1] = 0
    sim.direction[0, 0] = 0
    sim.prepare()

    opp = CTOpponentRuntime(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=list(range(4000, 4006)),
    )
    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=4,
        max_bytes=8 * 1024 * 1024,
        max_body_capacity=256,
    )
    local, glob, scalars, legal_t, q_b8, b8_act = observe_hero_bv8(sim, b8)
    planes, summaries = compute_topology_features(sim, [0, 1])
    for r in range(2):
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"parity-root-{r}",
                kind="ordinary",
                partition="assessment",
                world_id=r,
                mix=rosters[r].mix,
                roster_index=0,
                life_id=1,
                tick=10,
                hero_length=int(sim.length[r, 0].item()),
                death_event_id=None,
                lookback=None,
                death_tick=None,
                death_length=None,
                b8_action=int(b8_act[r].item()),
            ),
            snapshot=capture_world_snapshot(
                sim, opp, r, world_id=r, life_id=1, tick=10, max_body_capacity=256
            ),
            local_obs=local[r],
            glob_obs=glob[r],
            scalars_obs=scalars[r],
            legal_mask=legal_t[r],
            q_b8=q_b8[r],
            topo_planes_u16=planes[r],
            topo_summaries_f32=summaries[r],
        )

    ctrl = CTControl(
        worlds=6,
        collection_ticks=20,
        max_body_capacity=256,
        branch_horizon=5,
        scoring_chunks=1,
        branch_batch_size=16,
        tiny=True,
    )
    chunks = partition_roots_into_chunks(2, 1)
    c_compact = score_root_chunk(
        chunk_index=0,
        root_indices=chunks[0],
        store=store,
        control=ctrl,
        b8_network=b8,
        checkpoint_networks=ckpts,
        device="cpu",
        compaction_interval=2,
    )
    c_dense = score_root_chunk(
        chunk_index=0,
        root_indices=chunks[0],
        store=store,
        control=ctrl,
        b8_network=b8,
        checkpoint_networks=ckpts,
        device="cpu",
        compaction_interval=0,
    )
    assert c_compact is not None and c_dense is not None
    assert c_compact.telemetry["selection_frozen_before_scoring"] is True
    assert (
        c_compact.telemetry["selection_freeze_monotonic_ns"]
        <= c_compact.telemetry["scoring_start_monotonic_ns"]
    )
    assert c_compact.selection_freeze_sha256 == c_dense.selection_freeze_sha256
    np.testing.assert_array_equal(c_compact.survive_sel_raw, c_dense.survive_sel_raw)
    np.testing.assert_allclose(c_compact.mass_sel_raw, c_dense.mass_sel_raw, equal_nan=True)
    np.testing.assert_array_equal(c_compact.survive_score_raw, c_dense.survive_score_raw)
    np.testing.assert_allclose(c_compact.mass_score_raw, c_dense.mass_score_raw, equal_nan=True)
    np.testing.assert_array_equal(c_compact.a_star, c_dense.a_star)
    np.testing.assert_array_equal(c_compact.forbidden_sel, c_dense.forbidden_sel)


def test_three_arms_zero_residual_init_and_shared_schedule_fit() -> None:
    b8, ckpts = _models()
    init_proof = verify_zero_residual_init(b8, grid_h=12, grid_w=12, seed=20261028)
    assert init_proof["status"] == "PASS"
    assert all(init_proof["exact_step0_b8_identity"].values())
    assert init_proof["t_arm_step1_head_grad_max"] > 0.0
    assert init_proof["t_arm_step1_encoder_grad_max"] == 0.0
    assert init_proof["t_arm_step2_encoder_grad_max"] > 0.0

    # Finite numerics on extreme saturated logits + illegal masks.
    extreme_ref = torch.tensor([[180.0, -180.0, 0.0, -1e9, -1e9, -1e9]], dtype=torch.float64)
    extreme_pol = torch.tensor(
        [[-180.0, 180.0, 0.0, -1e9, -1e9, -1e9]], dtype=torch.float32, requires_grad=True
    )
    extreme_leg = torch.tensor([[True, True, True, False, False, False]], dtype=torch.bool)
    kl = float64_masked_kl_from_logits(extreme_ref, extreme_pol, extreme_leg)
    assert torch.isfinite(kl).all()
    kl.sum().backward()
    assert extreme_pol.grad is not None and torch.isfinite(extreme_pol.grad).all()
    assert float(extreme_pol.grad[0, 3:].abs().sum().item()) == 0.0

    # Synthetic train + assessment fit check verifying 3 arms + detached D100-D300 probe.
    n_roots = 12
    world_partitions = ("train", "train", "train", "train", "assessment", "assessment")
    world_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    kinds = ["death_lookback"] * 4 + ["ordinary"] * 4 + ["death_lookback"] * 2 + ["ordinary"] * 2
    lookbacks = [100, 185, 140, 235, None, None, None, None, 100, 235, None, None]
    death_ids = [0, 0, 1, 1, None, None, None, None, 2, 2, None, None]

    cfg1 = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=240,
        game_height=240,
        initial_food=8,
        max_food=12,
        max_capacity=64,
        body_storage_capacity=64,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg1, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([1234], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    sim.prepare()
    rosters = training_rosters(6)
    opp = CTOpponentRuntime(sim, (rosters[0],), b8_network=b8, checkpoint_networks=ckpts)
    snap = capture_world_snapshot(sim, opp, 0, world_id=0, life_id=1, tick=0, max_body_capacity=64)

    rng = torch.Generator().manual_seed(11)
    loc = torch.randint(0, 2, (4, 7, 31, 31), generator=rng, dtype=torch.uint8)
    glb = torch.randint(0, 2, (4, 4, 37, 37), generator=rng, dtype=torch.uint8)
    scl = torch.randn((4, 60), generator=rng, dtype=torch.float32)
    leg_t = torch.tensor([True, True, True, False, False, False], dtype=torch.bool)
    qb8_t = torch.zeros(6, dtype=torch.float32)

    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=n_roots,
        max_bytes=4 * 1024 * 1024,
        max_body_capacity=64,
    )
    for i in range(n_roots):
        w_id = world_ids[i]
        is_danger = kinds[i] == "death_lookback"
        planes_u16 = np.zeros((4, sim.grid_h, sim.grid_w), dtype=np.uint16)
        if is_danger:
            planes_u16[:, :3, :3] = 3000
            tsum = np.full((6,), 0.05, dtype=np.float32)
        else:
            planes_u16[:, :8, :8] = 55000
            tsum = np.full((6,), 0.85, dtype=np.float32)
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"fit-root-{i}",
                kind=kinds[i],
                partition=world_partitions[w_id],
                world_id=w_id,
                mix=rosters[w_id].mix,
                roster_index=0,
                life_id=1,
                tick=20 + i,
                hero_length=650,
                death_event_id=death_ids[i],
                lookback=lookbacks[i],
                death_tick=300 if is_danger else None,
                death_length=660 if is_danger else None,
                b8_action=0,
            ),
            snapshot=snap,
            local_obs=loc[i % 4],
            glob_obs=glb[i % 4],
            scalars_obs=scl[i % 4],
            legal_mask=leg_t,
            q_b8=qb8_t,
            topo_planes_u16=planes_u16,
            topo_summaries_f32=tsum,
        )

    leg_np = np.tile(np.array([[True, True, True, False, False, False]], dtype=bool), (n_roots, 1))
    sel_s = np.ones((n_roots, 6, 4), dtype=np.float32)
    sel_m = np.full((n_roots, 6, 4), 700.0, dtype=np.float32)
    sel_s[:, 2, :] = 0.0  # action 2 dies -> forbidden
    sel_s[:, 3:, :] = np.nan
    sel_m[:, 3:, :] = np.nan
    s_mean = np.where(leg_np, np.nan_to_num(sel_s.astype(np.float64), nan=0.0).mean(axis=2), np.nan)
    m_mean = np.where(leg_np, np.nan_to_num(sel_m.astype(np.float64), nan=0.0).mean(axis=2), np.nan)
    b8_acts = np.zeros(n_roots, dtype=np.int64)
    q_sel, a_st, forb = select_a_star_and_forbidden(
        s_sel=s_mean,
        m_sel=m_mean,
        legal_mask=leg_np,
        b8_actions=b8_acts,
        forbidden_gap=0.50,
    )
    chunk = ScoredRootChunk(
        chunk_index=0,
        root_indices=np.arange(n_roots, dtype=np.int64),
        legal_mask=leg_np,
        b8_action=b8_acts,
        survive_sel_raw=sel_s,
        mass_sel_raw=sel_m,
        s_sel=s_mean,
        m_sel=m_mean,
        q_sel=q_sel,
        a_star=a_st,
        forbidden_sel=forb,
        selection_freeze_sha256="a" * 64,
        survive_score_raw=sel_s.copy(),
        mass_score_raw=sel_m.copy(),
        s_score=s_mean.copy(),
        m_score=m_mean.copy(),
        q_score=q_sel.copy(),
        telemetry={"logical_branch_steps": 100, "executed_sim_steps": 90, "wall_sec": 0.1},
    )
    merged = merge_scored_chunks([chunk], expected_num_roots=n_roots)

    n_anc = 16
    anc_worlds = np.array(
        [(i % 4) if i < 12 else (4 + (i % 2)) for i in range(n_anc)], dtype=np.int32
    )
    anc_assess = np.array(
        [world_partitions[int(w)] == "assessment" for w in anc_worlds], dtype=bool
    )
    a_topo_builder = IncrementalSparseTopologyBuilder(sim.grid_h, sim.grid_w)
    p_anc = np.zeros((n_anc, 4, sim.grid_h, sim.grid_w), dtype=np.uint16)
    s_anc = np.full((n_anc, 6), 0.80, dtype=np.float32)
    a_topo_builder.append_batch(p_anc, s_anc)
    anchors = AnchorArchive(
        num_rows=n_anc,
        world_id=anc_worlds,
        mix_id=np.zeros(n_anc, dtype=np.int8),
        is_assessment=anc_assess,
        life_id=np.ones(n_anc, dtype=np.int32),
        tick=np.arange(n_anc, dtype=np.int32),
        hero_length=np.full(n_anc, 650, dtype=np.int32),
        local=torch.stack([loc[i % 4] for i in range(n_anc)], dim=0),
        glob=torch.stack([glb[i % 4] for i in range(n_anc)], dim=0),
        scalars=torch.stack([scl[i % 4] for i in range(n_anc)], dim=0),
        legal_mask=leg_t.unsqueeze(0).expand(n_anc, -1).clone(),
        q_b8=torch.zeros((n_anc, 6), dtype=torch.float32),
        b8_action=np.zeros(n_anc, dtype=np.int8),
        topology=a_topo_builder.build(),
    )

    fit_ctrl = CTControl(
        worlds=6,
        collection_ticks=20,
        max_body_capacity=64,
        minibatches=4,
        searched_batch_size=4,
        anchor_batch_size=6,
        tiny=True,
    )
    fit_res = fit_all_three_arms(
        control=fit_ctrl,
        store=store,
        scored=merged,
        anchors=anchors,
        world_partitions=world_partitions,
        b8_model=b8,
        device="cpu",
    )
    assert fit_res is not None
    arm_preds, shared_meta = fit_res
    assert shared_meta["zero_residual_init_verification"]["status"] == "PASS"
    assert shared_meta["train_searched_roots"] == 8
    assert shared_meta["train_anchor_rows"] == 12
    assert set(arm_preds) == {"frozen_phi", "full_bv8", "full_bv8_plus_topology"}
    assert all(p.danger_probe_fitted is True for p in arm_preds.values())


def test_ordered_gates_world_bootstrap_auc_and_holm() -> None:
    # Build synthetic store + anchors across 200 worlds (0..159 train, 160..199 assessment = 40)
    # to test validate_ct_support and all ordered decision gates in evaluate_ct_gates.
    ctrl = CTControl(
        worlds=200,
        collection_ticks=50,
        max_body_capacity=64,
        bootstrap_resamples=200,
        tiny=True,
    )
    world_partitions = tuple(["train"] * 160 + ["assessment"] * 40)
    rosters = training_rosters(200)

    cfg1 = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=240,
        game_height=240,
        max_capacity=64,
        body_storage_capacity=64,
        mechanics_version=1,
    )
    b8, ckpts = _models()
    sim = CTDeviceSim(cfg1, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([91], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    sim.prepare()
    opp = CTOpponentRuntime(sim, (rosters[0],), b8_network=b8, checkpoint_networks=ckpts)
    base_snap = capture_world_snapshot(
        sim, opp, 0, world_id=0, life_id=1, tick=0, max_body_capacity=64
    )
    loc0, glb0, scl0, leg0, qb80, _ = observe_hero_bv8(sim, b8)
    pl0, sm0 = compute_topology_features(sim, [0])

    # 1 train root + 40 assessment death roots (1 per assessment world 160..199, lookback=100)
    # + 260 assessment ordinary roots -> total 301 roots!
    n_deaths = 40
    n_ord = 260
    n_total = 1 + n_deaths + n_ord
    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=n_total + 8,
        max_bytes=64 * 1024 * 1024,
        max_body_capacity=64,
    )

    def _snap_for_roster(w_id: int) -> object:
        return type(base_snap)(
            world_id=w_id,
            life_id=1,
            tick=100,
            prepared=True,
            num_snakes=base_snap.num_snakes,
            body_cap=base_snap.body_cap,
            food_cap=base_snap.food_cap,
            body_offsets=base_snap.body_offsets,
            body_segments_xy=base_snap.body_segments_xy,
            food_slot_indices=base_snap.food_slot_indices,
            food_xy=base_snap.food_xy,
            food_corpse=base_snap.food_corpse,
            small_tensors=base_snap.small_tensors,
            opponent_state={
                **base_snap.opponent_state,
                "roster_identity": rosters[w_id].identity,
            },
        )

    store.add_root(
        meta=OmitIndexRootMeta(
            root_uid="train-0",
            kind="ordinary",
            partition="train",
            world_id=0,
            mix=rosters[0].mix,
            roster_index=0,
            life_id=1,
            tick=10,
            hero_length=700,
            death_event_id=None,
            lookback=None,
            death_tick=None,
            death_length=None,
            b8_action=0,
        ),
        snapshot=_snap_for_roster(0),  # type: ignore[arg-type]
        local_obs=loc0[0],
        glob_obs=glb0[0],
        scalars_obs=scl0[0],
        legal_mask=leg0[0],
        q_b8=qb80[0],
        topo_planes_u16=pl0[0],
        topo_summaries_f32=sm0[0],
    )
    for d in range(n_deaths):
        w_id = 160 + d
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"assess-death-{d}",
                kind="death_lookback",
                partition="assessment",
                world_id=w_id,
                mix=rosters[w_id].mix,
                roster_index=0,
                life_id=1,
                tick=100,
                hero_length=700,
                death_event_id=d,
                lookback=100,
                death_tick=200,
                death_length=710,
                b8_action=0,
            ),
            snapshot=_snap_for_roster(w_id),  # type: ignore[arg-type]
            local_obs=loc0[0],
            glob_obs=glb0[0],
            scalars_obs=scl0[0],
            legal_mask=leg0[0],
            q_b8=qb80[0],
            topo_planes_u16=pl0[0],
            topo_summaries_f32=sm0[0],
        )
    for o in range(n_ord):
        w_id = 160 + (o % 40)
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"assess-ord-{o}",
                kind="ordinary",
                partition="assessment",
                world_id=w_id,
                mix=rosters[w_id].mix,
                roster_index=0,
                life_id=1,
                tick=150 + o,
                hero_length=700,
                death_event_id=None,
                lookback=None,
                death_tick=None,
                death_length=None,
                b8_action=0,
            ),
            snapshot=_snap_for_roster(w_id),  # type: ignore[arg-type]
            local_obs=loc0[0],
            glob_obs=glb0[0],
            scalars_obs=scl0[0],
            legal_mask=leg0[0],
            q_b8=qb80[0],
            topo_planes_u16=pl0[0],
            topo_summaries_f32=sm0[0],
        )

    # Anchor archive: 100 train rows + 10,000 assessment rows -> passes >= 10,000 bar!
    n_anc = 10_100
    anc_worlds = np.array(
        [0 if i < 100 else (160 + ((i - 100) % 40)) for i in range(n_anc)], dtype=np.int32
    )
    anc_assess = anc_worlds >= 160
    empty_topo = IncrementalSparseTopologyBuilder(sim.grid_h, sim.grid_w)
    empty_topo.append_batch(
        np.zeros((n_anc, 4, sim.grid_h, sim.grid_w), dtype=np.uint16),
        np.zeros((n_anc, 6), dtype=np.float32),
    )
    anchors = AnchorArchive(
        num_rows=n_anc,
        world_id=anc_worlds,
        mix_id=np.zeros(n_anc, dtype=np.int8),
        is_assessment=anc_assess,
        life_id=np.ones(n_anc, dtype=np.int32),
        tick=np.zeros(n_anc, dtype=np.int32),
        hero_length=np.full(n_anc, 700, dtype=np.int32),
        local=torch.zeros((n_anc, 7, 31, 31), dtype=torch.uint8),
        glob=torch.zeros((n_anc, 4, 37, 37), dtype=torch.uint8),
        scalars=torch.zeros((n_anc, 60), dtype=torch.float32),
        legal_mask=torch.ones((n_anc, 6), dtype=torch.bool),
        q_b8=torch.zeros((n_anc, 6), dtype=torch.float32),
        b8_action=np.zeros(n_anc, dtype=np.int8),
        topology=empty_topo.build(),
    )

    ok_sup = validate_ct_support(
        control=ctrl,
        store=store,
        anchors=anchors,
        world_partitions=world_partitions,
    )
    assert ok_sup["status"] == "PASS"

    bad_sup = validate_ct_support(
        control=CTControl(**{**ctrl.to_dict(), "min_held_out_deaths": 45}),
        store=store,
        anchors=anchors,
        world_partitions=world_partitions,
    )
    assert bad_sup["status"] == "INCOMPLETE_SUPPORT"

    # Build MergedScoredRoots where assessment deaths 0..19 (store indices 1..20, 50% >= 25%)
    # are rescued by action 1 (S_score=1.0, M=850 vs B8 action 0 S_score=0.0, M=400).
    leg_np = np.tile(np.array([[True, True, True, False, False, False]], dtype=bool), (n_total, 1))
    b8_act = np.zeros(n_total, dtype=np.int64)
    a_star = np.zeros(n_total, dtype=np.int64)
    a_star[1:21] = 1
    s_sc = np.ones((n_total, 6), dtype=np.float64)
    m_sc = np.full((n_total, 6), 800.0, dtype=np.float64)
    s_sc[:, 3:] = np.nan
    m_sc[:, 3:] = np.nan
    s_sc[1:21, 0] = 0.0
    m_sc[1:21, 0] = 400.0
    s_sc[1:21, 1] = 1.0
    m_sc[1:21, 1] = 850.0
    s_sc[21 : 1 + n_deaths, :3] = 0.0
    m_sc[21 : 1 + n_deaths, :3] = 350.0
    s_sc[1 + n_deaths :, 2] = 0.0  # action 2 on ordinary roots is lethal (drop = 1.0 >= 0.50)
    forb = np.zeros((n_total, 6), dtype=bool)
    forb[:, 2] = True

    scored = MergedScoredRoots(
        num_roots=n_total,
        legal_mask=leg_np,
        b8_action=b8_act,
        s_sel=s_sc.copy(),
        m_sel=m_sc.copy(),
        q_sel=s_sc.copy(),
        a_star=a_star,
        forbidden_sel=forb,
        s_score=s_sc,
        m_score=m_sc,
        q_score=s_sc.copy(),
        chunk_sha256s=["f" * 64],
        total_logical_branch_steps=1000,
        total_executed_sim_steps=900,
        total_scoring_wall_sec=0.5,
    )

    # Case A: INVALID precedence
    inv = evaluate_ct_gates(
        control=ctrl,
        support_report=ok_sup,
        store=store,
        scored=scored,
        anchors=anchors,
        arm_predictions=None,
        fault_reason="synthetic_fault",
    )
    assert inv["verdict"] == "INVALID"

    # Case B: INCOMPLETE_SUPPORT precedence
    inc = evaluate_ct_gates(
        control=ctrl,
        support_report=bad_sup,
        store=store,
        scored=scored,
        anchors=anchors,
        arm_predictions=None,
    )
    assert inc["verdict"] == "INCOMPLETE_SUPPORT"

    # Case C: High probe AUC (1.00 >= 0.83) WITHOUT +15pp T-arm retention over full_bv8!
    # frozen_phi rescues 4/20 (20%), full_bv8 rescues 12/20 (60% -> +40pp >= +15pp and >=30%),
    # T-arm rescues 13/20 (65% -> +5pp < +15pp) -> GO_BV8_ONLY (AUC alone never buys training!).
    probe_scores = np.array(
        [2.0 if (1 <= i < 1 + n_deaths) else -2.0 for i in range(n_total)], dtype=np.float64
    )
    anc_zero = np.zeros(10_000, dtype=np.int64)

    def _mk_pred(name: str, n_rescued: int) -> ArmEvaluationPredictions:
        acts = np.zeros(n_total, dtype=np.int64)
        acts[1 : 1 + n_rescued] = 1
        return ArmEvaluationPredictions(
            arm_name=name,
            root_logits=np.zeros((n_total, 6), dtype=np.float32),
            root_greedy_actions=acts,
            root_b8_neg_log_prob=probe_scores.copy(),
            root_danger_probe_scores=probe_scores.copy(),
            danger_probe_fitted=True,
            assessment_anchor_greedy_actions=anc_zero.copy(),
            train_telemetry={},
        )

    res_bare = evaluate_ct_gates(
        control=ctrl,
        support_report=ok_sup,
        store=store,
        scored=scored,
        anchors=anchors,
        arm_predictions={
            "frozen_phi": _mk_pred("frozen_phi", 4),
            "full_bv8": _mk_pred("full_bv8", 12),
            "full_bv8_plus_topology": _mk_pred("full_bv8_plus_topology", 13),
        },
    )
    assert res_bare["verdict"] == "GO_BV8_ONLY"
    assert res_bare["gates"]["gate_t_dynamic_topology"]["d100_d300_auc_gate_pass"] is True
    assert res_bare["gates"]["gate_t_dynamic_topology"]["retention_gate_pass"] is False
    assert res_bare["gates"]["gate_t_dynamic_topology"]["auc_alone_buys_training"] is False

    # Case D: T-arm rescues 18/20 (+30pp >= +15pp) AND AUC=1.00 >= 0.83 -> GO_BV8_AND_TOPOLOGY!
    res_topo = evaluate_ct_gates(
        control=ctrl,
        support_report=ok_sup,
        store=store,
        scored=scored,
        anchors=anchors,
        arm_predictions={
            "frozen_phi": _mk_pred("frozen_phi", 4),
            "full_bv8": _mk_pred("full_bv8", 12),
            "full_bv8_plus_topology": _mk_pred("full_bv8_plus_topology", 18),
        },
    )
    assert res_topo["verdict"] == "GO_BV8_AND_TOPOLOGY"
    assert res_topo["gates"]["gate_t_dynamic_topology"]["status"] == "PASS"

    # Direct Holm step-down check (m=2)
    h = holm_step_down({"full_bv8": 0.02, "full_bv8_plus_topology": 0.04}, family_alpha=0.10)
    assert h["full_bv8"]["statistically_superior"] is True
    assert h["full_bv8_plus_topology"]["statistically_superior"] is True

    # World-grouped cross-world AUC check (excludes same-world pair (w=1, w=1);
    # 5 cross-world pairs remain: (1,3)->1, (1,4)->1, (2,3)->1, (2,4)->1, (2,1)->0 => 4/5 = 0.80)
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float64)
    is_pos = np.array([True, True, False, False, False], dtype=bool)
    is_neg = ~is_pos
    wids = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    auc_res = world_grouped_mann_whitney_auc(scores, is_pos, is_neg, wids, resamples=100)
    np.testing.assert_allclose(auc_res["world_grouped_auc"], 0.80, atol=1e-9)
    np.testing.assert_allclose(auc_res["root_level_auc"], 4.0 / 6.0, atol=1e-9)
