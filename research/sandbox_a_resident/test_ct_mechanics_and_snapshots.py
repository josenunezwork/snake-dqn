"""Unit tests for C/T mechanics adapter, forced-state determinism, snapshots, and topology."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import (
    CT_SKILL_TASK_ID,
    SEPARATED_WATCH_V2_TASK_ID,
    CTDeviceSim,
    CTOpponentRuntime,
    forced_state_determinism_suite,
    task_separation_report,
)
from research.sandbox_a_resident.ct_snapshots import (
    MAX_LEGAL_BODY_CAPACITY,
    LookbackRing,
    LosslessBoundedSnapshotStore,
    OmitIndexRootMeta,
    SnapshotOverflowError,
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.ct_topology import (
    TOPOLOGY_SPEC,
    compute_topology_features,
    compute_topology_features_reference,
    pack_sparse_topology,
    validate_geometric_fixtures,
)
from research.sandbox_a_resident.device_sim import _ENV_TENSOR_FIELDS
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


def _tiny_models() -> tuple[Ego2sNet, dict[str, torch.nn.Module]]:
    torch.manual_seed(17)
    b8 = Ego2sNet(local_channels=7, n_scalars=60).eval()
    ckpts: dict[str, torch.nn.Module] = {
        FRP3_S12_SHA256: torch.nn.Linear(61, 6).eval(),
        CHAMPION_SHA256: torch.nn.Linear(61, 6).eval(),
    }
    return b8, ckpts


def test_task_identities_and_devicesim_untouched() -> None:
    report = task_separation_report()
    assert report["active_task_id"] == CT_SKILL_TASK_ID
    assert report["separated_product_task_id"] == SEPARATED_WATCH_V2_TASK_ID
    assert report["v2_transfer_claim"] is False
    assert report["train01_packet15_modified"] is False
    assert report["train_mode"] is False
    assert report["hero_respawn"] is False
    assert report["opponent_respawn"] is True


def test_forced_state_determinism_suite_all_seven_cases() -> None:
    report = forced_state_determinism_suite("cpu")
    assert report["status"] == "PASS"
    expected_cases = {
        "eat_and_death",
        "simultaneous_contested_food",
        "boost_mechanics",
        "corpse_ordering",
        "respawn_and_terminal_lock",
        "prepared_state_and_carry",
        "root_restoration",
    }
    assert set(report["cases"]) == expected_cases
    for name, verdict in report["cases"].items():
        assert verdict == "PASS", f"Forced-state case failed: {name}: {verdict}"


@torch.inference_mode()
def test_instep_food_refill_row_independence() -> None:
    cfg2 = BatchSimConfig(
        num_envs=2,
        num_snakes=2,
        game_width=600,
        game_height=600,
        segment_size=20,
        wall_thickness=20,
        initial_food=4,
        max_food=8,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    cfg1 = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=600,
        game_height=600,
        segment_size=20,
        wall_thickness=20,
        initial_food=4,
        max_food=8,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim_ab = CTDeviceSim(cfg2, device="cpu", train_mode=False, allow_respawn=True)
    sim_b_only = CTDeviceSim(cfg1, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim_ab)
    r2_cache.enable(sim_b_only)

    sim_ab.alive.fill_(False)
    sim_ab.terminal_slots.fill_(False)
    sim_ab.rng_key.copy_(torch.tensor([111111, 222222], dtype=torch.long))
    sim_ab.rng_counter.zero_()
    sim_ab.reset_fresh_worlds()
    sim_ab.terminal_slots[:, 0] = True
    sim_ab.prepare()

    # Place ambient food directly ahead of row 0 snake 0 so row 0 triggers in-step refill
    # while row 1 does not; verify row 1 state + RNG counter remain bitwise identical.
    hp0 = int(sim_ab.head_ptr[0, 0].item())
    hx0 = int(sim_ab.bodies[0, 0, hp0, 0].item())
    hy0 = int(sim_ab.bodies[0, 0, hp0, 1].item())
    d0 = int(sim_ab.direction[0, 0].item())
    dx = [0, 1, 0, -1][d0]
    dy = [-1, 0, 1, 0][d0]
    fx, fy = hx0 + dx, hy0 + dy
    sim_ab.food_cells[0, 0] = torch.tensor([fx, fy], dtype=torch.long)
    sim_ab.food_present[0, 0] = True
    sim_ab.corpse_food[0, 0] = False

    for field in _ENV_TENSOR_FIELDS:
        getattr(sim_b_only, field)[0].copy_(getattr(sim_ab, field)[1])
    sim_b_only._prepared = sim_ab._prepared
    r2_cache.bump(sim_b_only)

    sim_ab.step(torch.tensor([[1, 1], [1, 1]], dtype=torch.long))
    sim_b_only.step(torch.tensor([[1, 1]], dtype=torch.long))

    assert int(sim_ab._last_food_ate[0, 0].item()) >= 1
    for field in _ENV_TENSOR_FIELDS:
        assert torch.equal(
            getattr(sim_ab, field)[1], getattr(sim_b_only, field)[0]
        ), f"Row 1 diverged on {field} when Row 0 triggered in-step food refill"


def test_lookback_ring_exact_replay_all_three_mixes() -> None:
    b8, ckpts = _tiny_models()
    rosters = training_rosters(6)
    assert {r.mix for r in rosters} == {"scripted", "frozen", "mixed"}

    cfg = BatchSimConfig(
        num_envs=6,
        num_snakes=6,
        game_width=400,
        game_height=400,
        initial_food=8,
        max_food=14,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.arange(1000, 1006, dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True

    opp = CTOpponentRuntime(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=list(range(2000, 2006)),
    )

    ring = LookbackRing(num_worlds=6, stride=5, max_lookback=25, max_body_capacity=256)
    life_ids = [1] * 6
    target_records: dict[
        int, list[tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, int]]
    ] = {}

    for tick in range(9):
        sim.prepare()
        ring.maybe_capture_checkpoint(sim, opp, tick=tick, life_ids=life_ids)
        local, glob, _scalars, legal, _q_b8, b8_act = observe_hero_bv8(sim, b8)
        ring.record_tick_actions(
            tick=tick,
            hero_actions=b8_act.detach().cpu().numpy(),
            life_ids=life_ids,
        )
        if tick in (5, 7, 8):
            target_records[tick] = [
                (
                    capture_world_snapshot(
                        sim, opp, w, world_id=w, life_id=1, tick=tick, max_body_capacity=256
                    ),
                    local[w].clone(),
                    glob[w].clone(),
                    legal[w].clone(),
                    int(b8_act[w].item()),
                )
                for w in range(6)
            ]
        full_actions = opp.actions(sim, b8_act)
        sim.step(full_actions)
        opp.after_step(sim)

    assert all(
        ring.has_same_life_lookbacks(w, death_tick=9, life_id=1, lookbacks=(1, 2, 4))
        for w in range(6)
    )
    replay_cache: dict[tuple[object, ...], tuple[CTDeviceSim, CTOpponentRuntime]] = {}
    for target_tick in (5, 7, 8):
        for w in range(6):
            (
                rec_snap,
                r_loc,
                r_glob,
                _r_scl,
                r_leg,
                _r_qb8,
                r_act,
                r_planes,
                r_sums,
            ) = ring.reconstruct_lookback_root(
                world_id=w,
                target_tick=target_tick,
                expected_life_id=1,
                roster=rosters[w],
                cfg=cfg,
                b8_network=b8,
                checkpoint_networks=ckpts,
                device="cpu",
                replay_cache=replay_cache,
            )
            exp_snap, exp_loc, exp_glob, exp_leg, exp_act = target_records[target_tick][w]
            assert np.array_equal(rec_snap.body_segments_xy, exp_snap.body_segments_xy)
            assert rec_snap.opponent_state["rng_key"] == exp_snap.opponent_state["rng_key"]
            assert rec_snap.opponent_state["rng_step"] == exp_snap.opponent_state["rng_step"]
            assert torch.equal(r_loc, exp_loc)
            assert torch.equal(r_glob, exp_glob)
            assert torch.equal(r_leg, exp_leg)
            assert r_act == exp_act
            assert r_planes.shape == (4, sim.grid_h, sim.grid_w)
            assert r_sums.shape == (6,)


@torch.inference_mode()
def test_lossless_snapshot_store_long_body_10400_and_overflow_refusal() -> None:
    b8, ckpts = _tiny_models()
    rosters = training_rosters(6)
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        max_capacity=MAX_LEGAL_BODY_CAPACITY,
        body_storage_capacity=MAX_LEGAL_BODY_CAPACITY,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([55555], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True

    # Populate hero snake 0 with a full 10,400-segment snake body wrapped around head_ptr=321!
    sim.seg_count[0, 0] = MAX_LEGAL_BODY_CAPACITY
    sim.length[0, 0] = MAX_LEGAL_BODY_CAPACITY
    sim.head_ptr[0, 0] = 321
    idx = torch.arange(MAX_LEGAL_BODY_CAPACITY, dtype=torch.long)
    sim.bodies[0, 0, :, 0] = idx % sim.grid_w
    sim.bodies[0, 0, :, 1] = (idx // sim.grid_w) % sim.grid_h
    sim.prepare()

    opp = CTOpponentRuntime(
        sim,
        (rosters[0],),
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[66666],
    )
    snap = capture_world_snapshot(
        sim,
        opp,
        0,
        world_id=0,
        life_id=1,
        tick=100,
        max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
    )
    assert snap.body_segments_xy.shape[0] >= MAX_LEGAL_BODY_CAPACITY
    local, glob, scalars, legal, q_b8, b8_act = observe_hero_bv8(sim, b8)
    planes, summaries = compute_topology_features(sim, [0])

    # Lossless sparse CSR topology round-trip
    topo_batch = pack_sparse_topology(planes, summaries)
    np.testing.assert_array_equal(topo_batch.unpack_raw_u16([0]), planes)

    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=2,
        max_bytes=16 * 1024 * 1024,
        max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
    )
    meta0 = OmitIndexRootMeta(
        root_uid="root-long-0",
        kind="death_lookback",
        partition="assessment",
        world_id=0,
        mix=rosters[0].mix,
        roster_index=0,
        life_id=1,
        tick=100,
        hero_length=MAX_LEGAL_BODY_CAPACITY,
        death_event_id=0,
        lookback=15,
        death_tick=115,
        death_length=MAX_LEGAL_BODY_CAPACITY,
        b8_action=int(b8_act[0].item()),
    )
    bound0 = store.add_root(
        meta=meta0,
        snapshot=snap,
        local_obs=local[0],
        glob_obs=glob[0],
        scalars_obs=scalars[0],
        legal_mask=legal[0],
        q_b8=q_b8[0],
        topo_planes_u16=planes[0],
        topo_summaries_f32=summaries[0],
    )
    assert bound0.root_index == 0

    # Verify lossless 10,400-segment restoration into a fresh simulator!
    fresh_sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(fresh_sim)
    fresh_opp = CTOpponentRuntime(
        fresh_sim,
        (rosters[0],),
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[1],
    )
    restore_world_snapshots_into([store.snapshots[0]], fresh_sim, fresh_opp)
    assert torch.equal(fresh_sim.bodies[0, 0], sim.bodies[0, 0])
    assert int(fresh_sim.head_ptr[0, 0].item()) == 321
    assert int(fresh_sim.seg_count[0, 0].item()) == MAX_LEGAL_BODY_CAPACITY

    # 1) Body span > 10,400 refused with SnapshotOverflowError (never truncated).
    bad_small = {k: v.clone() for k, v in snap.small_tensors.items()}
    bad_small["length"][0] = MAX_LEGAL_BODY_CAPACITY + 1
    bad_snap = type(snap)(
        world_id=snap.world_id,
        life_id=snap.life_id,
        tick=snap.tick,
        prepared=snap.prepared,
        num_snakes=snap.num_snakes,
        body_cap=snap.body_cap,
        food_cap=snap.food_cap,
        body_offsets=snap.body_offsets,
        body_segments_xy=snap.body_segments_xy,
        food_slot_indices=snap.food_slot_indices,
        food_xy=snap.food_xy,
        food_corpse=snap.food_corpse,
        small_tensors=bad_small,
        opponent_state=snap.opponent_state,
    )
    with pytest.raises(SnapshotOverflowError, match="exceeds max_body_capacity"):
        store.add_root(
            meta=OmitIndexRootMeta(**{**meta0.__dict__, "root_uid": "root-overflow"}),
            snapshot=bad_snap,
            local_obs=local[0],
            glob_obs=glob[0],
            scalars_obs=scalars[0],
            legal_mask=legal[0],
            q_b8=q_b8[0],
            topo_planes_u16=planes[0],
            topo_summaries_f32=summaries[0],
        )

    # 2) max_roots limit refused cleanly.
    store.add_root(
        meta=OmitIndexRootMeta(**{**meta0.__dict__, "root_uid": "root-long-1"}),
        snapshot=snap,
        local_obs=local[0],
        glob_obs=glob[0],
        scalars_obs=scalars[0],
        legal_mask=legal[0],
        q_b8=q_b8[0],
        topo_planes_u16=planes[0],
        topo_summaries_f32=summaries[0],
    )
    with pytest.raises(SnapshotOverflowError, match="root limit reached"):
        store.add_root(
            meta=OmitIndexRootMeta(**{**meta0.__dict__, "root_uid": "root-long-2"}),
            snapshot=snap,
            local_obs=local[0],
            glob_obs=glob[0],
            scalars_obs=scalars[0],
            legal_mask=legal[0],
            q_b8=q_b8[0],
            topo_planes_u16=planes[0],
            topo_summaries_f32=summaries[0],
        )

    # 3) Byte budget overflow refused cleanly.
    tiny_byte_store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=4,
        max_bytes=128,
        max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
    )
    with pytest.raises(SnapshotOverflowError, match="byte limit exceeded"):
        tiny_byte_store.add_root(
            meta=meta0,
            snapshot=snap,
            local_obs=local[0],
            glob_obs=glob[0],
            scalars_obs=scalars[0],
            legal_mask=legal[0],
            q_b8=q_b8[0],
            topo_planes_u16=planes[0],
            topo_summaries_f32=summaries[0],
        )


def test_dynamic_topology_approximations_geometric_fixtures_and_numba_parity() -> None:
    assert TOPOLOGY_SPEC["name"] == "dynamic-topology approximations"
    assert TOPOLOGY_SPEC["summaries"]["bfs_horizon"] == 128
    assert TOPOLOGY_SPEC["planes"]["shape_production"] == [4, 83, 145]
    assert TOPOLOGY_SPEC["summaries"]["count"] == 6

    fixture_report = validate_geometric_fixtures()
    assert fixture_report["status"] == "PASS"

    # Random populated multi-snake world parity between Numba @njit and pure-Python reference.
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=6,
        max_capacity=128,
        body_storage_capacity=128,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([7001, 7002], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    sim.prepare()

    p_nb, s_nb = compute_topology_features(sim)
    p_ref, s_ref = compute_topology_features_reference(sim)
    np.testing.assert_array_equal(p_nb, p_ref)
    np.testing.assert_allclose(s_nb, s_ref, atol=1e-6)
