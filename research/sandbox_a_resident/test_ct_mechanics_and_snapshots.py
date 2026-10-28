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


@torch.inference_mode()
def test_production_refill_and_respawn_rejection_without_draw_feeder() -> None:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=240,
        game_height=240,
        segment_size=20,
        wall_thickness=20,
        initial_food=2,
        max_food=2,
        max_capacity=128,
        body_storage_capacity=128,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    assert not hasattr(sim, "_draw_feeder")
    assert not hasattr(sim, "set_draw_feeder")

    # 1) Astra's exact duplicate-draw refill probe on the production _random_cells path:
    # max_food=2, empty food, draws (2,2), (2,2), (3,3), (4,4) -> must reject duplicate (2,2),
    # place food at (2,2) and (3,3), and leave (4,4) unconsumed.
    sim.alive.fill_(True)
    sim.terminal_slots.fill_(False)
    sim.terminal_slots[:, 0] = True
    sim.bodies.zero_()
    sim.head_ptr.zero_()
    sim.seg_count.fill_(1)
    sim.length.fill_(10)
    # Place snakes away from (2,2), (3,3), (4,4)
    sim.bodies[0, 0, 0] = torch.tensor([8, 8], dtype=torch.long)
    sim.bodies[0, 1, 0] = torch.tensor([8, 9], dtype=torch.long)
    sim.bodies[0, 2, 0] = torch.tensor([9, 8], dtype=torch.long)
    sim.food_present.fill_(False)
    sim.corpse_food.fill_(False)
    sim.food_cells.zero_()
    sim.food_count.zero_()

    draw_queue = [(2, 2), (2, 2), (3, 3), (4, 4)]
    orig_random_cells = sim._random_cells

    def _patched_random_cells() -> torch.Tensor:
        if draw_queue:
            cx, cy = draw_queue.pop(0)
            sim.rng_counter += 1
            return torch.tensor([[cx, cy]], dtype=torch.long)
        return orig_random_cells()

    sim._random_cells = _patched_random_cells  # type: ignore[method-assign]
    sim._prepared = False
    sim.prepare()
    assert int(sim.food_count[0].item()) == 2
    active = sim.food_cells[0][sim.food_present[0]].tolist()
    assert sorted(active) == [[2, 2], [3, 3]]
    assert draw_queue == [(4, 4)]

    # 2) Snake-occupied + duplicate food rejection during in-step refill:
    # Snake 0 occupies (8,8); clear food and feed (8,8) [occ], (5,5), (5,5) [dup], (6,6), (7,7).
    sim.food_present.fill_(False)
    sim.food_count.zero_()
    draw_queue[:] = [(8, 8), (5, 5), (5, 5), (6, 6), (7, 7)]
    sim._refill_food_rows(torch.tensor([True], dtype=torch.bool))
    assert int(sim.food_count[0].item()) == 2
    active2 = sim.food_cells[0][sim.food_present[0]].tolist()
    assert sorted(active2) == [[5, 5], [6, 6]]
    assert draw_queue == [(7, 7)]

    # 3) Production respawn collision rejection via _random_cells:
    # Snake 0 alive at (5,5),(4,5),(3,5); snake 1 dead with respawn_timer=1; snake 2 alive at (8,9).
    # First respawn candidate draw (5,5) collides with snake 0 body; second draw (6,7) is free;
    # third draw (9,9) stays unconsumed.
    sim.bodies[0, 0, 0] = torch.tensor([3, 5], dtype=torch.long)
    sim.bodies[0, 0, 1] = torch.tensor([4, 5], dtype=torch.long)
    sim.bodies[0, 0, 2] = torch.tensor([5, 5], dtype=torch.long)
    sim.head_ptr[0, 0] = 2
    sim.seg_count[0, 0] = 3
    sim.alive[0, 1] = False
    sim.respawn_timer[0, 1] = 1
    draw_queue[:] = [(5, 5), (6, 7), (9, 9)]
    sim._prepared = False
    sim.prepare()
    assert bool(sim.alive[0, 1].item()) is True
    assert int(sim.direction[0, 1].item()) == 1
    hp1 = int(sim.head_ptr[0, 1].item())
    assert sim.bodies[0, 1, hp1].tolist() == [6, 7]
    assert draw_queue == [(9, 9)]


def test_compact_v2_serialization_roundtrip_and_export_size_overflow_guards(tmp_path) -> None:
    from research.sandbox_a_resident.ct_task import AnchorArchive
    from research.sandbox_a_resident.ct_topology import SparseTopologyBatch
    from research.sandbox_a_resident.ct_units import (
        StageExportOverflowError,
        publish_stage_snapshot,
    )
    from research.sandbox_a_resident.training_compact import describe_state

    b8, ckpts = _tiny_models()
    rosters = training_rosters(6)
    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        max_capacity=256,
        body_storage_capacity=256,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.tensor([9101, 9102, 9103], dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    sim.prepare()
    opp = CTOpponentRuntime(
        sim,
        (rosters[0], rosters[1], rosters[2]),  # heterogeneous scripted + frozen + mixed carries
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[11, 22, 33],
    )
    local, glob, scalars, legal, q_b8, b8_act = observe_hero_bv8(sim, b8)
    planes, summaries = compute_topology_features(sim)

    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=8,
        max_bytes=16 * 1024 * 1024,
        max_body_capacity=256,
    )
    for e in range(3):
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"v2-rt-{e}",
                kind="ordinary",
                partition="train",
                world_id=e,
                mix=rosters[e].mix,
                roster_index=e,
                life_id=1,
                tick=10 + e,
                hero_length=int(sim.length[e, 0].item()),
                death_event_id=None,
                lookback=None,
                death_tick=None,
                death_length=None,
                b8_action=int(b8_act[e].item()),
            ),
            snapshot=capture_world_snapshot(
                sim, opp, e, world_id=e, life_id=1, tick=10 + e, max_body_capacity=256
            ),
            local_obs=local[e],
            glob_obs=glob[e],
            scalars_obs=scalars[e],
            legal_mask=legal[e],
            q_b8=q_b8[e],
            topo_planes_u16=planes[e],
            topo_summaries_f32=summaries[e],
        )

    st_dict = store.to_state_dict()
    assert st_dict["schema"] == "ct-lossless-snapshot-store/v2"
    assert describe_state(st_dict)["tensor_count"] > 0
    restored_store = LosslessBoundedSnapshotStore.from_state_dict(st_dict)
    assert len(restored_store.snapshots) == 3
    for e in range(3):
        np.testing.assert_array_equal(
            restored_store.snapshots[e].body_segments_xy, store.snapshots[e].body_segments_xy
        )
        assert set(restored_store.snapshots[e].opponent_state["carry"]) == set(
            store.snapshots[e].opponent_state["carry"]
        )
        for ck, cv in store.snapshots[e].opponent_state["carry"].items():
            assert torch.equal(restored_store.snapshots[e].opponent_state["carry"][ck], cv)
    np.testing.assert_array_equal(
        restored_store.finalize_topology().unpack_raw_u16([0, 1, 2]), planes
    )

    # Custom non-derivable Plane 0 lossless downgrade check:
    custom_planes = planes.copy()
    custom_planes[0, 0, 0, 0] = 12345
    tb_custom = pack_sparse_topology(custom_planes, summaries)
    assert tb_custom.p0_derived is False
    tb_custom_rt = SparseTopologyBatch.from_state_dict(tb_custom.to_state_dict())
    np.testing.assert_array_equal(tb_custom_rt.unpack_raw_u16([0, 1, 2]), custom_planes)

    # AnchorArchive v2 lossless round-trip + training_compact describe_state check:
    obs = store.stacked_observations()
    anc = AnchorArchive(
        num_rows=3,
        world_id=np.array([0, 1, 2], dtype=np.int32),
        mix_id=np.array([0, 1, 2], dtype=np.int8),
        is_assessment=np.array([False, False, True], dtype=bool),
        life_id=np.array([1, 2, 1], dtype=np.int32),
        tick=np.array([5, 6, 7], dtype=np.int32),
        hero_length=np.array([50, 60, 70], dtype=np.int32),
        local=obs["local"],
        glob=obs["glob"],
        scalars=obs["scalars"],
        legal_mask=obs["legal_mask"],
        q_b8=obs["q_b8"],
        b8_action=np.array([0, 1, 2], dtype=np.int8),
        topology=store.finalize_topology(),
    )
    anc_sd = anc.to_state_dict()
    assert anc_sd["schema"] == "ct-anchor-archive/v2"
    assert describe_state(anc_sd)["tensor_count"] > 0
    anc_rt = AnchorArchive.from_state_dict(anc_sd)
    assert torch.equal(anc_rt.local, anc.local)
    assert torch.equal(anc_rt.glob, anc.glob)
    assert torch.equal(anc_rt.scalars, anc.scalars)
    assert torch.equal(anc_rt.q_b8, anc.q_b8)
    np.testing.assert_array_equal(anc_rt.topology.unpack_raw_u16([0, 1, 2]), planes)

    # Stage export hard cap refusal & atomic partial cleanup:
    out_dir = tmp_path / "cap_exports"
    with pytest.raises(StageExportOverflowError, match="max_stage_export_bytes"):
        publish_stage_snapshot(
            out_dir,
            1,
            lambda dst: dst.write_bytes(b"x" * 2048),
            metadata={"allocation": "test"},
            max_stage_bytes=1024,
            max_total_bytes=4096,
        )
    assert not list(out_dir.glob("*"))

    publish_stage_snapshot(
        out_dir,
        1,
        lambda dst: dst.write_bytes(b"x" * 900),
        metadata={"allocation": "test"},
        max_stage_bytes=1024,
        max_total_bytes=1800,
    )
    with pytest.raises(StageExportOverflowError, match="max_total_export_bytes"):
        publish_stage_snapshot(
            out_dir,
            2,
            lambda dst: dst.write_bytes(b"y" * 800),
            metadata={"allocation": "test"},
            max_stage_bytes=1024,
            max_total_bytes=1800,
        )
    assert not (out_dir / "stage-0002.pt").exists()
    assert not (out_dir / "stage-0002.pt.complete").exists()
    assert not list(out_dir.glob("*.partial"))


def test_recursive_total_export_cap_counts_sidecars_headers_summaries_and_diagnostics(
    tmp_path: object,
) -> None:
    """Re-review Finding 3 regression: max_total_bytes bounds all 6 stage files + diagnostics."""
    from pathlib import Path

    from research.sandbox_a_resident.ct_units import (
        PUBLISH_SIDECAR_BYTES,
        StageExportOverflowError,
        publish_bounded_export,
        publish_stage_snapshot,
        recursive_export_bytes,
    )

    assert isinstance(tmp_path, Path)

    # 1) Astra's exact counterexample: 100-byte payload w/ max_stage_bytes=100, max_total_bytes=100
    #    must be refused BEFORE os.link because the 6-file stage bundle exceeds 100 bytes.
    counter_dir = tmp_path / "astra_100b_counterexample"
    with pytest.raises(StageExportOverflowError, match="max_total_export_bytes"):
        publish_stage_snapshot(
            counter_dir,
            1,
            lambda dst: dst.write_bytes(b"z" * 100),
            metadata={"allocation": "test"},
            max_stage_bytes=100,
            max_total_bytes=100,
        )
    assert recursive_export_bytes(counter_dir) == 0
    assert not list(counter_dir.glob("*"))

    # 2) Exact 6-file byte boundary (path-length independent via identical basename length "bnd_X").
    probe_dir = tmp_path / "bnd_0"
    publish_stage_snapshot(
        probe_dir,
        1,
        lambda dst: dst.write_bytes(b"z" * 100),
        metadata={"allocation": "test"},
        max_stage_bytes=100,
    )
    exact_six_bytes = recursive_export_bytes(probe_dir)
    assert exact_six_bytes > 500
    assert len([p for p in probe_dir.rglob("*") if p.is_file()]) == 6

    under_dir = tmp_path / "bnd_1"
    with pytest.raises(StageExportOverflowError, match="max_total_export_bytes"):
        publish_stage_snapshot(
            under_dir,
            1,
            lambda dst: dst.write_bytes(b"z" * 100),
            metadata={"allocation": "test"},
            max_stage_bytes=100,
            max_total_bytes=exact_six_bytes - 1,
        )
    assert recursive_export_bytes(under_dir) == 0

    exact_dir = tmp_path / "bnd_2"
    publish_stage_snapshot(
        exact_dir,
        1,
        lambda dst: dst.write_bytes(b"z" * 100),
        metadata={"allocation": "test"},
        max_stage_bytes=100,
        max_total_bytes=exact_six_bytes,
    )
    assert recursive_export_bytes(exact_dir) == exact_six_bytes

    # 3) Reserved tail allowance + nested stall-watchdog diagnostic bytes are enforced recursively.
    reserve_fail_dir = tmp_path / "bnd_3"
    with pytest.raises(StageExportOverflowError, match="reserved_tail=51"):
        publish_stage_snapshot(
            reserve_fail_dir,
            1,
            lambda dst: dst.write_bytes(b"z" * 100),
            metadata={"allocation": "test"},
            max_stage_bytes=100,
            max_total_bytes=exact_six_bytes + 50,
            reserved_tail_bytes=51,
        )
    assert recursive_export_bytes(reserve_fail_dir) == 0

    diag_dir = tmp_path / "bnd_4"
    (diag_dir / "stall-watchdog").mkdir(parents=True)
    (diag_dir / "stall-watchdog/stall-watchdog.json").write_bytes(b"d" * 200)
    with pytest.raises(StageExportOverflowError, match="existing=200"):
        publish_stage_snapshot(
            diag_dir,
            1,
            lambda dst: dst.write_bytes(b"z" * 100),
            metadata={"allocation": "test"},
            max_stage_bytes=100,
            max_total_bytes=exact_six_bytes + 199,
        )
    assert not (diag_dir / "stage-0001.pt").exists()

    # 4) Terminal summary via publish_bounded_export enforces payload + 74 B sidecars + reserve.
    summary_bytes = b'{"verdict": "STOP"}\n'
    summary_bundle = len(summary_bytes) + PUBLISH_SIDECAR_BYTES
    with pytest.raises(StageExportOverflowError, match="max_total_export_bytes"):
        publish_bounded_export(
            exact_dir / "summary.json",
            lambda dst: dst.write_bytes(summary_bytes),
            export_root=exact_dir,
            max_total_bytes=exact_six_bytes + summary_bundle + 19,
            reserved_tail_bytes=20,
        )
    assert not (exact_dir / "summary.json").exists()

    publish_bounded_export(
        exact_dir / "summary.json",
        lambda dst: dst.write_bytes(summary_bytes),
        export_root=exact_dir,
        max_total_bytes=exact_six_bytes + summary_bundle + 20,
        reserved_tail_bytes=20,
    )
    assert (exact_dir / "summary.json.complete").is_file()
    assert recursive_export_bytes(exact_dir) == exact_six_bytes + summary_bundle
