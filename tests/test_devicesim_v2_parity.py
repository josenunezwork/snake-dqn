"""Local CPU mechanics-v2 parity tests for DeviceSim and CTDeviceSim vs GridBatchSim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.devicesim_v2_parity.harness import (
    build_product_world_bundle,
    build_watch6_v2_cpu_sim,
    compare_discrete_sim_state,
    compare_ego_bv8_observations,
    derive_world_seeds,
    install_splitmix_rng_proxies,
    prepare_cpu_policy_phase,
    run_v2_parity_rollout,
    splitmix_positive_scalar,
    watch6_v2_config,
)
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import (
    DEATH_HEAD,
    DeviceSim,
    _splitmix_positive,
)
from research.gpu_sim_spike.ego_device import build_observation
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from src.simd_env import v8_nb
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import FOOD_AMBIENT, FOOD_CORPSE, GridBatchSim

SAVED_SNAKES = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
B8_CKPT = SAVED_SNAKES / "b8_s3_t7_20261016.pth"
FRP3_CKPT = SAVED_SNAKES / "frp3_m3_s12_u60000_20261005.pth"


def test_splitmix_scalar_matches_device_tensor_including_uint64_high_bit() -> None:
    raw_seeds = [0, 1, 42, (1 << 63) - 1, 1 << 63, (1 << 63) + 1337, (1 << 64) - 1]
    cfg = BatchSimConfig(num_envs=len(raw_seeds), num_snakes=2, initial_food=0, max_food=0)
    cpu_sim = GridBatchSim(cfg, seeds=raw_seeds, train_mode=False, allow_respawn=True)
    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    ct_sim = CTDeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    assert torch.equal(gpu_sim.rng_key, ct_sim.rng_key)

    for counter in (1, 2, 17, 999):
        cnt_t = torch.full_like(gpu_sim.rng_key, counter)
        vec = _splitmix_positive(gpu_sim.rng_key, cnt_t).tolist()
        for idx, key_signed in enumerate(gpu_sim.rng_key.tolist()):
            assert splitmix_positive_scalar(key_signed, counter) == int(vec[idx])


@pytest.mark.parametrize("backend", ["reference", "indexed"])
def test_bug1_instep_ambient_refill_per_slot_before_later_eats_and_corpse_drops(
    backend: str,
) -> None:
    """Slot 0 & slot 1 eat ambient food -> each refills immediately in-step BEFORE slot 2 dies."""
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=200,
        game_height=120,
        segment_size=10,
        wall_thickness=10,
        initial_food=0,
        max_food=2,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        max_capacity=64,
        frame_rate=1,
    )
    cpu_sim = GridBatchSim(cfg, seeds=[77], train_mode=False, allow_respawn=True)
    # Place slot 0 at (5, 5) moving right onto ambient food at (6, 5).
    # Place slot 1 at (10, 5) moving right onto ambient food at (11, 5).
    # Place slot 2 at (19, 8) moving right into wall x=20 so its corpse drops AFTER both refills!
    cpu_sim.bodies.fill(0)
    cpu_sim.bodies[0, 0, 0] = (5, 5)
    cpu_sim.bodies[0, 1, 0] = (10, 5)
    cpu_sim.bodies[0, 2, :3] = ((17, 8), (18, 8), (19, 8))
    cpu_sim.head_ptr[0] = (0, 0, 2)
    cpu_sim.seg_count[0] = (1, 1, 3)
    cpu_sim.length[0] = (1, 1, 3)
    cpu_sim._reward_prev_length[0] = (1, 1, 3)
    cpu_sim.direction[0] = (1, 1, 1)
    cpu_sim.alive[0] = (True, True, True)
    cpu_sim.food_cells[0] = [(6, 5), (11, 5)]
    cpu_sim.food_set[0] = {(6, 5), (11, 5)}
    cpu_sim.corpse_cells[0] = set()
    cpu_sim._rebuild_env_grids(0, food=True)
    cpu_sim._policy_phase_prepared = True

    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    gpu_sim.move_backend = backend
    ct_sim = CTDeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    ct_sim.move_backend = backend

    # Inject deterministic draw tape:
    # Draw 1 (after slot 0 eats (6,5)): hits (18, 8), occupied by slot 2 pre-collision -> rejected!
    # Draw 2: (14, 6) -> placed as ambient food #1!
    # Draw 3 (after slot 1 eats (11,5)): hits (18, 8), STILL occupied by slot 2 -> rejected!
    # Draw 4: (15, 6) -> placed as ambient food #2 BEFORE slot 2 dies & drops corpse!
    tape_cells = [(18, 8), (14, 6), (18, 8), (15, 6)]

    class TapeRng:
        def __init__(self, cells: list[tuple[int, int]]) -> None:
            self.cells = list(cells)
            self.idx = 0

        def _draw_find_empty_candidate(self) -> tuple[int, int]:
            cell = self.cells[self.idx]
            self.idx += 1
            return (cell[0] * 10, cell[1] * 10)

    cpu_tape = TapeRng(tape_cells)
    cpu_sim._rngs[0] = cpu_tape  # type: ignore[assignment]

    for sim in (gpu_sim, ct_sim):
        state = {"i": 0}

        def _fake_random(s=sim, st=state) -> torch.Tensor:
            c = tape_cells[st["i"]]
            st["i"] += 1
            s.rng_counter += 1
            return torch.tensor([[c[0], c[1]]], dtype=torch.long)

        sim._random_cells = _fake_random  # type: ignore[method-assign]

    actions = np.array([[1, 1, 1]], dtype=np.int64)
    cpu_sim.step(actions)
    gpu_sim.step(torch.as_tensor(actions))
    ct_sim.step(torch.as_tensor(actions))

    assert bool(cpu_sim._last_ambient_food_ate[0, 0])
    assert bool(cpu_sim._last_ambient_food_ate[0, 1])
    assert cpu_tape.idx == 4
    assert int(gpu_sim.rng_counter[0].item()) == 4
    assert int(ct_sim.rng_counter[0].item()) == 4
    assert cpu_sim.food_cells[0] == [(14, 6), (15, 6), (19, 8), (18, 8)]
    assert compare_discrete_sim_state(cpu_sim, gpu_sim) == []
    assert compare_discrete_sim_state(cpu_sim, ct_sim) == []

    # Also verify wall_thickness=10 margin excludes wall cells {0, grid_w-1} and {0, grid_h-1}.
    margin_sim = DeviceSim(watch6_v2_config(4), device="cpu", hero_respawn=True)
    for _ in range(250):
        drawn = margin_sim._random_cells()
        assert bool((drawn[:, 0] >= 1).all() and (drawn[:, 0] <= margin_sim.grid_w - 2).all())
        assert bool((drawn[:, 1] >= 1).all() and (drawn[:, 1] <= margin_sim.grid_h - 2).all())


@pytest.mark.parametrize("backend", ["reference", "indexed"])
def test_bug2_boost_trail_fifo_evicts_oldest_corpse_at_450_cap_and_duplicate_no_evict(
    backend: str,
) -> None:
    """At full 450 pellets (300 ambient + 150 corpse), boost burn FIFO-evicts oldest corpse."""
    cfg = watch6_v2_config(1, max_capacity=128)
    cpu_sim = GridBatchSim(cfg, seeds=[101], train_mode=False, allow_respawn=True)
    cpu_sim.bodies.fill(0)
    # Slot 0 length 6 at y=10: head (30,10)..tail (25,10), boost_frames=2 -> burns tail on boost
    # while board is at full 450 pellets (300 ambient + 150 corpse), verifying FIFO corpse eviction!
    for slot in range(6):
        y = 10 + slot * 8
        for k in range(6):
            cpu_sim.bodies[0, slot, 5 - k] = (30 - k, y)
        cpu_sim.head_ptr[0, slot] = 5
        cpu_sim.seg_count[0, slot] = 6
        cpu_sim.length[0, slot] = 6
        cpu_sim._reward_prev_length[0, slot] = 6
        cpu_sim.direction[0, slot] = 1
        cpu_sim.alive[0, slot] = True
        cpu_sim.boost_frames[0, slot] = 2

    snake_occupied = {tuple(map(int, cpu_sim.bodies[0, s, k])) for s in range(6) for k in range(6)}
    free_cells = [
        (x, y) for y in range(2, 81) for x in range(40, 140) if (x, y) not in snake_occupied
    ]
    ambient_300 = free_cells[:300]
    corpse_150 = free_cells[300:450]
    oldest_corpse = corpse_150[0]
    second_oldest_corpse = corpse_150[1]

    ordered = ambient_300 + corpse_150
    cpu_sim.food_cells[0] = list(ordered)
    cpu_sim.food_set[0] = set(ordered)
    cpu_sim.corpse_cells[0] = set(corpse_150)
    cpu_sim._rebuild_env_grids(0, food=True)
    cpu_sim._food_array[0, :450] = np.asarray(ordered, dtype=np.int64)
    cpu_sim._food_n[0] = 450
    cpu_sim._amb_count[0] = 300
    for cx, cy in ambient_300:
        cpu_sim._food[0, cy, cx] = FOOD_AMBIENT
    for cx, cy in corpse_150:
        cpu_sim._food[0, cy, cx] = FOOD_CORPSE
    cpu_sim._policy_phase_prepared = True

    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    gpu_sim.move_backend = backend
    ct_sim = CTDeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    ct_sim.move_backend = backend
    proxies = install_splitmix_rng_proxies(cpu_sim, gpu_sim)

    # Slot 0 boosts (action 4 = straight boost); slots 1..5 go straight normal (action 1).
    actions = np.array([[4, 1, 1, 1, 1, 1]], dtype=np.int64)
    cpu_sim.step(actions)
    gpu_sim.step(torch.as_tensor(actions))
    ct_sim.step(torch.as_tensor(actions))

    assert len(cpu_sim.food_cells[0]) == 450
    assert len(cpu_sim.corpse_cells[0]) == 150
    assert oldest_corpse not in cpu_sim.food_set[0]
    assert second_oldest_corpse in cpu_sim.food_set[0]
    assert cpu_sim.food_cells[0][-1] == (27, 10)
    assert (27, 10) in cpu_sim.corpse_cells[0]
    assert compare_discrete_sim_state(cpu_sim, gpu_sim, proxies=proxies) == []
    assert compare_discrete_sim_state(cpu_sim, ct_sim, proxies=proxies) == []

    # Duplicate check BEFORE eviction: calling _append_food with an existing coordinate (even
    # second_oldest_corpse itself!) at full 450 cap must NOT evict second_oldest_corpse!
    before_cells = gpu_sim.food_cells.clone()
    before_corpse = gpu_sim.corpse_food.clone()
    dup_cell = torch.tensor([[second_oldest_corpse[0], second_oldest_corpse[1]]], dtype=torch.long)
    gpu_sim._append_food(dup_cell, torch.tensor([True]), True)
    assert torch.equal(gpu_sim.food_cells, before_cells)
    assert torch.equal(gpu_sim.corpse_food, before_corpse)


@pytest.mark.parametrize("backend", ["reference", "indexed"])
@pytest.mark.parametrize(
    "len_a,len_b,expect_a_alive,expect_b_alive,expect_a_kills",
    [
        (23, 20, True, False, 1),  # 23 >= 1.15 * 20 = 23.0 -> slot 0 wins head-on!
        (22, 20, False, False, 0),  # 22 < 23.0 -> mutual head-on death!
    ],
)
def test_size_ratio_115_headon_win_and_mutual_tie(
    backend: str,
    len_a: int,
    len_b: int,
    expect_a_alive: bool,
    expect_b_alive: bool,
    expect_a_kills: int,
) -> None:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=600,
        game_height=200,
        segment_size=10,
        wall_thickness=10,
        initial_food=0,
        max_food=0,
        mechanics_version=2,
        max_capacity=64,
        frame_rate=1,
    )
    cpu_sim = GridBatchSim(cfg, seeds=[303], train_mode=False, allow_respawn=True)
    cpu_sim.bodies.fill(0)
    for k in range(len_a):
        cpu_sim.bodies[0, 0, len_a - 1 - k] = (29 - k, 10)
    for k in range(len_b):
        cpu_sim.bodies[0, 1, len_b - 1 - k] = (31 + k, 10)
    cpu_sim.head_ptr[0] = (len_a - 1, len_b - 1)
    cpu_sim.seg_count[0] = (len_a, len_b)
    cpu_sim.length[0] = (len_a, len_b)
    cpu_sim._reward_prev_length[0] = (len_a, len_b)
    cpu_sim.direction[0] = (1, 3)  # Both move onto (30, 10)
    cpu_sim.alive[0] = (True, True)
    cpu_sim._rebuild_env_grids(0, food=True)
    cpu_sim._policy_phase_prepared = True

    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    gpu_sim.move_backend = backend
    ct_sim = CTDeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    ct_sim.move_backend = backend

    actions = np.array([[1, 1]], dtype=np.int64)
    cpu_sim.step(actions)
    gpu_sim.step(torch.as_tensor(actions))
    ct_sim.step(torch.as_tensor(actions))

    assert bool(cpu_sim.alive[0, 0]) is expect_a_alive
    assert bool(cpu_sim.alive[0, 1]) is expect_b_alive
    assert int(cpu_sim._last_kills[0, 0]) == expect_a_kills
    assert int(cpu_sim._last_death_cause[0, 1]) == DEATH_HEAD
    assert compare_discrete_sim_state(cpu_sim, gpu_sim) == []
    assert compare_discrete_sim_state(cpu_sim, ct_sim) == []


@pytest.mark.parametrize("backend", ["reference", "indexed"])
def test_bug3_continuous_respawn_hero_and_opponents_frame_rate_1(backend: str) -> None:
    """Hero slot 0 and opponent slots respawn on next prepare() with 100-trial margined draws."""
    cfg = watch6_v2_config(2, max_capacity=64)
    cpu_sim = GridBatchSim(cfg, seeds=[501, 502], train_mode=False, allow_respawn=True)
    # Mark hero slot 0 and opponents 2, 4 dead with respawn_timer=1 (just died under frame_rate=1).
    for env in range(2):
        for dead_slot in (0, 2, 4):
            cpu_sim.alive[env, dead_slot] = False
            cpu_sim.respawn_timer[env, dead_slot] = 1
        cpu_sim._rebuild_env_grids(env, food=True)

    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    gpu_sim.move_backend = backend
    ct_sim = CTDeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    ct_sim.move_backend = backend
    proxies = install_splitmix_rng_proxies(cpu_sim, gpu_sim)

    # Also verify terminal-hero default (hero_respawn=False) keeps slot 0 dead, respawning 2 & 4.
    term_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=False)
    term_sim.move_backend = backend
    term_sim.prepare()
    assert not bool(term_sim.alive[:, 0].any())
    assert bool(term_sim.alive[:, 2].all() and term_sim.alive[:, 4].all())

    # Continuous respawn (hero_respawn=True) respawns hero slot 0 + opponents 2, 4 in lockstep!
    prepare_cpu_policy_phase(cpu_sim)
    gpu_sim.prepare()
    ct_sim.prepare()

    assert bool(cpu_sim.alive.all())
    assert bool(gpu_sim.alive.all())
    assert bool(ct_sim.alive.all())
    assert (
        compare_discrete_sim_state(cpu_sim, gpu_sim, proxies=proxies, check_step_outputs=False)
        == []
    )
    assert (
        compare_discrete_sim_state(cpu_sim, ct_sim, proxies=proxies, check_step_outputs=False) == []
    )


def test_bug4_mid_boost_death_then_respawn_ego2s_bv8_observation_clears_boost_flag() -> None:
    """A snake that died mid-boost and respawned at length=1 must not show boost=True in bv8."""
    cfg = watch6_v2_config(1, max_capacity=64)
    cpu_sim = GridBatchSim(cfg, seeds=[808], train_mode=False, allow_respawn=True)
    # Place hero (slot 0) at (50, 40) length 10, and opponent (slot 1) at (54, 3) length 6
    # boosting UP (direction=0, action=4) into wall y=0!
    cpu_sim.bodies.fill(0)
    for k in range(10):
        cpu_sim.bodies[0, 0, 9 - k] = (50 - k, 40)
    for k in range(6):
        cpu_sim.bodies[0, 1, 5 - k] = (54, 1 + k)
    for s in range(2, 6):
        cpu_sim.bodies[0, s, 0] = (80 + s * 5, 60)
    cpu_sim.head_ptr[0] = (9, 5, 0, 0, 0, 0)
    cpu_sim.seg_count[0] = (10, 6, 1, 1, 1, 1)
    cpu_sim.length[0] = (10, 6, 1, 1, 1, 1)
    cpu_sim._reward_prev_length[0] = (10, 6, 1, 1, 1, 1)
    cpu_sim.direction[0] = (1, 0, 1, 1, 1, 1)
    cpu_sim.alive[0] = True
    cpu_sim._rebuild_env_grids(0, food=True)
    cpu_sim._policy_phase_prepared = True

    gpu_sim = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True, prepared=True)
    proxies = install_splitmix_rng_proxies(cpu_sim, gpu_sim)

    # Step 1: slot 1 boosts into wall and dies -> _boosted_this_step[0, 1] becomes True!
    actions = np.array([[1, 4, 1, 1, 1, 1]], dtype=np.int64)
    cpu_sim.step(actions)
    gpu_sim.step(torch.as_tensor(actions))
    assert not bool(cpu_sim.alive[0, 1])
    assert bool(cpu_sim._boosted_this_step[0, 1])
    assert bool(gpu_sim._boosted_this_step[0, 1].item())

    # Prepare next tick: slot 1 respawns at length=1, while _boosted_this_step[0, 1] is STILL True!
    # Force respawn close to hero (52, 40) so slot 1 is hero's nearest enemy (scalars 17, 18) AND
    # inside hero's 31x31 local crop (local plane 6)!
    cpu_sim.bodies[0, 1, 0] = (53, 40)
    cpu_sim.head_ptr[0, 1] = 0
    cpu_sim.seg_count[0, 1] = 1
    cpu_sim.length[0, 1] = 1
    cpu_sim.direction[0, 1] = 1
    cpu_sim.alive[0, 1] = True
    cpu_sim.respawn_timer[0, 1] = 0
    cpu_sim._rebuild_env_grids(0, food=True)

    gpu_sim.bodies[0, 1, 0] = torch.tensor([53, 40], dtype=torch.long)
    gpu_sim.head_ptr[0, 1] = 0
    gpu_sim.seg_count[0, 1] = 1
    gpu_sim.length[0, 1] = 1
    gpu_sim.direction[0, 1] = 1
    gpu_sim.alive[0, 1] = True
    gpu_sim.respawn_timer[0, 1] = 0

    assert bool(cpu_sim._boosted_this_step[0, 1]) and int(cpu_sim.length[0, 1]) == 1
    assert bool(gpu_sim._boosted_this_step[0, 1].item()) and int(gpu_sim.length[0, 1].item()) == 1

    rows = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    got_local, got_glob, got_scalars = build_observation(gpu_sim, rows)
    exp = build_ego_raster(
        cpu_sim, EgoRasterConfig(version="b"), backend="numba", rows=rows.numpy()
    )
    # Non-boosting enemy_next encodes step=1 as 128 (never 255 boosting!); scalar 17 is 0.0.
    assert int(got_local[:, 6].max().item()) == 128
    assert float(got_scalars[0, 17].item()) == 0.0
    np.testing.assert_array_equal(got_local.numpy(), exp["local"])
    np.testing.assert_array_equal(got_glob.numpy(), exp["global"])
    np.testing.assert_allclose(got_scalars.numpy(), exp["scalars"], rtol=0.0, atol=1e-6)
    assert compare_ego_bv8_observations(cpu_sim, gpu_sim) == []
    del proxies


@pytest.mark.skipif(
    not (B8_CKPT.is_file() and FRP3_CKPT.is_file()),
    reason="product B8 and FRP3 checkpoints required",
)
def test_bug5_selector_seam_prepare_and_vector61_carry_across_5x_frp3_v8() -> None:
    """Verify step_with_policy seam, CapturedWorld prepared seam, and post-observe carry seam."""
    cpu_sim = build_watch6_v2_cpu_sim([9101, 9102], max_capacity=256, mature_fraction=0.5)
    world = build_product_world_bundle(cpu_sim, B8_CKPT, FRP3_CKPT)
    proxies = install_splitmix_rng_proxies(
        cpu_sim, DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    )
    del proxies

    # Advance 2 steps on CPU so Vector61Runtime builds live prev_dist / prev_id carry AND
    # check that v8_nb.v8_decide matches Vector61SimdPolicy.actions bit-for-bit!
    from dataclasses import replace
    from unittest.mock import patch

    from src.core import game_config as gc_mod

    base_app = gc_mod.get_config()
    v2_app = replace(
        base_app,
        game=replace(
            base_app.game,
            mechanics_version=int(cpu_sim.cfg.mechanics_version),
            min_boost_length=int(cpu_sim.cfg.min_boost_length),
            boost_length_cost_frames=int(cpu_sim.cfg.boost_length_cost_frames),
            arena_type=str(cpu_sim.cfg.arena_type),
        ),
    )

    for _ in range(2):
        prepare_cpu_policy_phase(cpu_sim)
        world.runtime.prepare(cpu_sim)
        opp_rows = np.array([[e, s] for s in range(1, 6) for e in range(2) if cpu_sim.alive[e, s]])
        states, exact = world.runtime.selection(opp_rows)
        q_opp = world.opponents[0].q_values(states).detach().cpu().numpy().astype(np.float32)
        nb_act = v8_nb.v8_decide(cpu_sim, opp_rows, q_opp, exact, lam=8.0, features=False).action
        with patch.object(gc_mod, "get_config", return_value=v2_app):
            for slot in range(1, 6):
                slot_rows = opp_rows[opp_rows[:, 1] == slot]
                ref_act = world.opponents[slot - 1].actions(
                    cpu_sim.get_resolved_action_mask(), cpu_sim, slot_rows
                )
                np.testing.assert_array_equal(nb_act[opp_rows[:, 1] == slot], ref_act)
        act_all = np.ones((2, 6), dtype=np.int64)
        act_all[opp_rows[:, 0], opp_rows[:, 1]] = nb_act
        cpu_sim.step(act_all)
        world.runtime.observe_step(cpu_sim)

    # Seam A: post-observe_step (unprepared sim, valid carry in runtime).
    gpu_unprep = DeviceSim.from_reference(cpu_sim, device="cpu", hero_respawn=True)
    pol_unprep = DevicePolicy.from_world(world, device="cpu", sim=gpu_unprep)
    assert gpu_unprep._prepared is False
    assert pol_unprep.vector_prepared is False

    # Seam B: inside step_with_policy selector callback (_policy_step_in_progress=True,
    # _policy_phase_prepared not yet set, runtime.prepare already called).
    captured: dict[str, object] = {}

    def _selector(sim_in_seam: GridBatchSim) -> np.ndarray:
        assert sim_in_seam._policy_step_in_progress is True
        assert getattr(sim_in_seam, "_policy_phase_prepared", False) is False
        world.runtime.prepare(sim_in_seam)
        g_sim = DeviceSim.from_reference(sim_in_seam, device="cpu", hero_respawn=True)
        g_pol = DevicePolicy.from_world(world, device="cpu", sim=g_sim)
        captured["sim_prepared"] = g_sim._prepared
        captured["vec_prepared"] = g_pol.vector_prepared
        captured["frame_before"] = g_sim.frame.clone()
        _q, _leg, h_act = g_pol.evaluate(g_sim)
        g_actions = g_pol.actions(g_sim, h_act)
        captured["gpu_actions"] = g_actions.numpy().copy()
        captured["g_sim"] = g_sim
        captured["g_pol"] = g_pol
        return g_actions.numpy()

    cpu_sim.step_with_policy(_selector)
    world.runtime.observe_step(cpu_sim)

    assert captured["sim_prepared"] is True
    assert captured["vec_prepared"] is True
    g_sim_b = captured["g_sim"]
    g_pol_b = captured["g_pol"]
    assert isinstance(g_sim_b, DeviceSim) and isinstance(g_pol_b, DevicePolicy)
    # Verify frame was NOT double-incremented when g_sim_b steps from the prepared selector seam.
    g_sim_b.step(torch.as_tensor(captured["gpu_actions"]))
    g_pol_b.after_step(g_sim_b)
    np.testing.assert_array_equal(cpu_sim.frame, g_sim_b.frame.numpy())


def test_end_to_end_watch6_v2_scripted_stress_rollout_cpu() -> None:
    seeds = derive_world_seeds(0, 12)
    report = run_v2_parity_rollout(
        seeds=seeds,
        ticks=60,
        device="cpu",
        b8_checkpoint=None,
        frp3_checkpoint=None,
        check_ct_sim=True,
        ego_check_stride=15,
        mature_fraction=0.5,
        max_capacity=256,
    )
    assert report["verdict"] == "PASS", report["mismatch_samples"]
    assert report["unexplained_mismatches"] == 0
    assert report["discrete_mismatches"] == 0
    assert report["ct_discrete_mismatches"] == 0
    assert report["ego_obs_mismatches"] == 0
    assert report["events"]["ambient_eats_and_instep_refills"] > 0
    assert report["events"]["boost_trail_burns"] > 0
    assert report["events"]["boost_burns_at_full_450_cap"] > 0
    assert report["events"]["deaths_total"] > 0
    assert report["events"]["respawns_total"] > 0


@pytest.mark.skipif(
    not (B8_CKPT.is_file() and FRP3_CKPT.is_file()),
    reason="product B8 and FRP3 checkpoints required",
)
def test_end_to_end_watch6_v2_product_b8_frp3_v8_rollout_cpu() -> None:
    seeds = derive_world_seeds(1, 6)
    report = run_v2_parity_rollout(
        seeds=seeds,
        ticks=25,
        device="cpu",
        b8_checkpoint=B8_CKPT,
        frp3_checkpoint=FRP3_CKPT,
        check_ct_sim=True,
        ego_check_stride=10,
        mature_fraction=0.5,
        max_capacity=256,
    )
    assert report["verdict"] == "PASS", report["mismatch_samples"]
    assert report["unexplained_mismatches"] == 0
    assert report["discrete_mismatches"] == 0
    assert report["ct_discrete_mismatches"] == 0
    assert report["ego_obs_mismatches"] == 0
    assert report["total_live_decisions"] > 0
