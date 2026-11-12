"""Exact bitwise parity tests between NumPy ReferenceWorld and Torch GameV3Sim (CPU)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
import torch

from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs, build_obs_reference
from research.game_v3.reference import ReferenceStepInfo, ReferenceWorld
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import STATE_FIELDS, GameV3Sim, StepInfo

_SNAKE_STATE_FIELDS: tuple[str, ...] = tuple(
    field.name for field in STATE_FIELDS if field.in_reference_snake
)
_MASS_LEDGER_FIELDS: tuple[str, ...] = (
    "total_spawned_snake_mass",
    "total_spawned_ambient_mass",
    "total_corpse_decay_sink",
    "total_boost_decay_sink",
    "total_length_cap_sink",
    "total_truncation_sink",
)


def assert_step_info_parity(
    info: StepInfo, ref_infos: Sequence[ReferenceStepInfo], tick: int, label: str = "cpu"
) -> None:
    """Assert every StepInfo hook field is bitwise-identical to ReferenceWorld."""
    for field_name in info.__dataclass_fields__:
        got = getattr(info, field_name).detach().cpu().numpy()
        exp = np.stack([getattr(ri, field_name) for ri in ref_infos], axis=0)
        if not np.array_equal(got, exp):
            diff = np.argwhere(got != exp)[0]
            raise AssertionError(
                f"[{label}] StepInfo.{field_name} diverged at tick={tick}, idx={tuple(diff)}: "
                f"got={got[tuple(diff)]}, exp={exp[tuple(diff)]}"
            )


def assert_full_state_and_obs_parity(
    sim: GameV3Sim,
    refs: Sequence[ReferenceWorld],
    tick: int,
    *,
    check_rings: bool = False,
    check_obs: bool = False,
    learner_mask: torch.Tensor | None = None,
    label: str = "cpu",
) -> None:
    """Assert bitwise-identical full state (and optional rings/obs) vs ReferenceWorld."""
    got_wticks = sim.world_ticks.detach().cpu().numpy()
    exp_wticks = np.array([r.tick for r in refs], dtype=np.int64)
    assert np.array_equal(
        got_wticks, exp_wticks
    ), f"[{label}] world_ticks mismatch at {tick}: {got_wticks} != {exp_wticks}"

    got_occ = sim.active_occupancy_grid().detach().cpu().numpy()
    exp_occ = np.stack([r.occ_grid for r in refs], axis=0)
    if not np.array_equal(got_occ, exp_occ):
        diff = np.argwhere(got_occ != exp_occ)[0]
        raise AssertionError(
            f"[{label}] active_occupancy_grid diverged at tick={tick}, idx={tuple(diff)}: "
            f"got={got_occ[tuple(diff)]}, exp={exp_occ[tuple(diff)]}"
        )

    for attr in _SNAKE_STATE_FIELDS:
        got = getattr(sim, attr).detach().cpu().numpy()
        exp = np.stack([getattr(r, attr) for r in refs], axis=0)
        if not np.array_equal(got, exp):
            diff = np.argwhere(got != exp)[0]
            raise AssertionError(
                f"[{label}] sim.{attr} diverged at tick={tick}, idx={tuple(diff)}: "
                f"got={got[tuple(diff)]}, exp={exp[tuple(diff)]}"
            )

    got_amb_cnt = sim.ambient_count.detach().cpu().numpy()
    exp_amb_cnt = np.array([r.ambient_count for r in refs], dtype=np.int32)
    assert np.array_equal(got_amb_cnt, exp_amb_cnt), f"[{label}] ambient_count mismatch at {tick}"

    for grid_attr in ("ambient_grid", "corpse_grid"):
        got_g = getattr(sim, grid_attr).detach().cpu().numpy()
        exp_g = np.stack([getattr(r, grid_attr) for r in refs], axis=0)
        if not np.array_equal(got_g, exp_g):
            diff = np.argwhere(got_g != exp_g)[0]
            raise AssertionError(
                f"[{label}] sim.{grid_attr} diverged at tick={tick}, idx={tuple(diff)}: "
                f"got={got_g[tuple(diff)]}, exp={exp_g[tuple(diff)]}"
            )

    for ledger_attr in _MASS_LEDGER_FIELDS:
        got_l = getattr(sim, ledger_attr).detach().cpu().numpy()
        exp_l = np.array([getattr(r, ledger_attr) for r in refs], dtype=np.int64)
        if not np.array_equal(got_l, exp_l):
            diff = np.argwhere(got_l != exp_l)[0]
            raise AssertionError(
                f"[{label}] sim.{ledger_attr} diverged at tick={tick}, idx={tuple(diff)}: "
                f"got={got_l[tuple(diff)]}, exp={exp_l[tuple(diff)]}"
            )

    for e, ref in enumerate(refs):
        ref_snake_mass = int(np.sum(np.where(ref.alive, ref.length, 0), dtype=np.int64))
        ref_amb_mass = int(np.sum(ref.ambient_grid, dtype=np.int64))
        ref_cor_mass = int(np.sum(ref.corpse_grid, dtype=np.int64))
        lhs = ref_snake_mass + ref_amb_mass + ref_cor_mass
        rhs = (
            ref.total_spawned_snake_mass
            + ref.total_spawned_ambient_mass
            - ref.total_corpse_decay_sink
            - ref.total_boost_decay_sink
            - ref.total_length_cap_sink
            - ref.total_truncation_sink
        )
        assert (
            lhs == rhs
        ), f"[{label}] ref[{e}] mass conservation failed at tick={tick}: {lhs} != {rhs}"

    if check_rings:
        max_len = sim.cfg.max_length
        ring_pos_np = sim.ring_pos.detach().cpu().numpy()
        ring_cell_np = sim.ring_cell.detach().cpu().numpy().astype(np.int32)
        ring_r_sq_np = sim.ring_r_sq.detach().cpu().numpy().astype(np.int32)
        head_ptr_np = sim.head_ptr.detach().cpu().numpy()
        seg_cnt_np = sim.seg_count.detach().cpu().numpy()
        alive_np = sim.alive.detach().cpu().numpy()

        for e, ref in enumerate(refs):
            for i in range(sim.N):
                if not alive_np[e, i]:
                    continue
                cnt = int(seg_cnt_np[e, i])
                sim_idxs = (int(head_ptr_np[e, i]) - np.arange(cnt, dtype=np.int32)) % max_len
                ref_idxs = (int(ref.head_ptr[i]) - np.arange(cnt, dtype=np.int32)) % max_len
                assert np.array_equal(ring_pos_np[e, i, sim_idxs], ref.ring_pos[i, ref_idxs])
                assert np.array_equal(ring_cell_np[e, i, sim_idxs], ref.ring_cell[i, ref_idxs])
                assert np.array_equal(ring_r_sq_np[e, i, sim_idxs], ref.ring_r_sq[i, ref_idxs])

    if check_obs:
        obs_sim = build_obs(sim)
        obs_refs = [build_obs_reference(r) for r in refs]
        for obs_attr in ("raster_s1", "raster_s4", "raster_s16", "action_mask"):
            got_o = getattr(obs_sim, obs_attr).detach().cpu().numpy()
            exp_o = np.stack([getattr(or_i, obs_attr) for or_i in obs_refs], axis=0)
            if not np.array_equal(got_o, exp_o):
                diff = np.argwhere(got_o != exp_o)[0]
                raise AssertionError(
                    f"[{label}] obs.{obs_attr} diverged at tick={tick}, idx={tuple(diff)}: "
                    f"got={got_o[tuple(diff)]}, exp={exp_o[tuple(diff)]}"
                )

        # Bitwise IEEE-754 float32 comparison (zero tolerance) on scalars as well!
        got_sc = obs_sim.scalars.detach().cpu().numpy().view(np.uint32)
        exp_sc = np.stack([or_i.scalars for or_i in obs_refs], axis=0).view(np.uint32)
        if not np.array_equal(got_sc, exp_sc):
            diff = np.argwhere(got_sc != exp_sc)[0]
            got_f = obs_sim.scalars.detach().cpu().numpy()[tuple(diff)]
            exp_f = np.stack([or_i.scalars for or_i in obs_refs], axis=0)[tuple(diff)]
            raise AssertionError(
                f"[{label}] obs.scalars diverged bitwise at tick={tick}, idx={tuple(diff)}: "
                f"got={got_f!r}, exp={exp_f!r}"
            )

        if learner_mask is not None:
            mask_np = learner_mask.detach().cpu().numpy()
            idx = LearnerIndex(sim, learner_mask)
            sub_out = allocate_obs_batch(sim, idx)
            obs_sub = build_obs(sim, idx, out=sub_out)
            for obs_attr in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
                got_sub = getattr(obs_sub, obs_attr).detach().cpu().numpy()
                exp_sub = getattr(obs_sim, obs_attr).detach().cpu().numpy()[mask_np]
                if obs_attr == "scalars":
                    assert np.array_equal(got_sub.view(np.uint32), exp_sub.view(np.uint32))
                else:
                    assert np.array_equal(got_sub, exp_sub)


def test_fast_reference_vs_torch_cpu_full_state_and_obs_parity() -> None:
    """Fast smoke: 8 worlds x 120 ticks at N=24 with all 10 actions, collisions, and truncations."""
    cfg = GameV3Config(
        arena_radius=120.0,
        num_snakes=24,
        respawn_delay=8,
        max_life_ticks=55,
        ambient_food_density=0.015,
    )
    num_envs = 8
    seeds = [2000 + e for e in range(num_envs)]
    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    rng = np.random.default_rng(314159)
    learner_mask = torch.from_numpy(rng.random((num_envs, cfg.num_snakes)) < (8.0 / 24.0))
    assert_full_state_and_obs_parity(
        sim, refs, 0, check_rings=True, check_obs=True, learner_mask=learner_mask
    )

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_deaths = 0
    total_truncs = 0

    for t in range(1, 121):
        mode = t % 3
        if mode == 0:
            actions = food_seeking_actions(sim)
        elif mode == 1:
            actions = random_safe_actions(sim)
        else:
            actions = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = actions.numpy()
        np.add.at(action_counts, act_np.ravel(), 1)

        info = sim.step(actions)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t)

        total_deaths += int(info.died.sum().item())
        total_truncs += int(info.truncated.sum().item())

        if t in (1, 15, 30, 55, 56, 90, 120):
            assert_full_state_and_obs_parity(
                sim, refs, t, check_rings=True, check_obs=True, learner_mask=learner_mask
            )

    assert np.all(action_counts > 0), f"All 10 actions must be exercised, got {action_counts}"
    assert total_deaths >= 20
    assert total_truncs >= 10


def test_scripted_policies_respect_action_mask_and_determinism() -> None:
    """Scripted baseline policies must be deterministic and never request boost when masked."""
    cfg = GameV3Config(arena_radius=120.0, num_snakes=24, ambient_food_density=0.008)
    sim = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[10, 20, 30, 40])

    for _ in range(15):
        obs = build_obs(sim)
        a_rs1 = random_safe_actions(sim)
        a_rs2 = random_safe_actions(sim)
        a_fs1 = food_seeking_actions(sim)
        a_fs2 = food_seeking_actions(sim)

        assert torch.equal(a_rs1, a_rs2)
        assert torch.equal(a_fs1, a_fs2)
        assert int(a_rs1.min().item()) >= 0 and int(a_rs1.max().item()) < NUM_ACTIONS
        assert int(a_fs1.min().item()) >= 0 and int(a_fs1.max().item()) < NUM_ACTIONS

        # Wherever boost is masked (actions 5..9 illegal), scripted actions must stay in 0..4.
        boost_forbidden = ~obs.action_mask[:, :, 5]
        assert torch.all(a_rs1[boost_forbidden] < 5)
        assert torch.all(a_fs1[boost_forbidden] < 5)
        sim.step(a_fs1)


@pytest.mark.slow
def test_32_worlds_x_1000_ticks_n24_exact_bitwise_parity_cpu() -> None:
    """Contract (Rule 8): 32 worlds x 1,000 ticks at N=24 on CPU with 0 state/obs divergence."""
    cfg = GameV3Config(
        arena_radius=180.0,
        num_snakes=24,
        respawn_delay=15,
        max_life_ticks=350,
        ambient_food_density=0.006,
    )
    num_envs = 32
    num_ticks = 1000
    seeds = [10_000 + e for e in range(num_envs)]

    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    rng = np.random.default_rng(20261110)
    learner_mask = torch.tensor(
        [[(e + j) % 3 == 0 for j in range(cfg.num_snakes)] for e in range(num_envs)],
        dtype=torch.bool,
    )
    assert_full_state_and_obs_parity(
        sim, refs, 0, check_rings=True, check_obs=True, learner_mask=learner_mask
    )

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_deaths = 0
    total_kills = 0
    total_truncs = 0

    for t in range(1, num_ticks + 1):
        mode = t % 4
        if mode == 0:
            actions = food_seeking_actions(sim)
        elif mode == 1:
            actions = random_safe_actions(sim)
        elif mode == 2:
            # Bias toward boost actions (5..9) to exercise multi-point boost steps & shedding.
            actions = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        else:
            # Mix scripted steering on slots 0..15 with uniform 10-action exploration on 16..23.
            actions = food_seeking_actions(sim).clone()
            actions[:, 16:] = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, 8), dtype=np.int64)
            )

        act_np = actions.numpy()
        np.add.at(action_counts, act_np.ravel(), 1)

        info = sim.step(actions)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="cpu-32x1000")

        total_deaths += int(info.died.sum().item())
        total_kills += int(info.kill_credit.sum().item())
        total_truncs += int(info.truncated.sum().item())

        if t % 100 == 0 or t in (1, 350, 351, num_ticks):
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=(t in (350, num_ticks)),
                check_obs=True,
                learner_mask=learner_mask,
                label="cpu-32x1000",
            )

    assert np.all(action_counts >= 10_000), f"Expected all 10 actions heavily used: {action_counts}"
    assert total_deaths >= 200
    assert total_kills >= 50
    assert total_truncs >= 50


def test_parity_large_snakes_all_body_radius_tiers() -> None:
    """Parity across large snakes exercising every distinct body-radius tier."""
    cfg = GameV3Config(
        arena_radius=180.0,
        num_snakes=20,
        respawn_delay=6,
        ambient_food_density=0.01,
    )
    num_envs = 2
    seeds = [4101, 4102]
    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]
    sim_mps = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
        if torch.backends.mps.is_available()
        else None
    )

    # Collect one representative length for EVERY distinct r_sq_cells tier in [start..max_length].
    r_sq_list = sim._r_sq_table.tolist()
    tier_to_len: dict[int, int] = {}
    for slen in range(cfg.start_length, cfg.max_length + 1):
        tier_val = int(r_sq_list[slen])
        if tier_val not in tier_to_len:
            tier_to_len[tier_val] = slen
    all_tiers = set(tier_to_len.keys())
    rep_lengths = sorted(tier_to_len.values())
    assert len(rep_lengths) >= 12
    slot_lengths = [
        rep_lengths[i % len(rep_lengths)] if i < len(rep_lengths) else cfg.max_length
        for i in range(cfg.num_snakes)
    ]

    # Assign identical multi-tier target lengths to initial alive snakes in sim, mps, and refs;
    # subsequent steps emit points at every body-radius tier, collide, boost, and drop corpses.
    for e in range(num_envs):
        for i in range(cfg.num_snakes):
            if bool(sim.alive[e, i].item()):
                target_l = slot_lengths[i]
                delta = target_l - int(sim.length[e, i].item())
                sim.length[e, i] = target_l
                sim.prev_length[e, i] = target_l
                sim.total_spawned_snake_mass[e] += delta
                refs[e].length[i] = target_l
                refs[e].prev_length[i] = target_l
                refs[e].total_spawned_snake_mass += delta
                if sim_mps is not None:
                    sim_mps.length[e, i] = target_l
                    sim_mps.prev_length[e, i] = target_l
                    sim_mps.total_spawned_snake_mass[e] += delta

    rng = np.random.default_rng(202603)
    seen_tiers: set[int] = set()
    for t in range(1, 65):
        if t % 3 == 0:
            act = food_seeking_actions(sim)
        elif t % 3 == 1:
            act = random_safe_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = act.numpy()
        info = sim.step(act)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="large-tiers")
        if sim_mps is not None:
            info_m = sim_mps.step(act.to("mps"))
            for k in info.__dataclass_fields__:
                assert torch.equal(getattr(info, k), getattr(info_m, k).cpu())

        alive_lens = sim.length[sim.alive].tolist()
        for slen in alive_lens:
            seen_tiers.add(int(r_sq_list[slen]))

        if t in (1, 20, 40, 64):
            assert_full_state_and_obs_parity(
                sim, refs, t, check_rings=True, check_obs=True, label="large-tiers"
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps, refs, t, check_rings=True, check_obs=True, label="large-tiers-mps"
                )

    assert seen_tiers == all_tiers, f"Missing body-radius tiers: {all_tiers - seen_tiers}"


def test_parity_ring_wraparound_max_length_40_for_400_ticks() -> None:
    """Ring buffer wraparound parity with max_length=40 across 400 ticks."""
    cfg = GameV3Config(
        arena_radius=90.0,
        num_snakes=12,
        max_length=40,
        respawn_delay=5,
        ambient_food_density=0.025,
    )
    num_envs = 4
    seeds = [5100 + e for e in range(num_envs)]
    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]
    sim_mps = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
        if torch.backends.mps.is_available()
        else None
    )

    rng = np.random.default_rng(40400)
    max_life_age_seen = 0
    for t in range(1, 401):
        if t % 4 != 3:
            act = food_seeking_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = act.numpy()
        info = sim.step(act)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="ring-wrap-40")
        if sim_mps is not None:
            info_m = sim_mps.step(act.to("mps"))
            for k in info.__dataclass_fields__:
                assert torch.equal(getattr(info, k), getattr(info_m, k).cpu())

        max_life_age_seen = max(max_life_age_seen, int(info.life_age.max().item()))
        if t in (45, 100, 200, 300, 400):
            assert_full_state_and_obs_parity(
                sim, refs, t, check_rings=True, check_obs=True, label="ring-wrap-40"
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps, refs, t, check_rings=True, check_obs=True, label="ring-wrap-40-mps"
                )

    assert (
        max_life_age_seen > cfg.max_length * 2
    ), f"Expected multiple 40-slot ring wraparounds, got max_life_age={max_life_age_seen}"
    assert int(sim.seg_count.max().item()) == cfg.max_length


def test_parity_default_r600_n24() -> None:
    """Bitwise parity at the exact default GameV3Config (R=600, N=24)."""
    cfg = GameV3Config()
    assert cfg.arena_radius == 600.0 and cfg.num_snakes == 24
    num_envs = 2
    seeds = [6101, 6102]
    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]
    sim_mps = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
        if torch.backends.mps.is_available()
        else None
    )

    mask_cpu = torch.tensor(
        [[j < 8 for j in range(cfg.num_snakes)] for _ in range(num_envs)], dtype=torch.bool
    )
    assert_full_state_and_obs_parity(
        sim, refs, 0, check_rings=True, check_obs=True, learner_mask=mask_cpu, label="r600-n24"
    )

    rng = np.random.default_rng(60024)
    for t in range(1, 51):
        if t % 2 == 0:
            act = food_seeking_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = act.numpy()
        info = sim.step(act)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="r600-n24")
        if sim_mps is not None:
            info_m = sim_mps.step(act.to("mps"))
            for k in info.__dataclass_fields__:
                assert torch.equal(getattr(info, k), getattr(info_m, k).cpu())

        if t in (1, 25, 50):
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=mask_cpu,
                label="r600-n24",
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=mask_cpu.to("mps"),
                    label="r600-n24-mps",
                )


def test_parity_truncate_masks_all_shapes_vs_reference() -> None:
    """Parity vs ReferenceWorld for bool, (E,), and (E, N) truncate= masks."""
    cfg = GameV3Config(
        arena_radius=100.0,
        num_snakes=16,
        respawn_delay=5,
        max_life_ticks=40,
        ambient_food_density=0.02,
    )
    num_envs = 6
    seeds = [7100 + e for e in range(num_envs)]
    sim = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]
    sim_mps = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
        if torch.backends.mps.is_available()
        else None
    )

    rng = np.random.default_rng(710016)
    learner_mask = torch.from_numpy(rng.random((num_envs, cfg.num_snakes)) < 0.4)
    total_rollout_truncs = 0

    for t in range(1, 91):
        if t % 3 == 0:
            act = food_seeking_actions(sim)
        elif t % 3 == 1:
            act = random_safe_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = act.numpy()

        # Cycle across all supported truncate= forms: False, True, (E,), (E, N), and post-step flag.
        use_post_flag = False
        if t % 11 == 0:
            trunc_sim: bool | torch.Tensor = True
            ref_truncs: list[bool | np.ndarray] = [True for _ in range(num_envs)]
        elif t % 7 == 0:
            w_mask_np = rng.random((num_envs,)) < 0.5
            w_mask_np[t % num_envs] = True
            trunc_sim = torch.from_numpy(w_mask_np)
            ref_truncs = [bool(w_mask_np[e]) for e in range(num_envs)]
        elif t % 5 == 0:
            s_mask_np = rng.random((num_envs, cfg.num_snakes)) < 0.35
            trunc_sim = torch.from_numpy(s_mask_np)
            ref_truncs = [s_mask_np[e] for e in range(num_envs)]
        elif t % 13 == 0:
            use_post_flag = True
            post_np = rng.random((num_envs, cfg.num_snakes)) < 0.45
            trunc_sim = False
            ref_truncs = [post_np[e] for e in range(num_envs)]
        else:
            trunc_sim = False
            ref_truncs = [False for _ in range(num_envs)]

        info = sim.step(act, truncate=trunc_sim)
        if use_post_flag:
            post_t = torch.from_numpy(np.stack(ref_truncs, axis=0))
            info.flag_truncation(post_t)
        ref_infos = [refs[e].step(act_np[e], truncate=ref_truncs[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="truncate-masks")

        if sim_mps is not None:
            trunc_mps = trunc_sim.to("mps") if isinstance(trunc_sim, torch.Tensor) else trunc_sim
            info_m = sim_mps.step(act.to("mps"), truncate=trunc_mps)
            if use_post_flag:
                info_m.flag_truncation(torch.from_numpy(np.stack(ref_truncs, axis=0)).to("mps"))
            for k in info.__dataclass_fields__:
                assert torch.equal(getattr(info, k), getattr(info_m, k).cpu())

        if t % 11 == 0 or t % 7 == 0 or t % 5 == 0 or use_post_flag:
            total_rollout_truncs += int(info.truncated.sum().item())

        if t in (1, 15, 35, 40, 41, 65, 90):
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask,
                label="truncate-masks",
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=learner_mask.to("mps"),
                    label="truncate-masks-mps",
                )

    assert total_rollout_truncs >= 100


def test_parity_forced_rebases_and_repeated_reset_worlds_600_ticks_vs_reference() -> None:
    """Parity vs ReferenceWorld across forced rebases and repeated reset_worlds (600 ticks).

    Exercises:
      - Low ``auto_rebase_threshold=96`` triggering repeated automatic rebases during
        ``step()`` and inside ``reset_worlds()``, plus explicit ``rebase_subticks()`` calls;
      - 8 ``reset_worlds`` calls across 600 ticks (single-world, subset, alternating, all-world,
        and empty masks; ``seeds=None``, ``(E,)`` seeds, and ``(m_envs,)`` seeds) while un-reset
        worlds continue running across multiple rebases;
      - Heavy boosting (actions 5..9), fractional boost shedding/drops, kills, deaths, respawns,
        and ``truncate=`` masks, verifying full game state, ``StepInfo``, rings, disjoint surviving
        discs, and ``build_obs`` vs ``build_obs_reference``.
    """
    cfg = GameV3Config(
        arena_radius=105.0,
        num_snakes=18,
        max_length=48,
        respawn_delay=6,
        ambient_food_density=0.022,
    )
    num_envs = 8
    num_ticks = 600
    init_seeds = [8200 + e * 13 for e in range(num_envs)]
    sim = GameV3Sim(
        cfg, num_envs=num_envs, device="cpu", seeds=init_seeds, auto_rebase_threshold=96
    )
    refs = [ReferenceWorld(cfg, seed=s) for s in init_seeds]
    sim_mps = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=init_seeds, auto_rebase_threshold=96)
        if torch.backends.mps.is_available()
        else None
    )

    rng = np.random.default_rng(20261112)
    learner_mask = torch.tensor(
        [[(e + j) % 3 == 0 for j in range(cfg.num_snakes)] for e in range(num_envs)],
        dtype=torch.bool,
    )
    assert_full_state_and_obs_parity(
        sim, refs, 0, check_rings=True, check_obs=True, learner_mask=learner_mask, label="rebase-rw"
    )

    reset_schedule: dict[int, tuple[list[int], str]] = {
        60: ([0], "subset_seeds"),
        135: ([1, 3, 5, 7], "full_seeds"),
        210: ([], "none_seeds"),  # empty world_mask no-op check
        285: ([2, 4, 6], "subset_seeds"),
        360: ([0, 3, 7], "none_seeds"),  # re-seed from current sim.seeds
        435: (list(range(num_envs)), "full_seeds"),  # all-worlds reset_worlds
        510: ([1, 2, 5, 6], "force_reset_rebase"),  # forces auto-rebase inside reset_worlds
        560: ([4], "subset_seeds"),
    }

    rebase_events = 0
    reset_calls = 0
    total_deaths = 0
    total_kills = 0

    for t in range(1, num_ticks + 1):
        # Mix food-seeking with heavy boost actions (5..9) whenever length > min_boost_length.
        if t % 3 == 0:
            act = food_seeking_actions(sim).clone()
            can_boost = sim.length > cfg.min_boost_length
            boost_draw = torch.from_numpy(
                rng.integers(5, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
            boost_slots = torch.from_numpy(rng.random((num_envs, cfg.num_snakes)) < 0.65)
            act = torch.where(can_boost & boost_slots, boost_draw, act)
        elif t % 3 == 1:
            act = random_safe_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            )
        act_np = act.numpy()

        if t % 80 == 0:
            s_mask_np = rng.random((num_envs, cfg.num_snakes)) < 0.4
            trunc_arg: bool | torch.Tensor = torch.from_numpy(s_mask_np)
            ref_truncs: list[bool | np.ndarray] = [s_mask_np[e] for e in range(num_envs)]
        elif t % 45 == 0:
            w_mask_np = np.array([(e + t) % 2 == 0 for e in range(num_envs)], dtype=bool)
            trunc_arg = torch.from_numpy(w_mask_np)
            ref_truncs = [bool(w_mask_np[e]) for e in range(num_envs)]
        else:
            trunc_arg = False
            ref_truncs = [False for _ in range(num_envs)]

        off_before = sim._subtick_offset
        info = sim.step(act, truncate=trunc_arg)
        if sim._subtick_offset > off_before:
            rebase_events += 1
        ref_infos = [refs[e].step(act_np[e], truncate=ref_truncs[e]) for e in range(num_envs)]
        assert_step_info_parity(info, ref_infos, t, label="rebase-rw")

        if sim_mps is not None:
            t_mps = trunc_arg.to("mps") if isinstance(trunc_arg, torch.Tensor) else trunc_arg
            info_m = sim_mps.step(act.to("mps"), truncate=t_mps)
            assert sim_mps._subtick_offset == sim._subtick_offset
            for k in info.__dataclass_fields__:
                assert torch.equal(getattr(info, k), getattr(info_m, k).cpu())

        total_deaths += int(info.died.sum().item())
        total_kills += int(info.kill_credit.sum().item())

        # Also run explicit manual rebase_subticks() mid-flight at ticks 170 and 475.
        if t in (170, 475):
            sh = sim.rebase_subticks()
            if sh > 0:
                rebase_events += 1
            if sim_mps is not None:
                sh_m = sim_mps.rebase_subticks()
                assert sh_m == sh
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask,
                label="manual-rebase",
            )

        if t in reset_schedule:
            sel_envs, seed_mode = reset_schedule[t]
            wmask = torch.zeros((num_envs,), dtype=torch.bool)
            for e_idx in sel_envs:
                wmask[e_idx] = True

            off_pre_reset = sim._subtick_offset
            if seed_mode == "none_seeds":
                sim.reset_worlds(wmask)
                if sim_mps is not None:
                    sim_mps.reset_worlds(wmask.to("mps"))
                for e_idx in sel_envs:
                    refs[e_idx] = ReferenceWorld(cfg, seed=int(sim.seeds[e_idx].item()))
            elif seed_mode == "full_seeds":
                full_seeds = [50_000 + t * 29 + e for e in range(num_envs)]
                sim.reset_worlds(wmask, seeds=full_seeds)
                if sim_mps is not None:
                    sim_mps.reset_worlds(wmask.to("mps"), seeds=full_seeds)
                for e_idx in sel_envs:
                    refs[e_idx] = ReferenceWorld(cfg, seed=full_seeds[e_idx])
            elif seed_mode == "subset_seeds":
                sub_seeds = [70_000 + t * 37 + k for k in range(len(sel_envs))]
                sim.reset_worlds(wmask, seeds=sub_seeds)
                if sim_mps is not None:
                    sim_mps.reset_worlds(wmask.to("mps"), seeds=sub_seeds)
                for k, e_idx in enumerate(sel_envs):
                    refs[e_idx] = ReferenceWorld(cfg, seed=sub_seeds[k])
            elif seed_mode == "force_reset_rebase":
                sub_seeds = [90_000 + t * 41 + k for k in range(len(sel_envs))]
                saved_th = sim.auto_rebase_threshold
                try:
                    init_st = max(1, sim.max_pts * sim.tick + 1 - sim._subtick_offset)
                    forced_th = max(4, init_st)
                    sim.auto_rebase_threshold = forced_th
                    if sim_mps is not None:
                        sim_mps.auto_rebase_threshold = forced_th
                    sim.reset_worlds(wmask, seeds=sub_seeds)
                    if sim_mps is not None:
                        sim_mps.reset_worlds(wmask.to("mps"), seeds=sub_seeds)
                finally:
                    sim.auto_rebase_threshold = saved_th
                    if sim_mps is not None:
                        sim_mps.auto_rebase_threshold = saved_th
                assert sim._subtick_offset > off_pre_reset
                for k, e_idx in enumerate(sel_envs):
                    refs[e_idx] = ReferenceWorld(cfg, seed=sub_seeds[k])

            if sim._subtick_offset > off_pre_reset:
                rebase_events += 1
            reset_calls += 1
            sim.assert_surviving_discs_disjoint()
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask,
                label=f"post-reset-{t}",
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=learner_mask.to("mps"),
                    label=f"post-reset-mps-{t}",
                )

        if t % 50 == 0 or t == num_ticks:
            sim.assert_surviving_discs_disjoint()
            assert_full_state_and_obs_parity(
                sim,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask,
                label="rebase-rw",
            )
            if sim_mps is not None:
                assert_full_state_and_obs_parity(
                    sim_mps,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=learner_mask.to("mps"),
                    label="rebase-rw-mps",
                )

    assert reset_calls == 8
    assert rebase_events >= 10
    assert sim._subtick_offset >= 500
    assert total_deaths >= 150
    assert total_kills >= 25
    assert int(sim.total_boost_decay_sink.sum().item()) > 0


def _sample_random_config(rng: np.random.Generator) -> GameV3Config:
    """Draw a valid small-arena GameV3Config across diverse rule parameters."""
    spawn_wall_margin = float(rng.integers(12, 24))
    arena_radius = float(rng.integers(int(spawn_wall_margin) + 44, 145))
    num_snakes = int(rng.integers(6, 49))
    start_length = int(rng.integers(8, 17))
    min_boost_length = int(start_length + rng.integers(0, 6))
    max_length = int(rng.integers(max(min_boost_length + 14, 34), 96))
    return GameV3Config(
        arena_radius=arena_radius,
        num_snakes=num_snakes,
        start_length=start_length,
        max_length=max_length,
        turn_rate=float(rng.uniform(0.08, 0.22)),
        turn_exponent=float(rng.uniform(0.12, 0.36)),
        min_boost_length=min_boost_length,
        boost_shed_ticks=int(rng.integers(2, 7)),
        boost_shed_mass=int(rng.integers(1, 3)),
        boost_drop_fraction=float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
        food_pickup_mult=float(rng.choice([1.25, 1.5, 1.75, 2.0])),
        corpse_mass_fraction=float(rng.choice([0.35, 0.6, 0.7, 0.9])),
        corpse_max_pellets=int(rng.integers(6, 28)),
        ambient_food_density=float(rng.uniform(0.004, 0.022)),
        ambient_pellet_value=int(rng.integers(1, 4)),
        ambient_spawn_budget_per_tick=int(rng.integers(8, 48)),
        respawn_delay=int(rng.integers(2, 8)),
        respawn_candidates=int(rng.integers(4, 10)),
        respawn_clearance_cells=int(rng.integers(3, 6)),
        spawn_wall_margin=spawn_wall_margin,
        max_life_ticks=int(rng.choice([0, 0, 35, 60])),
    )


def test_parity_randomized_configs_and_action_sequences() -> None:
    """Bitwise parity across randomized GameV3Config instances and random action sequences."""
    cfg_rng = np.random.default_rng(20261111)
    configs = [_sample_random_config(cfg_rng) for _ in range(6)]
    num_envs = 4
    num_ticks = 100
    has_mps = torch.backends.mps.is_available()

    for cfg_idx, cfg in enumerate(configs):
        seeds = [81000 + cfg_idx * 100 + e for e in range(num_envs)]
        sim_cpu = GameV3Sim(
            cfg,
            num_envs=num_envs,
            device="cpu",
            seeds=seeds,
            auto_rebase_threshold=96,
        )
        sim_gpu = (
            GameV3Sim(
                cfg,
                num_envs=num_envs,
                device="mps",
                seeds=seeds,
                auto_rebase_threshold=96,
            )
            if has_mps
            else None
        )
        refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

        act_rng = np.random.default_rng(99000 + cfg_idx)
        learner_mask = torch.from_numpy(act_rng.random((num_envs, cfg.num_snakes)) < 0.35)
        if not bool(learner_mask.any().item()):
            learner_mask[0, 0] = True

        label_cpu = f"rand-cfg-{cfg_idx}-cpu"
        label_gpu = f"rand-cfg-{cfg_idx}-mps"
        assert_full_state_and_obs_parity(
            sim_cpu,
            refs,
            0,
            check_rings=True,
            check_obs=True,
            learner_mask=learner_mask,
            label=label_cpu,
        )
        if sim_gpu is not None:
            assert_full_state_and_obs_parity(
                sim_gpu,
                refs,
                0,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask.to("mps"),
                label=label_gpu,
            )

        for t in range(1, num_ticks + 1):
            if t % 7 == 0:
                act_cpu = food_seeking_actions(sim_cpu)
                act_np = act_cpu.numpy()
            else:
                act_np = act_rng.integers(
                    0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64
                )
                act_cpu = torch.from_numpy(act_np)

            if t % 29 == 0:
                w_trunc = act_rng.random((num_envs,)) < 0.5
                trunc_cpu: bool | torch.Tensor = torch.from_numpy(w_trunc)
                ref_truncs: list[bool] = [bool(w_trunc[e]) for e in range(num_envs)]
            else:
                trunc_cpu = False
                ref_truncs = [False for _ in range(num_envs)]

            info_cpu = sim_cpu.step(act_cpu, truncate=trunc_cpu)
            ref_infos = [refs[e].step(act_np[e], truncate=ref_truncs[e]) for e in range(num_envs)]
            assert_step_info_parity(info_cpu, ref_infos, t, label=label_cpu)

            if sim_gpu is not None:
                trunc_gpu = (
                    trunc_cpu.to("mps") if isinstance(trunc_cpu, torch.Tensor) else trunc_cpu
                )
                info_gpu = sim_gpu.step(act_cpu.to("mps"), truncate=trunc_gpu)
                assert_step_info_parity(info_gpu, ref_infos, t, label=label_gpu)

            if t in (50, num_ticks):
                assert_full_state_and_obs_parity(
                    sim_cpu,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=learner_mask,
                    label=label_cpu,
                )
                if sim_gpu is not None:
                    assert_full_state_and_obs_parity(
                        sim_gpu,
                        refs,
                        t,
                        check_rings=True,
                        check_obs=True,
                        learner_mask=learner_mask.to("mps"),
                        label=label_gpu,
                    )


def test_parity_over_256_simultaneous_respawns_vs_reference() -> None:
    """Bitwise parity vs ReferenceWorld when >256 snakes become ready to respawn on the same tick."""
    cfg = GameV3Config(
        arena_radius=72.0,
        num_snakes=24,
        respawn_delay=2,
        max_life_ticks=1,
        ambient_food_density=0.004,
    )
    num_envs = 256
    seeds = [91000 + e for e in range(num_envs)]
    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    sim_gpu = (
        GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
        if torch.backends.mps.is_available()
        else None
    )
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    learner_mask = torch.zeros((num_envs, cfg.num_snakes), dtype=torch.bool)
    for k in range(24):
        learner_mask[(k * 11) % num_envs, k % cfg.num_snakes] = True

    rng = np.random.default_rng(256024)
    mass_respawn_ticks = 0
    conflict_deferral_seen = False

    for t in range(1, 7):
        act_np = rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
        act_cpu = torch.from_numpy(act_np)
        info_cpu = sim_cpu.step(act_cpu)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]
        assert_step_info_parity(info_cpu, ref_infos, t, label="respawn-256x24-cpu")

        if sim_gpu is not None:
            info_gpu = sim_gpu.step(act_cpu.to("mps"))
            assert_step_info_parity(info_gpu, ref_infos, t, label="respawn-256x24-mps")

        spawned_this_tick = int(info_cpu.life_start.sum().item())
        if spawned_this_tick > 256:
            mass_respawn_ticks += 1
        queued_at_zero = int(((~sim_cpu.alive) & (sim_cpu.respawn_timer == 0)).sum().item())
        if spawned_this_tick > 256 and queued_at_zero > 0:
            conflict_deferral_seen = True

        if t in (2, 3, 6):
            assert_full_state_and_obs_parity(
                sim_cpu,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=learner_mask,
                label="respawn-256x24-cpu",
            )
            if sim_gpu is not None:
                assert_full_state_and_obs_parity(
                    sim_gpu,
                    refs,
                    t,
                    check_rings=True,
                    check_obs=True,
                    learner_mask=learner_mask.to("mps"),
                    label="respawn-256x24-mps",
                )

    assert mass_respawn_ticks >= 3, f"Expected >=3 waves of >256 respawns, got {mass_respawn_ticks}"
    assert conflict_deferral_seen, "Expected same-tick clearance conflicts among >256 ready slots"
