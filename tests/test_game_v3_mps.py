"""Exact bitwise parity tests on Apple Silicon MPS vs CPU & ReferenceWorld, plus bench smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.game_v3.bench import STAGE_ORDER, run_benchmark
from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.reference import ReferenceWorld
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim
from tests.test_game_v3_parity import assert_full_state_and_obs_parity, assert_step_info_parity

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="PyTorch MPS backend is not available on this machine",
)


def _assert_mps_equals_cpu_bitwise(sim_mps: GameV3Sim, sim_cpu: GameV3Sim, tick: int) -> None:
    """Assert every state and pyramid tensor on MPS is bitwise-identical to CPU."""
    tensor_attrs = (
        "world_ticks",
        "head_pos",
        "theta",
        "alive",
        "boosting",
        "length",
        "prev_length",
        "seg_count",
        "head_ptr",
        "oldest_active_subtick",
        "dist_accum_fp",
        "boost_ticks",
        "boost_drop_fp_accum",
        "respawn_timer",
        "life_age",
        "life_start",
        "truncated_pending",
        "occ_hi",
        "occ_lo",
        "occ_subtick_4x",
        "occ_subtick_16x",
        "ambient_grid",
        "ambient_grid_4x",
        "ambient_grid_16x",
        "corpse_grid",
        "corpse_grid_4x",
        "corpse_grid_16x",
        "ambient_count",
        "total_spawned_snake_mass",
        "total_spawned_ambient_mass",
        "total_corpse_decay_sink",
        "total_boost_decay_sink",
        "total_length_cap_sink",
        "total_truncation_sink",
    )
    for attr in tensor_attrs:
        gm = getattr(sim_mps, attr).detach().cpu()
        gc = getattr(sim_cpu, attr).detach().cpu()
        assert torch.equal(gm, gc), f"MPS vs CPU tensor {attr} diverged at tick={tick}"


def test_fast_mps_vs_cpu_and_reference_bitwise_parity() -> None:
    """Fast MPS smoke: 8 worlds x 120 ticks at N=24 bitwise-exact vs CPU and ReferenceWorld."""
    cfg = GameV3Config(
        arena_radius=120.0,
        num_snakes=24,
        respawn_delay=8,
        max_life_ticks=55,
        ambient_food_density=0.015,
    )
    num_envs = 8
    seeds = [3000 + e for e in range(num_envs)]
    sim_mps = GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    rng = np.random.default_rng(271828)
    mask_cpu = torch.from_numpy(rng.random((num_envs, cfg.num_snakes)) < (8.0 / 24.0))
    mask_mps = mask_cpu.to("mps")

    _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, 0)
    assert_full_state_and_obs_parity(
        sim_mps, refs, 0, check_rings=True, check_obs=True, learner_mask=mask_mps, label="mps"
    )

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for t in range(1, 121):
        mode = t % 3
        if mode == 0:
            act_mps = food_seeking_actions(sim_mps)
            act_cpu = food_seeking_actions(sim_cpu)
            assert torch.equal(act_mps.cpu(), act_cpu)
        elif mode == 1:
            act_mps = random_safe_actions(sim_mps)
            act_cpu = random_safe_actions(sim_cpu)
            assert torch.equal(act_mps.cpu(), act_cpu)
        else:
            act_np = rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            act_cpu = torch.from_numpy(act_np)
            act_mps = act_cpu.to("mps")

        act_np = act_cpu.numpy()
        np.add.at(action_counts, act_np.ravel(), 1)

        info_mps = sim_mps.step(act_mps)
        info_cpu = sim_cpu.step(act_cpu)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]

        assert_step_info_parity(info_mps, ref_infos, t, label="mps")
        for k in info_cpu.__dataclass_fields__:
            assert torch.equal(getattr(info_mps, k).cpu(), getattr(info_cpu, k))

        if t in (1, 20, 55, 56, 90, 120):
            _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, t)
            assert_full_state_and_obs_parity(
                sim_mps,
                refs,
                t,
                check_rings=True,
                check_obs=True,
                learner_mask=mask_mps,
                label="mps",
            )

    assert np.all(action_counts > 0)


@pytest.mark.slow
def test_32_worlds_x_1000_ticks_n24_exact_bitwise_parity_mps() -> None:
    """Contract (Rule 8): 32 worlds x 1,000 ticks at N=24 on MPS vs CPU & ReferenceWorld."""
    cfg = GameV3Config(
        arena_radius=180.0,
        num_snakes=24,
        respawn_delay=15,
        max_life_ticks=350,
        ambient_food_density=0.006,
    )
    num_envs = 32
    num_ticks = 1000
    seeds = [20_000 + e * 19 for e in range(num_envs)]

    sim_mps = GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    rng = np.random.default_rng(987654321)
    mask_cpu = torch.tensor(
        [[j < 8 for j in range(cfg.num_snakes)] for _ in range(num_envs)], dtype=torch.bool
    )
    mask_mps = mask_cpu.to("mps")
    idx_mps = LearnerIndex(sim_mps, mask_mps)
    out_sub_mps = allocate_obs_batch(sim_mps, idx_mps)

    _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, 0)
    assert_full_state_and_obs_parity(
        sim_mps,
        refs,
        0,
        check_rings=True,
        check_obs=True,
        learner_mask=mask_mps,
        label="mps-32x1000",
    )

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_deaths = 0
    total_kills = 0
    total_truncs = 0

    for t in range(1, num_ticks + 1):
        mode = t % 4
        if mode == 0:
            act_mps = food_seeking_actions(sim_mps)
            act_cpu = food_seeking_actions(sim_cpu)
            assert torch.equal(act_mps.cpu(), act_cpu)
        elif mode == 1:
            act_mps = random_safe_actions(sim_mps)
            act_cpu = random_safe_actions(sim_cpu)
            assert torch.equal(act_mps.cpu(), act_cpu)
        elif mode == 2:
            act_np = rng.integers(0, NUM_ACTIONS, size=(num_envs, cfg.num_snakes), dtype=np.int64)
            act_cpu = torch.from_numpy(act_np)
            act_mps = act_cpu.to("mps")
        else:
            act_mps = food_seeking_actions(sim_mps).clone()
            act_cpu = food_seeking_actions(sim_cpu).clone()
            rand_tail = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, 8), dtype=np.int64)
            )
            act_cpu[:, 16:] = rand_tail
            act_mps[:, 16:] = rand_tail.to("mps")
            assert torch.equal(act_mps.cpu(), act_cpu)

        act_np = act_cpu.numpy()
        np.add.at(action_counts, act_np.ravel(), 1)

        info_mps = sim_mps.step(act_mps)
        info_cpu = sim_cpu.step(act_cpu)
        ref_infos = [refs[e].step(act_np[e]) for e in range(num_envs)]

        assert_step_info_parity(info_mps, ref_infos, t, label="mps-32x1000")
        for k in info_cpu.__dataclass_fields__:
            assert torch.equal(getattr(info_mps, k).cpu(), getattr(info_cpu, k))

        total_deaths += int(info_cpu.died.sum().item())
        total_kills += int(info_cpu.kill_credit.sum().item())
        total_truncs += int(info_cpu.truncated.sum().item())

        if t % 100 == 0 or t in (1, 350, 351, num_ticks):
            _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, t)
            assert_full_state_and_obs_parity(
                sim_mps,
                refs,
                t,
                check_rings=(t in (350, num_ticks)),
                check_obs=True,
                learner_mask=mask_mps,
                label="mps-32x1000",
            )
            obs_sub = build_obs(sim_mps, idx_mps, out=out_sub_mps)
            obs_cpu = build_obs(sim_cpu)
            assert torch.equal(obs_sub.raster_s1.cpu(), obs_cpu.raster_s1[mask_cpu])

    assert np.all(action_counts >= 10_000), f"Expected all 10 actions heavily used: {action_counts}"
    assert total_deaths >= 200
    assert total_kills >= 50
    assert total_truncs >= 50


def test_mps_benchmark_entrypoint_writes_valid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real benchmark entrypoint runs on MPS, reports all-learner & 8-learner, and writes JSON."""
    out_json = tmp_path / "bench_mps_smoke.json"
    summary = run_benchmark(
        device="mps",
        envs_list=(16,),
        num_snakes=24,
        learner_slots=8,
        arena_radius=300.0,
        warmup_steps=2,
        bench_steps=5,
        stage_steps=3,
        json_output=out_json,
    )
    captured = capsys.readouterr().out
    assert "START game_v3_bench" in captured
    assert "END game_v3_bench" in captured
    assert out_json.is_file()

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["schema"] == "game-v3-bench/v2"
    assert loaded["device"] in ("mps", "mps:0")
    assert len(loaded["cases"]) == 1
    case = loaded["cases"][0]
    assert case["agent_steps_per_sec"] > 0.0
    assert case["learner_masked_agent_steps_per_sec"] > 0.0
    assert case["learner_masked_learner_steps_per_sec"] > 0.0
    assert case["memory"]["peak_memory_mib"] > 0.0
    assert case["memory"]["footprint_mib_per_world"] > 0.0
    assert set(case["stage_ms_per_tick"].keys()) == set(STAGE_ORDER)
    assert summary == loaded


def test_mps_multi_block_env_chunking_parity_past_256_envs() -> None:
    """Verify 256-env block slicing in stamp & obs matches CPU bit-for-bit at E=300 (>256)."""
    cfg = GameV3Config(arena_radius=60.0, num_snakes=24, ambient_food_density=0.015)
    num_envs = 300
    seeds = [9000 + e for e in range(num_envs)]
    sim_mps = GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=seeds)
    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)

    # Sparse mask spanning env 0, env 255, env 256, and env 299 across block boundaries.
    mask_cpu = torch.zeros((num_envs, cfg.num_snakes), dtype=torch.bool)
    for e_idx in (0, 127, 255, 256, 299):
        mask_cpu[e_idx, :8] = True
    mask_mps = mask_cpu.to("mps")

    for _ in range(12):
        act_cpu = food_seeking_actions(sim_cpu)
        act_mps = food_seeking_actions(sim_mps)
        assert torch.equal(act_mps.cpu(), act_cpu)
        sim_cpu.step(act_cpu)
        sim_mps.step(act_mps)

    _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, 12)
    obs_all_mps = build_obs(sim_mps)
    obs_all_cpu = build_obs(sim_cpu)
    assert torch.equal(obs_all_mps.raster_s1.cpu(), obs_all_cpu.raster_s1)
    assert torch.equal(obs_all_mps.raster_s4.cpu(), obs_all_cpu.raster_s4)
    assert torch.equal(obs_all_mps.raster_s16.cpu(), obs_all_cpu.raster_s16)
    assert torch.equal(
        obs_all_mps.scalars.cpu().view(torch.uint32), obs_all_cpu.scalars.view(torch.uint32)
    )

    obs_sub_mps = build_obs(sim_mps, learner_mask=mask_mps)
    assert torch.equal(obs_sub_mps.raster_s1.cpu(), obs_all_cpu.raster_s1[mask_cpu])
    assert torch.equal(obs_sub_mps.raster_s4.cpu(), obs_all_cpu.raster_s4[mask_cpu])
    assert torch.equal(obs_sub_mps.raster_s16.cpu(), obs_all_cpu.raster_s16[mask_cpu])
    assert torch.equal(
        obs_sub_mps.scalars.cpu().view(torch.uint32),
        obs_all_cpu.scalars[mask_cpu].view(torch.uint32),
    )

    # Verify multi-block occ_subtick_4x returns on sim.device (mps:0), not CPU, and exercise
    # multi-block rebase_subticks + reset_worlds across the 256-env block boundary.
    assert len(sim_mps._occ_4x_blocks) == 2
    occ4_mps = sim_mps.occ_subtick_4x
    assert occ4_mps.device == sim_mps.device
    assert torch.equal(occ4_mps.cpu(), sim_cpu.occ_subtick_4x)

    # Advance stored subticks uniformly by +200 (as after earlier segments have expired) so
    # rebase_subticks performs a real non-zero shift across both 256-env blocks at E=300.
    for s in (sim_mps, sim_cpu):
        s.tick += 200
        s.oldest_active_subtick.add_(200)
        s.ring_subtick[s.ring_subtick > 0] += 200
        s.occ_hi[s.occ_hi >= 0] += 200 << 6
        s.occ_lo[s.occ_lo >= 0] += 200 << 6
        for blk in s._occ_4x_blocks:
            blk[blk >= 0] += 200
        s.occ_subtick_16x[s.occ_subtick_16x >= 0] += 200

    sh_mps = sim_mps.rebase_subticks()
    sh_cpu = sim_cpu.rebase_subticks()
    assert sh_mps == sh_cpu >= 200
    _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, sim_cpu.tick)

    rw_mask_cpu = torch.zeros((num_envs,), dtype=torch.bool)
    for e_idx in (0, 255, 256, 299):
        rw_mask_cpu[e_idx] = True
    rw_seeds = [19_000, 19_255, 19_256, 19_299]
    sim_mps.reset_worlds(rw_mask_cpu.to("mps"), seeds=rw_seeds)
    sim_cpu.reset_worlds(rw_mask_cpu, seeds=rw_seeds)
    _assert_mps_equals_cpu_bitwise(sim_mps, sim_cpu, 12)

    # Also verify state_dict / load_state_dict round-trip across >256-env sharded occ_4x.
    ckpt = sim_mps.state_dict()
    sim_mps_restored = GameV3Sim(cfg, num_envs=num_envs, device="mps", seeds=[0] * num_envs)
    sim_mps_restored.load_state_dict(ckpt)
    _assert_mps_equals_cpu_bitwise(sim_mps_restored, sim_cpu, 12)


def test_mps_large_int64_scalar_q10_div_avoids_mps_abs_float32_cast() -> None:
    """Regression guard: boundary ego Q10 division stays bit-exact past 2^24 on MPS."""
    from research.game_v3.obs import _signed_div_q10_torch

    # Exact case where |num| = 229,277,429,760 > 2^24: MPS Tensor.abs() rounds via float32
    # to 229,277,433,856 and wrongly produces -1024 instead of -1023.
    nums_cpu = torch.tensor(
        [-229_277_429_760, 229_277_429_760, -5_000_000_000_001], dtype=torch.int64
    )
    dens_cpu = torch.tensor([223_903_744, 223_903_744, 5_000_000_000_000], dtype=torch.int64)
    got_cpu = _signed_div_q10_torch(nums_cpu, dens_cpu)
    got_mps = _signed_div_q10_torch(nums_cpu.to("mps"), dens_cpu.to("mps")).cpu()
    assert got_cpu.tolist() == [-1023, 1023, -1]
    assert torch.equal(got_mps, got_cpu)
