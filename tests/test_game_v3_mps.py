"""Exact bitwise parity tests on Apple Silicon MPS vs CPU & ReferenceWorld, plus bench smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.game_v3.bench import STAGE_ORDER, run_benchmark
from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import allocate_obs_batch, build_obs
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
        "head_pos",
        "theta",
        "alive",
        "boosting",
        "length",
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
    out_sub_mps = allocate_obs_batch(sim_mps, learner_mask=mask_mps)

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
            obs_sub = build_obs(sim_mps, learner_mask=mask_mps, out=out_sub_mps)
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
