"""MPS backend execution, discrete-outcome parity vs CPU, and benchmark smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.game_v3.bench import STAGE_ORDER, run_benchmark
from research.game_v3.config import GameV3Config
from research.game_v3.obs import build_obs
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="PyTorch MPS backend is not available on this machine",
)


def measure_mps_vs_cpu_agreement(
    *, num_envs: int = 32, num_snakes: int = 24, num_ticks: int = 500
) -> dict[str, float]:
    """Run paired CPU and MPS simulations and measure discrete agreement and float drift."""
    cfg = GameV3Config(num_snakes=num_snakes)
    seeds = [20_000 + e * 19 for e in range(num_envs)]

    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, seeds=seeds, device="cpu")
    sim_mps = GameV3Sim(cfg, num_envs=num_envs, seeds=seeds, device="mps")

    total_snake_ticks = num_envs * num_snakes * num_ticks
    alive_matches = 0
    length_matches = 0
    died_matches = 0
    killer_matches = 0
    credit_matches = 0
    eaten_matches = 0
    pos_bitwise_diffs = 0
    theta_bitwise_diffs = 0
    max_pos_abs_diff = 0.0

    half_n = num_snakes // 2
    for t in range(1, num_ticks + 1):
        # Compute scripted actions independently on each device to test full closed-loop parity.
        act_cpu = torch.cat(
            (food_seeking_actions(sim_cpu)[:, :half_n], random_safe_actions(sim_cpu)[:, half_n:]),
            dim=1,
        )
        act_mps = torch.cat(
            (food_seeking_actions(sim_mps)[:, :half_n], random_safe_actions(sim_mps)[:, half_n:]),
            dim=1,
        )

        info_cpu = sim_cpu.step(act_cpu)
        info_mps = sim_mps.step(act_mps)

        alive_matches += int((info_cpu.alive == info_mps.alive.cpu()).sum().item())
        length_matches += int((info_cpu.length == info_mps.length.cpu()).sum().item())
        died_matches += int((info_cpu.died == info_mps.died.cpu()).sum().item())
        killer_matches += int((info_cpu.killer_id == info_mps.killer_id.cpu()).sum().item())
        credit_matches += int((info_cpu.kill_credit == info_mps.kill_credit.cpu()).sum().item())
        eaten_matches += int((info_cpu.mass_eaten == info_mps.mass_eaten.cpu()).sum().item())

        pos_cpu_np = sim_cpu.head_pos.numpy()
        pos_mps_np = sim_mps.head_pos.cpu().numpy()
        pos_bitwise_diffs += int(
            np.any(pos_cpu_np.view(np.uint32) != pos_mps_np.view(np.uint32), axis=-1).sum()
        )
        max_pos_abs_diff = max(max_pos_abs_diff, float(np.max(np.abs(pos_cpu_np - pos_mps_np))))

        th_cpu_np = sim_cpu.theta.numpy()
        th_mps_np = sim_mps.theta.cpu().numpy()
        theta_bitwise_diffs += int((th_cpu_np.view(np.uint32) != th_mps_np.view(np.uint32)).sum())

    occ_match_rate = float(
        (sim_cpu.active_occupancy_grid() == sim_mps.active_occupancy_grid().cpu())
        .to(torch.float64)
        .mean()
        .item()
    )
    amb_match_rate = float(
        (sim_cpu.ambient_grid == sim_mps.ambient_grid.cpu()).to(torch.float64).mean().item()
    )
    cor_match_rate = float(
        (sim_cpu.corpse_grid == sim_mps.corpse_grid.cpu()).to(torch.float64).mean().item()
    )

    obs_cpu = build_obs(sim_cpu)
    obs_mps = build_obs(sim_mps)
    assert torch.isfinite(obs_mps.raster_s1).all()
    assert torch.isfinite(obs_mps.raster_s4).all()
    assert torch.isfinite(obs_mps.raster_s16).all()
    assert torch.isfinite(obs_mps.scalars).all()

    obs_s1_max_diff = float((obs_cpu.raster_s1 - obs_mps.raster_s1.cpu()).abs().max().item())
    obs_s4_max_diff = float((obs_cpu.raster_s4 - obs_mps.raster_s4.cpu()).abs().max().item())
    obs_s16_max_diff = float((obs_cpu.raster_s16 - obs_mps.raster_s16.cpu()).abs().max().item())
    obs_scalar_max_diff = float((obs_cpu.scalars - obs_mps.scalars.cpu()).abs().max().item())

    return {
        "total_snake_ticks": float(total_snake_ticks),
        "alive_agreement": alive_matches / total_snake_ticks,
        "length_agreement": length_matches / total_snake_ticks,
        "died_agreement": died_matches / total_snake_ticks,
        "killer_id_agreement": killer_matches / total_snake_ticks,
        "kill_credit_agreement": credit_matches / total_snake_ticks,
        "mass_eaten_agreement": eaten_matches / total_snake_ticks,
        "final_occ_grid_agreement": occ_match_rate,
        "final_ambient_grid_agreement": amb_match_rate,
        "final_corpse_grid_agreement": cor_match_rate,
        "pos_bitwise_divergence_rate": pos_bitwise_diffs / total_snake_ticks,
        "theta_bitwise_divergence_rate": theta_bitwise_diffs / total_snake_ticks,
        "max_pos_abs_diff_units": max_pos_abs_diff,
        "obs_s1_max_abs_diff": obs_s1_max_diff,
        "obs_s4_max_abs_diff": obs_s4_max_diff,
        "obs_s16_max_abs_diff": obs_s16_max_diff,
        "obs_scalar_max_abs_diff": obs_scalar_max_diff,
    }


def test_mps_sim_obs_and_discrete_agreement() -> None:
    """MPS sim + obs executes cleanly and matches CPU discrete outcomes."""
    metrics = measure_mps_vs_cpu_agreement(num_envs=32, num_snakes=24, num_ticks=500)
    print("MPS_VS_CPU_METRICS:", json.dumps(metrics, sort_keys=True))

    assert metrics["alive_agreement"] >= 0.999
    assert metrics["length_agreement"] >= 0.999
    assert metrics["died_agreement"] >= 0.999
    assert metrics["killer_id_agreement"] >= 0.999
    assert metrics["kill_credit_agreement"] >= 0.999
    assert metrics["mass_eaten_agreement"] >= 0.999
    assert metrics["final_occ_grid_agreement"] >= 0.9999
    assert metrics["final_ambient_grid_agreement"] >= 0.9999
    assert metrics["final_corpse_grid_agreement"] >= 0.9999


def test_mps_benchmark_entrypoint_writes_valid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real benchmark entrypoint runs on MPS, prints START/END stamps, and writes JSON."""
    out_json = tmp_path / "bench_mps_smoke.json"
    summary = run_benchmark(
        device="mps",
        envs_list=(16,),
        num_snakes=12,
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
    assert loaded["schema"] == "game-v3-bench/v1"
    assert loaded["device"] in ("mps", "mps:0")
    assert len(loaded["cases"]) == 1
    case = loaded["cases"][0]
    assert case["agent_steps_per_sec"] > 0.0
    assert case["memory"]["peak_memory_mib"] > 0.0
    assert set(case["stage_ms_per_tick"].keys()) == set(STAGE_ORDER)
    assert summary == loaded
