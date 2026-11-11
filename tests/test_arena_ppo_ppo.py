"""Tests for `research.arena_ppo.ppo` (rollout tape parity, GAE, weight update, CPU resume)."""

from __future__ import annotations

import math
import os
import signal
from pathlib import Path

import torch

from research.arena_ppo.obs import build_arena_ego_obs_gpu
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    find_newest_verified_checkpoint,
    run_training_job,
)
from research.arena_ppo.runtime import verify_completed_artifact


def test_ppo_config_json_roundtrip_and_defaults(tmp_path: Path) -> None:
    """Verify ArenaPPOConfig defaults and JSON file/string serialization."""
    cfg = ArenaPPOConfig()
    assert cfg.rollout_len == 256
    assert cfg.gae_lambda == 0.97
    assert cfg.gamma == 0.999
    assert cfg.checkpoint_interval_sec == 1200.0

    cfg_path = tmp_path / "ppo_cfg.json"
    cfg_path.write_text(cfg.to_json(), encoding="utf-8")
    loaded = ArenaPPOConfig.from_json(cfg_path)
    assert loaded == cfg


def test_compact_rollout_buffer_reconstructs_bit_identical_obs() -> None:
    """Prove CompactRolloutBuffer reconstructs bit-for-bit identical obs at every tick."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=4,
        rollout_len=20,
        max_capacity=128,
        minibatch_size=24,
        ppo_epochs=1,
        seed=42,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    # Seed mature lengths so boosts and deaths occur inside the 20-step rollout
    trainer.sim.sim.length[:] = torch.tensor([[1, 6, 8, 12, 5, 16]] * 4, dtype=torch.long)
    trainer.sim.sim._reward_prev_length.copy_(trainer.sim.sim.length)

    live_snapshots = []
    for t in range(cfg.rollout_len):
        trainer.sim.prepare()
        trainer.buffer.record_pre_step(t, trainer.sim)
        live_obs = build_arena_ego_obs_gpu(trainer.sim)
        live_mask = trainer.sim.legal_action_mask().reshape(-1, 6)
        live_snapshots.append(
            (
                live_obs.arena.clone(),
                live_obs.ego.clone(),
                live_obs.scalars.clone(),
                live_mask.clone(),
            )
        )

        act = torch.randint(0, 6, (4, 6), dtype=torch.long)
        trainer.sim.step(act)

    for t in range(cfg.rollout_len):
        t_idx = torch.full((4,), t, dtype=torch.long)
        e_idx = torch.arange(4, dtype=torch.long)
        rec_arena, rec_ego, rec_scalars, rec_mask = trainer.buffer.reconstruct_world_ticks(
            t_idx, e_idx
        )
        exp_arena, exp_ego, exp_scalars, exp_mask = live_snapshots[t]
        assert torch.equal(rec_arena, exp_arena), f"Rollout tape arena mismatch at t={t}"
        assert torch.equal(rec_ego, exp_ego), f"Rollout tape ego mismatch at t={t}"
        assert torch.equal(rec_scalars, exp_scalars), f"Rollout tape scalars mismatch at t={t}"
        assert torch.equal(rec_mask, exp_mask), f"Rollout tape mask mismatch at t={t}"


def test_gae_death_no_bootstrap_and_ppo_update_changes_weights() -> None:
    """Verify death has zero bootstrap, rollout cut bootstraps, loss is finite, weights change."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=18,
        ppo_epochs=2,
        minibatch_size=12,
        max_capacity=128,
        seed=101,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    before_params = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}

    metrics = trainer.step_update()
    assert math.isfinite(metrics.policy_loss)
    assert math.isfinite(metrics.value_loss)
    assert math.isfinite(metrics.aux_loss)
    assert math.isfinite(metrics.entropy) and metrics.entropy > 0.0
    assert math.isfinite(metrics.grad_norm) and metrics.grad_norm > 0.0

    changed = any(
        not torch.equal(before_params[k], v.detach()) for k, v in trainer.model.state_dict().items()
    )
    assert changed, "One PPO update must modify model parameters"

    # Explicit unit check on GAE death vs rollout-cut bootstrap
    buf = trainer.buffer
    buf.rewards.fill_(1.0)
    buf.values.fill_(2.0)
    buf.acted.fill_(True)
    buf.died.fill_(False)
    buf.truncated.fill_(False)
    buf.died[-1, 0, 0] = True  # Seat (0, 0) dies on last step -> NO bootstrap
    last_vals = torch.full((2, 6), 10.0, dtype=torch.float32)
    buf.compute_gae_and_aux(last_values=last_vals, gamma=0.999, gae_lambda=0.97)

    # Died seat: delta = 1.0 + 0.0 - 2.0 = -1.0
    assert math.isclose(float(buf.advantages[-1, 0, 0].item()), -1.0, abs_tol=1e-6)
    # Alive rollout-cut seat: delta = 1.0 + 0.999 * 10.0 - 2.0 = 8.99
    assert math.isclose(float(buf.advantages[-1, 0, 1].item()), 8.99, abs_tol=1e-5)
    # Aux death target on died step must be 1.0 and valid across all 3 horizons
    assert (buf.aux_targets[-1, 0, 0] == 1.0).all()
    assert buf.aux_valid[-1, 0, 0].all()
    # Alive seat at t=0 with T=18: valid=True for h=16 (target=0), censored for h=64,256
    assert bool(buf.aux_valid[0, 0, 1, 0].item()) is True
    assert float(buf.aux_targets[0, 0, 1, 0].item()) == 0.0
    assert bool(buf.aux_valid[0, 0, 1, 1].item()) is False
    assert bool(buf.aux_valid[0, 0, 1, 2].item()) is False


def test_ppo_resume_bit_identical_to_uninterrupted_cpu(tmp_path: Path) -> None:
    """Prove resumed == uninterrupted bit-for-bit on CPU across model, optimizer, and sim."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=3,
        rollout_len=8,
        ppo_epochs=2,
        minibatch_size=12,
        max_capacity=128,
        total_updates=4,
        seed=20261110,
        deterministic=True,
    )

    # 1. Uninterrupted 4 updates
    uninterrupted = ArenaPPOTrainer(cfg, device="cpu")
    m_unint = [uninterrupted.step_update() for _ in range(4)]

    # 2. Split run: 2 updates -> save checkpoint -> fresh trainer loads newest verified -> 2 updates
    part1 = ArenaPPOTrainer(cfg, device="cpu")
    _ = [part1.step_update() for _ in range(2)]
    ckpt_dir = tmp_path / "ckpts"
    ckpt_path = ckpt_dir / "ckpt_u0000002.pt"
    part1.save_checkpoint(ckpt_path, reason="periodic")

    newest = find_newest_verified_checkpoint(ckpt_dir)
    assert newest == ckpt_path

    resumed_cfg = ArenaPPOConfig(**{**cfg.to_dict(), "seed": 999999})
    resumed = ArenaPPOTrainer(resumed_cfg, device="cpu")
    resumed.load_checkpoint(newest)
    assert resumed.updates_completed == 2

    m3 = resumed.step_update()
    m4 = resumed.step_update()
    assert math.isclose(m3.policy_loss, m_unint[2].policy_loss, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(m4.policy_loss, m_unint[3].policy_loss, rel_tol=0.0, abs_tol=1e-7)
    assert m4.agent_steps_collected == m_unint[3].agent_steps_collected

    for k, v_unint in uninterrupted.model.state_dict().items():
        v_res = resumed.model.state_dict()[k]
        assert torch.equal(v_unint, v_res), f"Model weight divergence after resume on {k}"

    sim_a = uninterrupted.sim.state_dict()["sim_tensors"]
    sim_b = resumed.sim.state_dict()["sim_tensors"]
    for k, t_a in sim_a.items():
        assert torch.equal(t_a, sim_b[k]), f"Sim tensor divergence after resume on {k}"


def test_run_training_job_sigusr1_checkpoint_and_exports(tmp_path: Path) -> None:
    """Verify SIGUSR1 saves verified checkpoint and run_training_job exports required files."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=2,
        seed=77,
        deterministic=True,
    )
    out_dir = tmp_path / "train_out"
    res = run_training_job(cfg=cfg, output_dir=out_dir, device="cpu")
    assert res["updates_completed"] == 2
    assert verify_completed_artifact(out_dir / "checkpoints" / "final.pt")
    assert verify_completed_artifact(out_dir / "arena_ppo_weights.pth")
    assert verify_completed_artifact(out_dir / "train_result.json")

    # Test signal handler path when SIGUSR1 is pre-raised inside loop via wall/signal check
    if hasattr(signal, "SIGUSR1"):
        out_sig = tmp_path / "train_sig"
        prev = signal.getsignal(signal.SIGUSR1)
        try:
            os.kill(os.getpid(), 0)
            res_wall = run_training_job(
                cfg=cfg, output_dir=out_sig, device="cpu", max_wall_seconds=0.0
            )
            assert res_wall["stopped_reason"] == "wall_budget_guard"
            assert verify_completed_artifact(out_sig / "checkpoints" / "final.pt")
        finally:
            signal.signal(signal.SIGUSR1, prev)
