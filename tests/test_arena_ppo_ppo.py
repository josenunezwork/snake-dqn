"""Tests for `research.arena_ppo.ppo` (rollout tape parity, GAE, weight update, CPU resume)."""

from __future__ import annotations

import math
import os
import signal
from pathlib import Path

import pytest
import torch

from research.arena_ppo.obs import build_arena_ego_obs_gpu
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    estimate_checkpoint_bytes,
    estimate_max_export_bytes,
    find_newest_verified_checkpoint,
    run_training_job,
)
from research.arena_ppo.runtime import verify_completed_artifact
from research.arena_ppo.sim import _SIM_STATE_TENSORS, StepInfo, TrainSim


def test_ppo_config_json_roundtrip_and_defaults(tmp_path: Path) -> None:
    """Verify ArenaPPOConfig per-net defaults and JSON file/string serialization."""
    cfg_s = ArenaPPOConfig(net_size="S")
    assert cfg_s.rollout_len == 256
    assert cfg_s.gae_lambda == 0.97
    assert cfg_s.gamma == 0.999
    assert cfg_s.checkpoint_interval_sec == 1200.0
    assert cfg_s.minibatch_size == 1536
    assert cfg_s.actor_chunk_envs == 1024
    assert cfg_s.keep_periodic_checkpoints == 3
    assert cfg_s.max_update_seconds == 1200.0

    cfg_m = ArenaPPOConfig(net_size="M")
    assert cfg_m.minibatch_size == 768
    assert cfg_m.actor_chunk_envs == 512

    cfg_path = tmp_path / "ppo_cfg.json"
    cfg_path.write_text(cfg_s.to_json(), encoding="utf-8")
    loaded = ArenaPPOConfig.from_json(cfg_path)
    assert loaded == cfg_s


def test_compact_rollout_buffer_reconstructs_bit_identical_obs() -> None:
    """Prove CompactRolloutBuffer reconstructs exact obs across deaths, respawns, and burns."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=6,
        rollout_len=36,
        max_capacity=128,
        minibatch_size=36,
        ppo_epochs=1,
        seed=42,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    # Seed mature lengths and force one seat into a wall on tick 0 so death+respawn are guaranteed
    trainer.sim.prepare()
    trainer.sim.sim.length[:] = torch.tensor([[8, 10, 12, 14, 7, 16]] * 6, dtype=torch.long)
    trainer.sim.sim._reward_prev_length.copy_(trainer.sim.sim.length)
    trainer.sim.sim.bodies[0, 0, trainer.sim.sim.head_ptr[0, 0]] = torch.tensor(
        [trainer.sim.grid_w - 1, 10], dtype=torch.long
    )
    trainer.sim.sim.direction[0, 0] = 1

    gen = torch.Generator(device="cpu").manual_seed(20261110)
    live_snapshots = []
    total_deaths = 0
    total_respawns = 0
    total_boost_burns = 0
    was_dead = torch.zeros((6, 6), dtype=torch.bool)

    for t in range(cfg.rollout_len):
        trainer.sim.prepare()
        just_respawned = was_dead & trainer.sim.sim.alive & (trainer.sim.sim.seg_count == 1)
        total_respawns += int(just_respawned.sum().item())

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

        prev_len = trainer.sim.sim.length.clone()
        prev_alive = trainer.sim.sim.alive.clone()
        if t == 0:
            # Seat (0, 0) boosts straight into wall -> guaranteed death; others boost straight
            act = torch.full((6, 6), 4, dtype=torch.long)
        elif t < 6:
            # Sustained boost for >= 3 consecutive ticks triggers boost tail burns
            act = torch.full((6, 6), 4, dtype=torch.long)
        else:
            act = torch.randint(0, 6, (6, 6), generator=gen, dtype=torch.long)
        info = trainer.sim.step(act)
        was_dead = info.died.clone()
        total_deaths += int(info.died.sum().item())
        burned = prev_alive & info.survived & (info.post_length < prev_len)
        total_boost_burns += int(burned.sum().item())

    assert total_deaths > 0, "Rollout tape test must exercise mid-rollout deaths"
    assert total_respawns > 0, "Rollout tape test must exercise mid-rollout respawns (seg_count=1)"
    assert total_boost_burns > 0, "Rollout tape test must exercise boost tail burns"

    for t in range(cfg.rollout_len):
        t_idx = torch.full((6,), t, dtype=torch.long)
        e_idx = torch.arange(6, dtype=torch.long)
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
    assert math.isfinite(metrics.epoch0_approx_kl)

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


def test_epoch0_approx_kl_near_zero_on_single_fp32_minibatch() -> None:
    """Verify MINOR 11 epoch0_approx_kl telemetry is ~0 on first minibatch in fp32."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=6,
        ppo_epochs=2,
        minibatch_size=72,  # >= T * E * S so epoch 0 uses a single minibatch before any step
        actor_chunk_envs=2,
        max_capacity=128,
        seed=303,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    metrics = trainer.step_update()
    assert abs(metrics.epoch0_approx_kl) < 1e-5


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

    # Mutable continuation field (total_updates) is allowed; mutate trainer state before loading
    # to prove load_checkpoint overwrites model, optimizer, sim, and RNG streams completely.
    resumed_cfg = ArenaPPOConfig(**{**cfg.to_dict(), "total_updates": 6})
    resumed = ArenaPPOTrainer(resumed_cfg, device="cpu")
    _ = resumed.step_update()
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

    sd_a = uninterrupted.sim.state_dict()
    sd_b = resumed.sim.state_dict()
    assert torch.equal(sd_a["bodies_live_i16"], sd_b["bodies_live_i16"])
    sim_a = sd_a["sim_tensors"]
    sim_b = sd_b["sim_tensors"]
    for k, t_a in sim_a.items():
        assert torch.equal(t_a, sim_b[k]), f"Sim tensor divergence after resume on {k}"


def test_resume_rejects_mismatched_config_and_skips_corrupt_checkpoint(tmp_path: Path) -> None:
    """Verify MINOR 6: load_checkpoint rejects mismatched config and scanner skips corrupt files."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        seed=555,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    trainer.step_update()
    ckpt_dir = tmp_path / "ckpts_m6"
    u1 = ckpt_dir / "ckpt_u0000001.pt"
    trainer.save_checkpoint(u1, reason="periodic")
    trainer.step_update()
    u2 = ckpt_dir / "ckpt_u0000002.pt"
    trainer.save_checkpoint(u2, reason="periodic")

    # Corrupt newer u2 payload bytes while leaving .sha256/.complete present
    u2.write_bytes(b"corrupted-checkpoint-payload")
    assert find_newest_verified_checkpoint(ckpt_dir) == u1

    # Mismatched seed or lr must raise ValueError on load_checkpoint
    wrong_seed_trainer = ArenaPPOTrainer(
        ArenaPPOConfig(**{**cfg.to_dict(), "seed": 556}), device="cpu"
    )
    with pytest.raises(ValueError, match="Checkpoint config mismatch on 'seed'"):
        wrong_seed_trainer.load_checkpoint(u1)

    wrong_lr_trainer = ArenaPPOTrainer(
        ArenaPPOConfig(**{**cfg.to_dict(), "lr": 1e-3}), device="cpu"
    )
    with pytest.raises(ValueError, match="Checkpoint config mismatch on 'lr'"):
        wrong_lr_trainer.load_checkpoint(u1)


def test_run_training_job_real_sigusr1_boundary_and_mid_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MINOR 9a & M5: real SIGUSR1 stops cleanly at boundary AND mid-tick."""
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 not available on this platform")

    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=3,
        checkpoint_every_updates=1,
        seed=77,
        deterministic=True,
    )

    # 1. Real SIGUSR1 delivered at update boundary via step_hook
    out_boundary = tmp_path / "sig_boundary"
    res_b = run_training_job(
        cfg=cfg,
        output_dir=out_boundary,
        device="cpu",
        step_hook=lambda _tr, u: os.kill(os.getpid(), signal.SIGUSR1) if u == 1 else None,
    )
    assert res_b["stopped_reason"] == "signal:SIGUSR1"
    assert res_b["aborted_mid_update"] is False
    assert res_b["updates_completed"] == 1
    assert verify_completed_artifact(out_boundary / "checkpoints" / "ckpt_sig_u0000001.pt")
    assert verify_completed_artifact(out_boundary / "checkpoints" / "final.pt")
    assert verify_completed_artifact(out_boundary / "arena_ppo_weights.pth")
    assert verify_completed_artifact(out_boundary / "train_result.json")

    # 2. Real SIGUSR1 delivered mid-rollout tick during update 2
    orig_step = TrainSim.step
    step_calls = {"n": 0}

    def _step_with_mid_signal(self: TrainSim, actions: torch.Tensor) -> StepInfo:
        step_calls["n"] += 1
        out = orig_step(self, actions)
        # Update 1 takes 4 ticks; fire real SIGUSR1 on tick 1 of update 2 (call #5)
        if step_calls["n"] == 5:
            os.kill(os.getpid(), signal.SIGUSR1)
        return out

    monkeypatch.setattr(TrainSim, "step", _step_with_mid_signal)
    out_mid = tmp_path / "sig_mid_tick"
    res_m = run_training_job(cfg=cfg, output_dir=out_mid, device="cpu")
    assert res_m["stopped_reason"] == "signal:SIGUSR1"
    assert res_m["aborted_mid_update"] is True
    assert res_m["updates_completed"] == 1
    assert verify_completed_artifact(out_mid / "checkpoints" / "ckpt_u0000001.pt")
    assert not (out_mid / "checkpoints" / "final.pt").exists()
    assert verify_completed_artifact(out_mid / "train_result.json")


def test_m5_projected_wall_guard_and_max_update_seconds(tmp_path: Path) -> None:
    """Verify M5 projected wall-budget stop and max_update_seconds runaway refusal."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=5,
        seed=88,
        deterministic=True,
    )
    # 1 update takes < 1s on CPU, while export_reserve_sec >= 5.0s -> stops after update 1
    out_proj = tmp_path / "proj_wall"
    res_proj = run_training_job(cfg=cfg, output_dir=out_proj, device="cpu", max_wall_seconds=3.0)
    assert res_proj["updates_completed"] == 1
    assert res_proj["stopped_reason"] == "wall_budget_projected_guard"
    assert verify_completed_artifact(out_proj / "checkpoints" / "final.pt")

    # Exceeding max_update_seconds saves periodic checkpoint and raises RuntimeError
    cfg_slow = ArenaPPOConfig(**{**cfg.to_dict(), "max_update_seconds": 1e-6})
    out_slow = tmp_path / "slow_update"
    with pytest.raises(RuntimeError, match="max_update_seconds"):
        run_training_job(cfg=cfg_slow, output_dir=out_slow, device="cpu")
    assert verify_completed_artifact(out_slow / "checkpoints" / "ckpt_u0000001.pt")


def test_m6_periodic_checkpoint_pruning_and_export_sizing(tmp_path: Path) -> None:
    """Verify M6 keeps newest K periodic checkpoints and sizes max_export_bytes accurately."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=5,
        checkpoint_every_updates=1,
        keep_periodic_checkpoints=3,
        seed=91,
        deterministic=True,
    )
    out_dir = tmp_path / "prune_out"
    res = run_training_job(cfg=cfg, output_dir=out_dir, device="cpu")
    assert res["updates_completed"] == 5
    ckpt_dir = out_dir / "checkpoints"
    assert not (ckpt_dir / "ckpt_u0000001.pt").exists()
    assert not (ckpt_dir / "ckpt_u0000001.pt.complete").exists()
    assert not (ckpt_dir / "ckpt_u0000002.pt").exists()
    for kept in ("ckpt_u0000003.pt", "ckpt_u0000004.pt", "ckpt_u0000005.pt", "final.pt"):
        assert verify_completed_artifact(ckpt_dir / kept)

    measured_bytes = (ckpt_dir / "final.pt").stat().st_size
    est_bytes = estimate_checkpoint_bytes(cfg)
    assert est_bytes >= measured_bytes
    sized_cap = estimate_max_export_bytes(cfg, measured_checkpoint_bytes=measured_bytes)
    assert 64 * 1024 * 1024 <= sized_cap <= 2 * 1024 * 1024 * 1024


def test_m7_max_world_frames_keeps_non_hit_worlds_byte_identical() -> None:
    """Verify M7: max_world_frames prepares s_{t+1} first and leaves ~hit worlds byte-identical."""
    base_kwargs = dict(
        net_size="S",
        num_envs=2,
        rollout_len=5,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=2,
        max_capacity=128,
        seed=1234,
        deterministic=True,
    )
    t_no_limit = ArenaPPOTrainer(ArenaPPOConfig(**base_kwargs, max_world_frames=0), device="cpu")
    t_limited = ArenaPPOTrainer(ArenaPPOConfig(**base_kwargs, max_world_frames=8), device="cpu")

    # Configure both sims identically on world 0, while advancing world 1's frame counter to 6
    # so world 1 hits max_world_frames=8 mid-rollout (at t=1). Also force seat (0, 0) into a wall
    # on tick 1 so world 0 exercises death + respawn + food RNG on the exact truncation tick!
    for tr in (t_no_limit, t_limited):
        tr.sim.prepare()
        tr.sim.sim.frame[1] = 6
        hp00 = int(tr.sim.sim.head_ptr[0, 0].item())
        tr.sim.sim.bodies[0, 0, hp00] = torch.tensor([tr.sim.grid_w - 2, 10], dtype=torch.long)
        tr.sim.sim.direction[0, 0] = 1

    fixed_actions = torch.ones((5, 2, 6), dtype=torch.long)  # straight

    for t in range(5):
        for tr in (t_no_limit, t_limited):
            tr.sim.prepare()
            tr.buffer.record_pre_step(t, tr.sim)
            info = tr.sim.step(fixed_actions[t])
            tr.buffer.died[t] = info.died
            tr.buffer.acted[t] = info.acted
            tr._maybe_apply_world_time_limits(t, info.survived)

    assert bool(t_limited.buffer.truncated[:, 1].any().item()) is True
    assert bool(t_limited.buffer.truncated[:, 0].any().item()) is False
    assert bool(torch.isfinite(t_limited.buffer.trunc_bootstrap[:, 1]).all().item()) is True

    # World 0 state across ALL DeviceSim tensors must be byte-identical to max_world_frames=0
    for name in _SIM_STATE_TENSORS:
        v_no = getattr(t_no_limit.sim.sim, name)
        v_lim = getattr(t_limited.sim.sim, name)
        if isinstance(v_no, torch.Tensor) and v_no.ndim >= 1 and v_no.shape[0] == 2:
            assert torch.equal(v_no[0], v_lim[0]), f"Non-hit world 0 diverged on {name}"


def test_ppo_trainer_custom_board_and_snake_count(tmp_path: Path) -> None:
    """Verify Coordinator #7 end-to-end PPO update + checkpoint at non-default board & snakes."""
    cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=2,
        num_snakes=4,
        grid_h=48,
        grid_w=64,
        rollout_len=6,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=2,
        max_capacity=128,
        seed=404,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    metrics = trainer.step_update()
    assert math.isfinite(metrics.policy_loss)
    assert metrics.agent_steps_collected == 2 * 4 * 6

    ckpt_path = tmp_path / "custom_ckpt" / "ckpt_u0000001.pt"
    trainer.save_checkpoint(ckpt_path, reason="periodic")
    reloaded = ArenaPPOTrainer(cfg, device="cpu")
    reloaded.load_checkpoint(ckpt_path)
    assert reloaded.sim.S == 4
    assert (reloaded.sim.grid_h, reloaded.sim.grid_w) == (48, 64)

    # Also verify >256 board coordinate dtype switches food_xy from uint8 to int16
    wide_cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=1,
        num_snakes=4,
        grid_h=48,
        grid_w=260,
        rollout_len=2,
        max_capacity=128,
    )
    wide_trainer = ArenaPPOTrainer(wide_cfg, device="cpu")
    assert wide_trainer.buffer.food_xy.dtype == torch.int16
