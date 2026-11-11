"""Tests for `research.arena_ppo.ppo` (rollout tape parity, GAE, weight update, CPU resume)."""

from __future__ import annotations

import json
import math
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from research.arena_ppo import ppo as ppo_mod
from research.arena_ppo.net import NetArch, SharedTrunkArenaNet
from research.arena_ppo.obs import build_arena_ego_obs_gpu, build_shared_world_obs_gpu
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    build_training_plan,
    estimate_checkpoint_bytes,
    estimate_max_export_bytes,
    estimate_rollout_buffer_bytes,
    find_newest_verified_checkpoint,
    run_training_job,
)
from research.arena_ppo.runtime import verify_completed_artifact
from research.arena_ppo.sim import _SIM_STATE_TENSORS, StepInfo, TrainSim
from research.colab_cli.runtime import _exports as supervisor_scan_exports


def test_ppo_config_json_roundtrip_and_defaults(tmp_path: Path) -> None:
    """Verify ArenaPPOConfig per-arch/per-net defaults and JSON file/string serialization."""
    cfg_s = ArenaPPOConfig(net_size="S")
    assert cfg_s.net_arch == "shared_trunk"
    assert cfg_s.rollout_len == 256
    assert cfg_s.gae_lambda == 0.97
    assert cfg_s.gamma == 0.999
    assert cfg_s.checkpoint_interval_sec == 1200.0
    assert cfg_s.minibatch_size == 3072
    assert cfg_s.actor_chunk_envs == 2048
    assert cfg_s.keep_periodic_checkpoints == 3
    assert cfg_s.max_update_seconds == 1200.0
    assert cfg_s.output_publish_interval_sec == 1800.0
    assert cfg_s.eval_every_updates == 0
    assert cfg_s.late_reset_enabled is False
    assert cfg_s.late_reset_min_length == 600
    assert cfg_s.late_reset_prob == 0.5

    cfg_m = ArenaPPOConfig(net_size="M")
    assert cfg_m.minibatch_size == 1536
    assert cfg_m.actor_chunk_envs == 1024

    cfg_ps = ArenaPPOConfig(net_arch="per_seat", net_size="S")
    assert cfg_ps.minibatch_size == 768
    assert cfg_ps.actor_chunk_envs == 512

    cfg_path = tmp_path / "ppo_cfg.json"
    cfg_path.write_text(cfg_s.to_json(), encoding="utf-8")
    loaded = ArenaPPOConfig.from_json(cfg_path)
    assert loaded == cfg_s


def test_compact_rollout_buffer_reconstructs_bit_identical_obs() -> None:
    """Prove CompactRolloutBuffer reconstructs exact obs across deaths, respawns, and burns."""
    cfg = ArenaPPOConfig(
        net_arch="per_seat",
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
        live_sh = build_shared_world_obs_gpu(trainer.sim)
        live_mask = trainer.sim.legal_action_mask().reshape(-1, 6)
        live_snapshots.append(
            (
                live_obs.arena.clone(),
                live_obs.ego.clone(),
                live_obs.scalars.clone(),
                live_sh.world_arena.clone(),
                live_sh.head_xy.clone(),
                live_sh.direction.clone(),
                live_sh.alive.clone(),
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
        rec_sh, rec_sh_mask = trainer.buffer.reconstruct_shared_world_ticks(t_idx, e_idx)
        (
            exp_arena,
            exp_ego,
            exp_scalars,
            exp_world_arena,
            exp_head_xy,
            exp_dir,
            exp_alive,
            exp_mask,
        ) = live_snapshots[t]
        assert torch.equal(rec_arena, exp_arena), f"Rollout tape arena mismatch at t={t}"
        assert torch.equal(rec_ego, exp_ego), f"Rollout tape ego mismatch at t={t}"
        assert torch.equal(rec_scalars, exp_scalars), f"Rollout tape scalars mismatch at t={t}"
        assert torch.equal(rec_mask, exp_mask), f"Rollout tape mask mismatch at t={t}"
        assert torch.equal(rec_sh.world_arena, exp_world_arena)
        assert torch.equal(rec_sh.ego, exp_ego)
        assert torch.equal(rec_sh.scalars, exp_scalars)
        assert torch.equal(rec_sh.head_xy, exp_head_xy)
        assert torch.equal(rec_sh.direction, exp_dir)
        assert torch.equal(rec_sh.alive, exp_alive)
        assert torch.equal(rec_sh_mask, exp_mask)


def test_gae_death_no_bootstrap_and_ppo_update_changes_weights() -> None:
    """Verify death has zero bootstrap, rollout cut bootstraps, loss is finite, weights change."""
    cfg = ArenaPPOConfig(
        net_arch="per_seat",
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
        net_arch="per_seat",
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


@pytest.mark.parametrize("net_arch", ["per_seat", "shared_trunk", "ego_only"])
def test_ppo_resume_bit_identical_to_uninterrupted_cpu(tmp_path: Path, net_arch: NetArch) -> None:
    """Prove resumed == uninterrupted bit-for-bit on CPU across all 3 net_arch variants."""
    cfg = ArenaPPOConfig(
        net_arch=net_arch,
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
        net_arch="per_seat",
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
    """Verify MINOR 9a, M5 & R3 MINOR 2: boundary single-save hard-link + mid-tick SIGUSR1."""
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 not available on this platform")

    # 1. Real SIGUSR1 at boundary with checkpoint_every_updates=0 so export phase does ONE save
    # and hard-links ckpt_sig_u0000001.pt + final.pt to the exact same inode inside grace window.
    cfg_boundary = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=3,
        checkpoint_every_updates=0,
        seed=77,
        deterministic=True,
    )
    save_calls = {"n": 0}
    orig_save_ckpt = ArenaPPOTrainer.save_checkpoint

    def _spy_save_ckpt(self: ArenaPPOTrainer, path: Path | str, *, reason: str = "periodic") -> str:
        save_calls["n"] += 1
        return orig_save_ckpt(self, path, reason=reason)

    monkeypatch.setattr(ArenaPPOTrainer, "save_checkpoint", _spy_save_ckpt)
    cli_root_b = tmp_path / "sig_boundary_root"
    out_boundary = cli_root_b / "output"
    cache_b: dict[str, tuple[int, int, int, int, int]] = {}
    cap_b = estimate_max_export_bytes(cfg_boundary)

    res_b = run_training_job(
        cfg=cfg_boundary,
        output_dir=out_boundary,
        work_dir=cli_root_b / "work",
        device="cpu",
        step_hook=lambda _tr, u: os.kill(os.getpid(), signal.SIGUSR1) if u == 1 else None,
    )
    assert save_calls["n"] == 1, "Boundary SIGUSR1 must write one checkpoint and hard-link final.pt"
    assert res_b["stopped_reason"] == "signal:SIGUSR1"
    assert res_b["aborted_mid_update"] is False
    assert res_b["updates_completed"] == 1
    sig_pt = out_boundary / "checkpoints" / "ckpt_sig_u0000001.pt"
    fin_pt = out_boundary / "checkpoints" / "final.pt"
    assert verify_completed_artifact(sig_pt)
    assert verify_completed_artifact(fin_pt)
    assert sig_pt.stat().st_ino == fin_pt.stat().st_ino, "ckpt_sig and final.pt must share inode"
    assert verify_completed_artifact(out_boundary / "arena_ppo_weights.pth")
    assert verify_completed_artifact(out_boundary / "train_result.json")
    supervisor_scan_exports(cli_root_b, cap_b, cache_b)

    monkeypatch.setattr(ArenaPPOTrainer, "save_checkpoint", orig_save_ckpt)

    # 2. Real SIGUSR1 delivered mid-rollout tick during update 2 -> publishes newest periodic ckpt
    cfg_mid = ArenaPPOConfig(**{**cfg_boundary.to_dict(), "checkpoint_every_updates": 1})
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
    cli_root_m = tmp_path / "sig_mid_root"
    out_mid = cli_root_m / "output"
    cache_m: dict[str, tuple[int, int, int, int, int]] = {}
    res_m = run_training_job(
        cfg=cfg_mid,
        output_dir=out_mid,
        work_dir=cli_root_m / "work",
        device="cpu",
    )
    assert res_m["stopped_reason"] == "signal:SIGUSR1"
    assert res_m["aborted_mid_update"] is True
    assert res_m["updates_completed"] == 1
    assert verify_completed_artifact(out_mid / "checkpoints" / "ckpt_u0000001.pt")
    assert not (out_mid / "checkpoints" / "final.pt").exists()
    assert verify_completed_artifact(out_mid / "train_result.json")
    supervisor_scan_exports(cli_root_m, cap_b, cache_m)


def test_m5_projected_wall_guard_and_max_update_seconds(tmp_path: Path) -> None:
    """Verify M5 + R3 MAJOR-2: projected wall guard AND max_update_seconds stop cleanly & export."""
    cfg = ArenaPPOConfig(
        net_arch="per_seat",
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
    assert verify_completed_artifact(out_proj / "train_result.json")

    # MAJOR-2: Exceeding max_update_seconds stops cleanly (no RuntimeError!) and exports all files
    cfg_slow = ArenaPPOConfig(**{**cfg.to_dict(), "max_update_seconds": 1e-6})
    cli_slow = tmp_path / "slow_update_root"
    out_slow = cli_slow / "output"
    cache_slow: dict[str, tuple[int, int, int, int, int]] = {}
    res_slow = run_training_job(
        cfg=cfg_slow,
        output_dir=out_slow,
        work_dir=cli_slow / "work",
        device="cpu",
    )
    assert res_slow["stopped_reason"] == "max_update_seconds_guard"
    assert res_slow["aborted_mid_update"] is False
    assert res_slow["updates_completed"] == 1
    assert verify_completed_artifact(out_slow / "checkpoints" / "ckpt_u0000001.pt")
    assert verify_completed_artifact(out_slow / "checkpoints" / "final.pt")
    assert verify_completed_artifact(out_slow / "arena_ppo_weights.pth")
    assert verify_completed_artifact(out_slow / "train_result.json")
    supervisor_scan_exports(cli_slow, estimate_max_export_bytes(cfg_slow), cache_slow)


def test_t1_work_dir_pruning_45m_hardlink_snapshots_and_supervisor_exports_scanner(
    tmp_path: Path,
) -> None:
    """BLOCKER T1 + R3 MINOR 6: prune in work_dir only; drive colab_cli _exports across saves."""
    cfg = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=2,
        rollout_len=3,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=6,
        checkpoint_every_updates=1,
        keep_periodic_checkpoints=2,
        output_publish_interval_sec=1e-6,  # publish periodic hard-linked snapshot every update
        log_rotate_every_updates=2,
        seed=91,
        deterministic=True,
    )
    cli_root = tmp_path / "colab_t1_root"
    out_dir = cli_root / "output"
    work_dir = cli_root / "work" / "arena_ppo"
    cap = estimate_max_export_bytes(cfg)
    scanner_cache: dict[str, tuple[int, int, int, int, int]] = {}
    observed_counts: list[int] = []

    def _poll_supervisor_scanner(_tr: ArenaPPOTrainer, _u: int) -> None:
        inv = supervisor_scan_exports(cli_root, cap, scanner_cache)
        observed_counts.append(len(inv))

    res_all_pub = run_training_job(
        cfg=cfg,
        output_dir=out_dir,
        work_dir=work_dir,
        device="cpu",
        step_hook=_poll_supervisor_scanner,
    )
    final_inv = supervisor_scan_exports(cli_root, cap, scanner_cache)
    assert res_all_pub["updates_completed"] == 6
    assert observed_counts == sorted(observed_counts), "output/ completed exports must never shrink"
    assert len(final_inv) >= observed_counts[-1]
    for u in range(1, 7):
        assert verify_completed_artifact(out_dir / "checkpoints" / f"ckpt_u{u:07d}.pt")
    assert verify_completed_artifact(out_dir / "checkpoints" / "final.pt")
    assert verify_completed_artifact(out_dir / "arena_ppo_weights.pth")
    assert verify_completed_artifact(out_dir / "train_result.json")

    # Subcase B: long output_publish_interval_sec -> 6 saves in work_dir (pruned to keep=2),
    # 0 periodic files unlinked in output/, and scanner passes on every single update + end.
    cfg_prune = ArenaPPOConfig(**{**cfg.to_dict(), "output_publish_interval_sec": 2400.0})
    cli_root_b = tmp_path / "colab_t1_prune_root"
    out_dir_b = cli_root_b / "output"
    work_dir_b = cli_root_b / "work" / "arena_ppo"
    cache_b: dict[str, tuple[int, int, int, int, int]] = {}

    def _poll_b(_tr: ArenaPPOTrainer, _u: int) -> None:
        supervisor_scan_exports(cli_root_b, cap, cache_b)

    res_prune = run_training_job(
        cfg=cfg_prune,
        output_dir=out_dir_b,
        work_dir=work_dir_b,
        device="cpu",
        step_hook=_poll_b,
    )
    supervisor_scan_exports(cli_root_b, cap, cache_b)
    assert len(res_prune["pruned_periodic_checkpoints"]) == 4
    work_ckpts = sorted(p.name for p in (work_dir_b / "checkpoints").glob("ckpt_u*.pt"))
    assert work_ckpts == ["ckpt_u0000005.pt", "ckpt_u0000006.pt"]
    assert verify_completed_artifact(out_dir_b / "checkpoints" / "final.pt")

    # Refuse work_dir inside output_dir
    with pytest.raises(ValueError, match="outside output_dir"):
        run_training_job(
            cfg=cfg_prune,
            output_dir=tmp_path / "bad_out",
            work_dir=tmp_path / "bad_out" / "nested_work",
            device="cpu",
        )

    # MINOR 6: per-file sizing from full max_capacity x 1.25 (independent of keep count)
    measured_bytes = (out_dir_b / "checkpoints" / "final.pt").stat().st_size
    est_bytes = estimate_checkpoint_bytes(cfg_prune)
    assert est_bytes >= measured_bytes
    est_cap_small = estimate_checkpoint_bytes(
        ArenaPPOConfig(net_arch="per_seat", net_size="S", num_envs=1024, max_capacity=400)
    )
    est_cap_large = estimate_checkpoint_bytes(
        ArenaPPOConfig(net_arch="per_seat", net_size="S", num_envs=1024, max_capacity=1600)
    )
    assert est_cap_large > est_cap_small, "estimate_checkpoint_bytes must scale with max_capacity"

    cfg_e4096_k3 = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="M",
        num_envs=4096,
        max_capacity=1600,
        keep_periodic_checkpoints=3,
    )
    cfg_e4096_k9 = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="M",
        num_envs=4096,
        max_capacity=1600,
        keep_periodic_checkpoints=9,
    )
    cap_k3 = estimate_max_export_bytes(cfg_e4096_k3)
    cap_k9 = estimate_max_export_bytes(cfg_e4096_k9)
    assert cap_k3 == cap_k9
    assert 64 * 1024 * 1024 <= cap_k3 <= 2 * 1024 * 1024 * 1024


def test_m7_max_world_frames_keeps_non_hit_worlds_byte_identical() -> None:
    """Verify M7: max_world_frames prepares s_{t+1} first and leaves ~hit worlds byte-identical."""
    base_kwargs: dict[str, Any] = dict(
        net_arch="per_seat",
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
        net_arch="per_seat",
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
        net_arch="per_seat",
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
    assert (trainer.model.cfg.token_grid_h, trainer.model.cfg.token_grid_w) == (6, 8)
    assert trainer.buffer.nbytes() == estimate_rollout_buffer_bytes(
        rollout_len=cfg.rollout_len,
        num_envs=cfg.num_envs,
        num_snakes=cfg.num_snakes,
        max_capacity=cfg.max_capacity,
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        food_capacity=trainer.sim.sim.food_capacity,
    )


def test_minor7_r3_resume_allows_speed_keys_and_features_abc_plan_logs_eval_late_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify MINOR 7 speed keys + Features (a) plan, (b) JSONL logs/eval, (c) late resets."""
    monkeypatch.setattr(torch, "compile", lambda fn, **_kw: fn)

    # MINOR 7: save with actor_chunk_envs=2, ring_chunk=64, compile_model=False; resume with
    # actor_chunk_envs=1, ring_chunk=128, compile_model=True!
    cfg_orig = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=2,
        ring_chunk=64,
        compile_model=False,
        max_capacity=128,
        seed=515,
        deterministic=True,
    )
    tr_orig = ArenaPPOTrainer(cfg_orig, device="cpu")
    tr_orig.step_update()
    ckpt_m7 = tmp_path / "m7_speed" / "ckpt_u0000001.pt"
    tr_orig.save_checkpoint(ckpt_m7)

    cfg_speed_override = ArenaPPOConfig(
        **{
            **cfg_orig.to_dict(),
            "actor_chunk_envs": 1,
            "ring_chunk": 128,
            "compile_model": True,
        }
    )
    tr_override = ArenaPPOTrainer(cfg_speed_override, device="cpu")
    loaded = tr_override.load_checkpoint(ckpt_m7)
    assert loaded["counters"]["updates_completed"] == 1
    m2 = tr_override.step_update()
    assert math.isfinite(m2.policy_loss)

    # Feature (a) & (b): --hours + --expected-steps-per-sec plan printed before starting,
    # rotating create-only JSONL training logs written every update, eval hook OFF by default.
    plan = build_training_plan(
        ArenaPPOConfig(net_arch="per_seat", net_size="S", num_envs=1024, rollout_len=256),
        hours=6.0,
        expected_steps_per_sec=150_000.0,
    )
    assert plan["expected_agent_steps"] == 6 * 3600 * 150_000
    assert plan["expected_updates"] == (6 * 3600 * 150_000) // (256 * 1024 * 6)

    cli_root = tmp_path / "features_ab_root"
    out_ab = cli_root / "output"
    cache_ab: dict[str, tuple[int, int, int, int, int]] = {}
    eval_calls: list[int] = []
    cfg_ab = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=2,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        total_updates=5,
        log_rotate_every_updates=2,
        eval_every_updates=0,  # OFF by default
        seed=707,
        deterministic=True,
    )
    res_ab = run_training_job(
        cfg=cfg_ab,
        output_dir=out_ab,
        work_dir=cli_root / "work",
        device="cpu",
        hours=0.5,
        expected_steps_per_sec=2400.0,
        eval_hook=lambda _tr, u: eval_calls.append(u) or {"u": u},
        step_hook=lambda _tr, _u: supervisor_scan_exports(
            cli_root, estimate_max_export_bytes(cfg_ab), cache_ab
        )
        and None,
    )
    supervisor_scan_exports(cli_root, estimate_max_export_bytes(cfg_ab), cache_ab)
    captured = capsys.readouterr()
    assert "ARENA_PPO_PLAN " in captured.out
    assert res_ab["training_plan"] is not None
    assert res_ab["training_plan"]["expected_updates"] > 0
    assert eval_calls == [], "eval_hook must remain OFF when eval_every_updates == 0"

    # Verify rotating create-only JSONL segments (5 updates @ rotate=2 -> 3 sealed files)
    log_files = sorted((out_ab / "train_logs").glob("train_log_*.jsonl"))
    assert len(log_files) == 3
    parsed_records: list[dict[str, Any]] = []
    for lf in log_files:
        assert verify_completed_artifact(lf)
        for line in lf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed_records.append(json.loads(line))
    assert [r["update_idx"] for r in parsed_records] == [1, 2, 3, 4, 5]
    required_keys = {
        "update_idx",
        "agent_steps",
        "steps_per_sec",
        "mean_length",
        "p95_length",
        "deaths_per_1k_agent_steps",
        "mean_episode_return",
        "entropy",
        "approx_kl",
        "value_explained_variance",
        "clip_fraction",
        "aux_losses",
    }
    for rec in parsed_records:
        assert required_keys.issubset(rec.keys())
        assert set(rec["aux_losses"].keys()) == {"total", "h16", "h64", "h256"}

    # Also verify CLI ppo.main() lifts default total_updates when --hours drives the run
    cli_main_out = tmp_path / "cli_main_out"
    partial_cfg_path = tmp_path / "partial_cfg.json"
    partial_cfg_path.write_text(
        json.dumps(
            {
                "net_arch": "per_seat",
                "net_size": "S",
                "num_envs": 2,
                "rollout_len": 3,
                "ppo_epochs": 1,
                "minibatch_size": 12,
                "max_capacity": 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research.arena_ppo.ppo",
            "--config-json",
            str(partial_cfg_path),
            "--output-dir",
            str(cli_main_out),
            "--device",
            "cpu",
            "--hours",
            "2.0",
            "--expected-steps-per-sec",
            "50000",
            "--max-wall-seconds",
            "1.0",
        ],
    )
    assert ppo_mod.main() == 0
    cli_res = json.loads((cli_main_out / "train_result.json").read_text(encoding="utf-8"))
    assert cli_res["config"]["total_updates"] > 100

    # Feature (c): end-to-end training + checkpoint/resume with late_reset_enabled=True
    # (update 1 harvests mature states and queues rollout-start restore; update 2 restores at t=0)
    cfg_late = ArenaPPOConfig(
        net_arch="per_seat",
        net_size="S",
        num_envs=4,
        rollout_len=6,
        ppo_epochs=1,
        minibatch_size=12,
        max_capacity=128,
        max_world_frames=4,
        late_reset_enabled=True,
        late_reset_min_length=5,
        late_reset_prob=1.0,
        late_reset_reservoir_size=8,
        seed=909,
        deterministic=True,
    )
    # 1. Uninterrupted 3 updates with late_reset_enabled=True
    tr_late_unint = ArenaPPOTrainer(cfg_late, device="cpu")
    tr_late_unint.sim.prepare()
    tr_late_unint.sim.sim.length[:, 0] = 8
    tr_late_unint.sim.sim._reward_prev_length.copy_(tr_late_unint.sim.sim.length)
    _ = tr_late_unint.step_update()
    _ = tr_late_unint.step_update()
    m_next_a = tr_late_unint.step_update()

    # 2. Split 2 updates -> save_checkpoint -> fresh trainer load_checkpoint -> 3rd update
    tr_late_part1 = ArenaPPOTrainer(cfg_late, device="cpu")
    tr_late_part1.sim.prepare()
    tr_late_part1.sim.sim.length[:, 0] = 8
    tr_late_part1.sim.sim._reward_prev_length.copy_(tr_late_part1.sim.sim.length)
    tr_late_part1.step_update()
    assert tr_late_part1.late_reservoir is not None
    assert tr_late_part1.late_reservoir.total_harvested > 0
    assert tr_late_part1.late_reservoir.total_restored == 0
    tr_late_part1.step_update()
    assert tr_late_part1.late_reservoir.total_restored > 0

    late_ckpt = tmp_path / "late_ckpt" / "ckpt_u0000002.pt"
    tr_late_part1.save_checkpoint(late_ckpt)
    tr_late_resumed = ArenaPPOTrainer(cfg_late, device="cpu")
    tr_late_resumed.load_checkpoint(late_ckpt)
    m_next_b = tr_late_resumed.step_update()
    assert math.isclose(m_next_a.policy_loss, m_next_b.policy_loss, rel_tol=0.0, abs_tol=1e-7)


def test_shared_trunk_learner_samples_world_ticks_once_and_masks_dead_seats_exact() -> None:
    """Prove shared_trunk runs once per (world, tick) group and masks dead/non-learner seats."""
    cfg = ArenaPPOConfig(
        net_arch="shared_trunk",
        net_size="S",
        num_envs=4,
        rollout_len=6,
        ppo_epochs=2,
        minibatch_size=12,  # 2 world-ticks (12 seats) per minibatch
        actor_chunk_envs=2,
        max_capacity=128,
        seed=7171,
        deterministic=True,
    )
    tr_clean = ArenaPPOTrainer(cfg, device="cpu")
    assert isinstance(tr_clean.model, SharedTrunkArenaNet)
    tr_clean.collect_rollout()

    # Force half of seats in several world-ticks to be dead/non-learner (`acted=False` or
    # `learner_mask=False`) while keeping both trainers identical on valid learner seats.
    tr_clean.buffer.acted[0, 0, [1, 3, 5]] = False
    tr_clean.buffer.learner_mask[1, 2, [0, 2, 4]] = False
    tr_clean.buffer.acted[3, 1, :] = False  # whole world-tick inactive

    tr_poison = ArenaPPOTrainer(cfg, device="cpu")
    tr_poison.model.load_state_dict(tr_clean.model.state_dict())
    tr_poison.optimizer.load_state_dict(tr_clean.optimizer.state_dict())

    # Copy rollout buffer tensors into tr_poison, then poison EVERY inactive seat's targets/stats
    for attr in (
        "body_tape",
        "tape_cursor",
        "tape_head_idx",
        "seg_count",
        "length",
        "alive",
        "prev_alive",
        "boosted",
        "direction",
        "prev_heads",
        "food_xy",
        "food_kind",
        "actions",
        "log_probs",
        "values",
        "rewards",
        "died",
        "truncated",
        "trunc_bootstrap",
        "acted",
        "learner_mask",
        "advantages",
        "returns",
        "aux_targets",
        "aux_valid",
    ):
        getattr(tr_poison.buffer, attr).copy_(getattr(tr_clean.buffer, attr))

    invalid = ~(tr_poison.buffer.acted & tr_poison.buffer.learner_mask)
    assert int(invalid.sum().item()) > 0
    tr_poison.buffer.advantages[invalid] = float("nan")
    tr_poison.buffer.returns[invalid] = float("nan")
    tr_poison.buffer.values[invalid] = float("nan")
    tr_poison.buffer.log_probs[invalid] = float("nan")
    tr_poison.buffer.rewards[invalid] = 1.0e9
    tr_poison.buffer.aux_targets[invalid] = float("nan")

    # Spy on arena_patch_stem batch size during ppo_update: must be mb_worlds=2 (NOT 12 rows!)
    stem_batch_sizes: list[int] = []
    handle = tr_clean.model.arena_patch_stem.register_forward_hook(
        lambda _mod, inp, _out: stem_batch_sizes.append(int(inp[0].shape[0]))
    )
    torch.manual_seed(424242)
    stats_clean, _ = tr_clean.ppo_update()
    handle.remove()

    torch.manual_seed(424242)
    stats_poison, _ = tr_poison.ppo_update()

    # 4 worlds * 6 ticks = 24 world-ticks -> 12 minibatches/epoch * 2 epochs = 24 trunk calls,
    # each with batch = 2 world-ticks (never 12 per-seat arena passes!).
    assert len(stem_batch_sizes) == 24
    assert all(b == 2 for b in stem_batch_sizes)

    assert math.isfinite(stats_poison["policy_loss"])
    assert math.isfinite(stats_poison["value_loss"])
    assert math.isfinite(stats_poison["aux_loss"])
    assert stats_clean == stats_poison
    for k, v_c in tr_clean.model.state_dict().items():
        v_p = tr_poison.model.state_dict()[k]
        assert torch.equal(v_c, v_p), f"Dead-seat poisoning leaked into parameter {k}"


@pytest.mark.parametrize("net_arch", ["per_seat", "shared_trunk", "ego_only"])
def test_b1_late_reservoir_rollout_start_restore_rebuilt_obs_match_live_all_archs(
    net_arch: NetArch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: reservoir restores ONLY at rollout start t=0; rebuilt obs == live obs on all 3 archs."""
    # Exact reviewer repro geometry: E=4, max_world_frames=6, late_reset_prob=1.0, T=24
    cfg = ArenaPPOConfig(
        net_arch=net_arch,
        net_size="S",
        num_envs=4,
        rollout_len=24,
        ppo_epochs=1,
        minibatch_size=24,
        actor_chunk_envs=4,
        max_capacity=128,
        max_world_frames=6,
        late_reset_enabled=True,
        late_reset_min_length=5,
        late_reset_prob=1.0,
        late_reset_reservoir_size=16,
        seed=20261111,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    assert trainer.late_reservoir is not None

    # Grow multi-segment bodies (seg_count >= 5, length >= 8) before rollout 1 so harvested
    # reservoir states carry long bodies that exposed the old mid-rollout body_tape bug.
    trainer.sim.prepare()
    trainer.sim.sim.length[:] = 8
    trainer.sim.sim._reward_prev_length.copy_(trainer.sim.sim.length)
    for _ in range(4):
        trainer.sim.step(torch.ones((4, 6), dtype=torch.long))
    trainer.sim.sim.frame.zero_()
    trainer.sim.sim._prepared = False

    # Rollout 1 hits max_world_frames=6 four times, harvests mature states, and queues t=0 restore,
    # performing ZERO mid-rollout reservoir restores.
    trainer.step_update()
    assert trainer.late_reservoir.total_harvested > 0
    assert trainer.late_reservoir.total_restored == 0

    # Poison episode_return_acc before rollout 2 to prove restored worlds zero it at t=0.
    trainer.episode_return_acc.fill_(77.5)

    orig_record_pre_step = ppo_mod.CompactRolloutBuffer.record_pre_step
    live_per_tick: list[tuple[Any, torch.Tensor]] = []
    ep_acc_at_t0: list[torch.Tensor] = []

    def _spy_record_pre_step(
        buf_self: ppo_mod.CompactRolloutBuffer, t: int, train_sim: TrainSim
    ) -> None:
        orig_record_pre_step(buf_self, t, train_sim)
        if t == 0:
            ep_acc_at_t0.append(trainer.episode_return_acc.clone())
        live_mask = train_sim.legal_action_mask().reshape(-1, 6).clone()
        if net_arch == "per_seat":
            obs = build_arena_ego_obs_gpu(train_sim)
            live_per_tick.append((obs, live_mask))
        else:
            sh_obs = build_shared_world_obs_gpu(train_sim)
            live_per_tick.append((sh_obs, live_mask))

    monkeypatch.setattr(ppo_mod.CompactRolloutBuffer, "record_pre_step", _spy_record_pre_step)
    trainer.collect_rollout()
    monkeypatch.setattr(ppo_mod.CompactRolloutBuffer, "record_pre_step", orig_record_pre_step)

    # Exactly the 4 worlds restored at t=0 of rollout 2 (zero mid-rollout restores at t=5,11,17,23)
    assert trainer.late_reservoir.total_restored == 4
    assert len(ep_acc_at_t0) == 1
    assert torch.all(ep_acc_at_t0[0] == 0.0), "Restored worlds must zero episode_return_acc at t=0"
    assert len(live_per_tick) == cfg.rollout_len

    e_idx = torch.arange(cfg.num_envs, dtype=torch.long)
    for t in range(cfg.rollout_len):
        t_idx = torch.full((cfg.num_envs,), t, dtype=torch.long)
        live_obs, exp_mask = live_per_tick[t]
        if net_arch == "per_seat":
            rec_arena, rec_ego, rec_scalars, rec_mask = trainer.buffer.reconstruct_world_ticks(
                t_idx, e_idx
            )
            assert torch.equal(rec_arena, live_obs.arena), f"arena mismatch at t={t}"
            assert torch.equal(rec_ego, live_obs.ego), f"ego mismatch at t={t}"
            assert torch.equal(rec_scalars, live_obs.scalars), f"scalars mismatch at t={t}"
            assert torch.equal(rec_mask, exp_mask), f"mask mismatch at t={t}"
        else:
            rec_sh, rec_mask = trainer.buffer.reconstruct_shared_world_ticks(t_idx, e_idx)
            assert torch.equal(
                rec_sh.world_arena, live_obs.world_arena
            ), f"world_arena mismatch at t={t} ({net_arch})"
            assert torch.equal(rec_sh.ego, live_obs.ego), f"ego mismatch at t={t} ({net_arch})"
            assert torch.equal(
                rec_sh.scalars, live_obs.scalars
            ), f"scalars mismatch at t={t} ({net_arch})"
            assert torch.equal(rec_sh.head_xy, live_obs.head_xy), f"head_xy mismatch at t={t}"
            assert torch.equal(rec_sh.direction, live_obs.direction), f"direction mismatch at t={t}"
            assert torch.equal(rec_sh.alive, live_obs.alive), f"alive mismatch at t={t}"
            assert torch.equal(rec_mask, exp_mask), f"mask mismatch at t={t}"

    stats, _ = trainer.ppo_update()
    assert math.isfinite(stats["policy_loss"])
    assert math.isfinite(stats["value_loss"])


def test_minor1_late_reset_requires_positive_max_world_frames() -> None:
    """Minor (1): ArenaPPOConfig refuses late_reset_enabled=True with max_world_frames <= 0."""
    with pytest.raises(ValueError, match="late_reset_enabled=True requires max_world_frames > 0"):
        ArenaPPOConfig(late_reset_enabled=True, max_world_frames=0)
    with pytest.raises(ValueError, match="late_reset_enabled=True requires max_world_frames > 0"):
        ArenaPPOConfig(late_reset_enabled=True, max_world_frames=-10)


def test_minor2_published_checkpoint_gap_never_exceeds_45_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minor (2): worst-case gap between published checkpoints is <= 45 min (2700 s)."""
    assert ArenaPPOConfig().output_publish_interval_sec == 1800.0

    fake_now = {"t": 10_000.0}
    monkeypatch.setattr(ppo_mod.time, "monotonic", lambda: fake_now["t"])

    orig_step_update = ArenaPPOTrainer.step_update

    def _timed_step_update(
        self: ArenaPPOTrainer, *, should_stop: Any = None
    ) -> ppo_mod.PPOUpdateMetrics:
        m = orig_step_update(self, should_stop=should_stop)
        # Simulate a 1000 s (16.67 min) update: two updates = 2000 s (< 2400 s), while a third
        # update would reach 3000 s (50 min > 45 min) unless update 2 publishes proactively!
        fake_now["t"] += 1000.0
        return dataclasses.replace(m, collect_sec=500.0, learn_sec=500.0)

    import dataclasses

    monkeypatch.setattr(ArenaPPOTrainer, "step_update", _timed_step_update)

    publish_timestamps: list[float] = [fake_now["t"]]
    orig_pub = ppo_mod._publish_work_checkpoint_to_output

    def _spy_publish(*args: Any, **kwargs: Any) -> tuple[str | None, list[str]]:
        sha, names = orig_pub(*args, **kwargs)
        if names:
            publish_timestamps.append(fake_now["t"])
        return sha, names

    monkeypatch.setattr(ppo_mod, "_publish_work_checkpoint_to_output", _spy_publish)

    cfg = ArenaPPOConfig(
        net_arch="shared_trunk",
        net_size="S",
        num_envs=1,
        num_snakes=4,
        grid_h=24,
        grid_w=32,
        rollout_len=2,
        ppo_epochs=1,
        minibatch_size=8,
        max_capacity=32,
        total_updates=5,
        output_publish_interval_sec=2400.0,  # even with 40m interval, 45m predictive cap binds!
        max_update_seconds=3600.0,
        seed=321,
        deterministic=True,
    )
    out_dir = tmp_path / "pub_45m_out"
    res = run_training_job(
        cfg=cfg,
        output_dir=out_dir,
        work_dir=tmp_path / "pub_45m_work",
        device="cpu",
    )
    assert res["updates_completed"] == 5
    # Update 2 (2000 s) + Update 4 (4000 s) + final export (5000 s) -> every gap <= 2000 s <= 2700 s
    assert verify_completed_artifact(out_dir / "checkpoints" / "ckpt_u0000002.pt")
    assert verify_completed_artifact(out_dir / "checkpoints" / "ckpt_u0000004.pt")
    gaps = [
        publish_timestamps[i + 1] - publish_timestamps[i]
        for i in range(len(publish_timestamps) - 1)
    ]
    assert gaps and max(gaps) <= 2700.0, f"Published checkpoint gap exceeded 45 min: {gaps}"


def test_minor5_checkpoint_cross_arch_load_refused(tmp_path: Path) -> None:
    """Minor (5a): a checkpoint saved from one net_arch is refused by every other net_arch."""
    archs: tuple[NetArch, ...] = ("per_seat", "shared_trunk", "ego_only")
    saved_ckpts: dict[NetArch, Path] = {}
    for arch in archs:
        cfg = ArenaPPOConfig(
            net_arch=arch,
            net_size="S",
            num_envs=2,
            rollout_len=2,
            ppo_epochs=1,
            minibatch_size=12,
            actor_chunk_envs=2,
            max_capacity=64,
            seed=2026,
            deterministic=True,
        )
        tr = ArenaPPOTrainer(cfg, device="cpu")
        ckpt_path = tmp_path / f"ckpt_{arch}" / "ckpt_u0000000.pt"
        tr.save_checkpoint(ckpt_path)
        saved_ckpts[arch] = ckpt_path

    for src_arch in archs:
        for dst_arch in archs:
            if src_arch == dst_arch:
                continue
            dst_cfg = ArenaPPOConfig(
                net_arch=dst_arch,
                net_size="S",
                num_envs=2,
                rollout_len=2,
                ppo_epochs=1,
                minibatch_size=12,
                actor_chunk_envs=2,
                max_capacity=64,
                seed=2026,
                deterministic=True,
            )
            dst_tr = ArenaPPOTrainer(dst_cfg, device="cpu")
            with pytest.raises(ValueError, match="Checkpoint config mismatch on 'net_arch'"):
                dst_tr.load_checkpoint(saved_ckpts[src_arch])


def test_minor6_max_world_frames_checks_host_bound_once_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Minor (6): max_world_frames > 0 does 1 host sync per rollout, not 1 per tick."""
    base = dict(
        net_arch="shared_trunk",
        net_size="S",
        num_envs=4,
        rollout_len=16,
        ppo_epochs=1,
        minibatch_size=24,
        actor_chunk_envs=4,
        max_capacity=64,
        seed=8888,
        deterministic=True,
    )
    # Warm module-level ego/grid tables once before counting Tensor.item() host syncs.
    ArenaPPOTrainer(ArenaPPOConfig(**base, max_world_frames=0), device="cpu").collect_rollout()

    orig_item = torch.Tensor.item
    counts = {"n": 0}

    def _counting_item(self: torch.Tensor) -> Any:
        counts["n"] += 1
        return orig_item(self)

    tr_unbounded = ArenaPPOTrainer(ArenaPPOConfig(**base, max_world_frames=0), device="cpu")
    monkeypatch.setattr(torch.Tensor, "item", _counting_item)
    counts["n"] = 0
    tr_unbounded.collect_rollout()
    calls_unbounded = counts["n"]
    monkeypatch.setattr(torch.Tensor, "item", orig_item)

    tr_bounded = ArenaPPOTrainer(ArenaPPOConfig(**base, max_world_frames=100), device="cpu")
    monkeypatch.setattr(torch.Tensor, "item", _counting_item)
    counts["n"] = 0
    tr_bounded.collect_rollout()
    calls_bounded = counts["n"]

    # Old code added +16 Tensor.item() host syncs (one per tick); new code adds +1 per rollout.
    assert (calls_bounded - calls_unbounded) == 1
