"""Tests for `research/v3_ppo` LIGHT/FULL sharded checkpoints, bit-identical resume, and eval."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import research.v3_ppo.eval as eval_mod
import research.v3_ppo.exports as exports_mod
import research.v3_ppo.job as job_mod
import research.v3_ppo.ppo as ppo_mod
import research.v3_ppo.trainer as trainer_mod
from research.arena_ppo.runtime import (
    sha256_file,
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.game_v3.config import GameV3Config
from research.v3_ppo.config import V3_PPO_NAMESPACE, V3PPOConfig
from research.v3_ppo.eval import (
    _combine_matchups,
    _EvalMatchupRaw,
    evaluate_policy,
    evaluate_scripted_baseline,
)
from research.v3_ppo.ppo import (
    StopRequestedError,
    V3PPOTrainer,
    pack_sim_state_dict,
    prune_old_checkpoints,
    resolve_resume_checkpoint,
    run_training_job,
    unpack_sim_state_dict,
    verify_full_checkpoint_set,
    verify_light_checkpoint,
    verify_staged_checkpoint,
)
from research.v3_ppo.prepare import ROOT, prepare_v3_ppo_packet


def _tiny_cfg(
    *,
    num_envs: int = 4,
    total_updates: int = 6,
    eval_every_updates: int = 2,
    eval_every_minutes: float = 0.0,
    learner_slots: int | None = None,
    checkpoint_shard_bytes: int = 256 * 1024 * 1024,
    seed: int = 20261111,
) -> V3PPOConfig:
    gcfg = GameV3Config(
        arena_radius=60.0,
        num_snakes=6,
        ambient_food_density=0.02,
        respawn_delay=8,
    )
    return V3PPOConfig(
        game_config=gcfg,
        num_envs=num_envs,
        learner_slots=learner_slots,
        opponent_policy="food_seeking",
        net_size="S",
        amp=False,
        use_channels_last=False,
        rollout_len=6,
        ppo_epochs=2,
        minibatch_size=16,
        total_updates=total_updates,
        checkpoint_shard_bytes=checkpoint_shard_bytes,
        eval_every_minutes=eval_every_minutes,
        eval_every_updates=eval_every_updates,
        eval_envs=2,
        eval_burnin_ticks=8,
        eval_ticks=30,
        eval_learner_slots=2,
        eval_seed=77777,
        seed=seed,
    )


def test_sharded_full_checkpoint_multi_shard_reload_and_guards(tmp_path: Path) -> None:
    """Verify >1 shard FULL write/reload, uint8-chunk zlib packing, and foreign-config refusal."""
    cfg = _tiny_cfg(num_envs=4, total_updates=2, eval_every_updates=0)
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.train_step()

    # 1. Verify lossless `pack_sim_state_dict` stores `torch.uint8` tensors (no UTF-8 bytes bloat)
    raw_sd = trainer.sim.state_dict()
    packed_sd = pack_sim_state_dict(raw_sd, total_threshold_bytes=0, tensor_min_bytes=256)
    assert "_zlib_tensors" in packed_sd
    assert len(packed_sd["_zlib_tensors"]) > 0
    for meta in packed_sd["_zlib_tensors"].values():
        assert all(
            isinstance(ch, torch.Tensor) and ch.dtype == torch.uint8 for ch in meta["chunks"]
        )
    unpacked_sd = unpack_sim_state_dict(packed_sd)
    assert "_zlib_tensors" not in unpacked_sd
    assert set(unpacked_sd["tensors"]) == set(raw_sd["tensors"])
    for t_name, orig_tensor in raw_sd["tensors"].items():
        rec_tensor = unpacked_sd["tensors"][t_name]
        assert rec_tensor.dtype == orig_tensor.dtype
        assert rec_tensor.shape == orig_tensor.shape
        assert torch.equal(rec_tensor, orig_tensor), f"Mismatch in unpacked tensor {t_name}"

    # 2. Measure 1-shard size and choose `max_shard_bytes` so 4 worlds split into >= 2 shards
    probe_dir = tmp_path / "probe_1shard"
    one_shard_idx = trainer.save_full_checkpoint(probe_dir, tag="u0000001")
    one_meta = verify_full_checkpoint_set(one_shard_idx, require_markers=True)
    one_shard_size = one_meta["shard_paths"][0].stat().st_size
    # Force <4 worlds per shard (`one_shard_size - 1024` < 4-world shard, > 1-world) -> >=2 shards
    multi_shard_cap = one_shard_size - 1024

    sharded_dir = tmp_path / "multi_shard_ckpts"
    idx_path = trainer.save_full_checkpoint(
        sharded_dir, tag="u0000001", max_shard_bytes=multi_shard_cap
    )
    full_meta = verify_full_checkpoint_set(idx_path, require_markers=True)
    assert full_meta["num_shards"] >= 2
    for sp in full_meta["shard_paths"]:
        verify_completed_artifact(sp)
        assert sp.stat().st_size <= multi_shard_cap

    staged_meta = verify_staged_checkpoint(idx_path, verify_markers=True)
    assert staged_meta["checkpoint_kind"] == "full"
    assert staged_meta["num_shards"] == full_meta["num_shards"]
    assert staged_meta["updates_completed"] == 1

    # Reload into a fresh trainer and confirm simulator & model state match identically
    reloaded = V3PPOTrainer(cfg, device="cpu")
    load_info = reloaded.load_checkpoint(idx_path)
    assert load_info["checkpoint_kind"] == "full"
    assert load_info["worlds_reinitialized"] is False
    assert load_info["num_shards"] >= 2
    for name, orig_t in raw_sd["tensors"].items():
        assert torch.equal(reloaded.sim.state_dict()["tensors"][name], orig_t)

    # Refuses overwrite on create-only LIGHT checkpoint path
    light_p = sharded_dir / "light_u0000001.pt"
    assert verify_light_checkpoint(light_p)["updates_completed"] == 1
    with pytest.raises(FileExistsError):
        trainer.save_light_checkpoint(light_p)

    # Refuses mismatched immutable config keys on load
    wrong_cfg = _tiny_cfg(num_envs=4, total_updates=2, eval_every_updates=0, learner_slots=3)
    other_trainer = V3PPOTrainer(wrong_cfg, device="cpu")
    with pytest.raises(ValueError, match="Checkpoint config mismatch"):
        other_trainer.load_checkpoint(idx_path)


def test_corrupted_or_missing_shard_falls_back_to_older_full_or_light(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Corrupting/removing a shard skips broken FULL sets and falls back to older FULL or LIGHT."""
    cfg = _tiny_cfg(num_envs=4, total_updates=3, eval_every_updates=0)
    trainer = V3PPOTrainer(cfg, device="cpu")
    ckpt_dir = tmp_path / "ckpts"

    # Save FULL at u=1 (forced multi-shard)
    trainer.train_step()
    probe_idx = trainer.save_full_checkpoint(tmp_path / "probe", tag="u0000001")
    shard_cap = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size - 1024
    idx_u1 = trainer.save_full_checkpoint(ckpt_dir, tag="u0000001", max_shard_bytes=shard_cap)
    meta_u1 = verify_full_checkpoint_set(idx_u1)
    assert meta_u1["num_shards"] >= 2

    # Save LIGHT only at u=2
    trainer.train_step()
    trainer.save_light_checkpoint(ckpt_dir / "light_u0000002.pt")

    # Save FULL at u=3 (forced multi-shard)
    trainer.train_step()
    idx_u3 = trainer.save_full_checkpoint(ckpt_dir, tag="u0000003", max_shard_bytes=shard_cap)
    meta_u3 = verify_full_checkpoint_set(idx_u3)
    assert meta_u3["num_shards"] >= 2

    # 1. Intact directory resolves to newest complete FULL (`u=3`)
    best0 = resolve_resume_checkpoint(ckpt_dir)
    assert best0 is not None
    assert best0["kind"] == "full"
    assert best0["updates_completed"] == 3

    # 2. Corrupt 1 byte inside shard 1 of `u=3` -> skips `full_u0000003` & falls back to FULL `u=1`!
    victim_shard_u3 = meta_u3["shard_paths"][1]
    corrupted_bytes = bytearray(victim_shard_u3.read_bytes())
    corrupted_bytes[-1] ^= 0xFF
    victim_shard_u3.write_bytes(bytes(corrupted_bytes))

    best1 = resolve_resume_checkpoint(ckpt_dir)
    captured1 = capsys.readouterr().out
    assert "V3_PPO_SKIP_INCOMPLETE_FULL" in captured1
    assert best1 is not None
    assert best1["kind"] == "full"
    assert best1["updates_completed"] == 1

    # 3. Remove a shard from `u=1` too -> zero complete FULL sets remain; falls back to newest LIGHT
    #    (`light_u0000003.pt`, `u=3`) with `reason=only_light_available` and logs informational
    #    `"worlds reinitialised"` message on load (without error-like WARN/WARNING tags)!
    meta_u1["shard_paths"][0].unlink()
    best2 = resolve_resume_checkpoint(ckpt_dir, log_choice=True)
    choice_log = capsys.readouterr().out
    assert best2 is not None
    assert best2["kind"] == "light"
    assert best2["updates_completed"] == 3
    assert "reason=only_light_available" in choice_log
    assert "WARN" not in choice_log and "ERROR" not in choice_log

    fresh_trainer = V3PPOTrainer(cfg, device="cpu")
    fresh_trainer.episode_return_acc.fill_(99.0)
    load_light = fresh_trainer.load_checkpoint(best2["path"])
    captured2 = capsys.readouterr().out
    assert load_light["checkpoint_kind"] == "light"
    assert load_light["worlds_reinitialized"] is True
    assert "V3_PPO_LIGHT_RESUME " in captured2
    assert "worlds reinitialised" in captured2
    assert "WARNING" not in captured2 and "WARN_" not in captured2
    assert fresh_trainer.updates_completed == 3
    assert float(fresh_trainer.episode_return_acc.abs().sum().item()) == 0.0


def test_prune_old_checkpoints_protects_lights_referenced_by_retained_fulls(
    tmp_path: Path,
) -> None:
    """Retention pruning (`keep_full=2, keep_light=4`) keeps LIGHTs needed by retained FULLs."""
    cfg = _tiny_cfg(num_envs=2, total_updates=6, eval_every_updates=0)
    trainer = V3PPOTrainer(cfg, device="cpu")
    prune_dir = tmp_path / "prune_ckpts"
    for u in range(1, 7):
        t_tag = f"u{u:07d}"
        lp = prune_dir / f"light_{t_tag}.pt"
        trainer.updates_completed = u
        trainer.save_light_checkpoint(lp)
        if u in (1, 3, 5):
            trainer.save_full_checkpoint(prune_dir, tag=t_tag, light_path=lp)
    prune_old_checkpoints(prune_dir, keep_full=2, keep_light=4)
    remaining_full = sorted(p.name for p in prune_dir.glob("full_*.index.json"))
    assert remaining_full == ["full_u0000003.index.json", "full_u0000005.index.json"]
    # `light_u0000003.pt` & `light_u0000005.pt` stay protected because retained FULL sets use them
    assert (prune_dir / "light_u0000003.pt").is_file()
    assert (prune_dir / "light_u0000005.pt").is_file()
    assert not (prune_dir / "full_u0000001.index.json").exists()


def test_bit_identical_cpu_resume_on_multishard_full_with_clean_step_streak(tmp_path: Path) -> None:
    """Train k=3, save >1-shard FULL (non-zero clean_step_streak), resume k=3: bit-identical!"""
    cfg = _tiny_cfg(num_envs=4, total_updates=6, eval_every_updates=2, learner_slots=4, seed=314159)

    trainer_a = V3PPOTrainer(cfg, device="cpu")
    for _ in range(3):
        trainer_a.train_step()
        if trainer_a.updates_completed % cfg.eval_every_updates == 0:
            trainer_a.run_evaluation()

    # Explicitly set non-zero `clean_step_streak` to verify preservation across FULL resume!
    trainer_a.sim.clean_step_streak = 1234

    probe_idx = trainer_a.save_full_checkpoint(tmp_path / "probe", tag="u0000003")
    shard_cap = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size - 1024
    ckpt_dir = tmp_path / "resume_k3"
    full_idx_path = trainer_a.save_full_checkpoint(
        ckpt_dir, tag="u0000003", max_shard_bytes=shard_cap
    )
    assert verify_full_checkpoint_set(full_idx_path)["num_shards"] >= 2

    # Continue uninterrupted for k=3 more updates (updates 4, 5, 6 with eval at 4 and 6)
    metrics_a = []
    evals_a = []
    for _ in range(3):
        metrics_a.append(trainer_a.train_step())
        if trainer_a.updates_completed % cfg.eval_every_updates == 0:
            evals_a.append(trainer_a.run_evaluation())

    # Scramble all global RNG streams before constructing & restoring `trainer_b`
    random.seed(999999)
    np.random.seed(888888)
    torch.manual_seed(777777)
    _ = torch.randn(137)

    trainer_b = V3PPOTrainer(cfg, device="cpu")
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    load_res = trainer_b.load_checkpoint(full_idx_path)
    assert load_res["checkpoint_kind"] == "full"
    assert load_res["num_shards"] >= 2
    assert trainer_b.sim.clean_step_streak == 1234

    metrics_b = []
    evals_b = []
    for _ in range(3):
        metrics_b.append(trainer_b.train_step())
        if trainer_b.updates_completed % cfg.eval_every_updates == 0:
            evals_b.append(trainer_b.run_evaluation())

    # 1. Step scalar metrics match identically (excluding wall-clock timing keys)
    timing_keys = {
        "collect_seconds",
        "learn_seconds",
        "update_seconds",
        "agent_steps_per_sec",
        "learner_steps_per_sec",
    }
    for ma, mb in zip(metrics_a, metrics_b, strict=True):
        clean_a = {k: v for k, v in ma.items() if k not in timing_keys}
        clean_b = {k: v for k, v in mb.items() if k not in timing_keys}
        assert clean_a == clean_b

    # 2. Model weights & buffers match bit-for-bit
    sd_a = trainer_a.raw_model.state_dict()
    sd_b = trainer_b.raw_model.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"Model tensor mismatch on {k}"

    # 3. Optimizer moment tensors match bit-for-bit
    opt_a = trainer_a.optimizer.state_dict()["state"]
    opt_b = trainer_b.optimizer.state_dict()["state"]
    assert opt_a.keys() == opt_b.keys()
    for idx in opt_a:
        for sk in opt_a[idx]:
            va, vb = opt_a[idx][sk], opt_b[idx][sk]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"Optimizer tensor mismatch on param {idx}.{sk}"
            else:
                assert va == vb

    # 4. Simulator fixed-point state tensors & rebase counters match bit-for-bit
    sim_a = trainer_a.sim.state_dict()
    sim_b = trainer_b.sim.state_dict()
    assert sim_a["tick"] == sim_b["tick"]
    assert sim_a["subtick_offset"] == sim_b["subtick_offset"]
    assert sim_a["clean_step_streak"] == sim_b["clean_step_streak"]
    assert trainer_a.sim.clean_step_streak == trainer_b.sim.clean_step_streak
    assert trainer_b.sim.clean_step_streak == 1234 + 3 * cfg.rollout_len
    for t_name in sim_a["tensors"]:
        assert torch.equal(
            sim_a["tensors"][t_name], sim_b["tensors"][t_name]
        ), f"Sim tensor mismatch on {t_name}"

    # 5. Built-in evaluation metrics match identically
    for ea, eb in zip(evals_a, evals_b, strict=True):
        assert ea["combined"] == eb["combined"]
        for m_key in ("vs_food_seeking", "vs_random_safe"):
            assert ea["matchups"][m_key] == eb["matchups"][m_key]


def test_evaluate_policy_preserves_caller_rng_and_is_deterministic_after_load(
    tmp_path: Path,
) -> None:
    """Verify `evaluate_policy` leaves caller RNG untouched and matches after checkpoint reload."""
    cfg = _tiny_cfg(total_updates=1, eval_every_updates=1)
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.train_step()
    ckpt_dir = tmp_path / "eval_ckpt"
    full_idx = trainer.save_full_checkpoint(ckpt_dir, tag="u0000001")

    torch.manual_seed(424242)
    np.random.seed(424242)
    random.seed(424242)
    expected_next_rand = torch.rand(5)

    # Reset to same state, run evaluate_policy, and confirm next draw is untouched!
    torch.manual_seed(424242)
    np.random.seed(424242)
    random.seed(424242)
    ev1 = evaluate_policy(trainer.raw_model, cfg, device="cpu")
    actual_next_rand = torch.rand(5)
    assert torch.equal(actual_next_rand, expected_next_rand), "evaluate_policy mutated caller RNG!"

    # Load checkpoint into a fresh trainer after arbitrary RNG noise and re-evaluate
    torch.manual_seed(987654)
    _ = torch.randn(500)
    fresh = V3PPOTrainer(cfg, device="cpu")
    fresh.load_checkpoint(full_idx, restore_rng=False)
    ev2 = evaluate_policy(fresh.raw_model, cfg, device="cpu")

    assert ev1["combined"] == ev2["combined"]
    assert ev1["eval_burnin_ticks"] == cfg.eval_burnin_ticks
    assert ev1["eval_ticks"] == cfg.eval_ticks
    assert set(ev1["matchups"]) == {"vs_food_seeking", "vs_random_safe"}
    for m_key in ("vs_food_seeking", "vs_random_safe"):
        assert ev1["matchups"][m_key] == ev2["matchups"][m_key]
        row = ev1["matchups"][m_key]
        for req_metric in (
            "completed_lives",
            "censored_lives",
            "mean_censored_length",
            "mean_death_length",
            "median_death_length",
            "p95_death_length",
            "mean_lifetime_ticks",
            "median_lifetime_ticks",
            "p95_lifetime_ticks",
            "time_averaged_length",
            "mass_gained_per_1k_agent_steps",
            "net_length_delta_per_1k_agent_steps",
            "total_deaths",
            "deaths_per_1k_agent_steps",
            "kills",
            "kills_per_1k_agent_steps",
            "mean_completed_episode_return",
        ):
            assert req_metric in row


def test_evaluate_scripted_baseline_separates_censored_lives_from_deaths() -> None:
    """Verify window-censored survivors are never mixed into `mean_death_length`."""
    # In a huge R=600 arena with 6 snakes over 2 ticks, 0 deaths occur -> all lives are censored.
    censor_cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=600.0, num_snakes=6),
        num_envs=2,
        eval_envs=2,
        eval_burnin_ticks=0,
        eval_ticks=2,
        eval_learner_slots=1,
    )
    rs_censor = evaluate_scripted_baseline("random_safe", censor_cfg, device="cpu")["combined"]
    assert rs_censor["completed_lives"] == 0
    assert rs_censor["total_deaths"] == 0
    assert rs_censor["censored_lives"] > 0
    assert rs_censor["mean_death_length"] == 0.0
    assert rs_censor["mean_censored_length"] >= 10.0
    assert rs_censor["time_averaged_length"] >= 10.0


def test_combine_matchups_computes_exact_float64_time_averaged_length() -> None:
    """Verify `_combine_matchups` sums exact float64 `alive_len_sum` without rounding drift."""
    m_a = _EvalMatchupRaw(
        summary={"time_averaged_length": 13.3333},
        death_lengths=[13.0],
        lifetime_ticks=[3.0],
        completed_returns=[1.0],
        censored_lengths=[],
        acted_steps=3,
        mass_gained=3,
        net_len_delta=3,
        deaths=1,
        kills=0,
        alive_len_sum=40.0,  # 13.333333...
        alive_ticks=3,
    )
    m_b = _EvalMatchupRaw(
        summary={"time_averaged_length": 13.6667},
        death_lengths=[14.0],
        lifetime_ticks=[3.0],
        completed_returns=[2.0],
        censored_lengths=[],
        acted_steps=3,
        mass_gained=6,
        net_len_delta=6,
        deaths=1,
        kills=1,
        alive_len_sum=41.0,  # 13.666666... -> exact combined = 81 / 6 = 13.5 exactly!
        alive_ticks=3,
    )
    comb = _combine_matchups([m_a, m_b])
    assert comb["time_averaged_length"] == 13.5


def test_cooperative_stop_mid_update_discards_dirty_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cooperative stop mid-PPO-epoch rewinds half-stepped weights to last clean checkpoint."""
    cfg_stop = _tiny_cfg(num_envs=2, total_updates=3, eval_every_updates=0)
    cfg_stop = V3PPOConfig.from_dict(
        {**cfg_stop.to_dict(), "full_checkpoint_every_updates": 1, "ppo_epochs": 2}
    )
    out_stop = tmp_path / "mid_stop_out"
    work_stop = tmp_path / "mid_stop_work"

    # Capture clean update-1 weights & force stop mid-ppo_update on update 2 after 1 step
    u1_weights: dict[str, torch.Tensor] = {}
    orig_ppo_update = V3PPOTrainer.ppo_update

    def _mid_epoch_stop_ppo_update(
        self: V3PPOTrainer,
        *,
        should_stop=None,  # type: ignore[no-untyped-def]
    ) -> dict[str, float]:
        if self.updates_completed == 1:
            # Mutate model weights as if minibatch 1 stepped, then raise StopRequestedError!
            with torch.no_grad():
                for p in self.raw_model.parameters():
                    p.add_(42.0)
            raise StopRequestedError("Simulated signal stop after minibatch 1 optimizer.step()")
        return orig_ppo_update(self, should_stop=should_stop)

    def _capture_u1_hook(tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 2:
            for k, v in tr.raw_model.state_dict().items():
                u1_weights[k] = v.detach().cpu().clone()

    monkeypatch.setattr(V3PPOTrainer, "ppo_update", _mid_epoch_stop_ppo_update)
    res_stop = run_training_job(
        cfg=cfg_stop,
        output_dir=out_stop,
        work_dir=work_stop,
        device="cpu",
        step_hook=_capture_u1_hook,
    )
    monkeypatch.setattr(V3PPOTrainer, "ppo_update", orig_ppo_update)

    assert res_stop["aborted_mid_update"] is True
    assert res_stop["updates_completed"] == 1
    published_stop = resolve_resume_checkpoint(out_stop / "checkpoints")
    assert published_stop is not None
    assert published_stop["updates_completed"] == 1
    check_tr = V3PPOTrainer(cfg_stop, device="cpu")
    check_tr.load_checkpoint(published_stop["path"])
    for k, clean_t in u1_weights.items():
        assert torch.equal(
            check_tr.raw_model.state_dict()[k], clean_t
        ), f"Mid-update mutated parameter {k} leaked into exported checkpoint!"


def test_in_session_restart_recovers_from_checkpoint_and_stages_continuation(
    tmp_path: Path,
) -> None:
    """In-session restart recovers from clean checkpoint, and staged FULL set resumes cleanly."""
    cfg_rec = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0).to_dict(),
            "full_exports": "final",
            "checkpoint_every_updates": 2,
            "eval_every_minutes": 0.0001,  # also exercises wall-time eval cadence!
            "max_in_session_restarts": 2,
        }
    )
    out_rec = tmp_path / "rec_out"
    work_rec = tmp_path / "rec_work"
    injected = {"raised": False}

    def _transient_fault_hook(tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 3 and not injected["raised"]:
            injected["raised"] = True
            # Corrupt weights right before raising to prove restart reloads clean u=2 checkpoint!
            with torch.no_grad():
                for p in tr.raw_model.parameters():
                    p.fill_(float("nan"))
            raise RuntimeError("Transient recoverable runtime error at update 3")

    res_rec = run_training_job(
        cfg=cfg_rec,
        output_dir=out_rec,
        work_dir=work_rec,
        device="cpu",
        step_hook=_transient_fault_hook,
    )
    assert injected["raised"] is True
    assert res_rec["status"] == "PASS"
    assert res_rec["in_session_restarts"] == 1
    assert res_rec["updates_completed"] == 4
    assert res_rec["last_eval"] is not None
    verify_completed_artifact(out_rec / "config.json")
    verify_completed_artifact(out_rec / "train_result.json")
    verify_completed_artifact(out_rec / "checkpoints" / "latest_manifest.json")

    # Stage sharded FULL set via ledger `companion_files` and resume to update 6
    resolved_u4 = resolve_resume_checkpoint(out_rec / "checkpoints")
    assert resolved_u4 is not None
    assert resolved_u4["kind"] == "full"
    assert resolved_u4["updates_completed"] == 4

    in2 = tmp_path / "attempt2_in"
    in2.mkdir()
    idx_src = Path(resolved_u4["path"])
    idx_dst = in2 / idx_src.name
    idx_sha = write_create_only_artifact(idx_dst, lambda tmp: shutil.copyfile(idx_src, tmp))
    comp_list = []
    for c_src in resolved_u4["files"][1:]:
        c_p = Path(c_src)
        c_dst = in2 / c_p.name
        c_sha = write_create_only_artifact(c_dst, lambda tmp, s=c_p: shutil.copyfile(s, tmp))
        comp_list.append(
            {
                "relpath": c_p.name,
                "sha256": c_sha,
                "size_bytes": int(c_dst.stat().st_size),
            }
        )
    ledger_p = in2 / "verified-ledger.json"
    write_create_only_json(
        ledger_p,
        {
            "schema": "v3-ppo-verified-ledger/v1",
            "namespace": V3_PPO_NAMESPACE,
            "unit_kind": "checkpoint",
            "skip_verified": True,
            "resume_checkpoint": {
                "relpath": idx_dst.name,
                "checkpoint_kind": "full",
                "sha256": idx_sha,
                "size_bytes": int(idx_dst.stat().st_size),
                "companion_files": comp_list,
            },
            "verified_units": [],
        },
    )

    cfg_u6 = V3PPOConfig.from_dict(
        {**cfg_rec.to_dict(), "total_updates": 6, "eval_every_minutes": 0.0}
    )
    out_u6 = tmp_path / "attempt2_out"
    res_u6 = run_training_job(
        cfg=cfg_u6,
        output_dir=out_u6,
        work_dir=tmp_path / "w2",
        device="cpu",
        input_dir=in2,
        ledger_path=ledger_p,
    )
    assert res_u6["status"] == "PASS"
    assert res_u6["updates_completed"] == 6
    assert res_u6["updates_this_session"] == 2
    assert res_u6["resumed_from"]["checkpoint_kind"] == "full"
    assert res_u6["resumed_from"]["worlds_reinitialized"] is False
    assert res_u6["resumed_from"]["updates_completed_at_resume"] == 4

    assert hasattr(ppo_mod, "V3PPOTrainer") and hasattr(V3PPOTrainer, "save_full_checkpoint")
    if torch.backends.mps.is_available():
        mps_cfg = _tiny_cfg(num_envs=2, total_updates=1, eval_every_updates=1)
        mps_trainer = V3PPOTrainer(mps_cfg, device="mps")
        mps_step = mps_trainer.train_step()
        mps_eval = mps_trainer.run_evaluation()
        assert (
            mps_step["agent_steps"] == mps_cfg.num_envs * mps_cfg.num_snakes * mps_cfg.rollout_len
        )
        assert mps_eval["combined"]["learner_agent_steps"] >= 1


def test_unthrottled_light_exports_bounded_full_and_immutable_sha256(tmp_path: Path) -> None:
    """Every LIGHT interval exports, periodic FULL count obeys bound, & sha256 never mutates."""
    cfg = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=6, eval_every_updates=0).to_dict(),
            "full_exports": "periodic",
            "light_checkpoint_interval_sec": 0.04,
            "full_checkpoint_interval_sec": 0.11,
            "max_periodic_full_exports": 1,
            "keep_full_checkpoints": 1,
            "keep_light_checkpoints": 2,
        }
    )
    out_dir = tmp_path / "cadence_out"
    work_dir = tmp_path / "cadence_work"
    first_seen_sha256: dict[str, str] = {}
    first_seen_stat: dict[str, tuple[int, int, int, int]] = {}

    def _snapshot_completed_exports() -> None:
        if not out_dir.is_dir():
            return
        for marker in sorted(out_dir.rglob("*.complete")):
            artifact = Path(str(marker)[: -len(".complete")])
            if artifact.is_file():
                rel = artifact.relative_to(out_dir).as_posix()
                cur_sha = sha256_file(artifact)
                st = artifact.stat()
                cur_stat = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                if rel in first_seen_sha256:
                    assert cur_sha == first_seen_sha256[rel], f"Completed export mutated: {rel}"
                    assert cur_stat == first_seen_stat[rel], f"Export inode/ctime mutated: {rel}"
                else:
                    first_seen_sha256[rel] = cur_sha
                    first_seen_stat[rel] = cur_stat

    def _cadence_hook(_tr: V3PPOTrainer, _next_update: int) -> None:
        _snapshot_completed_exports()
        time.sleep(0.055)
        _snapshot_completed_exports()

    res = run_training_job(
        cfg=cfg,
        output_dir=out_dir,
        work_dir=work_dir,
        device="cpu",
        step_hook=_cadence_hook,
    )
    _snapshot_completed_exports()

    assert res["status"] == "PASS"
    # Every 0.04s interval across 6 updates (~0.055s+ each) exported a LIGHT (no 3-count cap!)
    exported_lights = sorted((out_dir / "checkpoints").glob("light_u*.pt"))
    assert (
        len(exported_lights) >= 5
    ), f"Expected >=5 unthrottled LIGHT exports, got {exported_lights}"
    assert res["periodic_light_exports_published"] >= 5

    # Periodic FULL count obeyed `max_periodic_full_exports = 1` (+ 1 final boundary stop FULL = 2)
    exported_fulls = sorted((out_dir / "checkpoints").glob("full_u*.index.json"))
    assert res["periodic_full_exports_published"] == 1
    assert len(exported_fulls) == 2

    # Retention pruned unshared (`st_nlink == 1`) `work/checkpoints/` (`keep_full=1, keep_light=2`)
    # while leaving hard-linked (`st_nlink > 1`) entries intact so `output/` `st_ctime_ns` stays
    # immutable!
    unshared_work_fulls = [
        p for p in (work_dir / "checkpoints").glob("full_u*.index.json") if p.stat().st_nlink <= 1
    ]
    unshared_work_lights = [
        p
        for p in (work_dir / "checkpoints").glob("light_u*.pt")
        if p.name != "light_init_u0000000.pt" and p.stat().st_nlink <= 1
    ]
    assert len(unshared_work_fulls) <= 1
    assert len(unshared_work_lights) <= 2

    # Every completed export recorded during the run still exists and has identical sha256 & stat!
    assert len(first_seen_sha256) >= 6
    for rel, orig_sha in first_seen_sha256.items():
        final_file = out_dir / rel
        assert final_file.is_file(), f"Completed export deleted: {rel}"
        verify_completed_artifact(final_file)
        assert sha256_file(final_file) == orig_sha
        st_f = final_file.stat()
        assert (st_f.st_ino, st_f.st_size, st_f.st_mtime_ns, st_f.st_ctime_ns) == first_seen_stat[
            rel
        ]


def test_in_session_restart_tags_generation_and_refuses_stale_light_on_full_save(
    tmp_path: Path,
) -> None:
    """In-session restart rewinds to FULL u=2 and tags post-restart checkpoints (`_r01`)."""
    cfg_gen = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0).to_dict(),
            "full_exports": "periodic",
            "checkpoint_every_updates": 1,
            "full_checkpoint_every_updates": 2,
            "light_resume_margin_minutes": 45.0,
            "max_in_session_restarts": 1,
        }
    )
    out_gen = tmp_path / "gen_restart_out"
    work_gen = tmp_path / "gen_restart_work"
    fault_state = {"injected": False}
    stale_u3_weights: dict[str, torch.Tensor] = {}
    perturbed_post_restart = {"done": False}

    def _post_restart_hook(tr: V3PPOTrainer, next_update: int) -> None:
        if not fault_state["injected"] and next_update == 4:
            # Pre-restart u=3 LIGHT is already saved & published to `out_gen/checkpoints/`;
            # transient fault on update 4 rewinds to FULL u=2 (since LIGHT u=3 is <45m newer).
            fault_state["injected"] = True
            assert (out_gen / "checkpoints" / "light_u0000003.pt").is_file()
            for k, v in tr.raw_model.state_dict().items():
                stale_u3_weights[k] = v.detach().cpu().clone()
            with torch.no_grad():
                for p in tr.raw_model.parameters():
                    p.add_(7.0)
            raise RuntimeError("Transient fault on update 4 after pre-restart light_u0000003.pt")
        if fault_state["injected"] and not perturbed_post_restart["done"] and next_update == 3:
            perturbed_post_restart["done"] = True
            # Request a FULL checkpoint at post-restart u=3 while pre-restart `light_u0000003.pt`
            # still sits in `out_gen/checkpoints/`.
            object.__setattr__(cfg_gen, "full_checkpoint_every_updates", 3)
            with torch.no_grad():
                next(iter(tr.raw_model.parameters())).view(-1)[0].add_(0.125)

    res_gen = run_training_job(
        cfg=cfg_gen,
        output_dir=out_gen,
        work_dir=work_gen,
        device="cpu",
        step_hook=_post_restart_hook,
    )

    assert res_gen["status"] == "PASS"
    assert res_gen["in_session_restarts"] == 1
    assert perturbed_post_restart["done"] is True
    assert (out_gen / "checkpoints" / "light_u0000003.pt").is_file()
    # Post-restart FULL at u=3 MUST point to `light_u0000003_r01.pt`, never `light_u0000003.pt`.
    r01_idx = out_gen / "checkpoints" / "full_u0000003_r01.index.json"
    assert r01_idx.is_file()
    r01_meta = verify_full_checkpoint_set(r01_idx)
    assert r01_meta["light_path"].name == "light_u0000003_r01.pt"
    assert r01_meta["restart_generation"] == 1
    check_r01 = V3PPOTrainer(cfg_gen, device="cpu")
    check_r01.load_checkpoint(r01_idx)
    first_param_key = next(iter(stale_u3_weights))
    assert not torch.equal(
        check_r01.raw_model.state_dict()[first_param_key], stale_u3_weights[first_param_key]
    ), "FULL checkpoint after restart erroneously reused pre-restart LIGHT weights!"

    # Explicit guard test: `save_full_checkpoint` refuses a `light_path` with mismatched generation
    stale_dir = tmp_path / "stale_guard"
    guard_tr = V3PPOTrainer(cfg_gen, device="cpu")
    guard_tr.train_step()
    guard_tr.restart_generation = 0
    stale_lp = stale_dir / "light_u0000001.pt"
    guard_tr.save_light_checkpoint(stale_lp)
    guard_tr.restart_generation = 1
    with pytest.raises(ValueError, match="Refusing to build FULL checkpoint"):
        guard_tr.save_full_checkpoint(stale_dir, tag="u0000001_r01", light_path=stale_lp)


def test_resume_prefers_full_until_light_exceeds_margin_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Resume prefers newest FULL unless LIGHT is newer by `> light_resume_margin_minutes`."""
    margin_dir = tmp_path / "margin_ckpts"
    tr = V3PPOTrainer(_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0), device="cpu")
    tr.train_step()
    tr.train_step()
    base_ts = 1_700_000_000.0
    monkeypatch.setattr(trainer_mod.time, "time", lambda: base_ts)
    tr.save_full_checkpoint(margin_dir, tag="u0000002")
    tr.train_step()
    tr.train_step()
    # 20 minutes newer (< default 45 min margin) -> chooses FULL u=2!
    monkeypatch.setattr(trainer_mod.time, "time", lambda: base_ts + 20.0 * 60.0)
    tr.save_light_checkpoint(margin_dir / "light_u0000004.pt")
    capsys.readouterr()

    chosen_within = resolve_resume_checkpoint(
        margin_dir, light_resume_margin_minutes=45.0, log_choice=True
    )
    out_within = capsys.readouterr().out
    assert chosen_within is not None
    assert chosen_within["kind"] == "full"
    assert chosen_within["updates_completed"] == 2
    assert "V3_PPO_RESUME_CHOICE context=resolve_dir chosen_kind=full" in out_within
    assert "reason=full_preferred_within_margin" in out_within

    # With 10 min margin (`20 min > 10 min`) -> chooses newer LIGHT u=4!
    chosen_exceeds = resolve_resume_checkpoint(
        margin_dir, light_resume_margin_minutes=10.0, log_choice=True
    )
    out_exceeds = capsys.readouterr().out
    assert chosen_exceeds is not None
    assert chosen_exceeds["kind"] == "light"
    assert chosen_exceeds["updates_completed"] == 4
    assert "V3_PPO_RESUME_CHOICE context=resolve_dir chosen_kind=light" in out_exceeds
    assert "reason=light_newer_beyond_margin" in out_exceeds


def test_finally_rollback_failure_still_writes_manifest_and_train_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protected rollback in `finally` writes `latest_manifest.json` & `train_result.json`."""
    cfg_fin = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=3, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "max_in_session_restarts": 0,
        }
    )
    out_fin = tmp_path / "finally_out"
    work_fin = tmp_path / "finally_work"
    orig_load_ckpt = V3PPOTrainer.load_checkpoint

    def _dead_cuda_load_ckpt(self: V3PPOTrainer, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA error: device-side assert triggered during finally rollback")

    def _boom_on_u2(_tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 2:
            monkeypatch.setattr(V3PPOTrainer, "load_checkpoint", _dead_cuda_load_ckpt)
            raise RuntimeError("Unrecoverable fault on update 2")

    with pytest.raises(RuntimeError, match="Unrecoverable fault on update 2"):
        run_training_job(
            cfg=cfg_fin,
            output_dir=out_fin,
            work_dir=work_fin,
            device="cpu",
            step_hook=_boom_on_u2,
        )
    monkeypatch.setattr(V3PPOTrainer, "load_checkpoint", orig_load_ckpt)
    out_log = capsys.readouterr().out
    assert "V3_PPO_FINALLY_WARN stage=rollback_load_checkpoint" in out_log
    verify_completed_artifact(out_fin / "checkpoints" / "latest_manifest.json")
    verify_completed_artifact(out_fin / "train_result.json")
    res_fin = json.loads((out_fin / "train_result.json").read_text(encoding="utf-8"))
    assert res_fin["status"] == "FAIL"


def test_update_based_periodic_checkpoints_obey_session_export_byte_cap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Update-based periodic checkpoints (`checkpoint_every_updates`) respect export byte cap."""
    probe_tr = V3PPOTrainer(
        _tiny_cfg(num_envs=2, total_updates=1, eval_every_updates=0), device="cpu"
    )
    probe_tr.train_step()
    probe_lp = tmp_path / "probe_light.pt"
    probe_tr.save_light_checkpoint(probe_lp)
    probe_light_bytes = probe_lp.stat().st_size

    cfg_cap = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=5, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "light_checkpoint_interval_sec": 3600.0,
            "full_checkpoint_interval_sec": 7200.0,
        }
    )
    out_cap = tmp_path / "cap_out"
    res_cap = run_training_job(
        cfg=cfg_cap,
        output_dir=out_cap,
        work_dir=tmp_path / "cap_work",
        device="cpu",
        max_session_export_bytes=int(probe_light_bytes * 3.2),  # 2 LIGHTs + 25% stop_reserve!
    )
    out_log = capsys.readouterr().out
    assert res_cap["status"] == "PASS"
    assert res_cap["periodic_light_exports_published"] == 2
    assert "V3_PPO_EXPORT_BUDGET_REACHED" in out_log


def test_periodic_evaluation_runs_single_sim_pass_and_leaves_training_bit_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Periodic evaluation constructs 1 simulator per eval and leaves training bit-identical."""
    sim_creations: list[int] = []
    real_eval_sim = eval_mod.GameV3Sim

    def _spy_eval_sim(*args, **kwargs):  # type: ignore[no-untyped-def]
        sim_creations.append(int(kwargs.get("num_envs", args[1] if len(args) > 1 else -1)))
        return real_eval_sim(*args, **kwargs)

    monkeypatch.setattr(eval_mod, "GameV3Sim", _spy_eval_sim)

    cfg_no_ev = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0, seed=90210).to_dict(),
            "full_exports": "final",
        }
    )
    cfg_with_ev = V3PPOConfig.from_dict(
        {
            **cfg_no_ev.to_dict(),
            "eval_every_updates": 2,
            "eval_envs": 4,  # 2 worlds vs food_seeking + 2 worlds vs random_safe in ONE pass!
            "eval_burnin_ticks": 4,
            "eval_ticks": 12,
        }
    )
    res_no_ev = run_training_job(
        cfg=cfg_no_ev,
        output_dir=tmp_path / "no_ev_out",
        work_dir=tmp_path / "no_ev_work",
        device="cpu",
    )
    capsys.readouterr()
    sim_creations.clear()

    res_with_ev = run_training_job(
        cfg=cfg_with_ev,
        output_dir=tmp_path / "with_ev_out",
        work_dir=tmp_path / "with_ev_work",
        device="cpu",
    )
    ev_stdout = capsys.readouterr().out

    assert res_no_ev["status"] == "PASS"
    assert res_with_ev["status"] == "PASS"
    assert res_with_ev["eval_runs_completed"] == 2
    # Strictly ONE GameV3Sim(num_envs=4) constructed per evaluation (2 evals => [4, 4], NOT 4 sims!)
    assert sim_creations == [4, 4]
    assert res_with_ev["last_eval"]["matchup_envs"] == {
        "vs_food_seeking": 2,
        "vs_random_safe": 2,
    }
    assert res_with_ev["agent_steps_per_sec_scope"] == "train_excl_eval"
    assert res_with_ev["eval_wall_seconds_total"] > 0.0
    assert res_with_ev["eval_wall_share"] > 0.0
    assert "sps_scope=train_excl_eval" in ev_stdout
    assert "eval_wall_s=" in ev_stdout
    assert "eval_wall_pct=" in ev_stdout

    # Verify training weights and simulator state at update 4 are BIT-IDENTICAL with vs without eval
    tr_a = V3PPOTrainer(cfg_no_ev, device="cpu")
    tr_b = V3PPOTrainer(cfg_with_ev, device="cpu")
    ckpt_a = resolve_resume_checkpoint(tmp_path / "no_ev_out" / "checkpoints")
    ckpt_b = resolve_resume_checkpoint(tmp_path / "with_ev_out" / "checkpoints")
    assert ckpt_a is not None and ckpt_b is not None
    tr_a.load_checkpoint(ckpt_a["path"])
    tr_b.load_checkpoint(ckpt_b["path"])
    for k, ta in tr_a.raw_model.state_dict().items():
        assert torch.equal(ta, tr_b.raw_model.state_dict()[k]), f"Eval perturbed weight {k}!"
    for t_name, sa in tr_a.sim.state_dict()["tensors"].items():
        assert torch.equal(
            sa, tr_b.sim.state_dict()["tensors"][t_name]
        ), f"Eval perturbed {t_name}!"


def test_evaluation_stop_signal_aborts_cleanly_without_logging_partial_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop signal polled during evaluation aborts cleanly and omits partial `eval/*.jsonl` rows."""
    base_cfg = _tiny_cfg(num_envs=2, total_updates=2, eval_every_updates=2, seed=90210)
    tr = V3PPOTrainer(base_cfg, device="cpu")

    poll_calls = {"n": 0}

    def _stop_after_2_polls() -> bool:
        poll_calls["n"] += 1
        return poll_calls["n"] >= 2

    aborted_ev = evaluate_policy(
        tr.raw_model,
        V3PPOConfig.from_dict(
            {
                **base_cfg.to_dict(),
                "eval_envs": 4,
                "eval_burnin_ticks": 40,
                "eval_ticks": 200,
            }
        ),
        device="cpu",
        should_stop=_stop_after_2_polls,
        poll_interval_ticks=8,
    )
    assert aborted_ev["aborted"] is True
    assert aborted_ev["ticks_completed"] < 240

    # Verify mid-eval abort inside `run_training_job` is NOT recorded in `eval/*.jsonl`
    real_eval_pol = trainer_mod.evaluate_policy

    def _aborting_eval_pol(*args, **kwargs):  # type: ignore[no-untyped-def]
        return real_eval_pol(*args, **{**kwargs, "should_stop": lambda: True})

    monkeypatch.setattr(trainer_mod, "evaluate_policy", _aborting_eval_pol)
    out_abort = tmp_path / "abort_ev_out"
    res_abort = run_training_job(
        cfg=V3PPOConfig.from_dict(
            {
                **base_cfg.to_dict(),
                "eval_envs": 4,
                "eval_burnin_ticks": 4,
                "eval_ticks": 12,
            }
        ),
        output_dir=out_abort,
        work_dir=tmp_path / "abort_ev_work",
        device="cpu",
    )
    abort_stdout = capsys.readouterr().out
    monkeypatch.setattr(trainer_mod, "evaluate_policy", real_eval_pol)

    assert res_abort["eval_runs_aborted"] == 1
    assert res_abort["eval_runs_completed"] == 0
    assert res_abort["last_eval"] is None
    assert "V3_PPO_EVAL_ABORTED" in abort_stdout
    eval_dir = out_abort / "eval"
    if eval_dir.exists():
        for jf in eval_dir.glob("*.jsonl"):
            assert jf.read_text(encoding="utf-8").strip() == ""


def test_evaluation_skips_when_remaining_wall_below_twice_estimated_duration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remaining wall `< 2x` eval estimate skips starting evaluation (`V3_PPO_EVAL_SKIPPED`)."""
    cfg_skip = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=2, eval_every_updates=1, seed=90210).to_dict(),
            "eval_envs": 4,
            "eval_burnin_ticks": 4,
            "eval_ticks": 12,
            "eval_estimated_seconds": 60.0,  # 2x = 120s > remaining wall (~90s)
        }
    )
    res_skip = run_training_job(
        cfg=cfg_skip,
        output_dir=tmp_path / "skip_ev_out",
        work_dir=tmp_path / "skip_ev_work",
        device="cpu",
        max_wall_seconds=100.0,
    )
    skip_stdout = capsys.readouterr().out
    assert res_skip["status"] == "PASS"
    assert res_skip["eval_runs_skipped"] >= 1
    assert res_skip["eval_runs_completed"] == 0
    assert "V3_PPO_EVAL_SKIPPED" in skip_stdout


def test_light_exports_maintain_real_interval_after_periodic_full_cap_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capped `work/` FULL saves never reset or stretch periodic LIGHT export intervals."""
    import research.arena_ppo.runtime as arena_rt
    from research.colab_cli.runtime import _exports as scan_supervisor_exports

    light_interval = 30.0
    full_interval = 75.0
    cfg_cap = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=20, eval_every_updates=0, seed=7331).to_dict(),
            "full_exports": "periodic",
            "light_checkpoint_interval_sec": light_interval,
            "full_checkpoint_interval_sec": full_interval,
            "max_periodic_full_exports": 1,
        }
    )

    cli_root = tmp_path / "cli_root"
    out_dir = cli_root / "output"
    work_dir = cli_root / "work" / "v3_ppo"

    supervisor_cache: dict[str, object] = {}
    orig_fsync = arena_rt._fsync_directory

    def _hooked_fsync(directory: Path) -> None:
        orig_fsync(directory)
        if out_dir.exists():
            scan_supervisor_exports(cli_root, 1024**3, supervisor_cache)

    monkeypatch.setattr(arena_rt, "_fsync_directory", _hooked_fsync)

    # Deterministic 10.0s/update clock:
    # - u=3 (30s), u=6 (60s): periodic LIGHT
    # - u=8 (80s >= 75s): 1st periodic FULL (hits `max_periodic_full_exports=1`)
    # - u=11 (110s), u=14 (140s): periodic LIGHT
    # - u=16 (160s, 160-80=80 >= 75s): `due_full_t` saves FULL in `work/`, cap blocks export;
    #   `last_light_export_mono` stays at 140s so u=17 (170s) and u=20 (200s) export LIGHT on time!
    fake_clock = {"t": 1000.0}
    monkeypatch.setattr(job_mod.time, "monotonic", lambda: fake_clock["t"])

    periodic_events: list[tuple[str, int, float]] = []
    real_publish = job_mod.publish_checkpoint_to_output

    def _spy_publish(
        work_ckpt_path: Path,
        out_ckpts: Path,
        *,
        reason: str,
        counters: dict[str, object],
        **kwargs: object,
    ) -> list[str]:
        published = real_publish(
            work_ckpt_path,
            out_ckpts,
            reason=reason,
            counters=counters,
            **kwargs,  # type: ignore[arg-type]
        )
        if reason in {"periodic_full", "periodic_light"}:
            periodic_events.append(
                (reason, int(counters["updates_completed"]), fake_clock["t"] - 1000.0)
            )
        return published

    monkeypatch.setattr(job_mod, "publish_checkpoint_to_output", _spy_publish)

    def _advance_clock(_tr: V3PPOTrainer, _next_u: int) -> None:
        fake_clock["t"] += 10.0

    res = run_training_job(
        cfg=cfg_cap,
        output_dir=out_dir,
        work_dir=work_dir,
        device="cpu",
        step_hook=_advance_clock,
    )
    scan_supervisor_exports(cli_root, 1024**3, supervisor_cache)

    assert res["status"] == "PASS"
    assert res["periodic_full_exports_published"] == 1
    assert res["periodic_light_exports_published"] == 6
    assert periodic_events == [
        ("periodic_light", 3, 30.0),
        ("periodic_light", 6, 60.0),
        ("periodic_full", 8, 80.0),
        ("periodic_light", 11, 110.0),
        ("periodic_light", 14, 140.0),
        ("periodic_light", 17, 170.0),
        ("periodic_light", 20, 200.0),
    ]
    # Capped FULL save at u=16 MUST exist in `work_dir` while u=17 LIGHT still exported at +30.0s!
    assert (work_dir / "checkpoints" / "full_u0000016.index.json").is_file()
    assert not (out_dir / "checkpoints" / "full_u0000016.index.json").exists()
    times = [ev[2] for ev in periodic_events]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) <= light_interval


def test_restart_and_final_failure_record_exception_class_and_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify full traceback and exception class are logged and recorded on restart and failure."""
    cfg_ok = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=2, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "max_in_session_restarts": 2,
        }
    )
    raised_once = {"done": False}

    def _restart_once_hook(_tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 2 and not raised_once["done"]:
            raised_once["done"] = True
            raise RuntimeError("injected transient restart boom")

    res_restart = run_training_job(
        cfg=cfg_ok,
        output_dir=tmp_path / "restart_out",
        work_dir=tmp_path / "restart_work",
        device="cpu",
        step_hook=_restart_once_hook,
    )
    restart_log = capsys.readouterr().out
    assert "V3_PPO_IN_SESSION_RESTART attempt=1/2 error_class=RuntimeError" in restart_log
    assert "Traceback (most recent call last):" in restart_log
    assert "_restart_once_hook" in restart_log
    assert res_restart["status"] == "PASS"
    assert res_restart["error"] is None
    assert res_restart["error_class"] is None
    assert res_restart["error_traceback"] is None
    assert len(res_restart["in_session_restart_errors"]) == 1
    rec_err = res_restart["in_session_restart_errors"][0]
    assert rec_err["attempt"] == 1
    assert rec_err["failed_update"] == 2
    assert rec_err["error_class"] == "RuntimeError"
    assert rec_err["error"] == "injected transient restart boom"
    assert "Traceback (most recent call last):" in rec_err["traceback"]
    assert "_restart_once_hook" in rec_err["traceback"]

    # Now test final failure after 1 restart attempt
    cfg_fail = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=2, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "max_in_session_restarts": 1,
        }
    )

    def _always_fail_u2(_tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 2:
            raise KeyError("fatal_missing_key_u2")

    with pytest.raises(KeyError, match="fatal_missing_key_u2"):
        run_training_job(
            cfg=cfg_fail,
            output_dir=tmp_path / "fail_out",
            work_dir=tmp_path / "fail_work",
            device="cpu",
            step_hook=_always_fail_u2,
        )
    fail_log = capsys.readouterr().out
    assert "V3_PPO_FINAL_FAILURE error_class=KeyError" in fail_log
    assert "Traceback (most recent call last):" in fail_log
    assert "_always_fail_u2" in fail_log

    res_fail = json.loads((tmp_path / "fail_out" / "train_result.json").read_text(encoding="utf-8"))
    assert res_fail["status"] == "FAIL"
    assert res_fail["stopped_reason"] == "exception:KeyError"
    assert res_fail["error_class"] == "KeyError"
    assert "fatal_missing_key_u2" in res_fail["error"]
    assert "Traceback (most recent call last):" in res_fail["error_traceback"]
    assert "_always_fail_u2" in res_fail["error_traceback"]


def _tensor_sha256_hex(t: torch.Tensor) -> str:
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(f"{arr.dtype}:{arr.shape}:".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _snapshot_trainer_hashes(trainer: V3PPOTrainer) -> dict[str, object]:
    net_hashes = {
        k: _tensor_sha256_hex(v) for k, v in sorted(trainer.raw_model.state_dict().items())
    }
    sim_sd = trainer.sim.state_dict()
    sim_hashes = {k: _tensor_sha256_hex(v) for k, v in sorted(sim_sd["tensors"].items())}
    return {
        "updates_completed": int(trainer.updates_completed),
        "world_steps_collected": int(trainer.world_steps_collected),
        "agent_steps_collected": int(trainer.agent_steps_collected),
        "sim_tick": int(sim_sd["tick"]),
        "sim_subtick_offset": int(sim_sd.get("subtick_offset", 0)),
        "sim_clean_step_streak": int(trainer.sim.clean_step_streak),
        "net_state_sha256": net_hashes,
        "sim_state_sha256": sim_hashes,
    }


def test_e2865f93_checkpoints_resume_bit_identically_on_cpu(tmp_path: Path) -> None:
    """Checkpoints written by commit `e2865f93` load and resume bit-identically on CPU."""
    baseline_root = tmp_path / "e2865f93_root"
    baseline_root.mkdir(parents=True)
    (baseline_root / "src").symlink_to(ROOT / "src")
    (baseline_root / "research").mkdir()
    for sibling in (ROOT / "research").iterdir():
        if sibling.name != "v3_ppo":
            (baseline_root / "research" / sibling.name).symlink_to(sibling)

    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "e2865f93", "research/v3_ppo"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(baseline_root)],
        input=archive.stdout,
        check=True,
    )

    ckpt_dir = tmp_path / "e2865f93_ckpts"
    golden_json = tmp_path / "golden_e2865f93.json"
    gen_script = tmp_path / "gen_e2865f93.py"
    gen_script.write_text(
        """from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from research.game_v3.config import GameV3Config
from research.v3_ppo.config import V3PPOConfig
from research.v3_ppo.ppo import V3PPOTrainer, verify_full_checkpoint_set

def tensor_sha(t: torch.Tensor) -> str:
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(f"{arr.dtype}:{arr.shape}:".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()

def snap(tr: V3PPOTrainer) -> dict:
    sim_sd = tr.sim.state_dict()
    return {
        "updates_completed": int(tr.updates_completed),
        "world_steps_collected": int(tr.world_steps_collected),
        "agent_steps_collected": int(tr.agent_steps_collected),
        "sim_tick": int(sim_sd["tick"]),
        "sim_subtick_offset": int(sim_sd.get("subtick_offset", 0)),
        "sim_clean_step_streak": int(tr.sim.clean_step_streak),
        "net_state_sha256": {
            k: tensor_sha(v) for k, v in sorted(tr.raw_model.state_dict().items())
        },
        "sim_state_sha256": {k: tensor_sha(v) for k, v in sorted(sim_sd["tensors"].items())},
    }

def make_cfg() -> V3PPOConfig:
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=6, ambient_food_density=0.02, respawn_delay=8)
    return V3PPOConfig(
        game_config=gcfg, num_envs=3, learner_slots=4, opponent_policy="food_seeking",
        net_size="S", amp=False, use_channels_last=False, rollout_len=4, ppo_epochs=2,
        minibatch_size=16, total_updates=5, eval_every_minutes=0.0, eval_every_updates=0,
        seed=20261111,
    )

ckpt_dir = Path(sys.argv[2])
probe_dir = ckpt_dir.parent / "probe"
cfg = make_cfg()
tr = V3PPOTrainer(cfg, device=torch.device("cpu"))
tr.train_step()
tr.train_step()
tr.sim.clean_step_streak = 77

probe_idx = tr.save_full_checkpoint(probe_dir, tag="u0000002")
one_size = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size
ckpt_dir.mkdir(parents=True, exist_ok=True)
full_idx = tr.save_full_checkpoint(ckpt_dir, tag="u0000002", max_shard_bytes=one_size - 1024)
assert verify_full_checkpoint_set(full_idx)["num_shards"] >= 2

k_updates = 3
tr_l = V3PPOTrainer(cfg, device=torch.device("cpu"))
tr_l.load_checkpoint(ckpt_dir / "light_u0000002.pt")
for _ in range(k_updates):
    tr_l.train_step()

tr_f = V3PPOTrainer(cfg, device=torch.device("cpu"))
tr_f.load_checkpoint(full_idx)
for _ in range(k_updates):
    tr_f.train_step()

Path(sys.argv[3]).write_text(
    json.dumps({"k_updates": k_updates, "light": snap(tr_l), "full": snap(tr_f)}, sort_keys=True),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )

    sub_env = {
        **os.environ,
        "PYTHONPATH": str(baseline_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    subprocess.run(
        [sys.executable, str(gen_script), str(baseline_root), str(ckpt_dir), str(golden_json)],
        env=sub_env,
        check=True,
    )
    golden = json.loads(golden_json.read_text(encoding="utf-8"))
    k_updates = int(golden["k_updates"])

    cfg = V3PPOConfig(
        game_config=GameV3Config(
            arena_radius=60.0, num_snakes=6, ambient_food_density=0.02, respawn_delay=8
        ),
        num_envs=3,
        learner_slots=4,
        opponent_policy="food_seeking",
        net_size="S",
        amp=False,
        use_channels_last=False,
        rollout_len=4,
        ppo_epochs=2,
        minibatch_size=16,
        total_updates=5,
        eval_every_minutes=0.0,
        eval_every_updates=0,
        seed=20261111,
    )

    new_light = V3PPOTrainer(cfg, device=torch.device("cpu"))
    meta_light = new_light.load_checkpoint(ckpt_dir / "light_u0000002.pt")
    assert meta_light["checkpoint_kind"] == "light"
    for _ in range(k_updates):
        new_light.train_step()
    assert _snapshot_trainer_hashes(new_light) == golden["light"]

    new_full = V3PPOTrainer(cfg, device=torch.device("cpu"))
    meta_full = new_full.load_checkpoint(ckpt_dir / "full_u0000002.index.json")
    assert meta_full["checkpoint_kind"] == "full"
    assert meta_full["num_shards"] == 2
    for _ in range(k_updates):
        new_full.train_step()
    assert _snapshot_trainer_hashes(new_full) == golden["full"]


def test_final_light_completes_in_output_before_final_full_shards_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final `LIGHT` must be `.complete` in `output/` before final `FULL` shards start building."""
    probe_cfg = _tiny_cfg(num_envs=4, total_updates=2, eval_every_updates=0)
    probe_tr = V3PPOTrainer(probe_cfg, device="cpu")
    probe_tr.train_step()
    probe_idx = probe_tr.save_full_checkpoint(tmp_path / "probe_1shard", tag="u0000001")
    one_shard_size = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size
    multi_shard_cap = one_shard_size - 1024

    real_publish_sidecars = exports_mod._publish_sidecars_for_existing_payload
    real_save_full = V3PPOTrainer.save_full_checkpoint
    real_monotonic = job_mod.time.monotonic

    for case_name, expected_reason, expected_tag, total_updates, full_every_u, max_wall_s in (
        ("wall_guard_coincident_periodic_full", "wall_clock_guard", "u0000001", 4, 1, 20.0),
        ("completed_multi_shard", "completed", "u0000002", 2, 0, None),
    ):
        case_root = tmp_path / case_name
        out_dir = case_root / "out"
        work_dir = case_root / "work"
        out_ckpts = out_dir / "checkpoints"
        work_ckpts = work_dir / "checkpoints"

        completed_targets: list[Path] = []
        full_save_calls: list[str] = []

        def _spy_sidecars(
            target: Path,
            digest: str,
            _sink: list[Path] = completed_targets,
        ) -> None:
            real_publish_sidecars(target, digest)
            _sink.append(Path(target))

        def _spy_save_full(
            self_tr: V3PPOTrainer,
            checkpoints_dir: Path,
            *,
            tag: str | None = None,
            _exp_tag: str = expected_tag,
            _out_ckpts: Path = out_ckpts,
            _work_ckpts: Path = work_ckpts,
            _calls: list[str] = full_save_calls,
            **kwargs: object,
        ) -> Path:
            eff_tag = tag if tag is not None else f"u{self_tr.updates_completed:07d}"
            _calls.append(eff_tag)
            if eff_tag == _exp_tag:
                out_light = _out_ckpts / f"light_{_exp_tag}.pt"
                assert out_light.is_file(), f"Missing exported {out_light} before FULL shard build"
                assert Path(f"{out_light}.sha256").is_file()
                assert Path(f"{out_light}.complete").is_file()
                verify_completed_artifact(out_light)
                verify_light_checkpoint(out_light, require_markers=True)
                assert list(_work_ckpts.glob(f"full_{_exp_tag}*")) == []
                assert list(_out_ckpts.glob(f"full_{_exp_tag}*")) == []
            return real_save_full(
                self_tr,
                checkpoints_dir,
                tag=tag,
                **kwargs,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(exports_mod, "_publish_sidecars_for_existing_payload", _spy_sidecars)
        monkeypatch.setattr(V3PPOTrainer, "save_full_checkpoint", _spy_save_full)

        fake_clock = {"t": 1000.0}
        if max_wall_s is not None:
            monkeypatch.setattr(job_mod.time, "monotonic", lambda _fc=fake_clock: _fc["t"])
        else:
            monkeypatch.setattr(job_mod.time, "monotonic", real_monotonic)

        def _step_hook(
            _tr: V3PPOTrainer,
            _next_u: int,
            _fc: dict[str, float] = fake_clock,
            _use_wall: bool = (max_wall_s is not None),
        ) -> None:
            if _use_wall:
                _fc["t"] += 16.0

        cfg_case = V3PPOConfig.from_dict(
            {
                **_tiny_cfg(
                    num_envs=4,
                    total_updates=total_updates,
                    eval_every_updates=0,
                    checkpoint_shard_bytes=multi_shard_cap,
                ).to_dict(),
                "full_exports": "periodic" if full_every_u > 0 else "final",
                "checkpoint_every_updates": 0,
                "full_checkpoint_every_updates": full_every_u,
                "light_checkpoint_interval_sec": 3600.0,
                "full_checkpoint_interval_sec": 7200.0,
            }
        )
        res = run_training_job(
            cfg=cfg_case,
            output_dir=out_dir,
            work_dir=work_dir,
            device="cpu",
            max_wall_seconds=max_wall_s,
            step_hook=_step_hook,
        )
        assert res["status"] == "PASS"
        assert res["stopped_reason"] == expected_reason
        assert full_save_calls == [expected_tag]

        full_meta = verify_full_checkpoint_set(
            out_ckpts / f"full_{expected_tag}.index.json",
            require_markers=True,
        )
        assert full_meta["num_shards"] >= 2

        out_completed_names = [
            p.name for p in completed_targets if p.resolve().parent == out_ckpts.resolve()
        ]
        light_pos = out_completed_names.index(f"light_{expected_tag}.pt")
        shard_positions = [out_completed_names.index(sp.name) for sp in full_meta["shard_paths"]]
        index_pos = out_completed_names.index(f"full_{expected_tag}.index.json")
        assert all(
            light_pos < s_pos for s_pos in shard_positions
        ), f"Expected light_{expected_tag}.pt before shards in {out_completed_names}"
        assert max(shard_positions) < index_pos

        # Simulate supervisor timeout mid-FULL download (LIGHT + shard 0 arrived; shard 1 & index
        # missing): final weights in `light_<tag>.pt` are already complete and resumable.
        partial_dl = case_root / "partial_download"
        partial_dl.mkdir(parents=True, exist_ok=True)
        first_shard = full_meta["shard_paths"][0]
        for base_file in (out_ckpts / f"light_{expected_tag}.pt", first_shard):
            for suffix in ("", ".sha256", ".complete"):
                src_f = Path(f"{base_file}{suffix}")
                shutil.copy2(src_f, partial_dl / src_f.name)

        resolved = resolve_resume_checkpoint(partial_dl, require_markers=True)
        assert resolved is not None
        assert resolved["kind"] == "light"
        assert Path(resolved["path"]).name == f"light_{expected_tag}.pt"
        assert resolved["updates_completed"] == res["updates_completed"]

        reloaded = V3PPOTrainer(cfg_case, device="cpu")
        loaded_meta = reloaded.load_checkpoint(Path(resolved["path"]))
        assert loaded_meta["checkpoint_kind"] == "light"
        assert reloaded.updates_completed == res["updates_completed"]


def test_full_exports_modes_none_final_periodic_exported_filenames_and_work_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `none`, `final`, and `periodic` export exact expected filenames in tiny CPU runs."""
    assert V3PPOConfig().full_exports == "none"

    probe_cfg = _tiny_cfg(num_envs=4, total_updates=4, eval_every_updates=0)
    probe_tr = V3PPOTrainer(probe_cfg, device="cpu")
    for _ in range(4):
        probe_tr.train_step()
    probe_idx = probe_tr.save_full_checkpoint(tmp_path / "probe_modes", tag="u0000004")
    one_shard_bytes = verify_full_checkpoint_set(probe_idx)["shard_paths"][0].stat().st_size
    multi_shard_cap = one_shard_bytes - 1024

    base_dict = {
        **_tiny_cfg(
            num_envs=4,
            total_updates=4,
            eval_every_updates=0,
            checkpoint_shard_bytes=multi_shard_cap,
        ).to_dict(),
        "checkpoint_every_updates": 1,
        "full_checkpoint_every_updates": 2,
        "light_checkpoint_interval_sec": 3600.0,
        "full_checkpoint_interval_sec": 7200.0,
        "max_in_session_restarts": 1,
    }

    def _payload_names(ckpts_dir: Path) -> list[str]:
        return sorted(
            p.name
            for p in ckpts_dir.iterdir()
            if p.is_file() and not p.name.endswith((".sha256", ".complete", ".meta.json"))
        )

    # 1. `none` (default): NO periodic FULL and NO final FULL under `output/`, while `work/` FULL
    #    checkpoints are still written and used by in-session restart without world re-init!
    cfg_none = V3PPOConfig.from_dict({**base_dict, "full_exports": "none"})
    out_none = tmp_path / "out_none"
    work_none = tmp_path / "work_none"
    fault_once = {"fired": False}

    def _fault_at_u3_hook(_tr: V3PPOTrainer, next_u: int) -> None:
        if next_u == 3 and not fault_once["fired"]:
            fault_once["fired"] = True
            raise RuntimeError("Transient fault at u=3 to verify work/ FULL restart in none mode")

    capsys.readouterr()
    res_none = run_training_job(
        cfg=cfg_none,
        output_dir=out_none,
        work_dir=work_none,
        device="cpu",
        step_hook=_fault_at_u3_hook,
    )
    none_stdout = capsys.readouterr().out
    assert res_none["status"] == "PASS"
    assert res_none["in_session_restarts"] == 1
    assert res_none["periodic_full_exports_published"] == 0
    assert res_none["periodic_light_exports_published"] == 4
    assert list(out_none.rglob("full_*")) == []
    assert _payload_names(out_none / "checkpoints") == [
        "latest_manifest.json",
        "light_u0000001.pt",
        "light_u0000002.pt",
        "light_u0000003_r01.pt",
        "light_u0000004_r01.pt",
    ]
    # `work/` FULL at u=2 WAS written and restored on in-session restart (mature worlds preserved!)
    assert (work_none / "checkpoints" / "full_u0000002.index.json").is_file()
    assert "V3_PPO_RESUME_CHOICE context=in_session_restart_1 chosen_kind=full" in none_stdout
    assert "worlds reinitialised" not in none_stdout

    # Also verify mid-update cooperative stop rollback in `none` mode NEVER exports `full_*`!
    orig_ppo_upd = V3PPOTrainer.ppo_update

    def _stop_mid_u3(self_tr: V3PPOTrainer, *, should_stop=None):  # type: ignore[no-untyped-def]
        if self_tr.updates_completed == 2:
            raise StopRequestedError("Cooperative stop mid-u3 after work/ saved full_u0000002")
        return orig_ppo_upd(self_tr, should_stop=should_stop)

    monkeypatch.setattr(V3PPOTrainer, "ppo_update", _stop_mid_u3)
    out_none_stop = tmp_path / "out_none_stop"
    res_none_stop = run_training_job(
        cfg=cfg_none,
        output_dir=out_none_stop,
        work_dir=tmp_path / "work_none_stop",
        device="cpu",
    )
    monkeypatch.setattr(V3PPOTrainer, "ppo_update", orig_ppo_upd)
    assert res_none_stop["aborted_mid_update"] is True
    assert res_none_stop["updates_completed"] == 2
    assert list(out_none_stop.rglob("full_*")) == []
    assert _payload_names(out_none_stop / "checkpoints") == [
        "latest_manifest.json",
        "light_u0000001.pt",
        "light_u0000002.pt",
    ]

    # 2. `final`: zero periodic FULL under `output/`, and exactly ONE final FULL set completed
    #    strictly after the final LIGHT (`light_u0000004.pt`).
    real_sidecars = exports_mod._publish_sidecars_for_existing_payload
    final_completion_order: list[str] = []
    out_final = tmp_path / "out_final"
    work_final = tmp_path / "work_final"

    def _record_final_sidecars(target: Path, digest: str) -> None:
        real_sidecars(target, digest)
        if target.resolve().parent == (out_final / "checkpoints").resolve():
            final_completion_order.append(target.name)

    monkeypatch.setattr(
        exports_mod, "_publish_sidecars_for_existing_payload", _record_final_sidecars
    )
    cfg_final = V3PPOConfig.from_dict({**base_dict, "full_exports": "final"})
    res_final = run_training_job(
        cfg=cfg_final,
        output_dir=out_final,
        work_dir=work_final,
        device="cpu",
    )
    monkeypatch.setattr(exports_mod, "_publish_sidecars_for_existing_payload", real_sidecars)
    assert res_final["status"] == "PASS"
    assert res_final["periodic_full_exports_published"] == 0
    assert res_final["periodic_light_exports_published"] == 4
    assert (work_final / "checkpoints" / "full_u0000002.index.json").is_file()
    assert list((out_final / "checkpoints").glob("full_u0000002*")) == []
    final_full_indices = sorted(
        p.name for p in (out_final / "checkpoints").glob("full_*.index.json")
    )
    assert final_full_indices == ["full_u0000004.index.json"]
    final_meta = verify_full_checkpoint_set(
        out_final / "checkpoints" / "full_u0000004.index.json", require_markers=True
    )
    assert final_meta["num_shards"] >= 2
    expected_final_shards = [p.name for p in final_meta["shard_paths"]]
    assert _payload_names(out_final / "checkpoints") == sorted(
        [
            "latest_manifest.json",
            "light_u0000001.pt",
            "light_u0000002.pt",
            "light_u0000003.pt",
            "light_u0000004.pt",
            "full_u0000004.index.json",
            *expected_final_shards,
        ]
    )
    pos_final_light = final_completion_order.index("light_u0000004.pt")
    pos_final_shards = [final_completion_order.index(s) for s in expected_final_shards]
    pos_final_idx = final_completion_order.index("full_u0000004.index.json")
    assert all(pos_final_light < p for p in pos_final_shards)
    assert max(pos_final_shards) < pos_final_idx

    # 3. `periodic`: periodic FULL (`u=2`) + final FULL (`u=4`) + all LIGHTs (`u=1..4`).
    cfg_periodic = V3PPOConfig.from_dict(
        {**base_dict, "full_exports": "periodic", "max_periodic_full_exports": 1}
    )
    out_periodic = tmp_path / "out_periodic"
    work_periodic = tmp_path / "work_periodic"
    res_periodic = run_training_job(
        cfg=cfg_periodic,
        output_dir=out_periodic,
        work_dir=work_periodic,
        device="cpu",
    )
    assert res_periodic["status"] == "PASS"
    assert res_periodic["periodic_full_exports_published"] == 1
    assert res_periodic["periodic_light_exports_published"] == 3
    periodic_full_indices = sorted(
        p.name for p in (out_periodic / "checkpoints").glob("full_*.index.json")
    )
    assert periodic_full_indices == ["full_u0000002.index.json", "full_u0000004.index.json"]
    u2_shards = [
        p.name
        for p in verify_full_checkpoint_set(
            out_periodic / "checkpoints" / "full_u0000002.index.json", require_markers=True
        )["shard_paths"]
    ]
    u4_shards = [
        p.name
        for p in verify_full_checkpoint_set(
            out_periodic / "checkpoints" / "full_u0000004.index.json", require_markers=True
        )["shard_paths"]
    ]
    assert _payload_names(out_periodic / "checkpoints") == sorted(
        [
            "latest_manifest.json",
            "light_u0000001.pt",
            "light_u0000002.pt",
            "light_u0000003.pt",
            "light_u0000004.pt",
            "full_u0000002.index.json",
            "full_u0000004.index.json",
            *u2_shards,
            *u4_shards,
        ]
    )


def test_continuation_recipe_change_refusal_allowed_history_and_optimizer_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test continuation refusal by default, `--allow-recipe-change`, history, and AdamW state."""
    base_cfg = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(
                num_envs=4, learner_slots=4, total_updates=2, eval_every_updates=0
            ).to_dict(),
            "checkpoint_every_updates": 2,
            "full_checkpoint_every_updates": 0,
            "light_checkpoint_interval_sec": 3600.0,
        }
    )
    out1 = tmp_path / "run1_out"
    res1 = run_training_job(
        cfg=base_cfg,
        output_dir=out1,
        work_dir=tmp_path / "run1_work",
        device="cpu",
    )
    assert res1["status"] == "PASS"
    assert res1["updates_completed"] == 2
    ckpt1_path = out1 / "checkpoints" / "light_u0000002.pt"
    ckpt1_raw = torch.load(ckpt1_path, map_location="cpu", weights_only=False)
    step_u2 = int(next(iter(ckpt1_raw["optimizer_state_dict"]["state"].values()))["step"])
    assert step_u2 > 16  # > max possible steps in 2 updates at rollout_len=4 (<= 16)

    # 1. Default continuation refuses recipe changes in both `prepare` and `run_training_job`.
    base_cfg_u4 = V3PPOConfig.from_dict({**base_cfg.to_dict(), "total_updates": 4})
    with pytest.raises(ValueError, match="Continuation checkpoint config mismatch on 'gamma'"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_refused_default",
            mode="train",
            gpu="L4",
            output_dir=tmp_path / "remote_refused_default",
            prior_output_dir=out1,
            allow_missing_canary_receipt=True,
            allow_unmerged_commit=True,
            ppo_config=base_cfg_u4,
            explicit_total_updates=True,
            recipe_json='{"gamma": 0.995, "reward_alive_length_coef": 0.0}',
            commit="HEAD",
        )

    changed_cfg = V3PPOConfig.from_dict(
        {
            **base_cfg_u4.to_dict(),
            "reward_alive_length_coef": 0.0,
            "reward_death_penalty": 2.0,
            "gamma": 0.995,
            "lr": 1.0e-4,
            "rollout_len": 4,
        }
    )
    probe_tr = V3PPOTrainer(changed_cfg, device="cpu")
    with pytest.raises(ValueError, match="Checkpoint config mismatch on 'gamma'"):
        probe_tr.load_checkpoint(ckpt1_path, allow_recipe_change=False)

    # 2. Even with `--allow-recipe-change`, non-allow-listed fields (`num_envs`, `value_coef`) fail.
    with pytest.raises(ValueError, match="Continuation checkpoint config mismatch on 'num_envs'"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_refused_num_envs",
            mode="train",
            gpu="L4",
            output_dir=tmp_path / "remote_refused_num_envs",
            prior_output_dir=out1,
            allow_missing_canary_receipt=True,
            allow_unmerged_commit=True,
            num_envs=6,
            ppo_config=base_cfg_u4,
            explicit_total_updates=True,
            allow_recipe_change=True,
            commit="HEAD",
        )
    bad_non_recipe_cfg = V3PPOConfig.from_dict({**base_cfg_u4.to_dict(), "value_coef": 0.25})
    with pytest.raises(ValueError, match="Continuation checkpoint config mismatch on 'value_coef'"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_refused_value_coef",
            mode="train",
            gpu="L4",
            output_dir=tmp_path / "remote_refused_value_coef",
            prior_output_dir=out1,
            allow_missing_canary_receipt=True,
            allow_unmerged_commit=True,
            ppo_config=bad_non_recipe_cfg,
            explicit_total_updates=True,
            allow_recipe_change=True,
            commit="HEAD",
        )

    # 3. Direct `load_checkpoint(..., allow_recipe_change=True)` keeps AdamW moments and updates lr.
    loaded_meta = probe_tr.load_checkpoint(ckpt1_path, allow_recipe_change=True)
    assert loaded_meta["recipe_change"] == {
        "update": 2,
        "changed_fields": {
            "gamma": {"old": 0.99, "new": 0.995},
            "lr": {"old": 3.0e-4, "new": 1.0e-4},
            "reward_alive_length_coef": {"old": 0.001, "new": 0.0},
            "reward_death_penalty": {"old": 1.0, "new": 2.0},
            "rollout_len": {"old": 6, "new": 4},
        },
    }
    assert probe_tr.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    first_state = next(iter(probe_tr.optimizer.state.values()))
    assert int(first_state["step"]) == step_u2
    assert float(first_state["exp_avg"].abs().sum().item()) > 0.0
    assert float(first_state["exp_avg_sq"].abs().sum().item()) > 0.0

    # 4. End-to-end `prepare --allow-recipe-change` + tiny CPU continuation run 2 (`u=2..4`).
    pkt2_dir = tmp_path / "pkt_allowed_hop2"
    summary2 = prepare_v3_ppo_packet(
        pkt2_dir,
        mode="train",
        gpu="L4",
        output_dir=tmp_path / "remote_allowed_hop2",
        prior_output_dir=out1,
        allow_missing_canary_receipt=True,
        allow_unmerged_commit=True,
        ppo_config=base_cfg_u4,
        explicit_total_updates=True,
        recipe_json=(
            '{"reward_alive_length_coef": 0.0, "reward_death_penalty": 2.0, '
            '"gamma": 0.995, "lr": 0.0001, "rollout_len": 4}'
        ),
        allow_recipe_change=True,
        commit="HEAD",
    )
    assert summary2["allow_recipe_change"] is True
    assert summary2["recipe_change"] == loaded_meta["recipe_change"]
    manifest2 = json.loads(Path(summary2["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest2["env"]["V3_PPO_ALLOW_RECIPE_CHANGE"] == "1"
    staged_cfg2 = V3PPOConfig.from_dict(json.loads(manifest2["env"]["V3_PPO_CONFIG_JSON"]))
    assert list(staged_cfg2.recipe_change_history) == [loaded_meta["recipe_change"]]

    capsys.readouterr()
    out2 = tmp_path / "run2_out"
    res2 = run_training_job(
        cfg=staged_cfg2,
        output_dir=out2,
        work_dir=tmp_path / "run2_work",
        device="cpu",
        input_dir=pkt2_dir / "inputs",
        ledger_path=pkt2_dir / "inputs" / "verified-ledger.json",
        allow_recipe_change=True,
    )
    stdout2 = capsys.readouterr().out
    assert res2["status"] == "PASS"
    assert res2["updates_completed"] == 4
    assert "V3_PPO_RECIPE_CHANGE old=" in stdout2
    assert " update=2" in stdout2

    expected_hist2 = [loaded_meta["recipe_change"]]
    out2_cfg_json = json.loads((out2 / "config.json").read_text(encoding="utf-8"))
    assert out2_cfg_json["recipe_change_history"] == expected_hist2
    assert res2["config"]["recipe_change_history"] == expected_hist2

    ckpt2_path = out2 / "checkpoints" / "light_u0000004.pt"
    ckpt2_raw = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
    assert ckpt2_raw["recipe_change_history"] == expected_hist2
    assert ckpt2_raw["config"]["recipe_change_history"] == expected_hist2
    assert ckpt2_raw["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(1.0e-4)
    step_u4 = int(next(iter(ckpt2_raw["optimizer_state_dict"]["state"].values()))["step"])
    # Run 2 alone can take at most (4*4*4 / 16) * 2 * 2 == 16 optimizer steps; step_u4 > step_u2 > 16
    # proves AdamW step/moment buffers from run 1 were preserved across the recipe change!
    assert step_u4 > step_u2

    # 5. Multi-hop: from `out2`, a new recipe change WITHOUT `--allow-recipe-change` is still refused,
    #    same-recipe continuation preserves history, and a second `--allow-recipe-change` appends!
    staged_cfg2_u6 = V3PPOConfig.from_dict({**staged_cfg2.to_dict(), "total_updates": 6})
    with pytest.raises(
        ValueError, match="Continuation checkpoint config mismatch on 'entropy_coef'"
    ):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_hop3_refused",
            mode="train",
            gpu="L4",
            output_dir=tmp_path / "remote_hop3_refused",
            prior_output_dir=out2,
            allow_missing_canary_receipt=True,
            allow_unmerged_commit=True,
            ppo_config=staged_cfg2_u6,
            explicit_total_updates=True,
            recipe_json='{"entropy_coef": 0.002}',
            allow_recipe_change=False,
            commit="HEAD",
        )

    pkt3_same = tmp_path / "pkt_hop3_same"
    summary3_same = prepare_v3_ppo_packet(
        pkt3_same,
        mode="train",
        gpu="L4",
        output_dir=tmp_path / "remote_hop3_same",
        prior_output_dir=out2,
        allow_missing_canary_receipt=True,
        allow_unmerged_commit=True,
        ppo_config=staged_cfg2_u6,
        explicit_total_updates=True,
        allow_recipe_change=False,
        commit="HEAD",
    )
    manifest3_same = json.loads(Path(summary3_same["manifest_path"]).read_text(encoding="utf-8"))
    assert "V3_PPO_ALLOW_RECIPE_CHANGE" not in manifest3_same["env"]
    assert (
        json.loads(manifest3_same["env"]["V3_PPO_CONFIG_JSON"])["recipe_change_history"]
        == expected_hist2
    )

    pkt3_chg = tmp_path / "pkt_hop3_chg"
    summary3_chg = prepare_v3_ppo_packet(
        pkt3_chg,
        mode="train",
        gpu="L4",
        output_dir=tmp_path / "remote_hop3_chg",
        prior_output_dir=out2,
        allow_missing_canary_receipt=True,
        allow_unmerged_commit=True,
        ppo_config=staged_cfg2_u6,
        explicit_total_updates=True,
        recipe_json='{"entropy_coef": 0.002}',
        allow_recipe_change=True,
        commit="HEAD",
    )
    manifest3_chg = json.loads(Path(summary3_chg["manifest_path"]).read_text(encoding="utf-8"))
    staged_cfg3 = V3PPOConfig.from_dict(json.loads(manifest3_chg["env"]["V3_PPO_CONFIG_JSON"]))
    out3 = tmp_path / "run3_out"
    res3 = run_training_job(
        cfg=staged_cfg3,
        output_dir=out3,
        work_dir=tmp_path / "run3_work",
        device="cpu",
        input_dir=pkt3_chg / "inputs",
        ledger_path=pkt3_chg / "inputs" / "verified-ledger.json",
        allow_recipe_change=True,
    )
    assert res3["status"] == "PASS"
    assert res3["config"]["recipe_change_history"] == [
        loaded_meta["recipe_change"],
        {
            "update": 4,
            "changed_fields": {
                "entropy_coef": {"old": 0.01, "new": 0.002},
            },
        },
    ]


def test_evaluation_metrics_invariant_across_reward_recipes_except_episode_return() -> None:
    """Confirm built-in eval length/death/kill/growth metrics do not depend on reward recipe."""
    gcfg = GameV3Config(
        arena_radius=55.0,
        num_snakes=8,
        ambient_food_density=0.03,
        respawn_delay=6,
    )
    cfg_recipe_a = V3PPOConfig(
        game_config=gcfg,
        num_envs=4,
        learner_slots=4,
        net_size="S",
        amp=False,
        use_channels_last=False,
        eval_envs=8,
        eval_burnin_ticks=20,
        eval_ticks=80,
        eval_learner_slots=4,
        eval_seed=424242,
        reward_alive_length_coef=0.001,
        reward_alive_const=0.0,
        reward_length_delta_coef=0.1,
        reward_kill_coef=0.5,
        reward_death_penalty=1.0,
        gamma=0.99,
    )
    cfg_recipe_b = V3PPOConfig(
        game_config=gcfg,
        num_envs=4,
        learner_slots=4,
        net_size="S",
        amp=False,
        use_channels_last=False,
        eval_envs=8,
        eval_burnin_ticks=20,
        eval_ticks=80,
        eval_learner_slots=4,
        eval_seed=424242,
        reward_alive_length_coef=0.0,
        reward_alive_const=0.25,
        reward_length_delta_coef=0.5,
        reward_kill_coef=2.5,
        reward_death_penalty=4.0,
        gamma=0.997,
        gae_lambda=0.98,
        entropy_coef=0.002,
    )

    tr = V3PPOTrainer(cfg_recipe_a, device="cpu")
    eval_a = evaluate_policy(tr.raw_model, cfg_recipe_a, device="cpu")
    eval_b = evaluate_policy(tr.raw_model, cfg_recipe_b, device="cpu")

    assert eval_a["aborted"] is False
    assert eval_b["aborted"] is False
    assert eval_a["combined"]["total_deaths"] > 0
    assert eval_a["combined"]["mass_gained"] > 0

    def _strip_recipe_dependent(block: dict[str, object]) -> dict[str, object]:
        return {k: v for k, v in block.items() if k != "mean_completed_episode_return"}

    assert _strip_recipe_dependent(eval_a["combined"]) == _strip_recipe_dependent(
        eval_b["combined"]
    )
    for suite_key in ("vs_food_seeking", "vs_random_safe"):
        assert _strip_recipe_dependent(eval_a["matchups"][suite_key]) == _strip_recipe_dependent(
            eval_b["matchups"][suite_key]
        )

    assert (
        eval_a["combined"]["mean_completed_episode_return"]
        != eval_b["combined"]["mean_completed_episode_return"]
    )


def test_periodic_mode_wall_guard_race_during_checkpoint_never_publishes_full_before_final_light(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing one stop decision per update prevents a due FULL from racing ahead of final LIGHT."""
    fake_clock = {"t": 1000.0}
    monkeypatch.setattr(job_mod.time, "monotonic", lambda: fake_clock["t"])

    real_save_light = V3PPOTrainer.save_light_checkpoint
    real_save_full = V3PPOTrainer.save_full_checkpoint
    out_dir = tmp_path / "race_out"
    out_ckpts = out_dir / "checkpoints"
    full_tags_built: list[str] = []

    def _spy_save_light(
        self_tr: V3PPOTrainer,
        path: Path,
        **kwargs: object,
    ) -> str:
        digest = real_save_light(self_tr, path, **kwargs)  # type: ignore[arg-type]
        # On u=1 (`t=1010 < 1015` before checkpointing), advance clock past `max_wall - reserve`
        # (`1020 >= 1015`) DURING checkpoint save! Because `post_stop_reason` was evaluated once
        # (`None`), the loop MUST NOT exit before u=2, so u=2 becomes the clean `final_boundary`.
        if self_tr.updates_completed == 1 and path.parent.parent.name != out_dir.name:
            fake_clock["t"] += 10.0
        return digest

    def _spy_save_full(
        self_tr: V3PPOTrainer,
        checkpoints_dir: Path,
        *,
        tag: str | None = None,
        **kwargs: object,
    ) -> Path:
        eff_tag = tag if tag is not None else f"u{self_tr.updates_completed:07d}"
        full_tags_built.append(eff_tag)
        if eff_tag == "u0000002":
            final_light = out_ckpts / "light_u0000002.pt"
            assert final_light.is_file()
            assert Path(f"{final_light}.complete").is_file()
            verify_light_checkpoint(final_light, require_markers=True)
        return real_save_full(self_tr, checkpoints_dir, tag=tag, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(V3PPOTrainer, "save_light_checkpoint", _spy_save_light)
    monkeypatch.setattr(V3PPOTrainer, "save_full_checkpoint", _spy_save_full)

    def _step_hook(_tr: V3PPOTrainer, next_u: int) -> None:
        if next_u == 1:
            fake_clock["t"] += 10.0  # t=1010 < 1015 (not expired yet at u=1 decision)
        else:
            fake_clock["t"] += 1.0  # t=1021 >= 1015 (expired at u=2 decision!)

    cfg_race = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0).to_dict(),
            "full_exports": "periodic",
            "max_periodic_full_exports": 2,
            "checkpoint_every_updates": 0,
            "full_checkpoint_every_updates": 1,
            "light_checkpoint_interval_sec": 3600.0,
            "full_checkpoint_interval_sec": 7200.0,
        }
    )
    res = run_training_job(
        cfg=cfg_race,
        output_dir=out_dir,
        work_dir=tmp_path / "race_work",
        device="cpu",
        max_wall_seconds=20.0,
        step_hook=_step_hook,
    )
    assert res["status"] == "PASS"
    assert res["stopped_reason"] == "wall_clock_guard"
    assert res["updates_completed"] == 2
    assert res["periodic_full_exports_published"] == 1
    assert full_tags_built == ["u0000001", "u0000002"]
