"""Tests for `research/v3_ppo` LIGHT/FULL sharded checkpoints, bit-identical resume, and eval."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import research.v3_ppo.eval as eval_mod
import research.v3_ppo.ppo as ppo_mod
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


def test_corrupted_or_missing_shard_falls_back_safely_and_prunes(
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
    #    (`light_u0000003.pt`, `u=3`) and logs world re-initialisation warning on load!
    meta_u1["shard_paths"][0].unlink()
    best2 = resolve_resume_checkpoint(ckpt_dir)
    assert best2 is not None
    assert best2["kind"] == "light"
    assert best2["updates_completed"] == 3

    fresh_trainer = V3PPOTrainer(cfg, device="cpu")
    fresh_trainer.episode_return_acc.fill_(99.0)
    load_light = fresh_trainer.load_checkpoint(best2["path"])
    captured2 = capsys.readouterr().out
    assert load_light["checkpoint_kind"] == "light"
    assert load_light["worlds_reinitialized"] is True
    assert "V3_PPO_LIGHT_RESUME_WARNING" in captured2
    assert fresh_trainer.updates_completed == 3
    assert float(fresh_trainer.episode_return_acc.abs().sum().item()) == 0.0

    # 4. Verify retention pruning (`keep_full=2, keep_light=4`) protects FULL-referenced LIGHTs
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
    """Train k=3, save >1-shard FULL (non-zero _clean_step_streak), resume k=3: bit-identical!"""
    cfg = _tiny_cfg(num_envs=4, total_updates=6, eval_every_updates=2, learner_slots=4, seed=314159)

    trainer_a = V3PPOTrainer(cfg, device="cpu")
    for _ in range(3):
        trainer_a.train_step()
        if trainer_a.updates_completed % cfg.eval_every_updates == 0:
            trainer_a.run_evaluation()

    # Explicitly set non-zero `_clean_step_streak` to prove Minor 10 preservation across resume!
    trainer_a.sim._clean_step_streak = 1234

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
    assert trainer_b.sim._clean_step_streak == 1234

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
    assert trainer_a.sim._clean_step_streak == trainer_b.sim._clean_step_streak
    assert trainer_b.sim._clean_step_streak == 1234 + 3 * cfg.rollout_len
    for t_name in sim_a["tensors"]:
        assert torch.equal(
            sim_a["tensors"][t_name], sim_b["tensors"][t_name]
        ), f"Sim tensor mismatch on {t_name}"

    # 5. Built-in evaluation metrics match identically
    for ea, eb in zip(evals_a, evals_b, strict=True):
        assert ea["combined"] == eb["combined"]
        for m_key in ("vs_food_seeking", "vs_random_safe"):
            assert ea["matchups"][m_key] == eb["matchups"][m_key]


def test_redesigned_evaluation_censored_separation_exact_combine_and_rng(tmp_path: Path) -> None:
    """Verify burn-in + survivorship-free metrics, censored separation, exact combine, & RNG."""
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

    # Prove censored lives are NEVER mixed into `mean_death_length`:
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

    # Minor 9: `_combine_matchups` sums exact float64 `alive_len_sum` without 4-decimal drift
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


def test_cooperative_stop_mid_update_discards_dirty_weights_and_in_session_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-update stop discards half-stepped weights; recoverable error restarts inside session."""
    # --- Part A: Cooperative stop mid-update (Minor 6) ---
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

    # --- Part B: In-session recoverable-error restart + idempotent config.json (Minor 7) ---
    cfg_rec = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0).to_dict(),
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

    # --- Part C: Stage sharded FULL set via ledger `companion_files` and resume to update 6 ---
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

    # Verify `ppo_mod` module import & optional MPS smoke
    assert hasattr(ppo_mod, "save_full_checkpoint") or hasattr(V3PPOTrainer, "save_full_checkpoint")
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
    """M1-r2: every LIGHT interval exports, FULL count obeys bound, & sha256 never mutates."""
    cfg = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=6, eval_every_updates=0).to_dict(),
            "light_checkpoint_interval_sec": 0.04,
            "full_checkpoint_interval_sec": 0.11,
            "max_periodic_full_exports": 1,
            "keep_full_checkpoints": 1,
            "keep_light_checkpoints": 2,
        }
    )
    out_dir = tmp_path / "m1_out"
    work_dir = tmp_path / "m1_work"
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


def test_restart_generation_light_margin_protected_finally_and_update_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover Minors 1, 2, 3, and 4: restart FULL/LIGHT safety, 45m margin, finally, & update cap."""
    # Minor 1: In-session restart rewinding from u=3 to FULL u=2 never reuses stale u=3 LIGHT.
    cfg_m1 = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "full_checkpoint_every_updates": 2,
            "light_resume_margin_minutes": 45.0,
            "max_in_session_restarts": 1,
        }
    )
    out_m1 = tmp_path / "m1_restart_out"
    work_m1 = tmp_path / "m1_restart_work"
    fault_state = {"injected": False}
    stale_u3_weights: dict[str, torch.Tensor] = {}
    perturbed_post_restart = {"done": False}

    def _post_restart_hook(tr: V3PPOTrainer, next_update: int) -> None:
        if not fault_state["injected"] and next_update == 4:
            # Pre-restart u=3 LIGHT is already saved & published to `out_m1/checkpoints/`;
            # transient fault on update 4 rewinds to FULL u=2 (since LIGHT u=3 is <45m newer).
            fault_state["injected"] = True
            assert (out_m1 / "checkpoints" / "light_u0000003.pt").is_file()
            for k, v in tr.raw_model.state_dict().items():
                stale_u3_weights[k] = v.detach().cpu().clone()
            with torch.no_grad():
                for p in tr.raw_model.parameters():
                    p.add_(7.0)
            raise RuntimeError("Transient fault on update 4 after pre-restart light_u0000003.pt")
        if fault_state["injected"] and not perturbed_post_restart["done"] and next_update == 3:
            perturbed_post_restart["done"] = True
            # Request a FULL checkpoint at post-restart u=3 while pre-restart `light_u0000003.pt`
            # still sits in `out_m1/checkpoints/`.
            object.__setattr__(cfg_m1, "full_checkpoint_every_updates", 3)
            with torch.no_grad():
                next(iter(tr.raw_model.parameters())).view(-1)[0].add_(0.125)

    res_m1 = run_training_job(
        cfg=cfg_m1,
        output_dir=out_m1,
        work_dir=work_m1,
        device="cpu",
        step_hook=_post_restart_hook,
    )

    assert res_m1["status"] == "PASS"
    assert res_m1["in_session_restarts"] == 1
    assert perturbed_post_restart["done"] is True
    assert (out_m1 / "checkpoints" / "light_u0000003.pt").is_file()
    # Post-restart FULL at u=3 MUST point to `light_u0000003_r01.pt`, never `light_u0000003.pt`.
    r01_idx = out_m1 / "checkpoints" / "full_u0000003_r01.index.json"
    assert r01_idx.is_file()
    r01_meta = verify_full_checkpoint_set(r01_idx)
    assert r01_meta["light_path"].name == "light_u0000003_r01.pt"
    assert r01_meta["restart_generation"] == 1
    check_r01 = V3PPOTrainer(cfg_m1, device="cpu")
    check_r01.load_checkpoint(r01_idx)
    first_param_key = next(iter(stale_u3_weights))
    assert not torch.equal(
        check_r01.raw_model.state_dict()[first_param_key], stale_u3_weights[first_param_key]
    ), "FULL checkpoint after restart erroneously reused pre-restart LIGHT weights!"

    # Explicit guard test: `save_full_checkpoint` refuses a `light_path` with mismatched generation
    stale_dir = tmp_path / "stale_guard"
    guard_tr = V3PPOTrainer(cfg_m1, device="cpu")
    guard_tr.train_step()
    guard_tr.restart_generation = 0
    stale_lp = stale_dir / "light_u0000001.pt"
    guard_tr.save_light_checkpoint(stale_lp)
    guard_tr.restart_generation = 1
    with pytest.raises(ValueError, match="Refusing to build FULL checkpoint"):
        guard_tr.save_full_checkpoint(stale_dir, tag="u0000001_r01", light_path=stale_lp)

    # Minor 2: Resume prefers LIGHT over FULL only when newer by > `light_resume_margin_minutes`.
    margin_dir = tmp_path / "margin_ckpts"
    m2_tr = V3PPOTrainer(_tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0), device="cpu")
    m2_tr.train_step()
    m2_tr.train_step()
    base_ts = 1_700_000_000.0
    monkeypatch.setattr(ppo_mod.time, "time", lambda: base_ts)
    m2_tr.save_full_checkpoint(margin_dir, tag="u0000002")
    m2_tr.train_step()
    m2_tr.train_step()
    # 20 minutes newer (< default 45 min margin) -> chooses FULL u=2!
    monkeypatch.setattr(ppo_mod.time, "time", lambda: base_ts + 20.0 * 60.0)
    m2_tr.save_light_checkpoint(margin_dir / "light_u0000004.pt")
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

    # --- Minor 3: Protected rollback in `finally` still writes manifest & `train_result.json` ---
    cfg_m3 = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=3, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "max_in_session_restarts": 0,
        }
    )
    out_m3 = tmp_path / "m3_finally_out"
    work_m3 = tmp_path / "m3_finally_work"
    orig_load_ckpt = V3PPOTrainer.load_checkpoint

    def _dead_cuda_load_ckpt(self: V3PPOTrainer, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA error: device-side assert triggered during finally rollback")

    def _boom_on_u2(_tr: V3PPOTrainer, next_update: int) -> None:
        if next_update == 2:
            monkeypatch.setattr(V3PPOTrainer, "load_checkpoint", _dead_cuda_load_ckpt)
            raise RuntimeError("Unrecoverable fault on update 2")

    with pytest.raises(RuntimeError, match="Unrecoverable fault on update 2"):
        run_training_job(
            cfg=cfg_m3,
            output_dir=out_m3,
            work_dir=work_m3,
            device="cpu",
            step_hook=_boom_on_u2,
        )
    monkeypatch.setattr(V3PPOTrainer, "load_checkpoint", orig_load_ckpt)
    out_m3_log = capsys.readouterr().out
    assert "V3_PPO_FINALLY_WARN stage=rollback_load_checkpoint" in out_m3_log
    verify_completed_artifact(out_m3 / "checkpoints" / "latest_manifest.json")
    verify_completed_artifact(out_m3 / "train_result.json")
    res_m3 = json.loads((out_m3 / "train_result.json").read_text(encoding="utf-8"))
    assert res_m3["status"] == "FAIL"

    # --- Minor 4: Update-based checkpoint settings obey `max_session_export_bytes` ---
    probe_light_bytes = (margin_dir / "light_u0000004.pt").stat().st_size
    cfg_m4 = V3PPOConfig.from_dict(
        {
            **_tiny_cfg(num_envs=2, total_updates=5, eval_every_updates=0).to_dict(),
            "checkpoint_every_updates": 1,
            "light_checkpoint_interval_sec": 3600.0,
            "full_checkpoint_interval_sec": 7200.0,
        }
    )
    out_m4 = tmp_path / "m4_cap_out"
    res_m4 = run_training_job(
        cfg=cfg_m4,
        output_dir=out_m4,
        work_dir=tmp_path / "m4_cap_work",
        device="cpu",
        max_session_export_bytes=int(probe_light_bytes * 3.2),  # 2 LIGHTs + 25% stop_reserve!
    )
    out_m4_log = capsys.readouterr().out
    assert res_m4["status"] == "PASS"
    assert res_m4["periodic_light_exports_published"] == 2
    assert "V3_PPO_EXPORT_BUDGET_REACHED" in out_m4_log


def test_single_pass_eval_bit_identical_training_stop_abort_and_wall_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M2-r2: 1-pass dual-matchup eval, bit-identical train, clean stop abort, & <2x wall skip."""
    # 1. Single simulator pass per evaluation + bit-identical training with vs without eval
    sim_creations: list[int] = []
    real_eval_sim = eval_mod.GameV3Sim

    def _spy_eval_sim(*args, **kwargs):  # type: ignore[no-untyped-def]
        sim_creations.append(int(kwargs.get("num_envs", args[1] if len(args) > 1 else -1)))
        return real_eval_sim(*args, **kwargs)

    monkeypatch.setattr(eval_mod, "GameV3Sim", _spy_eval_sim)

    cfg_no_ev = _tiny_cfg(num_envs=2, total_updates=4, eval_every_updates=0, seed=90210)
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

    # 2. Stop flag polled during evaluation aborts cleanly (`aborted=True`, never logged as data)
    poll_calls = {"n": 0}

    def _stop_after_2_polls() -> bool:
        poll_calls["n"] += 1
        return poll_calls["n"] >= 2

    aborted_ev = evaluate_policy(
        tr_b.raw_model,
        V3PPOConfig.from_dict(
            {**cfg_with_ev.to_dict(), "eval_burnin_ticks": 40, "eval_ticks": 200}
        ),
        device="cpu",
        should_stop=_stop_after_2_polls,
        poll_interval_ticks=8,
    )
    assert aborted_ev["aborted"] is True
    assert aborted_ev["ticks_completed"] < 240

    # Verify mid-eval abort inside `run_training_job` is NOT recorded in `eval/*.jsonl`
    real_eval_pol = ppo_mod.evaluate_policy

    def _aborting_eval_pol(*args, **kwargs):  # type: ignore[no-untyped-def]
        return real_eval_pol(*args, **{**kwargs, "should_stop": lambda: True})

    monkeypatch.setattr(ppo_mod, "evaluate_policy", _aborting_eval_pol)
    out_abort = tmp_path / "abort_ev_out"
    res_abort = run_training_job(
        cfg=V3PPOConfig.from_dict({**cfg_with_ev.to_dict(), "total_updates": 2}),
        output_dir=out_abort,
        work_dir=tmp_path / "abort_ev_work",
        device="cpu",
    )
    abort_stdout = capsys.readouterr().out
    monkeypatch.setattr(ppo_mod, "evaluate_policy", real_eval_pol)

    assert res_abort["eval_runs_aborted"] == 1
    assert res_abort["eval_runs_completed"] == 0
    assert res_abort["last_eval"] is None
    assert "V3_PPO_EVAL_ABORTED" in abort_stdout
    eval_dir = out_abort / "eval"
    if eval_dir.exists():
        for jf in eval_dir.glob("*.jsonl"):
            assert jf.read_text(encoding="utf-8").strip() == ""

    # 3. Remaining wall `< 2x` eval estimate skips starting evaluation (`V3_PPO_EVAL_SKIPPED`)
    cfg_skip = V3PPOConfig.from_dict(
        {
            **cfg_with_ev.to_dict(),
            "total_updates": 2,
            "eval_every_updates": 1,
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
