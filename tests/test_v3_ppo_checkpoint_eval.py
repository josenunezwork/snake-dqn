"""Tests for `research/v3_ppo` checkpoint round-trip, bit-identical CPU resume, and evaluation."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from research.arena_ppo.runtime import (
    verify_completed_artifact,
    write_create_only_artifact,
    write_create_only_json,
)
from research.game_v3.config import GameV3Config
from research.v3_ppo.config import V3_PPO_NAMESPACE, V3PPOConfig
from research.v3_ppo.eval import evaluate_policy, evaluate_scripted_baseline
from research.v3_ppo.ppo import (
    V3PPOTrainer,
    pack_sim_state_dict,
    run_training_job,
    unpack_sim_state_dict,
    verify_staged_checkpoint,
)


def _tiny_cfg(
    *,
    total_updates: int = 6,
    eval_every_updates: int = 2,
    learner_slots: int | None = None,
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
        num_envs=3,
        learner_slots=learner_slots,
        opponent_policy="food_seeking",
        net_size="S",
        amp=False,
        use_channels_last=False,
        rollout_len=6,
        ppo_epochs=2,
        minibatch_size=16,
        total_updates=total_updates,
        eval_every_updates=eval_every_updates,
        eval_envs=2,
        eval_ticks=25,
        eval_learner_slots=2,
        eval_seed=77777,
        seed=seed,
    )


def test_checkpoint_roundtrip_zlib_packing_and_guards(tmp_path: Path) -> None:
    """Verify atomic checkpoint sidecars, lossless zlib sim packing, and foreign-config refusal."""
    cfg = _tiny_cfg(total_updates=2, eval_every_updates=0)
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.train_step()

    # Explicitly test lossless `pack_sim_state_dict` / `unpack_sim_state_dict` with low thresholds
    raw_sd = trainer.sim.state_dict()
    packed_sd = pack_sim_state_dict(raw_sd, total_threshold_bytes=1024, tensor_min_bytes=256)
    assert "_zlib_tensors" in packed_sd
    assert len(packed_sd["_zlib_tensors"]) > 0
    unpacked_sd = unpack_sim_state_dict(packed_sd)
    assert "_zlib_tensors" not in unpacked_sd
    assert set(unpacked_sd["tensors"]) == set(raw_sd["tensors"])
    for t_name, orig_tensor in raw_sd["tensors"].items():
        rec_tensor = unpacked_sd["tensors"][t_name]
        assert rec_tensor.dtype == orig_tensor.dtype
        assert rec_tensor.shape == orig_tensor.shape
        assert torch.equal(rec_tensor, orig_tensor), f"Mismatch in unpacked tensor {t_name}"

    ckpt_path = tmp_path / "ckpt_u0000001.pt"
    digest = trainer.save_checkpoint(ckpt_path)
    assert len(digest) == 64
    assert verify_completed_artifact(ckpt_path) == digest
    meta = verify_staged_checkpoint(ckpt_path, verify_markers=True)
    assert meta["updates_completed"] == 1
    assert meta["sha256"] == digest

    # Refuses overwrite on create-only checkpoint path
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(ckpt_path)

    # Refuses mismatched immutable config keys on load
    wrong_cfg = _tiny_cfg(total_updates=2, eval_every_updates=0, learner_slots=3)
    other_trainer = V3PPOTrainer(wrong_cfg, device="cpu")
    with pytest.raises(ValueError, match="Checkpoint config mismatch"):
        other_trainer.load_checkpoint(ckpt_path)


def test_bit_identical_cpu_resume_across_updates_and_eval(tmp_path: Path) -> None:
    """Train k=3 updates, save, train k=3 more vs resume and train k=3 more: bit-identical!"""
    cfg = _tiny_cfg(total_updates=6, eval_every_updates=2, learner_slots=4, seed=314159)

    trainer_a = V3PPOTrainer(cfg, device="cpu")
    for _ in range(3):
        trainer_a.train_step()
        if trainer_a.updates_completed % cfg.eval_every_updates == 0:
            trainer_a.run_evaluation()

    ckpt_path = tmp_path / "resume_k3.pt"
    trainer_a.save_checkpoint(ckpt_path)

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
    trainer_b.load_checkpoint(ckpt_path)

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
    for t_name in sim_a["tensors"]:
        assert torch.equal(
            sim_a["tensors"][t_name], sim_b["tensors"][t_name]
        ), f"Sim tensor mismatch on {t_name}"

    # 5. Built-in evaluation metrics match identically
    for ea, eb in zip(evals_a, evals_b, strict=True):
        assert ea["combined"] == eb["combined"]
        for m_key in ("vs_food_seeking", "vs_random_safe"):
            assert ea["matchups"][m_key] == eb["matchups"][m_key]


def test_deterministic_evaluation_and_rng_preservation(tmp_path: Path) -> None:
    """Evaluation must be deterministic given `(seed, checkpoint)` and preserve caller RNG."""
    cfg = _tiny_cfg(total_updates=1, eval_every_updates=1)
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.train_step()
    ckpt_path = tmp_path / "eval_ckpt.pt"
    trainer.save_checkpoint(ckpt_path)

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
    fresh.load_checkpoint(ckpt_path, restore_rng=False)
    ev2 = evaluate_policy(fresh.raw_model, cfg, device="cpu")

    assert ev1["combined"] == ev2["combined"]
    assert set(ev1["matchups"]) == {"vs_food_seeking", "vs_random_safe"}
    for m_key in ("vs_food_seeking", "vs_random_safe"):
        assert ev1["matchups"][m_key] == ev2["matchups"][m_key]
        row = ev1["matchups"][m_key]
        for req_metric in (
            "mean_final_length",
            "p95_final_length",
            "deaths_per_1k_agent_steps",
            "kills",
            "kills_per_1k_agent_steps",
            "mean_episode_return",
        ):
            assert req_metric in row

    base_rs1 = evaluate_scripted_baseline("random_safe", cfg, device="cpu")
    base_rs2 = evaluate_scripted_baseline("random_safe", cfg, device="cpu")
    assert base_rs1["combined"] == base_rs2["combined"]


def test_run_training_job_resume_and_mps_smoke(tmp_path: Path) -> None:
    """Verify `run_training_job` create-only exports, ledger resume, and optional MPS smoke."""
    cfg1 = _tiny_cfg(total_updates=2, eval_every_updates=2)
    out1 = tmp_path / "attempt1_out"
    res1 = run_training_job(cfg=cfg1, output_dir=out1, work_dir=tmp_path / "w1", device="cpu")
    assert res1["status"] == "PASS"
    assert res1["updates_completed"] == 2
    verify_completed_artifact(out1 / "config.json")
    verify_completed_artifact(out1 / "train_result.json")
    verify_completed_artifact(out1 / "checkpoints" / "final.pt")
    verify_completed_artifact(out1 / "checkpoints" / "latest_manifest.json")

    # Stage `final.pt` into a fresh ledger input dir and resume to update 4
    in2 = tmp_path / "attempt2_in"
    in2.mkdir()
    src_pt = out1 / "checkpoints" / "final.pt"
    dst_pt = in2 / "ckpt_u0000002.pt"
    sha = write_create_only_artifact(dst_pt, lambda tmp: shutil.copyfile(src_pt, tmp))
    ledger_p = in2 / "verified-ledger.json"
    write_create_only_json(
        ledger_p,
        {
            "schema": "v3-ppo-verified-ledger/v1",
            "namespace": V3_PPO_NAMESPACE,
            "unit_kind": "checkpoint",
            "skip_verified": True,
            "resume_checkpoint": {
                "relpath": dst_pt.name,
                "sha256": sha,
                "size_bytes": int(dst_pt.stat().st_size),
            },
            "verified_units": [],
        },
    )

    cfg2 = _tiny_cfg(total_updates=4, eval_every_updates=2)
    out2 = tmp_path / "attempt2_out"
    res2 = run_training_job(
        cfg=cfg2,
        output_dir=out2,
        work_dir=tmp_path / "w2",
        device="cpu",
        input_dir=in2,
        ledger_path=ledger_p,
    )
    assert res2["status"] == "PASS"
    assert res2["updates_completed"] == 4
    assert res2["updates_this_session"] == 2
    assert res2["resumed_from"]["updates_completed_at_resume"] == 2

    if torch.backends.mps.is_available():
        mps_cfg = _tiny_cfg(total_updates=1, eval_every_updates=1)
        mps_trainer = V3PPOTrainer(mps_cfg, device="mps")
        mps_step = mps_trainer.train_step()
        mps_eval = mps_trainer.run_evaluation()
        assert (
            mps_step["agent_steps"] == mps_cfg.num_envs * mps_cfg.num_snakes * mps_cfg.rollout_len
        )
        assert mps_eval["combined"]["episodes"] >= 1
