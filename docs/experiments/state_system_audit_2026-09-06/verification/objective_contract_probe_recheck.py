#!/usr/bin/env python3
"""Small deterministic probes for the 2026-09-06 PQN objective audit."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import PyRefGame, _batch_snapshot, _cmp_frame, _install_v2_config
from src.scripts.train_pqn import (
    apply_resume_checkpoint,
    train_loop,
    validate_pqn_resume_checkpoint_config,
)
from src.model.raster_network import SCALARS_DIM, STRATEGIC_SHAPE, TACTICAL_SHAPE
from src.training.pqn_trainer import PQNConfig, PQNTrainer


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_one_step_food_state(sim: BatchSim) -> None:
    """Install a safe one-snake state that eats once by going straight."""
    sim.cfg = replace(sim.cfg, initial_food=0, max_food=0)
    sim.bodies[:] = 0
    sim.head_ptr[:] = 0
    sim.seg_count[:] = 1
    sim.length[:] = 1
    sim.alive[:] = True
    sim.direction[:] = 1
    sim.boost_frames[:] = 0
    sim.frames_since_food[:] = 0
    sim._reward_prev_length[:] = 1
    sim.frame[:] = 0
    sim.bodies[0, 0, 0] = (10, 10)
    sim.food_cells[0] = [(11, 10)]
    sim.food_set[0] = {(11, 10)}
    sim.corpse_cells[0] = set()
    sim._rebuild_traversed_from_heads()
    sim._last_mask = sim._compute_action_masks()


def reward_version_probe() -> dict:
    """Show reward_version changes metadata while leaving reward arithmetic fixed."""
    rows = []
    for version in (1, 2):
        torch.manual_seed(314159)
        cfg = PQNConfig(
            num_envs=1,
            num_snakes=1,
            rollout_len=1,
            mechanics_version=1,
            reward_version=version,
            gamma=0.997,
            pool_capacity=0,
            flip_augment=False,
            seed=17,
        )
        trainer = PQNTrainer(cfg, device=torch.device("cpu"))
        set_one_step_food_state(trainer.sim)
        trainer.sim.step(np.array([[1]], dtype=np.int64))
        rows.append(
            {
                "configured_reward_version": version,
                "checkpoint_reward_version": trainer.checkpoint_state()["reward_version"],
                "step_reward": float(trainer.sim.get_reward()[0, 0]),
                "new_length": int(trainer.sim.get_lengths()[0, 0]),
                "batch_sim_has_reward_version_field": hasattr(trainer.sim.cfg, "reward_version"),
            }
        )
    return {"rows": rows, "rewards_equal": rows[0]["step_reward"] == rows[1]["step_reward"]}


def policy_reassignment_probe() -> dict:
    """Show policy IDs are redrawn at rollout boundaries inside one live episode."""
    torch.manual_seed(271828)
    cfg = PQNConfig(
        num_envs=1,
        num_snakes=6,
        rollout_len=1,
        hero_frac=0.5,
        pool_capacity=3,
        eps_start=0.0,
        eps_end=0.0,
        max_frames=1000,
        flip_augment=False,
        seed=23,
    )
    trainer = PQNTrainer(cfg, device=torch.device("cpu"))
    with torch.no_grad():
        for snapshot_idx in range(3):
            trainer.network.advantage_stream[-1].bias.zero_()
            trainer.network.advantage_stream[-1].bias[snapshot_idx] = 10.0
            trainer.pool.add_snapshot(trainer.network)

    before = int(trainer.sim.frame[0])
    alive_before = trainer.sim.get_alive()[0].copy()
    first = trainer._rollout()
    middle = int(trainer.sim.frame[0])
    alive_middle = trainer.sim.get_alive()[0].copy()
    second = trainer._rollout()
    after = int(trainer.sim.frame[0])
    ids1 = np.asarray(first["policy_ids"])[0]
    ids2 = np.asarray(second["policy_ids"])[0]
    stayed_alive = alive_before & alive_middle & trainer.sim.get_alive()[0]
    changed = stayed_alive & (ids1 != ids2)
    eligibility_changed = stayed_alive & ((ids1 == -1) != (ids2 == -1))
    return {
        "frames": [before, middle, after],
        "episode_reset_between_rollouts": after <= middle,
        "policy_ids_first": ids1.tolist(),
        "policy_ids_second": ids2.tolist(),
        "stayed_alive": stayed_alive.tolist(),
        "living_slots_whose_policy_changed": np.nonzero(changed)[0].tolist(),
        "living_slots_whose_hero_eligibility_changed": np.nonzero(eligibility_changed)[0].tolist(),
    }


def second_reset_parity_probe() -> dict:
    """Compare the parity reference and BatchSim after a second episode reset."""
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=4,
        game_width=300,
        game_height=220,
        initial_food=12,
        max_food=12,
        mechanics_version=2,
        gamma=0.997,
    )
    seed = 41
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(seed)
        ref = PyRefGame(cfg)
        bat = BatchSim(cfg, seeds=[seed], train_mode=True, allow_respawn=False)
        initial_div = _cmp_frame(
            ref.snapshot(),
            _batch_snapshot(bat, 0),
            ref.action_masks(),
            bat.get_action_mask()[0],
            seed,
            0,
            compare_step_outputs=False,
        )
        ref._reset_world()
        bat.reset()
        second_div = _cmp_frame(
            ref.snapshot(),
            _batch_snapshot(bat, 0),
            ref.action_masks(),
            bat.get_action_mask()[0],
            seed,
            0,
            compare_step_outputs=False,
        )
        return {
            "seed": seed,
            "initial_parity": initial_div is None,
            "second_reset_parity": second_div is None,
            "second_reset_divergence": None
            if second_div is None
            else {
                "field": second_div.field,
                "detail": second_div.detail,
            },
            "reference_heads": ref.snapshot()["heads"],
            "batch_heads": _batch_snapshot(bat, 0)["heads"],
        }
    finally:
        initialize_config(saved)


def configure_floored_food_state(sim: BatchSim) -> None:
    """Install a reachable floored state where a survivor will eat next step."""
    sim.bodies[:] = 0
    sim.head_ptr[:] = 0
    sim.seg_count[:] = 1
    sim.length[:] = 1
    sim.alive[:] = False
    sim.alive[0, :2] = True
    sim.direction[:] = 1
    sim._reward_prev_length[:] = 1
    sim.frame[:] = 7
    sim.bodies[0, 0, 0] = (10, 10)
    sim.bodies[0, 1, 0] = (20, 15)
    sim.food_cells[0] = [(11, 10)]
    sim.food_set[0] = {(11, 10)}
    sim.corpse_cells[0] = set()
    sim._rebuild_traversed_from_heads()
    sim._last_mask = sim._compute_action_masks()


def post_floor_rng_probe() -> dict:
    """Show an ignored post-floor step changes the following reset via RNG."""
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        game_width=300,
        game_height=220,
        initial_food=1,
        max_food=1,
        mechanics_version=2,
        gamma=0.997,
    )
    seed = 99
    immediate = BatchSim(cfg, seeds=[seed], train_mode=True, allow_respawn=False)
    delayed = BatchSim(cfg, seeds=[seed], train_mode=True, allow_respawn=False)
    configure_floored_food_state(immediate)
    configure_floored_food_state(delayed)
    assert immediate.population_floor_reached()[0]
    assert delayed.population_floor_reached()[0]

    immediate.reset()
    delayed.step(np.ones((1, 6), dtype=np.int64))
    consumed = (11, 10) not in delayed.food_set[0]
    delayed.reset()
    heads_immediate = immediate.get_heads()[0].tolist()
    heads_delayed = delayed.get_heads()[0].tolist()
    return {
        "seed": seed,
        "post_floor_survivor_consumed_food": consumed,
        "next_reset_heads_immediate": heads_immediate,
        "next_reset_heads_after_one_ignored_step": heads_delayed,
        "next_reset_changed": heads_immediate != heads_delayed,
    }


def resume_contract_probe() -> dict:
    """Show behavior/observation knobs can change while resume validation passes."""
    common = {
        "num_envs": 1,
        "num_snakes": 1,
        "rollout_len": 1,
        "pool_capacity": 0,
        "flip_augment": False,
        "seed": 7,
    }
    torch.manual_seed(1234)
    source = PQNTrainer(
        PQNConfig(
            **common,
            lr=1e-3,
            max_frames=100,
            eps_start=1.0,
            eps_end=0.0,
            eps_decay_steps=100,
        ),
        device=torch.device("cpu"),
    )
    source.agent_steps = 50
    blob = source.checkpoint_state()

    resumed_cfg = PQNConfig(
        **common,
        lr=2e-4,
        max_frames=40,
        eps_start=1.0,
        eps_end=0.0,
        eps_decay_steps=10,
    )
    validate_pqn_resume_checkpoint_config(blob, resumed_cfg, checkpoint_path="in-memory")
    torch.manual_seed(5678)
    resumed = PQNTrainer(resumed_cfg, device=torch.device("cpu"))
    configured_lr_before_apply = resumed.optimizer.param_groups[0]["lr"]
    apply_resume_checkpoint(resumed, blob)
    missing = [
        key
        for key in ("lr", "max_frames", "eps_start", "eps_end", "eps_decay_steps")
        if key not in blob
    ]
    return {
        "validation_accepted_changed_contract": True,
        "checkpoint_missing_keys": missing,
        "configured_lr_before_apply": configured_lr_before_apply,
        "optimizer_lr_after_apply": resumed.optimizer.param_groups[0]["lr"],
        "source_epsilon_at_step_50": source.epsilon(),
        "resumed_epsilon_at_step_50": resumed.epsilon(),
        "source_max_frames": source.cfg.max_frames,
        "resumed_max_frames": resumed.cfg.max_frames,
    }


def tripwire_checkpoint_probe() -> dict:
    """Inject a nonfinite target and show halt persists the already-poisoned net."""
    torch.manual_seed(4242)
    trainer = PQNTrainer(
        PQNConfig(
            num_envs=1,
            num_snakes=1,
            rollout_len=1,
            minibatches=1,
            minibatch_size=1,
            pool_capacity=0,
            flip_augment=False,
            seed=42,
        ),
        device=torch.device("cpu"),
    )
    roll = {
        "tactical": torch.zeros((1, 1, 1, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((1, 1, 1, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((1, 1, 1, SCALARS_DIM)),
        "actions": np.zeros((1, 1, 1), dtype=np.int64),
        "valid": np.ones((1, 1, 1), dtype=bool),
        "policy_ids": np.array([[-1]], dtype=np.int64),
        "rewards": np.zeros((1, 1, 1), dtype=np.float64),
        "boost": np.zeros((1, 1, 1), dtype=bool),
        "epsilon": 1.0,
        "kills_total": 0,
    }
    trainer._rollout = lambda: roll
    trainer._compute_targets = lambda _: torch.full((1, 1, 1), float("nan"))
    out_dir = Path(__file__).resolve().parent / "tripwire_poison"
    history, tripped = train_loop(
        trainer,
        total_steps=1,
        log_every=1,
        ckpt_every=0,
        out_dir=out_dir,
    )
    saved = torch.load(out_dir / "latest_pqn.pth", map_location="cpu", weights_only=False)
    finite = {
        key: bool(torch.isfinite(value).all()) for key, value in saved["dqn_state_dict"].items()
    }
    history_lines = (out_dir / "history.jsonl").read_text().splitlines()
    return {
        "tripped": tripped,
        "completed_history_rows": len(history),
        "persisted_history_lines": len(history_lines),
        "triggering_telemetry_retained_only_in_memory": trainer.last_telemetry is not None,
        "saved_checkpoint_exists": (out_dir / "latest_pqn.pth").exists(),
        "saved_parameter_tensors_nonfinite": [key for key, ok in finite.items() if not ok],
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    result = {
        "commit": "c1cf7e8c954239cb6773c3da24bd17f4ddb9b715",
        "source_hashes": {
            rel: sha256(ROOT / rel)
            for rel in (
                "src/training/pqn_trainer.py",
                "src/training/pqn_selfplay.py",
                "src/core/reward_events.py",
                "src/core/reward_contract.py",
                "src/training/curriculum.py",
                "src/simd_env/batch_sim.py",
                "src/simd_env/parity.py",
            )
        },
        "reward_version": reward_version_probe(),
        "policy_reassignment": policy_reassignment_probe(),
        "second_reset_parity": second_reset_parity_probe(),
        "post_floor_rng": post_floor_rng_probe(),
        "resume_contract": resume_contract_probe(),
        "tripwire_checkpoint": tripwire_checkpoint_probe(),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
