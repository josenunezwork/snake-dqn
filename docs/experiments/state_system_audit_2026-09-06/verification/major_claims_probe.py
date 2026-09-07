#!/usr/bin/env python3
"""Independent bounded probes for four state/system audit claims."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from src.core.config_loader import load_config
from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import (
    CODE_ENEMY_BODY,
    CODE_ENEMY_PRED,
    ObsInputs,
    TACTICAL_HEAD_COL,
    TACTICAL_HEAD_ROW,
    build_observations,
)
from src.model.raster_network import TACTICAL_SHAPE
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def reward_version_probe():
    rows = []
    for version in (1, 2):
        torch.manual_seed(314159)
        cfg = PQNConfig(
            num_envs=1,
            num_snakes=1,
            rollout_len=1,
            mechanics_version=1,
            reward_version=version,
            pool_capacity=0,
            flip_augment=False,
            seed=17,
        )
        trainer = PQNTrainer(cfg, device=torch.device("cpu"))
        sim = trainer.sim
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
        sim.step(np.array([[1]], dtype=np.int64))
        rows.append({
            "configured_reward_version": version,
            "checkpoint_reward_version": trainer.checkpoint_state()["reward_version"],
            "step_reward": float(sim.get_reward()[0, 0]),
            "new_length": int(sim.get_lengths()[0, 0]),
            "batch_sim_has_reward_version_field": hasattr(sim.cfg, "reward_version"),
        })
    return {"rows": rows, "rewards_equal": rows[0]["step_reward"] == rows[1]["step_reward"]}


def policy_reassignment_probe():
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
    alive_before = trainer.sim.get_alive()[0].copy()
    first = trainer._rollout()
    alive_middle = trainer.sim.get_alive()[0].copy()
    second = trainer._rollout()
    stayed_alive = alive_before & alive_middle & trainer.sim.get_alive()[0]
    ids1 = np.asarray(first["policy_ids"])[0]
    ids2 = np.asarray(second["policy_ids"])[0]
    changed = stayed_alive & (ids1 != ids2)
    eligibility_changed = stayed_alive & ((ids1 == -1) != (ids2 == -1))
    return {
        "frames": [0, int(trainer.sim.frame[0]) - 1, int(trainer.sim.frame[0])],
        "episode_reset_between_rollouts": int(trainer.sim.frame[0]) <= 1,
        "policy_ids_first": ids1.tolist(),
        "policy_ids_second": ids2.tolist(),
        "stayed_alive": stayed_alive.tolist(),
        "living_slots_whose_policy_changed": np.nonzero(changed)[0].tolist(),
        "living_slots_whose_hero_eligibility_changed": np.nonzero(eligibility_changed)[0].tolist(),
    }


def _obs_for_order(order):
    # Observer slot 0 at (10,10), heading right. Enemy A predicts (14,10)
    # from (13,10); enemy B's second body cell occupies the same contested cell.
    E, S, MAXLEN = 1, 3, 2
    heads = np.array([[[10, 10], [13, 10], [16, 10]]], dtype=np.int64)
    bodies = np.zeros((E, S, MAXLEN, 2), dtype=np.int64)
    bodies[0, 0, 0] = (10, 10)
    bodies[0, 1, 0] = (13, 10)
    bodies[0, 2, 0] = (16, 10)
    bodies[0, 2, 1] = (14, 10)
    # Slots 1 and 2 are either A/B or B/A; reorder all enemy fields together.
    if order == "B_then_A":
        heads[:, 1:] = heads[:, [2, 1]]
        bodies[:, 1:] = bodies[:, [2, 1]]
    return ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=np.array([[1, 1, 2]], dtype=np.int64),
        lengths=np.array([[1, 1, 2]], dtype=np.int64),
        alive=np.ones((E, S), dtype=bool),
        heading=np.ones((E, S), dtype=np.int64),
        boost_frames=np.zeros((E, S), dtype=np.int64),
        frames_since_food=np.zeros((E, S), dtype=np.int64),
        boosting=np.zeros((E, S), dtype=bool),
        food_cells=np.zeros((E, 1, 2), dtype=np.int64),
        food_mass=np.zeros((E, 1), dtype=np.float64),
        food_is_corpse=np.zeros((E, 1), dtype=bool),
        grid_w=50,
        grid_h=50,
        max_snakes=S,
        starvation_max=500,
        max_length=100,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([0], dtype=np.int64),
        max_frames=100,
    )


def enemy_slot_probe():
    row, col = TACTICAL_HEAD_ROW - 4, TACTICAL_HEAD_COL
    codes = {}
    for order in ("A_then_B", "B_then_A"):
        obs = build_observations(_obs_for_order(order))["tactical_uint8"]
        codes[order] = int(obs[0, 0, 0, row, col])
    return {
        "contested_raster_cell": [row, col],
        "code_A_then_B": codes["A_then_B"],
        "code_B_then_A": codes["B_then_A"],
        "expected_pred_code": CODE_ENEMY_PRED,
        "expected_body_code": CODE_ENEMY_BODY,
        "codes_differ": codes["A_then_B"] != codes["B_then_A"],
    }


class _RowNetwork(torch.nn.Module):
    def forward(self, tensors):
        n = int(tensors["tactical"].shape[0])
        return torch.arange(float(n), dtype=torch.float32).unsqueeze(1).repeat(1, 6)


def play_q_row_probe():
    from src.simd_env.live_adapter import game_state_to_obs_inputs
    from web.backend.raster_policy import RasterServingPolicy
    from src.game.game_state import GameState

    saved = get_config()
    initialize_config(load_config("configs/mechanics_v2.yaml"))
    try:
        agent = SimpleNamespace(
            obs_spec="raster31v2",
            device=torch.device("cpu"),
            output_size=6,
            network=_RowNetwork(),
        )
        policy = RasterServingPolicy(agent)
        game = GameState(headless=False, human_mode=True, num_snakes=3, shared_policy=policy)
        policy.attach_game(game)
        ids = [int(s.id) for s in game.snakes]
        kinds = [type(s).__name__ for s in game.snakes]
        human_id = ids[0]
        ai_ids = ids[1:]
        policy._ensure_frame()
        queue_before = list(policy._dispatch_queue)
        # This is the first dqn call made by GameState.update's first AI, because
        # HumanSnake.update does not call policy.dqn().
        first_ai_q = policy.dqn(torch.zeros((1, 61))).detach().cpu().tolist()[0]
        first_ai_expected = [1.0] * 6
        # Fresh frame: execute the real game loop with Q recording enabled.
        game.update(train_mode=False, learn=False, allow_respawn=True)
        ai_q = [getattr(game.snakes[i], "last_q_values", None) for i in range(1, 3)]
        return {
            "roster_ids": ids,
            "roster_types": kinds,
            "human_id": human_id,
            "ai_ids": ai_ids,
            "dispatch_queue_before_calls": queue_before,
            "first_ai_q_direct": first_ai_q,
            "first_ai_expected_row": first_ai_expected,
            "direct_first_ai_is_wrong": first_ai_q != first_ai_expected,
            "real_update_ai_q_rows": ai_q,
            "real_update_first_ai_is_wrong": ai_q[0] != first_ai_expected,
            "real_update_second_ai_is_wrong": ai_q[1] != [2.0] * 6,
        }
    finally:
        initialize_config(saved)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = {
        "reward_version": reward_version_probe(),
        "mid_episode_policy_reassignment": policy_reassignment_probe(),
        "raster_enemy_slot_permutation": enemy_slot_probe(),
        "play_serving_q_row": play_q_row_probe(),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
