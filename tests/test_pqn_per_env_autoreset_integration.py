"""Independent B4 integration oracles for per-environment PQN autoreset.

These checks exercise the trainer and BatchSim boundary together.  They keep
expected target arithmetic and lane snapshots in this file rather than
reusing B3T's lifecycle assertions or implementation helpers.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def _config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 2,
        "num_snakes": 3,
        "rollout_len": 4,
        "max_frames": 20,
        "minibatches": 1,
        "minibatch_size": 4,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
        "hero_frac": 1.0,
        "eps_start": 0.0,
        "eps_end": 0.0,
        "initial_food": 0,
        "max_food": 0,
        "game_width": 300,
        "game_height": 300,
        "seed": 1701,
    }
    values.update(overrides)
    return PQNConfig(**values)


def _trainer(**overrides: object) -> PQNTrainer:
    return PQNTrainer(_config(**overrides), device=torch.device("cpu"))


def _lane_snapshot(trainer: PQNTrainer, env: int) -> dict[str, Any]:
    """Copy every mutable BatchSim field that can affect the next transition."""
    sim = trainer.sim
    return {
        "bodies": sim.bodies[env].copy(),
        "head_ptr": sim.head_ptr[env].copy(),
        "seg_count": sim.seg_count[env].copy(),
        "length": sim.length[env].copy(),
        "alive": sim.alive[env].copy(),
        "direction": sim.direction[env].copy(),
        "boost_frames": sim.boost_frames[env].copy(),
        "frames_since_food": sim.frames_since_food[env].copy(),
        "respawn_timer": sim.respawn_timer[env].copy(),
        "reward_prev_length": sim._reward_prev_length[env].copy(),
        "trav": sim._trav[env].copy(),
        "trav_valid": sim._trav_valid[env].copy(),
        "last_reward": sim._last_reward[env].copy(),
        "last_done": sim._last_done[env].copy(),
        "last_transition_valid": sim._last_transition_valid[env].copy(),
        "last_food_ate": sim._last_food_ate[env].copy(),
        "last_death_cause": sim._last_death_cause[env].copy(),
        "last_kills": sim._last_kills[env].copy(),
        "last_kill_victim_len": tuple(tuple(value) for value in sim._last_kill_victim_len[env]),
        "seed": int(sim._seeds[env]),
        "frame": int(sim.frame[env]),
        "food_cells": tuple(sim.food_cells[env]),
        "food_set": frozenset(sim.food_set[env]),
        "corpse_cells": frozenset(sim.corpse_cells[env]),
        # BatchSim wraps CPython random.Random in EnvRng; getstate() is the
        # serialized per-environment world stream used by the simulator.
        "rng": copy.deepcopy(sim._rngs[env]._rng.getstate()),
        "episode_rng": copy.deepcopy(trainer._action_rngs[env].bit_generator.state),
        "episode_id": int(trainer._episode_ids[env]),
        "reset_count": int(trainer._episode_reset_counts[env]),
        "policy_row": trainer._episode_policy_ids[env].copy(),
    }


def _assert_lane_snapshot_equal(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "reward_prev_length",
        "trav",
        "trav_valid",
        "last_reward",
        "last_done",
        "last_transition_valid",
        "last_food_ate",
        "last_death_cause",
        "last_kills",
        "policy_row",
    ):
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    for key in (
        "frame",
        "food_cells",
        "food_set",
        "corpse_cells",
        "last_kill_victim_len",
        "seed",
        "episode_id",
        "reset_count",
    ):
        assert after[key] == before[key], key
    assert _state_digest(after["rng"]) == _state_digest(before["rng"]), "world RNG"
    assert _state_digest(after["episode_rng"]) == _state_digest(before["episode_rng"]), "action RNG"


def _state_digest(value: object) -> str:
    """Stable equality helper for nested NumPy RNG state dictionaries."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_state_digest(value[key])}" for key in sorted(value)) + "}"
    if isinstance(value, np.ndarray):
        return f"array:{value.dtype}:{value.shape}:{value.tobytes().hex()}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_state_digest(item) for item in value) + "]"
    return repr(value)


def test_floor_lane_uses_fixed_t_and_delayed_selected_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane that reaches the population floor freezes until the next boundary."""
    trainer = _trainer()
    original_floor = trainer.sim.population_floor_reached
    calls = 0

    def floor_after_first_step() -> np.ndarray:
        nonlocal calls
        calls += 1
        result = original_floor()
        # The trainer queries floor once before stepping and again after the
        # step.  Mark the lane complete only after frame 1 so the first row is
        # a real transition and later rows are invalid tail rows.
        if trainer.sim.frame[0] >= 1:
            result[0] = True
        return result

    monkeypatch.setattr(trainer.sim, "population_floor_reached", floor_after_first_step)
    first = trainer._rollout()
    assert first["actions"].shape[0] == trainer.cfg.rollout_len
    valid = np.asarray(first["valid"], dtype=bool)
    assert valid[0, 0].all()
    assert not valid[1:, 0].any()
    assert valid[:, 1].any()
    assert first["reset_env_indices"].tolist() == []
    assert first["episode_ids"].tolist() == [0, 0]
    assert first["newly_completed_episodes"] == 1

    monkeypatch.setattr(trainer.sim, "population_floor_reached", original_floor)
    trainer.sim.frame[0] = trainer.cfg.max_frames
    second = trainer._rollout()
    assert second["reset_env_indices"].tolist() == [0]
    assert second["episode_ids"].tolist() == [1, 0]
    assert second["episode_reset_counts"].tolist() == [1, 0]
    trainer.close()


def test_frame_cap_keeps_fixed_t_and_drops_only_post_cap_rows() -> None:
    """A frame cap is a lane boundary, with no fabricated post-cap transitions."""
    trainer = _trainer(max_frames=2, rollout_len=4)
    roll = trainer._rollout()
    valid = np.asarray(roll["valid"], dtype=bool)
    assert roll["actions"].shape[0] == 4
    for env in range(trainer.cfg.num_envs):
        assert valid[:2, env].any()
        assert not valid[2:, env].any()
    assert roll["episode_ids"].tolist() == [0, 0]
    assert roll["reset_env_indices"].tolist() == []
    trainer.close()


def test_targets_use_reward_only_death_and_masked_bootstrap_without_invalid_carry() -> None:
    """The independent oracle rejects death bootstrap and invalid successor reward leakage."""
    trainer = _trainer(num_envs=1, num_snakes=1, rollout_len=2, gamma=0.9, lambda_=1.0)
    all_actions = torch.ones((2, 1, 1, 6), dtype=torch.bool)
    roll: dict[str, object] = {
        "rewards": np.asarray([[[-7.0]], [[999.0]]]),
        "dones": np.asarray([[[True]], [[False]]]),
        "trapped": np.zeros((2, 1, 1), dtype=bool),
        "valid": np.asarray([[[True]], [[False]]]),
        "next_mask": all_actions,
        "hero_q": torch.full((2, 1, 1, 6), 7.0),
        "final_obs": trainer._current_obs()[0],
    }
    death_target = trainer._compute_targets(roll)
    assert death_target[0, 0, 0].item() == pytest.approx(-7.0)

    roll["dones"] = np.asarray([[[False]], [[False]]])
    roll["rewards"] = np.asarray([[[2.0]], [[999.0]]])
    live_target = trainer._compute_targets(roll)
    assert live_target[0, 0, 0].item() == pytest.approx(2.0 + 0.9 * 7.0)
    trainer.close()


@pytest.mark.parametrize("boundary", ("floor", "cap"))
def test_survivor_floor_and_cap_bootstrap_from_old_final_successor(boundary: str) -> None:
    """A live survivor at either boundary uses the hand-computed final Q bootstrap."""

    class ConstantQ(torch.nn.Module):
        """Small independent forward oracle returning a known masked max."""

        output_size = 6

        def __init__(self) -> None:
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self, tactical: torch.Tensor, strategic: torch.Tensor, scalars: torch.Tensor
        ) -> torch.Tensor:
            del strategic, scalars
            return torch.ones((tactical.shape[0], 6), dtype=tactical.dtype) * (7.0 + self.offset)

    trainer = PQNTrainer(
        _config(num_envs=1, num_snakes=1, rollout_len=1, gamma=0.9, lambda_=1.0),
        network=ConstantQ(),
        device=torch.device("cpu"),
    )
    reward = 2.0 if boundary == "floor" else 3.0
    roll: dict[str, object] = {
        "rewards": np.asarray([[[reward]]]),
        "dones": np.asarray([[[False]]]),
        "trapped": np.zeros((1, 1, 1), dtype=bool),
        "valid": np.asarray([[[True]]]),
        "next_mask": torch.ones((1, 1, 1, 6), dtype=torch.bool),
        # The independently hand-computed masked max is 7.0.
        "hero_q": torch.full((1, 1, 1, 6), 7.0),
        "final_obs": trainer._current_obs()[0],
    }
    target = trainer._compute_targets(roll)
    assert target[0, 0, 0].item() == pytest.approx(reward + 0.9 * 7.0)
    trainer.close()


def test_reset_of_completed_lane_preserves_continuing_lane_state_and_rng() -> None:
    """Soft reset mutates only the selected lane and starts its next episode."""
    trainer = _trainer(rollout_len=1)
    trainer._rollout()
    before = _lane_snapshot(trainer, 1)
    lane1_lease = trainer._episode_leases[1]
    trainer._episode_finished_env[:] = [True, False]
    trainer.sim.frame[0] = trainer.cfg.max_frames

    assert trainer._prepare_derived_rollout() == 1
    after = _lane_snapshot(trainer, 1)
    _assert_lane_snapshot_equal(before, after)
    assert trainer._episode_ids.tolist() == [1, 0]
    assert trainer._episode_reset_counts.tolist() == [1, 0]
    assert trainer._last_reset_env_indices.tolist() == [0]
    assert trainer._episode_leases[1] is lane1_lease
    trainer.close()


def test_selected_reset_replaces_one_lease_and_cleanup_releases_both() -> None:
    """Each lane owns an episode lease; reset releases only the completed lane."""
    trainer = _trainer(hero_frac=0.0, pool_capacity=2, rollout_len=1)
    assert trainer.pool.add_snapshot(trainer.network) == 0
    trainer._rollout()
    old_lane0 = trainer._episode_leases[0]
    lane1 = trainer._episode_leases[1]
    assert old_lane0 is not None and lane1 is not None
    assert trainer.pool.active_pin_count == 2

    trainer._episode_finished_env[:] = [True, False]
    trainer.sim.frame[0] = trainer.cfg.max_frames
    assert trainer._prepare_derived_rollout() == 1
    assert old_lane0.closed
    assert trainer._episode_leases[0] is not old_lane0
    assert trainer._episode_leases[1] is lane1
    assert not lane1.closed
    assert trainer.pool.active_pin_count == 2

    trainer.close()
    trainer.close()
    assert trainer.pool.active_pin_count == 0


def test_leases_stay_pinned_through_sgd_and_only_reset_lane_is_replaced() -> None:
    """A completed lane refreshes after an update while its peer lease survives."""
    trainer = _trainer(
        hero_frac=0.5,
        pool_capacity=2,
        rollout_len=2,
        minibatch_size=2,
        action_collapse_patience=99,
    )
    assert trainer.pool.add_snapshot(trainer.network) == 0
    first = trainer.update()
    assert first.optimizer_steps == 1
    lane1_lease = trainer._episode_leases[1]
    old_lane0 = trainer._episode_leases[0]
    assert lane1_lease is not None and old_lane0 is not None
    assert trainer.pool.active_pin_count == 2

    trainer._episode_finished_env[:] = [True, False]
    trainer.sim.frame[0] = trainer.cfg.max_frames
    second = trainer.update()
    assert second.optimizer_steps == 1
    assert old_lane0.closed
    assert trainer._episode_leases[1] is lane1_lease
    assert not lane1_lease.closed
    assert trainer.pool.active_pin_count == 2
    trainer.close()
    assert trainer.pool.active_pin_count == 0


def test_close_attempts_every_lease_after_first_release_fault() -> None:
    """Cleanup retains the first error while releasing all later lane pins."""
    trainer = _trainer(hero_frac=0.0, pool_capacity=1, rollout_len=1)
    assert trainer.pool.add_snapshot(trainer.network) == 0
    trainer._rollout()
    first, second = trainer._episode_leases
    assert first is not None and second is not None
    original_close = first.close

    def broken_close() -> None:
        original_close()
        raise RuntimeError("first close failed")

    first.close = broken_close  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="first close failed"):
        trainer.close()
    assert second.closed
    assert trainer.pool.active_pin_count == 0
