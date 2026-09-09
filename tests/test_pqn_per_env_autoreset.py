"""Integration-facing ownership tests for the opt-in PQN lane lifecycle.

These fixtures exercise the trainer boundary rather than reproducing B1/B2's
unit tests: the meaningful questions here are when a finished lane resets,
which episode it belongs to, and how source leases survive target/SGD work.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.core.runtime_contract import ModelHeadContract, canonical_digest
from src.model.obs_spec import RASTER31V3
from src.training.pqn_selfplay import HERO_POLICY_ID
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def _config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 2,
        "num_snakes": 3,
        "rollout_len": 3,
        "max_frames": 20,
        "minibatches": 1,
        "minibatch_size": 4,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
    }
    values.update(overrides)
    return PQNConfig(**values)


def test_per_env_mode_uses_fixed_rollout_length_and_delays_reset_to_next_rollout(monkeypatch):
    """A floor lane remains final through its T-step buffers, then resets next time."""
    trainer = PQNTrainer(_config())
    original_floor = trainer.sim.population_floor_reached
    calls = {"count": 0}

    def floor_once() -> np.ndarray:
        calls["count"] += 1
        floor = original_floor()
        if calls["count"] >= 2:
            floor[0] = True
        return floor

    monkeypatch.setattr(trainer.sim, "population_floor_reached", floor_once)
    first = trainer._rollout()
    assert first["actions"].shape[0] == trainer.cfg.rollout_len
    assert first["newly_completed_episodes"] >= 1
    assert trainer._episode_finished_env[0]
    assert trainer._episode_ids.tolist() == [0, 0]

    # The reset belongs at next rollout entry, after the preceding target work.
    monkeypatch.setattr(trainer.sim, "population_floor_reached", original_floor)
    second = trainer._rollout()
    assert second["reset_env_indices"].tolist() == [0]
    assert trainer._episode_ids.tolist() == [1, 0]
    assert trainer._episode_reset_counts.tolist() == [1, 0]


def test_barrier_derived_control_preserves_short_tail_behavior():
    """The matched derived control does not inherit the autoreset fixed-T rule."""
    trainer = PQNTrainer(
        _config(episode_reset_mode="batch_barrier_v1", max_frames=1, rollout_len=3)
    )
    roll = trainer._rollout()
    assert roll["actions"].shape[0] == 1


def test_barrier_derived_control_still_slices_when_every_lane_floors_early(monkeypatch):
    """Derived seeding changes worlds, not the batch-barrier completion rule."""
    trainer = PQNTrainer(
        _config(episode_reset_mode="batch_barrier_v1", max_frames=20, rollout_len=3)
    )
    monkeypatch.setattr(
        trainer.sim,
        "population_floor_reached",
        lambda: trainer.sim.frame >= 1,
    )
    roll = trainer._rollout()
    assert roll["actions"].shape[0] == 1


def test_corrected_continuous_barrier_records_actual_batch_reset_lanes() -> None:
    """The new corrected telemetry cannot leave reset arrays at their construction defaults."""
    trainer = PQNTrainer(
        _config(
            episode_reset_mode="batch_barrier_v1",
            episode_seed_mode="continuous_env_rng_v1",
            max_frames=1,
            rollout_len=1,
        )
    )
    trainer._rollout()
    second = trainer._rollout()
    assert second["reset_env_indices"].tolist() == [0, 1]
    assert trainer._episode_reset_counts.tolist() == [1, 1]


def test_per_env_rollout_keeps_completed_lane_target_in_its_old_episode():
    """Invalid post-completion rows cannot make Q(lambda) cross the reset boundary."""
    trainer = PQNTrainer(_config(num_envs=1, num_snakes=1, rollout_len=2, gamma=0.9, lambda_=1.0))
    roll = {
        "rewards": np.asarray([[[2.0]], [[99.0]]]),
        "dones": np.asarray([[[False]], [[False]]]),
        "trapped": np.zeros((2, 1, 1), dtype=bool),
        "valid": np.asarray([[[True]], [[False]]]),
        "next_mask": torch.ones((2, 1, 1, 6), dtype=torch.bool),
        "hero_q": torch.full((2, 1, 1, 6), 7.0),
        "final_obs": trainer._current_obs()[0],
    }
    target = trainer._compute_targets(roll)
    # t=0 bootstraps its final successor; it cannot carry the invalid t=1 reward.
    assert target[0, 0, 0].item() == pytest.approx(2.0 + 0.9 * 7.0)
    assert target[1, 0, 0].item() == 0.0


def test_derived_lane_rng_and_assignment_are_replaced_only_for_reset_lane(monkeypatch):
    """Resetting lane 0 cannot mutate lane 1's assignment or action stream."""
    trainer = PQNTrainer(_config(hero_frac=1.0))
    trainer._rollout()
    assert trainer._action_rngs is not None and trainer._episode_policy_ids is not None
    lane1_rng_before = copy.deepcopy(trainer._action_rngs[1].bit_generator.state)
    lane1_policy_before = trainer._episode_policy_ids[1].copy()
    trainer._episode_finished_env[:] = [True, False]
    trainer._prepare_derived_rollout()
    assert trainer._action_rngs[1].bit_generator.state == lane1_rng_before
    assert np.array_equal(trainer._episode_policy_ids[1], lane1_policy_before)


def test_preload_requires_pre_episode_identity_and_records_realized_snapshot_hash():
    """B5's fixed source enters the ordinary pinned pool before episode zero."""
    source_hash = "a" * 64
    trainer = PQNTrainer(
        _config(
            pool_capacity=1,
            hero_frac=0.0,
            pool_admission_mode="disabled_v1",
            initial_opponent_checkpoint_sha256=source_hash,
        )
    )
    policy_id = trainer.preload_opponent_snapshot(
        trainer.network,
        checkpoint_sha256=source_hash,
        model_head_digest=ModelHeadContract("pqn", "dueling_q", 6).digest,
    )
    assert policy_id == 0
    assert trainer.pool.snapshot_hash(policy_id) == trainer._initial_opponent_snapshot_state_sha256
    rollout = trainer._rollout()
    checkpoint = trainer.checkpoint_state()
    source = checkpoint["policy_source_contract"]
    assert source["initial_opponent_checkpoint_sha256"] == source_hash
    assert source["initial_opponent_snapshot_state_sha256"] == trainer.pool.snapshot_hash(policy_id)
    assert checkpoint["sampler_contract"]["policy_source_contract_digest"] == canonical_digest(source)
    assert rollout["policy_identities"][str(policy_id)] == (
        f"snapshot:{policy_id}:{trainer.pool.snapshot_hash(policy_id)}"
    )
    assert rollout["rollout_policy_source"]["policy_source_contract_digest"] == canonical_digest(
        source
    )


def test_lane_leases_remain_pinned_until_reset_and_close_is_idempotent():
    """Each environment holds its own lease; cleanup clears every surviving pin."""
    trainer = PQNTrainer(_config(pool_capacity=1, hero_frac=0.0))
    trainer.pool.add_snapshot(trainer.network)
    trainer._rollout()
    assert all(lease is not None for lease in trainer._episode_leases)
    assert trainer.pool.active_pin_count > 0
    trainer._episode_finished_env[:] = [True, False]
    trainer._prepare_derived_rollout()
    assert trainer._episode_leases[0] is not None and trainer._episode_leases[1] is not None
    trainer.close()
    trainer.close()
    assert trainer.pool.active_pin_count == 0


def test_stale_pending_lane_fails_before_any_lease_rng_or_episode_mutation():
    """A forged pending bit cannot release a live assignment or start a new world."""
    trainer = PQNTrainer(_config(pool_capacity=1, hero_frac=0.0))
    trainer.pool.add_snapshot(trainer.network)
    trainer._rollout()
    assert trainer._action_rngs is not None and trainer._episode_policy_ids is not None
    trainer._episode_finished_env[:] = [True, False]
    before = {
        "ids": trainer._episode_ids.copy(),
        "counts": trainer._episode_reset_counts.copy(),
        "policy": trainer._episode_policy_ids.copy(),
        "frames": trainer.sim.frame.copy(),
        "pins": trainer.pool.active_pin_count,
        "rng": copy.deepcopy(trainer._action_rngs[0].bit_generator.state),
        "lease": trainer._episode_leases[0],
    }
    with pytest.raises(RuntimeError, match="not at a completed final state"):
        trainer._prepare_derived_rollout()
    assert np.array_equal(trainer._episode_ids, before["ids"])
    assert np.array_equal(trainer._episode_reset_counts, before["counts"])
    assert np.array_equal(trainer._episode_policy_ids, before["policy"])
    assert np.array_equal(trainer.sim.frame, before["frames"])
    assert trainer.pool.active_pin_count == before["pins"]
    assert trainer._action_rngs[0].bit_generator.state == before["rng"]
    assert trainer._episode_leases[0] is before["lease"] and not before["lease"].closed


def test_close_attempts_later_leases_after_an_earlier_cleanup_failure():
    """A release fault cannot strand another lane's pin."""
    trainer = PQNTrainer(_config(pool_capacity=1, hero_frac=0.0))
    trainer.pool.add_snapshot(trainer.network)
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


def test_precomputed_derived_actions_only_eligible_original_hero_slots(monkeypatch):
    """Fixed-source all-hero Q forwarding cannot consume RNG for frozen slots."""
    trainer = PQNTrainer(_config(hero_frac=1.0))
    trainer._rollout()
    assert trainer._episode_policy_ids is not None
    assert np.all(trainer._episode_policy_ids == HERO_POLICY_ID)


def test_derived_fixed_policy_source_uses_its_declared_identity_without_a_lane_lease():
    """Fixed sources deliberately have no pool lease, including in derived mode."""
    class FixedPolicy:
        identity = "fixed:derived:test"

        def actions(self, masks, sim, slots):
            return np.zeros(len(slots), dtype=np.int64)

    from src.training.rollout_policies import FixedPolicySource

    policy = FixedPolicy()
    trainer = PQNTrainer(
        _config(
            rollout_policy_mode="fixed",
            fixed_policy_identity=policy.identity,
            hero_frac=0.0,
        ),
        fixed_policy=FixedPolicySource(policy, policy.identity),
    )
    roll = trainer._rollout()
    assert roll["policy_identities"] == {"0": policy.identity}
    assert all(lease is None for lease in trainer._episode_leases)
