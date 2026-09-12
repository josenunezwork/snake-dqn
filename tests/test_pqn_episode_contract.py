"""Independent corrected-v3 episode and target-contract oracles.

These fixtures deliberately hand-build transition arrays rather than deriving
expected values from the rollout implementation.  They protect the boundary
between a real terminal transition and a dead-at-entry invalid row.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.model.raster_network import SCALARS_DIM, STRATEGIC_SHAPE, TACTICAL_SHAPE
from src.training.pqn_trainer import (
    PQNConfig,
    PQNTrainer,
    pqn_sampler_contract,
    pqn_target_contract,
)
from src.training.rollout_policies import FixedPolicySource


def _corrected_trainer() -> PQNTrainer:
    return PQNTrainer(
        PQNConfig(
            num_envs=1,
            num_snakes=1,
            rollout_len=2,
            gamma=0.9,
            lambda_=0.5,
            recipe="corrected-v3",
            obs_spec=RASTER31V3,
            flip_augment=False,
        )
    )


def _rollout(rewards: list[float], dones: list[bool], valid: list[bool]) -> dict[str, object]:
    t, e, s = 2, 1, 1
    return {
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(t, e, s),
        "dones": np.asarray(dones, dtype=bool).reshape(t, e, s),
        "trapped": np.zeros((t, e, s), dtype=bool),
        "valid": np.asarray(valid, dtype=bool).reshape(t, e, s),
        "next_mask": torch.ones((t, e, s, 6), dtype=torch.bool),
        "hero_q": torch.full((t, e, s, 6), 2.0),
        "final_obs": {
            "tactical": torch.zeros((e, s, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((e, s, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((e, s, SCALARS_DIM)),
        },
    }


def test_two_step_terminal_reward_carries_to_previous_live_transition() -> None:
    """A death at t+1 is reward-only but still supplies t's lambda return."""
    trainer = _corrected_trainer()
    targets = trainer._compute_targets(_rollout([1.0, 9.0], [False, True], [True, True]))

    # Independent hand calculation: G1 = 9; G0 = 1 + .9 * (.5 * 2 + .5 * 9).
    assert targets[1, 0, 0].item() == pytest.approx(9.0)
    assert targets[0, 0, 0].item() == pytest.approx(5.95)


def test_corrected_v3_rejects_alive_empty_successor_mask() -> None:
    """Advice exhaustion cannot manufacture the legacy trapped/death target."""
    trainer = _corrected_trainer()
    roll = _rollout([0.0, 0.0], [False, False], [True, True])
    roll["next_mask"] = torch.zeros((2, 1, 1, 6), dtype=torch.bool)

    with pytest.raises(RuntimeError, match="alive successor"):
        trainer._compute_targets(roll)


def test_corrected_contract_declares_pinned_episode_and_real_death_semantics() -> None:
    """Checkpoint provenance must name the repaired target boundary."""
    contract = pqn_target_contract(_corrected_trainer().cfg)
    assert contract["version"] == "pqn-qlambda-corrected-v3-lifecycle-v1"
    assert contract["episode_lifecycle_contract_digest"]
    assert contract["death"] == "actual_done_reward_only"
    assert contract["lambda_carry"] == "next_in_rollout_valid_transition_including_death"
    assert contract["inactive_worlds"] == "active_env_mask_freezes_world_rng_and_events"


@pytest.mark.parametrize(
    ("overrides", "expected_rng"),
    [
        ({"episode_seed_mode": "continuous_env_rng_v1"}, "shared_with_rollout"),
        ({"episode_seed_mode": "derived_env_episode_v1"}, "trainer_rng_used_only_for_sgd"),
        ({"episode_seed_mode": "derived_env_episode_v1", "sgd_seed": 19}, "independent_seed"),
    ],
)
def test_sampler_contract_records_the_rng_that_the_selected_lifecycle_uses(
    overrides: dict[str, object], expected_rng: str
) -> None:
    """Derived episode streams leave the trainer generator available only to SGD."""
    config = PQNConfig(
        num_envs=1,
        num_snakes=1,
        rollout_len=1,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        **overrides,
    )

    assert pqn_sampler_contract(config)["sgd_rng"] == expected_rng


def test_sampler_contract_records_disabled_and_non_replayed_snapshot_admission() -> None:
    """Pool provenance distinguishes disabled admission from a pinned due-slot loss."""
    base = {
        "num_envs": 1,
        "num_snakes": 1,
        "rollout_len": 1,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
    }
    disabled = pqn_sampler_contract(PQNConfig(**base, pool_admission_mode="disabled_v1"))
    scheduled = pqn_sampler_contract(PQNConfig(**base, pool_admission_mode="scheduled_v1"))

    assert disabled["pool_mutation"] == "disabled_v1"
    assert disabled["snapshot_admission"] == "disabled_pool_admission_mode"
    assert scheduled["pool_mutation"] == (
        "admission_after_successful_update_deferred_when_all_snapshots_pinned_"
        "and_not_replayed_off_schedule"
    )


def test_corrected_max_frame_one_completes_and_resets_on_next_rollout() -> None:
    """The terminal frame is counted now, then the following rollout starts fresh."""
    trainer = _corrected_trainer()
    trainer.cfg.max_frames = 1
    first = trainer.update()
    assert first.completed_episodes == 1
    assert first.rollout_capacity == 1
    assert first.episode_reset_count == 0

    second = trainer.update()
    assert second.completed_episodes == 1
    assert second.episode_reset_count == 1


def test_corrected_all_floor_at_first_step_slices_the_rollout() -> None:
    """All worlds ending together must not emit a padded zombie tail."""
    trainer = _corrected_trainer()
    trainer.sim.population_floor_reached = lambda: trainer.sim.frame >= 1
    roll = trainer._rollout()

    assert roll["actions"].shape[0] == 1
    assert roll["newly_completed_episodes"] == 1
    assert roll["batch_episode_finished"] is True


def test_corrected_all_floor_at_first_step_updates_without_event_shape_leak() -> None:
    """The shortened rollout must keep every event tensor aligned for telemetry."""
    trainer = _corrected_trainer()
    trainer.sim.population_floor_reached = lambda: trainer.sim.frame >= 1

    telemetry = trainer.update()

    assert telemetry.completed_episodes == 1
    assert telemetry.rollout_capacity == 1
    assert telemetry.valid_transitions <= telemetry.rollout_capacity


class _FixedRight:
    identity = "fixed:right:v1"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        self.calls.append((masks.copy(), slots.copy()))
        return np.full(masks.shape[0], 2, dtype=np.int64)


def test_fixed_source_actions_and_identity_are_real_rollout_contract_inputs() -> None:
    """Fixed opponents receive sparse slots and are signed into sampler metadata."""
    policy = _FixedRight()
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=1,
        hero_frac=0.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        rollout_policy_mode="fixed",
        fixed_policy_identity=policy.identity,
    )
    trainer = PQNTrainer(config, fixed_policy=FixedPolicySource(policy, policy.identity))
    roll = trainer._rollout()

    assert policy.calls and policy.calls[0][1].tolist() == [[0, 1]]
    assert roll["actions"][0, 0, 1] == 2
    checkpoint = trainer.checkpoint_state()
    assert checkpoint["sampler_contract"]["rollout_policy_source"] == {
        "mode": "fixed",
        "identity": policy.identity,
    }
    sampler = checkpoint["sampler_contract"]
    assert sampler["policy_assignment"] == "common_fixed_policy_for_nonhero_slots"
    assert sampler["pool_capacity"] == 0
    assert sampler["pool_add_interval"] is None
    assert sampler["requested_pool_capacity"] == config.pool_capacity
    assert sampler["requested_pool_add_interval"] == config.pool_add_interval
    assert sampler["empty_pool"] == "not_applicable_fixed_source"
    assert sampler["snapshot_identity"] == "not_applicable_fixed_source"
    assert sampler["episode_pinning"] is False
    assert sampler["pool_mutation"] == "not_applicable_fixed_source"
    assert sampler["snapshot_admission"] == "disabled_fixed_source"
    assert checkpoint["rollout_policy_source"]["mode"] == "fixed"
    assert checkpoint["rollout_policy_source"]["identity"] == policy.identity
    assert (
        checkpoint["rollout_policy_source"]["policy_source_contract_digest"]
        == checkpoint["policy_source_contract_digest"]
    )


def test_fixed_source_only_draws_exploration_for_actual_hero_slots() -> None:
    """The all-row Q forward must not consume an epsilon draw for fixed slots."""
    policy = _FixedRight()
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=1,
        hero_frac=0.0,
        eps_start=1.0,
        eps_end=1.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        rollout_policy_mode="fixed",
        fixed_policy_identity=policy.identity,
    )
    trainer = PQNTrainer(config, fixed_policy=FixedPolicySource(policy, policy.identity))

    class RecordingRng:
        def __init__(self) -> None:
            self.random_shapes: list[object] = []
            self.integer_calls: list[tuple[object, object, object]] = []

        def random(self, size=None):
            self.random_shapes.append(size)
            return np.zeros(size) if size is not None else 0.0

        def integers(self, low, high=None, size=None):
            self.integer_calls.append((low, high, size))
            return np.zeros(size, dtype=np.int64) if size is not None else 0

    rng = RecordingRng()
    trainer.rng = rng
    trainer._rollout()

    # Assignment draws one E×S selector; only forced HERO slot 0 draws epsilon.
    assert rng.random_shapes == [(1, 2), 1]
    # Assignment chooses fixed source IDs, then hero exploration chooses one legal action.
    assert rng.integer_calls == [(0, 1, (1, 2)), (3, None, None)]


def test_fixed_source_update_uses_its_actions_without_snapshot_admission() -> None:
    """The public update path uses fixed actions and keeps its pool disabled."""
    policy = _FixedRight()
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=1,
        hero_frac=0.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        pool_capacity=1,
        pool_add_interval=1,
        rollout_policy_mode="fixed",
        fixed_policy_identity=policy.identity,
    )
    trainer = PQNTrainer(config, fixed_policy=FixedPolicySource(policy, policy.identity))

    telemetry = trainer.update()

    assert policy.calls and policy.calls[0][1].tolist() == [[0, 1]]
    assert len(trainer.pool) == 0
    assert telemetry.policy_exposure[policy.identity] > 0


def test_pinned_assignment_and_lease_survive_two_rollouts_until_batch_reset() -> None:
    """Snapshot identity cannot change while the batch episode remains live."""
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=10,
        hero_frac=0.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        pool_capacity=1,
    )
    trainer = PQNTrainer(config)
    trainer.pool.add_snapshot(trainer.network)
    first = trainer._rollout()
    lease = trainer._episode_lease
    second = trainer._rollout()

    assert lease is not None and trainer._episode_lease is lease and not lease.closed
    assert np.array_equal(first["policy_ids"], second["policy_ids"])
    assert first["policy_identities"] == second["policy_identities"]


def test_due_admission_waits_for_pinned_episode_then_succeeds_after_completion() -> None:
    """A cap-one pool cannot replace a leased policy until the episode finishes."""
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=2,
        hero_frac=0.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        pool_capacity=1,
        pool_add_interval=1,
    )
    trainer = PQNTrainer(config)
    first_id = trainer.pool.add_snapshot(trainer.network)
    assert first_id is not None
    trainer.update_idx = 1  # Make both update boundaries due for admission.

    trainer.update()
    assert trainer.pool.policy_ids() == [first_id]
    assert trainer._episode_lease is not None and not trainer._episode_lease.closed

    trainer.update()
    assert trainer._episode_lease is None
    assert trainer.pool.policy_ids() != [first_id]
