"""Regression coverage for the opt-in one-snake PQN episode lifecycle."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.core.runtime_contract import RunProvenance, canonical_digest
from src.model.obs_spec import RASTER31V3
from src.scripts.train_pqn import _telemetry_record
from src.training.pqn_lifecycle import (
    SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1,
    validate_pqn_episode_lifecycle_metadata,
)
from src.training.pqn_trainer import (
    PQNConfig,
    PQNPerEnvTelemetry,
    PQNSoloTelemetry,
    PQNTrainer,
    pqn_target_contract,
)


def _solo_config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 2,
        "num_snakes": 1,
        "rollout_len": 2,
        "max_frames": 20,
        "minibatches": 1,
        "minibatch_size": 4,
        "action_collapse_min_samples": 100_000,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "decision_phase_mode": "watch_pre_move_v1",
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
        "hero_frac": 1.0,
        "pool_capacity": 0,
        "pool_admission_mode": "disabled_v1",
        "episode_completion_mode": SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1,
    }
    values.update(overrides)
    return PQNConfig(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"num_snakes": 2},
        {"num_snakes": True},
        {"hero_frac": 0.8},
        {"hero_frac": True},
        {"pool_capacity": 1},
        {"pool_capacity": False},
        {"pool_admission_mode": "scheduled_v1"},
        {"decision_phase_mode": "pre_transition_v1"},
        {"episode_reset_mode": "batch_barrier_v1"},
        {"episode_seed_mode": "continuous_env_rng_v1"},
        {"rollout_policy_mode": "fixed", "fixed_policy_identity": "fixed:test"},
    ],
)
def test_solo_mode_rejects_invalid_combinations_before_trainer_allocation(
    override: dict,
):
    expected = (
        "per_env_autoreset_v1 requires corrected-v3 and derived_env_episode_v1"
        if override.get("episode_seed_mode") == "continuous_env_rng_v1"
        else "sole_snake_death_or_frame_cap_v1"
    )
    with pytest.raises(ValueError, match=expected):
        _solo_config(**override)


def test_solo_completion_marks_only_the_dead_lane_and_resets_next_rollout(monkeypatch):
    trainer = PQNTrainer(_solo_config())
    trainer._rollout()
    assert trainer._action_rngs is not None and trainer._episode_policy_ids is not None
    lane1_rng = copy.deepcopy(trainer._action_rngs[1].bit_generator.state)
    lane1_episode = int(trainer._episode_ids[1])
    lane1_policy = trainer._episode_policy_ids[1].copy()

    monkeypatch.setattr(trainer.sim, "get_done", lambda: np.asarray([[True], [False]]))
    completed = trainer._completed_environment_mask()
    assert completed.tolist() == [True, False]
    trainer._episode_finished_env[:] = completed
    trainer._prepare_derived_rollout()

    assert trainer._last_reset_env_indices.tolist() == [0]
    assert trainer._episode_ids.tolist() == [1, lane1_episode]
    assert trainer._action_rngs[1].bit_generator.state == lane1_rng
    assert np.array_equal(trainer._episode_policy_ids[1], lane1_policy)


def test_watch_solo_death_keeps_the_real_terminal_row_then_resets_only_next_boundary(
    monkeypatch,
):
    """A terminal event is stored before completion freezes its lane's padding."""
    trainer = PQNTrainer(_solo_config())
    real_step = trainer.sim.step_with_policy
    calls = 0

    def terminal_first_lane(selector, active_env_mask=None):
        nonlocal calls
        real_step(selector, active_env_mask)
        if calls == 0:
            # The real Watch callback and transition run first. This injects a
            # deterministic collision outcome at the post-step event seam so
            # the assertion does not depend on random arena geometry.
            trainer.sim.alive[0, 0] = False
            trainer.sim._last_done[0, 0] = True
            trainer.sim._last_reward[0, 0] = trainer.cfg.death_value
            trainer.sim._last_food_ate[0, 0] = True
            trainer.sim._last_transition_valid[0, 0] = True
        calls += 1

    monkeypatch.setattr(trainer.sim, "step_with_policy", terminal_first_lane)
    first = trainer._rollout()
    assert first["valid"][0, 0, 0]
    assert first["dones"][0, 0, 0]
    assert first["rewards"][0, 0, 0] == pytest.approx(trainer.cfg.death_value)
    assert first["food_ate"][0, 0, 0]
    assert first["newly_completed_episodes"] == 1
    # The completed lane receives no fresh transition in the padded tail.
    assert not first["valid"][1, 0].any()
    targets = trainer._compute_targets(first)
    assert targets[0, 0, 0].item() == pytest.approx(trainer.cfg.death_value)

    # Update aggregation only accepts valid hero rows. The injected contact on
    # the valid terminal row survives; stale padded rows cannot enter either
    # the useful-step odometer or the food count.
    expected_food = int(np.logical_and(first["food_ate"], first["valid"]).sum())
    monkeypatch.setattr(trainer, "_rollout", lambda: first)
    telemetry = trainer.update()
    assert telemetry.hero_food_contact_events == expected_food
    assert telemetry.agent_steps == int(first["valid"].sum())
    assert isinstance(telemetry, PQNSoloTelemetry)
    assert telemetry.episode_completion_mode == SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1
    assert _telemetry_record(telemetry)["episode_completion_mode"] == (
        SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1
    )

    assert trainer._action_rngs is not None and trainer._episode_policy_ids is not None
    lane1_rng = copy.deepcopy(trainer._action_rngs[1].bit_generator.state)
    lane1_policy = trainer._episode_policy_ids[1].copy()
    lane1_episode = int(trainer._episode_ids[1])
    assert trainer._episode_finished_env.tolist() == [True, False]

    # Delayed reset owns only the completed environment; its peer retains its
    # episode identity, assignment, and action RNG byte-for-byte.
    assert trainer._prepare_derived_rollout() == 1
    assert trainer._last_reset_env_indices.tolist() == [0]
    assert int(trainer._episode_ids[1]) == lane1_episode
    assert trainer._action_rngs[1].bit_generator.state == lane1_rng
    assert np.array_equal(trainer._episode_policy_ids[1], lane1_policy)


def test_default_s1_does_not_complete_on_death_before_the_frame_cap(monkeypatch):
    config = _solo_config(episode_completion_mode="population_floor_or_frame_cap_v1")
    trainer = PQNTrainer(config)
    monkeypatch.setattr(trainer.sim, "get_done", lambda: np.asarray([[True], [False]]))
    assert not trainer._completed_environment_mask().any()


def test_default_per_environment_telemetry_and_history_omit_solo_mode():
    trainer = PQNTrainer(_solo_config(episode_completion_mode="population_floor_or_frame_cap_v1"))
    telemetry = trainer.update()
    assert isinstance(telemetry, PQNPerEnvTelemetry)
    assert not isinstance(telemetry, PQNSoloTelemetry)
    assert "episode_completion_mode" not in telemetry.__dict__
    assert "episode_completion_mode" not in _telemetry_record(telemetry)


def test_solo_frame_cap_completion_does_not_depend_on_done(monkeypatch):
    trainer = PQNTrainer(_solo_config())
    trainer.sim.frame[:] = trainer.cfg.max_frames
    monkeypatch.setattr(trainer.sim, "get_done", lambda: np.zeros((2, 1), dtype=bool))
    assert trainer._completed_environment_mask().tolist() == [True, True]


def test_solo_frame_cap_keeps_the_nonterminal_successor_bootstrap_contract():
    config = _solo_config(max_frames=1, rollout_len=1)
    contract = pqn_target_contract(config)
    assert contract["death"] == "actual_done_reward_only"
    assert contract["truncation"] == "masked_max_q_of_successor"
    assert contract["reset"] == "selected_envs_at_next_rollout_boundary"


def test_frame_cap_rollout_uses_a_real_successor_bootstrap(monkeypatch):
    """The cap is a nonterminal truncation even in the one-snake lifecycle."""
    trainer = PQNTrainer(_solo_config(num_envs=1, max_frames=1, rollout_len=1, lambda_=0.0))
    real_step = trainer.sim.step_with_policy

    def cap_transition(selector, active_env_mask=None):
        real_step(selector, active_env_mask)
        trainer.sim._last_done[:] = False
        trainer.sim._last_transition_valid[:] = True
        trainer.sim._last_reward[:] = 2.0
        trainer.sim.alive[:] = True

    def known_q(tactical, strategic, scalars):
        return torch.arange(1, 7, dtype=torch.float32, device=tactical.device).repeat(
            tactical.shape[0], 1
        )

    monkeypatch.setattr(trainer.sim, "step_with_policy", cap_transition)
    monkeypatch.setattr(trainer.network, "forward", known_q)
    roll = trainer._rollout()
    roll["next_mask"].fill_(True)
    assert roll["valid"][0, 0, 0] and not roll["dones"][0, 0, 0]
    assert trainer._episode_finished_env.tolist() == [True]
    assert trainer._compute_targets(roll)[0, 0, 0].item() == pytest.approx(
        2.0 + trainer.cfg.gamma * 6
    )


def test_native_solo_checkpoint_round_trips_and_rejects_rehashed_non_solo_world():
    trainer = PQNTrainer(_solo_config(num_envs=1))
    checkpoint = trainer.checkpoint_state()
    validated = validate_pqn_episode_lifecycle_metadata(
        checkpoint, allow_corrected_v3_adapter=False
    )
    assert validated.descriptor["episode_completion_mode"] == SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1
    assert checkpoint["episode_completion_mode"] == SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1

    forged = copy.deepcopy(checkpoint)
    forged["effective_world"]["num_snakes"] = 2
    forged["effective_world_digest"] = canonical_digest(forged["effective_world"])
    forged["run_provenance"]["world_digest"] = forged["effective_world_digest"]
    forged.update(RunProvenance(**forged["run_provenance"]).to_metadata())
    with pytest.raises(ValueError, match="effective_world.num_snakes=1"):
        validate_pqn_episode_lifecycle_metadata(forged, allow_corrected_v3_adapter=False)

    forged_sampler = copy.deepcopy(checkpoint)
    forged_sampler["sampler_contract"]["num_snakes"] = 2
    forged_sampler["sampler_contract_digest"] = canonical_digest(forged_sampler["sampler_contract"])
    forged_sampler["run_provenance"]["sampler_digest"] = forged_sampler["sampler_contract_digest"]
    forged_sampler.update(RunProvenance(**forged_sampler["run_provenance"]).to_metadata())
    with pytest.raises(ValueError, match="required S1 Watch"):
        validate_pqn_episode_lifecycle_metadata(forged_sampler, allow_corrected_v3_adapter=False)
