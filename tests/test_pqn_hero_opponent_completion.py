"""Regression contracts for the S2 terminal-hero PQN lifecycle."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.simd_env.eval_engine import _TerminalHeroBatchSim
from src.training.pqn_lifecycle import (
    HERO_DEATH_OR_FRAME_CAP_S2_V1,
    validate_pqn_episode_lifecycle_metadata,
)
from src.training.pqn_trainer import PQNConfig, PQNTrainer, pqn_target_contract
from src.training.rollout_policies import FixedPolicySource, canonical_watch_scripted_source


def _s2_config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 2,
        "num_snakes": 2,
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
        "hero_frac": 0.0,
        "pool_capacity": 0,
        "pool_admission_mode": "disabled_v1",
        "rollout_policy_mode": "fixed",
        "fixed_policy_identity": "scripted-anchor/v1:greedy_food",
        "initial_opponent_checkpoint_sha256": None,
        "episode_completion_mode": HERO_DEATH_OR_FRAME_CAP_S2_V1,
    }
    values.update(overrides)
    return PQNConfig(**values)


def _trainer(**overrides: object) -> PQNTrainer:
    fixed = canonical_watch_scripted_source("greedy_food")
    return PQNTrainer(_s2_config(**overrides), fixed_policy=fixed)


@pytest.mark.parametrize(
    "override",
    [
        {"num_snakes": 1},
        {"hero_frac": 1.0},
        {"pool_capacity": 1},
        {"pool_admission_mode": "scheduled_v1"},
        {"rollout_policy_mode": "snapshot_pool", "fixed_policy_identity": None},
        {"fixed_policy_identity": None},
        {"initial_opponent_checkpoint_sha256": "0" * 64},
        {"decision_phase_mode": "pre_transition_v1"},
        {"episode_reset_mode": "batch_barrier_v1"},
        {"episode_seed_mode": "continuous_env_rng_v1"},
    ],
)
def test_s2_mode_rejects_contract_mutations_before_trainer_allocation(override: dict) -> None:
    expected = (
        "per_env_autoreset_v1 requires corrected-v3 and derived_env_episode_v1"
        if override.get("episode_seed_mode") == "continuous_env_rng_v1"
        else "hero_death_or_frame_cap_s2_v1"
    )
    with pytest.raises(ValueError, match=expected):
        _s2_config(**override)


def test_s2_config_builds_terminal_hero_sim_with_opponent_respawn_enabled() -> None:
    trainer = _trainer()
    assert isinstance(trainer.sim, _TerminalHeroBatchSim)
    assert trainer.sim.train_mode is False
    assert trainer.sim.allow_respawn is True
    assert trainer.cfg.hero_frac == 0.0
    assert trainer.cfg.pool_capacity == 0
    assert trainer.fixed_policy is not None
    assert not trainer._initial_snapshot_preloaded


def test_hero_death_is_one_valid_terminal_row_and_does_not_respawn(monkeypatch) -> None:
    trainer = _trainer()
    real_step, calls = trainer.sim.step_with_policy, 0

    def hero_terminal(selector, active_env_mask=None):
        nonlocal calls
        real_step(selector, active_env_mask)
        if calls == 0:
            trainer.sim.alive[0, 0] = False
            trainer.sim._last_done[0, 0] = True
            trainer.sim._last_transition_valid[0, 0] = True
            trainer.sim._last_reward[0, 0] = trainer.cfg.death_value
        calls += 1

    monkeypatch.setattr(trainer.sim, "step_with_policy", hero_terminal)
    roll = trainer._rollout()
    assert roll["valid"][0, 0, 0] and roll["dones"][0, 0, 0]
    assert not roll["valid"][1, 0, 0].any()
    assert trainer._compute_targets(roll)[0, 0, 0].item() == trainer.cfg.death_value
    assert roll["newly_completed_episodes"] == 1
    trainer.sim.respawn_timer[0, 0] = 0
    trainer.sim._respawn_dead()
    assert not trainer.sim.get_alive()[0, 0]


def test_opponent_death_does_not_complete_the_hero_episode_and_respawns(monkeypatch) -> None:
    trainer = _trainer()
    monkeypatch.setattr(
        trainer.sim, "get_done", lambda: np.asarray([[False, True], [False, False]])
    )
    assert not trainer._completed_environment_mask().any()
    trainer.sim.alive[0, 1] = False
    trainer.sim.respawn_timer[0, 1] = 0
    trainer.sim._respawn_dead()
    assert trainer.sim.get_alive()[0, 1]


def test_s2_frame_cap_completes_with_nonterminal_successor_bootstrap(monkeypatch) -> None:
    trainer = _trainer(num_envs=1, max_frames=1, rollout_len=1, lambda_=0.0)
    real_step = trainer.sim.step_with_policy

    def cap_transition(selector, active_env_mask=None):
        real_step(selector, active_env_mask)
        trainer.sim._last_done[:] = False
        trainer.sim._last_transition_valid[:] = True
        trainer.sim._last_reward[:] = 2.0
        trainer.sim.alive[:, 0] = True

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
    assert pqn_target_contract(trainer.cfg)["truncation"] == "masked_max_q_of_successor"


def test_selected_hero_reset_preserves_peer_episode_policy_and_rng(monkeypatch) -> None:
    trainer = _trainer()
    trainer._rollout()
    assert trainer._action_rngs is not None and trainer._episode_policy_ids is not None
    peer_rng = copy.deepcopy(trainer._action_rngs[1].bit_generator.state)
    peer_world_rng = copy.deepcopy(trainer.sim._rngs[1]._rng.getstate())
    peer_bodies = trainer.sim.bodies[1].copy()
    peer_food = copy.deepcopy(trainer.sim.food_cells[1])
    peer_episode = int(trainer._episode_ids[1])
    peer_policy = trainer._episode_policy_ids[1].copy()
    monkeypatch.setattr(
        trainer.sim, "get_done", lambda: np.asarray([[True, False], [False, False]])
    )
    trainer._episode_finished_env[:] = trainer._completed_environment_mask()
    assert trainer._prepare_derived_rollout() == 1
    assert trainer._last_reset_env_indices.tolist() == [0]
    assert trainer._action_rngs[1].bit_generator.state == peer_rng
    assert trainer.sim._rngs[1]._rng.getstate() == peer_world_rng
    assert np.array_equal(trainer.sim.bodies[1], peer_bodies)
    assert trainer.sim.food_cells[1] == peer_food
    assert int(trainer._episode_ids[1]) == peer_episode
    assert np.array_equal(trainer._episode_policy_ids[1], peer_policy)


def test_only_valid_hero_slot_can_enter_sgd(monkeypatch) -> None:
    trainer = _trainer()
    roll = trainer._rollout()
    assert np.all(roll["policy_ids"][:, 0] < 0)
    assert np.all(roll["policy_ids"][:, 1] >= 0)
    roll["valid"][1, 0, 0] = False
    expected = int(roll["valid"][:, :, 0].sum())
    monkeypatch.setattr(trainer, "_rollout", lambda: roll)
    telemetry = trainer.update()
    assert telemetry.agent_steps == expected
    assert trainer._last_sgd_sampling["eligible_hero_transitions"] == expected


def test_fixed_opponent_dispatch_skips_completed_envs_and_dead_opponents() -> None:
    class RecordingPolicy:
        identity = "fixed:recording-s2-v1"

        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def actions(self, masks, sim, slots):
            self.calls.append(slots.copy())
            return np.asarray([np.flatnonzero(row)[0] for row in masks], dtype=np.int64)

    policy = RecordingPolicy()
    source = FixedPolicySource(policy, policy.identity)
    trainer = PQNTrainer(_s2_config(fixed_policy_identity=policy.identity), fixed_policy=source)
    trainer._prepare_derived_rollout()
    assert trainer._episode_policy_ids is not None
    trainer.sim.alive[0, 1] = False
    obs, mask = trainer._current_obs()
    trainer._watch_actions(
        trainer._episode_policy_ids,
        trainer.pool,
        obs,
        mask,
        0.0,
        trainer.sim,
        np.asarray([False, True]),
    )
    assert len(policy.calls) == 1
    assert policy.calls[0].tolist() == [[1, 1]]
    trainer.sim.alive[1, 1] = False
    policy.calls.clear()
    obs, mask = trainer._current_obs()
    trainer._watch_actions(
        trainer._episode_policy_ids,
        trainer.pool,
        obs,
        mask,
        0.0,
        trainer.sim,
        np.asarray([False, True]),
    )
    assert policy.calls == []


def test_closed_s2_metadata_accepts_the_mode_and_rejects_lifecycle_mutation() -> None:
    trainer = _trainer(num_envs=1)
    checkpoint = trainer.checkpoint_state()
    accepted = validate_pqn_episode_lifecycle_metadata(checkpoint, allow_corrected_v3_adapter=False)
    assert accepted.descriptor["episode_completion_mode"] == HERO_DEATH_OR_FRAME_CAP_S2_V1
    forged = copy.deepcopy(checkpoint)
    forged["episode_lifecycle_contract"]["terminal_semantics"]["hero_death"] = "respawn"
    with pytest.raises(ValueError, match="episode_lifecycle_contract"):
        validate_pqn_episode_lifecycle_metadata(forged, allow_corrected_v3_adapter=False)


def test_historical_default_completion_mode_is_unchanged() -> None:
    assert PQNConfig().episode_completion_mode == "population_floor_or_frame_cap_v1"
