"""Contract tests for the constructor-only solo ambient-food objective."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.inference_agent import InferenceAgent
from src.model.obs_spec import RASTER31V3
from src.training.pqn_lifecycle import SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1
from src.training.pqn_trainer import (
    DECISION_PHASE_PRE_TRANSITION_V1,
    DECISION_PHASE_WATCH_PRE_MOVE_V1,
    PQNConfig,
    PQNTrainer,
    pqn_reward_contract,
    pqn_target_contract,
)


def _solo_config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 2,
        "num_snakes": 1,
        "rollout_len": 1,
        "max_frames": 5,
        "minibatches": 1,
        "minibatch_size": 2,
        "action_collapse_min_samples": 10_000,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "decision_phase_mode": DECISION_PHASE_WATCH_PRE_MOVE_V1,
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
        "hero_frac": 1.0,
        "pool_capacity": 0,
        "pool_admission_mode": "disabled_v1",
        "episode_completion_mode": SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1,
        "ambient_food_reward_coefficient": 0.25,
    }
    values.update(overrides)
    return PQNConfig(**values)


def _install_body(trainer: PQNTrainer, env: int, cells: list[tuple[int, int]]) -> None:
    """Install a live right-facing body in BatchSim's production ring order."""
    sim = trainer.sim
    sim.bodies[env, 0] = 0
    sim.head_ptr[env, 0] = 0
    for offset, cell in enumerate(cells):
        sim.bodies[env, 0, (-offset) % sim.cap] = cell
    sim.seg_count[env, 0] = len(cells)
    sim.length[env, 0] = len(cells)
    sim.alive[env, 0] = True
    sim.direction[env, 0] = 1
    sim.boost_frames[env, 0] = 0
    sim._reward_prev_length[env, 0] = len(cells)


def _set_food(
    trainer: PQNTrainer, env: int, cells: list[tuple[int, int]], corpse: set[tuple[int, int]]
) -> None:
    sim = trainer.sim
    sim.food_cells[env] = list(cells)
    sim.food_set[env] = set(cells)
    sim.corpse_cells[env] = set(corpse)


def test_ambient_overlay_uses_source_validity_and_post_step_liveness() -> None:
    trainer = PQNTrainer(_solo_config(), device=torch.device("cpu"))
    trainer.sim.alive[:] = [[True], [False]]

    overlay = trainer._ambient_food_reward_overlay(
        np.asarray([[True], [True]]), np.asarray([[True], [True]])
    )
    np.testing.assert_array_equal(overlay, [[0.25], [0.0]])
    np.testing.assert_array_equal(
        trainer._ambient_food_reward_overlay(
            np.asarray([[False], [True]]), np.asarray([[True], [False]])
        ),
        [[0.0], [0.0]],
    )


@pytest.mark.parametrize("coefficient", [-0.01, float("nan"), float("inf"), True])
def test_invalid_ambient_food_coefficients_are_rejected(coefficient: object) -> None:
    with pytest.raises(ValueError, match="ambient_food_reward_coefficient"):
        _solo_config(ambient_food_reward_coefficient=coefficient)


@pytest.mark.parametrize(
    "overrides",
    [
        {"recipe": "legacy", "obs_spec": "raster31v2"},
        {"num_snakes": 2},
        {"episode_completion_mode": "population_floor_or_frame_cap_v1"},
        {"living_mass_reward_coefficient": 0.01},
    ],
)
def test_ambient_treatment_rejects_unsupported_compositions(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _solo_config(**overrides)


def test_zero_mode_keeps_literal_base_contract_and_nonzero_binds_digests() -> None:
    zero = _solo_config(ambient_food_reward_coefficient=0.0)
    treated = _solo_config()
    assert pqn_reward_contract(zero) == {
        "version": "pqn-potential-reward-v2",
        "gamma": zero.gamma,
        "potential": "logical_length_divided_by_10",
        "terminal_potential": 0.0,
        "shaping": "gamma_times_next_potential_minus_previous_potential",
        "death_bonus": zero.death_value,
        "kill_scale": zero.kill_scale,
        "kill_sum": "victim_order_left_to_right",
        "clipping": None,
    }
    contract = pqn_reward_contract(treated)
    assert contract["event_source"] == "ambient_food_ate"
    assert contract["source_classification"] == "captured_before_consumed_cell_removal"
    assert contract["diagnostic_only"] is True
    assert contract["promotion_eligible"] is False
    assert (
        pqn_target_contract(treated)["reward_digest"] != pqn_target_contract(zero)["reward_digest"]
    )


def test_rollout_buffers_source_partition_and_reward_only_ambient_alive_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PQNTrainer(_solo_config(), device=torch.device("cpu"))
    events = {
        "food_ate": np.asarray([[True], [True]]),
        "ambient_food_ate": np.asarray([[True], [False]]),
        "corpse_food_ate": np.asarray([[False], [True]]),
    }
    monkeypatch.setattr(trainer.sim, "get_step_events", lambda: events)
    monkeypatch.setattr(trainer.sim, "get_transition_valid", lambda: np.ones((2, 1), dtype=bool))
    monkeypatch.setattr(trainer.sim, "get_alive", lambda: np.asarray([[True], [True]]))

    roll = trainer._rollout()
    np.testing.assert_array_equal(
        roll["food_ate"], roll["ambient_food_ate"] | roll["corpse_food_ate"]
    )
    np.testing.assert_array_equal(roll["ambient_food_rewards"], [[[0.25], [0.0]]])
    np.testing.assert_array_equal(
        roll["rewards"], roll["base_rewards"] + roll["ambient_food_rewards"]
    )


def test_enabled_treatment_rejects_missing_source_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PQNTrainer(_solo_config(), device=torch.device("cpu"))
    monkeypatch.setattr(
        trainer.sim,
        "get_step_events",
        lambda: {"food_ate": np.zeros((2, 1), dtype=bool)},
    )
    with pytest.raises(RuntimeError, match="requires BatchSim ambient_food_ate"):
        trainer._rollout()


def test_live_collector_rewards_only_surviving_ambient_pickup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch collection preserves source labels through terminal resolution."""
    trainer = PQNTrainer(
        _solo_config(
            num_envs=3,
            game_width=80,
            game_height=100,
            initial_food=0,
            max_food=0,
        ),
        device=torch.device("cpu"),
    )
    # Lane 0 eats ambient food and survives. Lane 1 eats ambient food at the
    # wall-impact cell, so consumption precedes an actual terminal collision.
    # Lane 2 consumes an actual corpse-tagged pellet and must earn no overlay.
    _install_body(trainer, 0, [(4, 4)])
    _install_body(trainer, 1, [(6, 4)])
    _install_body(trainer, 2, [(4, 6)])
    _set_food(trainer, 0, [(5, 4)], set())
    _set_food(trainer, 1, [(7, 4)], set())
    _set_food(trainer, 2, [(5, 6)], {(5, 6)})
    trainer.sim._rebuild_traversed_from_heads()
    trainer.sim._refresh_action_masks()

    def straight(*_args: object) -> tuple[np.ndarray, torch.Tensor]:
        return (
            np.ones((3, 1), dtype=np.int64),
            torch.zeros((3, 1, 6), dtype=torch.float32, device=trainer.device),
        )

    monkeypatch.setattr(trainer, "_watch_actions", straight)
    roll = trainer._rollout()

    np.testing.assert_array_equal(roll["food_ate"], [[[True], [True], [True]]])
    np.testing.assert_array_equal(roll["ambient_food_ate"], [[[True], [True], [False]]])
    np.testing.assert_array_equal(roll["corpse_food_ate"], [[[False], [False], [True]]])
    np.testing.assert_array_equal(roll["dones"], [[[False], [True], [False]]])
    np.testing.assert_array_equal(roll["ambient_food_rewards"], [[[0.25], [0.0], [0.0]]])


@pytest.mark.parametrize(
    "decision_phase", [DECISION_PHASE_PRE_TRANSITION_V1, DECISION_PHASE_WATCH_PRE_MOVE_V1]
)
def test_zero_coefficient_collects_passive_source_events_in_both_collectors(
    monkeypatch: pytest.MonkeyPatch, decision_phase: str
) -> None:
    values: dict[str, object] = {
        "num_envs": 1,
        "num_snakes": 2,
        "rollout_len": 1,
        "max_frames": 5,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "decision_phase_mode": decision_phase,
        "ambient_food_reward_coefficient": 0.0,
    }
    trainer = PQNTrainer(PQNConfig(**values), device=torch.device("cpu"))
    events = {
        "food_ate": np.asarray([[True, True]]),
        "ambient_food_ate": np.asarray([[True, False]]),
        "corpse_food_ate": np.asarray([[False, True]]),
    }
    monkeypatch.setattr(trainer.sim, "get_step_events", lambda: events)
    roll = trainer._rollout()
    np.testing.assert_array_equal(
        roll["food_ate"], roll["ambient_food_ate"] | roll["corpse_food_ate"]
    )
    np.testing.assert_array_equal(roll["rewards"], roll["base_rewards"])
    assert not np.asarray(roll["ambient_food_rewards"]).any()


def test_treatment_checkpoint_records_contract_and_native_reload_identity(tmp_path) -> None:
    trainer = PQNTrainer(_solo_config(), device=torch.device("cpu"))
    checkpoint = trainer.checkpoint_state()
    assert checkpoint["ambient_food_reward_coefficient"] == 0.25
    assert checkpoint["reward_contract"]["event_source"] == "ambient_food_ate"
    path = tmp_path / "ambient-objective.pth"
    trainer.save_checkpoint(str(path))
    agent = InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))
    assert agent.obs_spec == RASTER31V3
    assert agent.network.output_size == checkpoint["output_size"]
