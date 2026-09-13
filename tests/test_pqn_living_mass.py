"""Focused contract coverage for PQN's opt-in living-mass base objective."""

from __future__ import annotations

import numpy as np
import pytest

from src.scripts import train_pqn
from src.scripts.train_pqn import validate_pqn_resume_checkpoint_config
from src.training.pqn_trainer import (
    DECISION_PHASE_WATCH_PRE_MOVE_V1,
    PQNConfig,
    PQNTrainer,
    pqn_reward_contract,
    pqn_target_contract,
)


def _config(**overrides: object) -> PQNConfig:
    values: dict[str, object] = {
        "num_envs": 1,
        "num_snakes": 2,
        "rollout_len": 1,
        "minibatches": 1,
        "minibatch_size": 2,
        "hero_frac": 1.0,
        "pool_capacity": 0,
        "max_frames": 1,
        "recipe": "corrected-v3",
        "obs_spec": "raster31v3",
        "flip_augment": False,
        "living_mass_reward_coefficient": 0.25,
    }
    values.update(overrides)
    return PQNConfig(**values)


@pytest.mark.parametrize("decision_phase", [None, DECISION_PHASE_WATCH_PRE_MOVE_V1])
def test_both_rollout_paths_apply_post_step_living_mass_overlay(decision_phase: str | None) -> None:
    """The ordinary and WATCH seams use valid, surviving post-step logical mass."""
    overrides = {} if decision_phase is None else {"decision_phase_mode": decision_phase}
    trainer = PQNTrainer(_config(**overrides))

    roll = trainer._rollout()

    valid = np.asarray(roll["valid"], dtype=bool)
    survived = ~np.asarray(roll["dones"], dtype=bool)
    overlay = np.asarray(roll["living_mass_rewards"])
    # At T=1 BatchSim's current lengths are precisely the stored post-step lengths.
    expected = 0.25 * valid[0] * survived[0] * trainer.sim.get_lengths()
    np.testing.assert_array_equal(overlay[0], expected)
    np.testing.assert_array_equal(
        np.asarray(roll["rewards"]),
        np.asarray(roll["base_rewards"]) + np.asarray(roll["living_mass_rewards"]),
    )


def test_overlay_boundaries_zero_invalid_and_death_but_include_floor_and_cap_survivors() -> None:
    """Boundary selection depends only on post-step validity/aliveness, before reset."""
    trainer = PQNTrainer(_config())
    trainer.sim.alive[:] = [[True, False]]
    trainer.sim.length[:] = [[7, 11]]

    # The first row stands for either a population-floor or frame-cap survivor:
    # both boundaries are after the completed transition and therefore receive mass.
    np.testing.assert_array_equal(
        trainer._living_mass_reward_overlay(np.array([[True, True]])), [[1.75, 0.0]]
    )
    np.testing.assert_array_equal(
        trainer._living_mass_reward_overlay(np.array([[False, True]])), [[0.0, 0.0]]
    )


def test_zero_mode_preserves_legacy_contract_and_rollout_reward_stream() -> None:
    """Disabled mode leaves both the old descriptor and simulator rewards unchanged."""
    config = _config(living_mass_reward_coefficient=0.0)
    trainer = PQNTrainer(config)
    roll = trainer._rollout()

    assert pqn_reward_contract(config) == {
        "version": "pqn-potential-reward-v2",
        "gamma": config.gamma,
        "potential": "logical_length_divided_by_10",
        "terminal_potential": 0.0,
        "shaping": "gamma_times_next_potential_minus_previous_potential",
        "death_bonus": config.death_value,
        "kill_scale": config.kill_scale,
        "kill_sum": "victim_order_left_to_right",
        "clipping": None,
    }
    np.testing.assert_array_equal(roll["rewards"], roll["base_rewards"])
    assert not np.asarray(roll["living_mass_rewards"]).any()


def test_objective_contract_changes_target_and_rejects_continuation_mismatch() -> None:
    """A continuation cannot silently cross the base-objective boundary."""
    trained_config = _config(living_mass_reward_coefficient=0.25)
    checkpoint = PQNTrainer(trained_config).checkpoint_state()
    zero_config = _config(living_mass_reward_coefficient=0.0)

    assert pqn_reward_contract(trained_config)["base_reward_contract"] == pqn_reward_contract(
        zero_config
    )
    assert (
        pqn_target_contract(trained_config)["reward_digest"]
        != pqn_target_contract(zero_config)["reward_digest"]
    )
    with pytest.raises(ValueError, match="reward_contract.*semantics"):
        validate_pqn_resume_checkpoint_config(checkpoint, zero_config)


@pytest.mark.parametrize("coefficient", [-0.001, float("nan"), float("inf"), True])
def test_invalid_living_mass_coefficients_are_rejected(coefficient: object) -> None:
    """The treatment cannot be enabled with an ambiguous or unsafe coefficient."""
    with pytest.raises(ValueError, match="living_mass_reward_coefficient"):
        _config(living_mass_reward_coefficient=coefficient)


def test_nonzero_living_mass_requires_corrected_v3() -> None:
    """The additive objective is intentionally unavailable to the legacy recipe."""
    with pytest.raises(ValueError, match="requires recipe='corrected-v3'"):
        PQNConfig(living_mass_reward_coefficient=0.25)


def test_yaml_living_mass_coefficient_flows_through_the_existing_pqn_mapping(tmp_path) -> None:
    """The typed optional YAML override reaches the trainer config mapping."""
    path = tmp_path / "living-mass.yaml"
    path.write_text("pqn:\n  living_mass_reward_coefficient: 0.003\n", encoding="utf-8")

    assert train_pqn._load_config_overrides(str(path)) == {"living_mass_reward_coefficient": 0.003}
