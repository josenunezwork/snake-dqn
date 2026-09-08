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
from src.training.pqn_trainer import PQNConfig, PQNTrainer, pqn_target_contract


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
    assert contract["version"] == "pqn-qlambda-corrected-v3"
    assert contract["death"] == "actual_done_reward_only"
    assert contract["lambda_carry"] == "next_in_rollout_valid_transition_including_death"
    assert contract["inactive_worlds"] == "active_env_mask_freezes_world_rng_and_events"
