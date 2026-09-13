"""Food-contact telemetry through real collectors, updates, and stale lanes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.training.pqn_trainer import (
    DECISION_PHASE_PRE_TRANSITION_V1,
    DECISION_PHASE_WATCH_PRE_MOVE_V1,
    HERO_POLICY_ID,
    PQNConfig,
    PQNTrainer,
)
from src.training.rollout_policies import FixedPolicySource


def _trainer(decision_phase: str) -> PQNTrainer:
    """Two lanes and one fixed opponent per lane; lane zero ends first."""

    class Fixed:
        identity = "fixed:food-telemetry-test:v1"

        def actions(self, masks, sim, slots):
            return np.argmax(masks, axis=1).astype(np.int64)

    policy = Fixed()
    source = FixedPolicySource(policy, policy.identity)
    config = PQNConfig(
        num_envs=2,
        num_snakes=2,
        rollout_len=2,
        max_frames=3,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        decision_phase_mode=decision_phase,
        episode_reset_mode="per_env_autoreset_v1",
        episode_seed_mode="derived_env_episode_v1",
        hero_frac=0.0,
        rollout_policy_mode="fixed",
        fixed_policy_identity=source.identity,
        pool_capacity=0,
        initial_food=0,
        max_food=0,
        game_width=300,
        game_height=300,
        sgd_epochs=1,
        minibatch_size=8,
        pad_sgd_batches=True,
        seed=19,
    )
    trainer = PQNTrainer(config, device=torch.device("cpu"), fixed_policy=source)
    trainer.sim.frame[0] = 2
    return trainer


@pytest.mark.parametrize(
    "decision_phase",
    [DECISION_PHASE_PRE_TRANSITION_V1, DECISION_PHASE_WATCH_PRE_MOVE_V1],
)
def test_update_counts_food_contacts_only_for_valid_heroes(
    monkeypatch: pytest.MonkeyPatch, decision_phase: str
) -> None:
    trainer = _trainer(decision_phase)
    # Model the event-provider contract: every slot has a boolean contact,
    # including a stale True retained after lane zero completes at t=0.
    # Opponent contacts must remain available in the raw collector buffer.
    monkeypatch.setattr(
        trainer.sim, "get_step_events", lambda: {"food_ate": np.ones((2, 2), dtype=bool)}
    )
    captured = {}
    original_rollout = trainer._rollout

    def capture_rollout():
        roll = original_rollout()
        captured.update(roll)
        return roll

    monkeypatch.setattr(trainer, "_rollout", capture_rollout)
    telemetry = trainer.update()

    assert captured["policy_ids"].tolist() == [[HERO_POLICY_ID, 0], [HERO_POLICY_ID, 0]]
    assert captured["food_ate"].shape == (2, 2, 2)
    assert captured["food_ate"].all()
    assert captured["valid"].tolist() == [
        [[True, True], [True, True]],
        [[False, False], [True, True]],
    ]
    # Three valid hero contacts; three valid opponent contacts and two stale
    # contacts are excluded by the production update aggregation.
    assert telemetry.hero_food_contact_events == 3
    assert telemetry.agent_steps == 3
    assert trainer.last_telemetry is telemetry


def test_update_accepts_historical_rollout_without_food_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(DECISION_PHASE_PRE_TRANSITION_V1)
    historical_roll = trainer._rollout()
    historical_roll.pop("food_ate")
    monkeypatch.setattr(trainer, "_rollout", lambda: historical_roll)

    telemetry = trainer.update()

    assert telemetry.hero_food_contact_events == 0
    assert telemetry.agent_steps == 3
