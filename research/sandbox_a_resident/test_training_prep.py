"""CPU-only bounded fixtures for TRAIN-01 preparation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic
from research.sandbox_a_resident.training_prep import (
    PreparationConfig,
    PreparationFailure,
    calibrate,
    prepare,
    solve_temperature,
)
from src.model.ego2s_network import Ego2sNet


class Task:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.tick = 0
        self.config = SimpleNamespace(horizon=5000)
        self.age = torch.zeros(3, dtype=torch.long)
        self.roster_id = torch.tensor([0, 1, 2])
        self.sim = SimpleNamespace(length=torch.tensor([[100], [250], [450]], dtype=torch.long))

    def observe(self):
        return (
            torch.zeros((3, 7, 31, 31), dtype=torch.uint8),
            torch.zeros((3, 4, 37, 37), dtype=torch.uint8),
            torch.zeros((3, 60)),
            torch.ones((3, 6), dtype=torch.bool),
        )

    def step(self, action):
        del action
        self.tick += 1
        self.age += 1
        self.sim.length[:, 0] = torch.tensor(
            ([650, 850, 100], [250, 450, 650], [850, 100, 250])[self.tick % 3]
        )
        reset = self.age.ge(2)
        self.age[reset] = 0
        return SimpleNamespace(
            reward=torch.ones(3), terminal=reset, cut=torch.zeros(3, dtype=torch.bool), reset=reset
        )

    def state_dict(self):
        return {"age": self.age.clone(), "length": self.sim.length.clone()}

    def load_state_dict(self, state):
        self.age.copy_(state["age"])
        self.sim.length.copy_(state["length"])


def factory(stage: str):
    return Task(stage)


def _models():
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    return B8Actor(b8_model=b8), ClockedPopArtCritic(b8)


def test_temperature_preserves_ties_as_the_greedy_set():
    logits = torch.tensor([[2.0, 2.0, 0.0, -1.0, -1.0, -1.0]])
    legal = torch.ones_like(logits, dtype=torch.bool)
    temperature = solve_temperature(logits, legal, 0.01)
    # Departure excludes both tied maximizing actions; a tie-break implementation
    # would calibrate a radically different temperature.
    p = torch.softmax(logits / temperature, -1)
    assert torch.isclose(1 - p[:, :2].sum(), torch.tensor(0.01), atol=2e-4)


def test_calibration_is_bounded_and_reinstalls_exact_b8_head_scale():
    actor, _ = _models()
    original = actor.reference.adv.weight.detach().clone()
    temperature, rows = calibrate(
        factory, actor, PreparationConfig.tiny(), torch.Generator().manual_seed(4)
    )
    assert rows == 10 and temperature > 0
    # prepare does installation; calibration itself never trains or changes B8.
    torch.testing.assert_close(actor.reference.adv.weight, original)


def test_calibration_requires_all_five_frozen_length_bins_before_fixed_cap():
    actor, _ = _models()
    calls = []

    class ShortTask(Task):
        def step(self, action):
            result = super().step(action)
            self.sim.length.fill_(100)
            return result

    def short_factory(stage):
        calls.append(stage)
        return ShortTask(stage)

    with pytest.raises(PreparationFailure, match="five frozen length-bin quotas"):
        calibrate(
            short_factory,
            actor,
            replace(PreparationConfig.tiny(), calibration_max_rows=12),
            torch.Generator().manual_seed(3),
        )
    assert calls == ["calibration"]


def test_failed_tiny_critic_endpoint_never_returns_actor_ready():
    actor, critic = _models()
    config = PreparationConfig.tiny()
    with pytest.raises(PreparationFailure, match="critic preparation endpoint failed"):
        prepare(factory, actor, critic, config, torch.Generator().manual_seed(5))
    # The fixed tiny task has constant return, so late EV cannot meet .10;
    # preparation failure is terminal before the PPO loop receives an actor.


def test_calibrated_callback_is_a_single_pre_critic_boundary():
    actor, critic = _models()
    seen = []

    def boundary(state, boundary_actor, boundary_critic, optimizer):
        seen.append((state.stage, state.critic_rows, boundary_actor.temperature, optimizer))
        assert boundary_critic is critic

    with pytest.raises(PreparationFailure):
        prepare(
            factory,
            actor,
            critic,
            PreparationConfig.tiny(),
            torch.Generator().manual_seed(8),
            on_calibrated=boundary,
        )
    assert len(seen) == 1
    assert seen[0][0:2] == ("calibrated", 0)
    assert seen[0][2] > 0


def test_tiny_fixture_limits_are_explicitly_not_production_values():
    tiny, production = PreparationConfig.tiny(), PreparationConfig()
    assert tiny.critic_min_fit_rows < production.critic_min_fit_rows
    assert tiny.min_holdout_lives < production.min_holdout_lives


def test_critic_cap_charges_every_simulated_world_not_only_fit_slots():
    actor, critic = _models()
    config = replace(
        PreparationConfig.tiny(),
        critic_max_rows=3,
        critic_min_fit_rows=3,
        min_holdout_lives=0,
        min_holdout_late_rows=0,
        min_holdout_lives_per_mix=0,
    )
    with pytest.raises(PreparationFailure) as failed:
        prepare(factory, actor, critic, config, torch.Generator().manual_seed(11))
    # E3 runs one whole world tick.  Only two slots are fit, but all three
    # simulator rows are charged before the cap blocks another tick.
    assert failed.value.state is not None
    assert failed.value.state.critic_rows == 3
    assert failed.value.state.fit_rows == 0
