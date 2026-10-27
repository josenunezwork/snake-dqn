"""Detached per-update PPO telemetry contracts for TRAIN-01-08."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import torch

from research.sandbox_a_resident.training_learner import PPOConfig, PPORollout, PPOTrainer
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic, categorical
from research.sandbox_a_resident.training_proxy import ProxyConfig, evaluate
from src.model.ego2s_network import Ego2sNet


def _rollout(actor: B8Actor, critic: ClockedPopArtCritic) -> PPORollout:
    torch.manual_seed(71)
    time, worlds = 3, 3
    local = torch.randint(0, 256, (time, worlds, 7, 31, 31), dtype=torch.uint8)
    glob = torch.randint(0, 256, (time, worlds, 4, 37, 37), dtype=torch.uint8)
    scalars = torch.rand(time, worlds, 60)
    masks = torch.ones(time, worlds, 6, dtype=torch.bool)
    clock = torch.linspace(1, 0.8, time).unsqueeze(1).expand(time, worlds)
    with torch.no_grad():
        logits = actor(local.flatten(0, 1), glob.flatten(0, 1), scalars.flatten(0, 1))
        distribution = categorical(logits, masks.flatten(0, 1))
        actions = distribution.sample().reshape(time, worlds)
        logp = distribution.log_prob(actions.flatten()).reshape(time, worlds)
        values = critic(
            local.flatten(0, 1), glob.flatten(0, 1), scalars.flatten(0, 1), clock.flatten()
        ).reshape(time, worlds)
    return PPORollout(
        local,
        glob,
        scalars,
        masks,
        actions,
        logp,
        torch.ones(time, worlds),
        values,
        torch.zeros(time, worlds, dtype=torch.bool),
        torch.zeros(time, worlds, dtype=torch.bool),
        torch.zeros(worlds),
        torch.ones(worlds, dtype=torch.bool),
        clock,
        torch.tensor([[0, 1, 2]]).expand(time, -1),
        torch.tensor([[650, 750, 900]]).expand(time, -1),
    )


def _assert_same(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right):
            _assert_same(item_left, item_right)
    else:
        assert left == right


def test_detached_monitoring_preserves_update_state_and_rng() -> None:
    """On/off telemetry changes observations only, including the next update RNG state."""
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor_on, critic_on = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    actor_off, critic_off = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    actor_off.load_state_dict(copy.deepcopy(actor_on.state_dict()))
    critic_off.load_state_dict(copy.deepcopy(critic_on.state_dict()))
    config = PPOConfig(minibatch=3)
    on, off = PPOTrainer(actor_on, critic_on, config), PPOTrainer(actor_off, critic_off, config)
    rollout = _rollout(actor_on, critic_on)

    torch.manual_seed(606)
    observed = on.update(rollout, diagnostics=True)
    rng_after_on = torch.get_rng_state()
    torch.manual_seed(606)
    unobserved = off.update(rollout, diagnostics=False)
    rng_after_off = torch.get_rng_state()

    _assert_same(actor_on.state_dict(), actor_off.state_dict())
    _assert_same(critic_on.state_dict(), critic_off.state_dict())
    _assert_same(on.state_dict(), off.state_dict())
    torch.testing.assert_close(rng_after_on, rng_after_off, rtol=0, atol=0)
    assert 0 < observed["actor_optimizer_steps"] <= observed["actor_optimizer_steps_denominator"]
    assert observed["old_kl_tested_max"] >= observed["old_kl_tested"]
    assert "preupdate_gae_bootstrap_dependent_critic_ev" in observed
    assert "stochastic_boost_rate_mix0_lt700" in observed
    assert "greedy_actor_reference_disagreement_mix2_ge800" in observed
    assert "preupdate_gae_bootstrap_dependent_critic_ev" not in unobserved

    # The next update consumes the same permutation stream too: detached
    # monitoring cannot defer an RNG/state perturbation to its successor.
    torch.manual_seed(607)
    on.update(rollout, diagnostics=True)
    rng_after_next_on = torch.get_rng_state()
    torch.manual_seed(607)
    off.update(rollout, diagnostics=False)
    rng_after_next_off = torch.get_rng_state()
    _assert_same(actor_on.state_dict(), actor_off.state_dict())
    _assert_same(critic_on.state_dict(), critic_off.state_dict())
    _assert_same(on.state_dict(), off.state_dict())
    torch.testing.assert_close(rng_after_next_on, rng_after_next_off, rtol=0, atol=0)


def test_logical_lengths_are_time_major_pre_action_facts() -> None:
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    critic = ClockedPopArtCritic(actor.reference)
    rollout = _rollout(actor, critic)
    rollout.logical_lengths = torch.ones(2, 3)
    try:
        PPORollout(**vars(rollout))
    except ValueError as error:
        assert "logical_lengths" in str(error)
    else:
        raise AssertionError("non-time-major logical lengths accepted")


def test_stop_minibatch_keeps_detached_reference_and_entropy_observations() -> None:
    """A first-minibatch KL latch still exposes its diagnostic saturation signal."""
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    rollout = _rollout(actor, critic)
    rollout.logp.sub_(10.0)
    result = trainer.update(rollout, diagnostics=True)
    assert result["actor_optimizer_steps"] == 0.0
    assert result["first_stop_epoch"] == 1.0
    assert result["first_stop_minibatch"] == 1.0
    assert result["reference_kl_minibatch_observations"] == 1.0
    assert result["entropy_minibatch_observations"] == 1.0


def test_detached_proxy_between_real_updates_preserves_continuation_identity() -> None:
    """A deterministic proxy cannot perturb the next PPO update or its RNG stream."""

    class ProxyTask:
        def __init__(self, worlds: int) -> None:
            self.worlds = worlds
            self.hero_length = torch.full((worlds,), 650, dtype=torch.long)

        def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.zeros((self.worlds, 7, 31, 31), dtype=torch.uint8),
                torch.zeros((self.worlds, 4, 37, 37), dtype=torch.uint8),
                torch.zeros((self.worlds, 60)),
                torch.ones((self.worlds, 6), dtype=torch.bool),
            )

        def step(self, _actions: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                mass=torch.ones(self.worlds), terminal=torch.zeros(self.worlds, dtype=torch.bool)
            )

    def make_task(_mix: str, worlds: int, _horizon: int, _seed: int) -> ProxyTask:
        return ProxyTask(worlds)

    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor_proxy, critic_proxy = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    actor_plain, critic_plain = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    actor_plain.load_state_dict(copy.deepcopy(actor_proxy.state_dict()))
    critic_plain.load_state_dict(copy.deepcopy(critic_proxy.state_dict()))
    config = PPOConfig(minibatch=3)
    with_proxy = PPOTrainer(actor_proxy, critic_proxy, config)
    without_proxy = PPOTrainer(actor_plain, critic_plain, config)
    first = _rollout(actor_proxy, critic_proxy)
    torch.manual_seed(701)
    with_proxy.update(first, diagnostics=False)
    torch.manual_seed(701)
    without_proxy.update(first, diagnostics=False)
    next_rollout = _rollout(actor_proxy, critic_proxy)
    rng_before_proxy = torch.get_rng_state()
    modes_before_proxy = {name: module.training for name, module in actor_proxy.named_modules()}
    evaluate(actor_proxy, make_task, ProxyConfig(horizon=1, worlds_per_mix=1, seed_base=17))
    torch.testing.assert_close(torch.get_rng_state(), rng_before_proxy, rtol=0, atol=0)
    assert modes_before_proxy == {
        name: module.training for name, module in actor_proxy.named_modules()
    }

    torch.manual_seed(702)
    with_proxy.update(next_rollout, diagnostics=False)
    rng_after_proxy = torch.get_rng_state()
    torch.manual_seed(702)
    without_proxy.update(next_rollout, diagnostics=False)
    rng_after_plain = torch.get_rng_state()
    _assert_same(actor_proxy.state_dict(), actor_plain.state_dict())
    _assert_same(critic_proxy.state_dict(), critic_plain.state_dict())
    _assert_same(with_proxy.state_dict(), without_proxy.state_dict())
    torch.testing.assert_close(rng_after_proxy, rng_after_plain, rtol=0, atol=0)
