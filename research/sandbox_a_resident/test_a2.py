"""A-2 hot-path equivalence tests: deferred bootstrap and direct categorical math."""

from __future__ import annotations

import torch

from research.sandbox_a_resident.core import (
    Recipe,
    ResidentAssay,
    bootstrap_subset,
    categorical_terms,
    distribution,
    world_subset,
)
from src.model.ego2s_network import Ego2sNet


def test_direct_categorical_matches_distribution_logprob_entropy_and_gradients() -> None:
    torch.set_num_threads(1)
    logits = torch.tensor([[0.2, -1.3, 0.7, 4.0, -2.0, 1.1]], requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    mask = torch.tensor([[True, False, True, False, False, False]])
    action = torch.tensor([2])
    _, entropy, logp = categorical_terms(logits, mask, action)
    direct_loss = entropy.sum() - logp.sum()
    direct_loss.backward()
    reference = distribution(reference_logits, mask)
    reference_loss = reference.entropy().sum() - reference.log_prob(action).sum()
    reference_loss.backward()
    torch.testing.assert_close(logp, reference.log_prob(action))
    torch.testing.assert_close(entropy, reference.entropy())
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_deferred_bootstrap_equals_per_event_reference_at_nonzero_critic() -> None:
    torch.set_num_threads(1)
    recipe = Recipe(worlds=2, horizon=3, minibatch=3, episode_limit=1, max_episodes=12, tiny=True)
    runner = ResidentAssay(recipe, Ego2sNet(local_channels=7, n_scalars=60))
    with torch.no_grad():
        runner.actor.critic.weight.normal_(generator=torch.Generator().manual_seed(9))
        runner.actor.critic.bias.fill_(0.37)
    data = runner.collect()
    bank = runner._bootstrap_bank
    assert bank is not None
    rows = torch.where(bank["valid"].flatten())[0]
    assert rows.numel() == int(data["truncated"].sum())
    for row in rows.tolist():
        reference = bootstrap_subset(runner.sim, bank, torch.tensor([row]))
        observation, _ = runner.observe(reference)
        expected = runner.actor.critic_only(*observation).item()
        tick = int(bank["tick"].flatten()[row])
        world = row % recipe.worlds
        # The deferred pass intentionally changes CNN batch shape.  CPU kernels
        # differ by a few float32 ulps from single-row evaluation; GAE semantics
        # are equal within the documented numerical reduction envelope.
        torch.testing.assert_close(
            data["next_value"][tick, world], torch.tensor(expected), rtol=2e-5, atol=2e-5
        )


def test_deferred_bank_matches_immediate_pre_reset_bootstrap_oracle(monkeypatch) -> None:
    """Freeze the pre-reset world before A-2 can replace it, as the old loop did."""
    torch.set_num_threads(1)
    recipe = Recipe(worlds=2, horizon=4, minibatch=4, episode_limit=2, max_episodes=12, tiny=True)
    runner = ResidentAssay(recipe, Ego2sNet(local_channels=7, n_scalars=60))
    runner.age.copy_(torch.tensor([0, 1]))  # Stagger first limits into distinct bank slots.
    with torch.no_grad():
        runner.actor.critic.weight.normal_(generator=torch.Generator().manual_seed(17))
        runner.actor.critic.bias.fill_(0.19)
    expected: dict[tuple[int, int], tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}
    original_step = runner.sim.step
    tick = 0

    def step(actions: torch.Tensor) -> None:
        nonlocal tick
        original_step(actions)
        next_age = runner.age + 1
        terminal = runner.sim._last_done[:, 0]
        truncated = (next_age >= recipe.episode_limit) & ~terminal
        indices = torch.where(truncated)[0]
        if indices.numel():
            subset = world_subset(runner.sim, indices)
            observation, _ = runner.observe(subset)
            values = runner.actor.critic_only(*observation).clone()
            for offset, world in enumerate(indices.tolist()):
                expected[(tick, world)] = (
                    tuple(x[offset : offset + 1].clone() for x in observation),
                    values[offset : offset + 1],
                )
        tick += 1

    monkeypatch.setattr(runner.sim, "step", step)
    data = runner.collect()
    bank = runner._bootstrap_bank
    assert bank is not None
    rows = torch.where(bank["valid"].flatten())[0]
    assert rows.numel() == len(expected)
    for selected in rows.tolist():
        deferred = bootstrap_subset(runner.sim, bank, torch.tensor([selected]))
        deferred_observation, _ = runner.observe(deferred)
        deferred_tick = int(bank["tick"].flatten()[selected])
        observation, values = expected[(deferred_tick, selected % recipe.worlds)]
        for left, right in zip(observation, deferred_observation):
            assert torch.equal(left, right)
        torch.testing.assert_close(
            data["next_value"][deferred_tick, selected % recipe.worlds],
            values[0],
            rtol=2e-5,
            atol=2e-5,
        )
