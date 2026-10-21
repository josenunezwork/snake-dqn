"""Focused synthetic math checks for the E2 PPO policy and learner."""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Categorical

from research.bl_ppo.learner import PPOLearner, PPORollout, compute_gae
from research.bl_ppo.model import PopArtValueHead, PPOPolicy, calibrate_temperature
from src.model.ego2s_network import Ego2sNet


def _policy(arm: str = "F") -> PPOPolicy:
    torch.manual_seed(7)
    return PPOPolicy(arm, None, 0.7, b8_model=Ego2sNet(7, 60))


def _obs(time: int = 1, batch: int = 2):
    return (
        torch.randint(0, 256, (time, batch, 7, 31, 31), dtype=torch.uint8),
        torch.randint(0, 256, (time, batch, 4, 37, 37), dtype=torch.uint8),
        torch.randn(time, batch, 60),
    )


def test_warm_start_logits_are_calibrated_b8_and_arm_counts_match() -> None:
    policy_r, policy_f = _policy("R").eval(), _policy("F").eval()
    local, global_, scalars = _obs()
    with torch.no_grad():
        logits, _, _ = policy_f(local[0], global_[0], scalars[0])
        expected = policy_f.teacher(local[0], global_[0], scalars[0]) / policy_f.temperature
    assert torch.equal(logits, expected)
    assert policy_r.recurrent_parameter_count == policy_f.recurrent_parameter_count == 394_752


def test_critic_clock_is_critic_only_and_one_world_value_stays_batched() -> None:
    policy = _policy("F").eval()
    local, global_, scalars = _obs(batch=1)
    with torch.no_grad():
        early_logits, early_value, _ = policy(
            local[0], global_[0], scalars[0], remaining_horizon=torch.tensor([1.0])
        )
        late_logits, late_value, _ = policy(
            local[0], global_[0], scalars[0], remaining_horizon=torch.tensor([0.1])
        )
    assert early_logits.shape == late_logits.shape == (1, 6)
    assert torch.equal(early_logits, late_logits)
    assert early_value.shape == late_value.shape == (1,)


def test_popart_stat_update_preserves_denormalised_values() -> None:
    head = PopArtValueHead(3, beta=0.5)
    features = torch.randn(4, 3)
    before = head(features).detach().clone()
    head.update_targets(torch.tensor([20.0, 30.0, 40.0, 50.0]))
    assert torch.allclose(head(features), before, atol=1e-6)


def test_calibration_reaches_one_percent_non_greedy() -> None:
    q_values = torch.tensor([[0.0, -2.0, -3.0, -4.0, -5.0, -6.0]]).repeat(8, 1)
    temperature = calibrate_temperature(q_values)
    probability = torch.softmax(q_values / temperature, dim=-1)
    assert torch.allclose(1.0 - probability.max(dim=-1).values, torch.full((8,), 0.01), atol=1e-5)


def test_calibration_refuses_tied_states_that_cannot_reach_target() -> None:
    with pytest.raises(ValueError, match="cannot reach"):
        calibrate_temperature(torch.zeros((3, 6)))


def test_gae_uses_zero_bootstrap_after_death() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    dones = torch.tensor([[False], [True], [False]])
    advantages, returns = compute_gae(rewards, values, dones, torch.tensor([40.0]))
    # At terminal t=1: delta = reward - V, and no later reward leaks backward.
    assert advantages[1, 0] == -18.0
    assert returns[1, 0] == 2.0
    assert torch.isclose(advantages[0, 0], torch.tensor(1.0 + 20.0 - 10.0 + 0.995 * -18.0))
    assert returns[2, 0] == 43.0


def test_gae_zeros_horizon_bootstrap_when_final_transition_is_terminal() -> None:
    advantages, returns = compute_gae(
        torch.tensor([[3.0]]), torch.tensor([[5.0]]), torch.tensor([[True]]), torch.tensor([999.0])
    )
    assert advantages.item() == -2.0
    assert returns.item() == 3.0


def test_gae_bootstraps_a_rollout_cut_but_not_horizon_or_death() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[4.0], [5.0]])
    dones = torch.tensor([[False], [False]])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        torch.tensor([9.0]),
        bootstrap_mask=torch.tensor([[True], [False]]),
    )
    assert returns[1, 0] == 2.0
    assert torch.isclose(advantages[0, 0], torch.tensor(1.0 + 5.0 - 4.0 + 0.995 * -3.0))


def test_recurrent_reset_removes_previous_hidden_state() -> None:
    policy = _policy("R").eval()
    local, global_, scalars = _obs(batch=1)
    hidden = torch.ones((1, 1, 256))
    with torch.no_grad():
        _, value_reset, hidden_reset = policy(
            local[0], global_[0], scalars[0], hidden, torch.tensor([True])
        )
        _, value_zero, hidden_zero = policy(local[0], global_[0], scalars[0], None, None)
    assert torch.equal(value_reset, value_zero)
    assert hidden_reset is not None and hidden_zero is not None
    assert torch.equal(hidden_reset, hidden_zero)


def test_critic_refit_is_required_before_actor_unfreeze() -> None:
    policy = _policy("F")
    learner = PPOLearner(policy, critic_refit_updates=1)
    local, global_, scalars = _obs(time=2, batch=2)
    with torch.no_grad():
        logits, values, _ = policy(local[0], global_[0], scalars[0])
    actions = Categorical(logits=logits).sample().repeat(2, 1)
    rollout = PPORollout(
        local=local,
        global_=global_,
        scalars=scalars,
        actions=actions,
        logp=Categorical(logits=logits).log_prob(actions[0]).repeat(2, 1),
        rewards=torch.ones((2, 2)),
        dones=torch.zeros((2, 2), dtype=torch.bool),
        values=values.repeat(2, 1),
        last_value=torch.zeros(2),
    )
    before = policy.residual_policy.weight.detach().clone()
    encoder_before = next(policy.encoder.parameters()).detach().clone()
    core_before = next(policy.core.parameters()).detach().clone()
    learner.update(rollout)
    assert torch.equal(before, policy.residual_policy.weight)
    assert torch.equal(encoder_before, next(policy.encoder.parameters()))
    assert torch.equal(core_before, next(policy.core.parameters()))
    assert learner.actor_frozen is False


def test_critic_qualification_requires_late_body_ev_and_baseline_gain() -> None:
    returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
    lengths = torch.tensor([800.0, 900.0, 1000.0, 1100.0])
    horizon = torch.tensor([0.8, 0.6, 0.4, 0.2])
    baseline = torch.tensor([0.0, 0.0, 0.0])
    metrics = PPOLearner.critic_quality(
        returns + torch.tensor([1.0, -1.0, 1.0, -1.0]),
        returns,
        lengths,
        horizon,
        baseline_coefficients=baseline,
    )
    assert metrics["explained_variance"] >= 0.1
    assert metrics["mse"] < metrics["baseline_mse"]
    assert metrics["passed"] == 1.0


def test_refit_transition_count_excludes_heldout_critic_rows() -> None:
    policy = _policy("F")
    learner = PPOLearner(policy, critic_refit_transitions=2_000_000)
    local, global_, scalars = _obs(time=2, batch=2)
    with torch.no_grad():
        logits, values, _ = policy(local[0], global_[0], scalars[0])
    actions = Categorical(logits=logits).sample().repeat(2, 1)
    rollout = PPORollout(
        local=local,
        global_=global_,
        scalars=scalars,
        actions=actions,
        logp=Categorical(logits=logits).log_prob(actions[0]).repeat(2, 1),
        rewards=torch.ones((2, 2)),
        dones=torch.zeros((2, 2), dtype=torch.bool),
        values=values.repeat(2, 1),
        last_value=torch.zeros(2),
        critic_fit_mask=torch.tensor([[True, False], [True, False]]),
    )
    learner.update(rollout, actor_enabled=False)
    assert learner.refit_transitions == 2


def test_recurrent_burn_in_is_real_context_and_has_no_gradient() -> None:
    policy = _policy("R")
    learner = PPOLearner(policy, critic_refit_updates=1)
    local, global_, scalars = _obs(time=34, batch=1)
    scalars.requires_grad_()
    actions = torch.zeros((34, 1), dtype=torch.long)
    hidden = policy.initial_hidden(1, "cpu")
    old_logp = []
    with torch.no_grad():
        for step in range(34):
            logits, _, hidden = policy(local[step], global_[step], scalars.detach()[step], hidden)
            old_logp.append(Categorical(logits=logits).log_prob(actions[step]))
    rollout = PPORollout(
        local=local,
        global_=global_,
        scalars=scalars,
        actions=actions,
        logp=torch.stack(old_logp),
        rewards=torch.ones((34, 1)),
        dones=torch.zeros((34, 1), dtype=torch.bool),
        values=torch.zeros((34, 1)),
        last_value=torch.zeros(1),
        loss_mask=torch.cat(
            [torch.zeros((32, 1), dtype=torch.bool), torch.ones((2, 1), dtype=torch.bool)]
        ),
        initial_hidden=policy.initial_hidden(1, "cpu"),
    )
    learner.update(rollout)
    assert scalars.grad is not None
    assert torch.count_nonzero(scalars.grad[:32]) == 0
    assert torch.count_nonzero(scalars.grad[32:]) > 0
