"""PPO math fixtures for P2; all run CPU single-threaded."""

from __future__ import annotations

import copy

import torch

from research.sandbox_a_resident.training_learner import (
    PPOConfig,
    PPORollout,
    PPOTrainer,
    _equal_mix_mean,
    compute_gae,
)
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic, categorical
from src.model.ego2s_network import Ego2sNet


def make_rollout(
    actor: B8Actor, critic: ClockedPopArtCritic, time: int = 3, worlds: int = 3
) -> PPORollout:
    torch.manual_seed(71)
    local = torch.randint(0, 256, (time, worlds, 7, 31, 31), dtype=torch.uint8)
    glob = torch.randint(0, 256, (time, worlds, 4, 37, 37), dtype=torch.uint8)
    scalars = torch.rand(time, worlds, 60)
    masks = torch.ones(time, worlds, 6, dtype=torch.bool)
    clock = torch.linspace(1, 0.8, time).unsqueeze(1).expand(time, worlds)
    with torch.no_grad():
        logits = actor(local.flatten(0, 1), glob.flatten(0, 1), scalars.flatten(0, 1)).reshape(
            time, worlds, 6
        )
        actions = (
            categorical(logits.flatten(0, 1), masks.flatten(0, 1)).sample().reshape(time, worlds)
        )
        logp = (
            categorical(logits.flatten(0, 1), masks.flatten(0, 1))
            .log_prob(actions.flatten())
            .reshape(time, worlds)
        )
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
    )


def test_gae_death_and_horizon_cut_bootstrap_and_trace() -> None:
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    rollout = make_rollout(actor, critic)
    rollout.terminal[1, 0] = True
    rollout.horizon[1, 1] = True
    rollout.last_value.fill_(7)
    advantages, returns = compute_gae(rollout)
    assert advantages[1, 0] == rollout.rewards[1, 0] - rollout.values[1, 0]
    assert advantages[1, 1] == rollout.rewards[1, 1] - rollout.values[1, 1]
    assert advantages[0, 0] == rollout.rewards[0, 0] + 0.995 * advantages[1, 0]
    assert returns.shape == rollout.values.shape


def test_two_epoch_ppo_updates_full_actor_and_separate_critic_and_roundtrips_state() -> None:
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    rollout = make_rollout(actor, critic)
    old_actor = actor.adv.weight.detach().clone()
    old_trunk = next(actor.local.parameters()).detach().clone()
    old_critic = critic.output.weight.detach().clone()
    result = trainer.update(rollout)
    assert torch.isfinite(torch.tensor(list(result.values()))).all()
    assert not torch.equal(old_actor, actor.adv.weight)
    assert not torch.equal(old_trunk, next(actor.local.parameters()))
    assert not torch.equal(old_critic, critic.output.weight)
    state = trainer.state_dict()
    restored = PPOTrainer(B8Actor(b8_model=b8), ClockedPopArtCritic(b8), PPOConfig(minibatch=3))
    restored.load_state_dict(state)
    assert (
        restored.actor_rows == trainer.actor_rows and restored.update_count == trainer.update_count
    )


def test_equal_mix_weighting_does_not_allow_long_mix_to_hide_loss() -> None:
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=9))
    rollout = make_rollout(actor, critic, time=3, worlds=3)
    result = trainer.update(rollout)
    assert "policy" in result and "reference_kl" in result


def test_equal_mix_weights_and_gradients_ignore_unequal_lifespans() -> None:
    values = torch.tensor([1.0] * 10 + [5.0, 5.0] + [9.0], requires_grad=True)
    mixes = torch.tensor([0] * 10 + [1] * 2 + [2])
    counts = torch.bincount(mixes, minlength=3)
    weighted = _equal_mix_mean(values, mixes, counts)
    torch.testing.assert_close(weighted, torch.tensor(5.0))
    weighted.backward()
    torch.testing.assert_close(values.grad[0], torch.tensor(1 / 30))
    torch.testing.assert_close(values.grad[10], torch.tensor(1 / 6))
    torch.testing.assert_close(values.grad[-1], torch.tensor(1 / 3))


def test_kl_holds_to_20m_then_decays_and_resume_is_bit_exact() -> None:
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    assert trainer.kl_beta(0) == 0.02
    assert trainer.kl_beta(20_000_000) == 0.02
    assert trainer.kl_beta(100_663_296) == 0.005
    rollout = make_rollout(actor, critic)
    torch.manual_seed(817)
    trainer.update(rollout)
    snapshot = copy.deepcopy(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "trainer": trainer.state_dict(),
        }
    )
    resumed_actor, resumed_critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    resumed_actor.load_state_dict(snapshot["actor"])
    resumed_critic.load_state_dict(snapshot["critic"])
    resumed = PPOTrainer(resumed_actor, resumed_critic, PPOConfig(minibatch=3))
    resumed.load_state_dict(snapshot["trainer"])
    torch.manual_seed(991)
    expected = trainer.update(rollout)
    torch.manual_seed(991)
    actual = resumed.update(rollout)
    assert expected == actual
    for left, right in zip(actor.state_dict().values(), resumed_actor.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(critic.state_dict().values(), resumed_critic.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
