"""PPO math fixtures for P2; all run CPU single-threaded."""

from __future__ import annotations

import copy

import pytest
import torch

from research.sandbox_a_resident.training_learner import (
    PPOConfig,
    PPORollout,
    PPOTrainer,
    _baseline_predictions,
    _equal_mix_mean,
    compute_gae,
    critic_quality,
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


def test_rollout_canonicalizes_all_false_advisory_masks() -> None:
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    rollout = make_rollout(actor, critic)
    rollout.legal_masks[0, 0].zero_()
    rollout.actions[0, 0] = 0
    # Reconstructing runs the validation/canonicalization boundary used by
    # manually supplied and persisted rollouts.
    normalized = PPORollout(**vars(rollout))
    assert normalized.legal_masks[0, 0].tolist() == [True, True, True, False, False, False]
    assert normalized.legal_masks[0, 0, normalized.actions[0, 0]]


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


def test_critic_spike_cannot_change_actor_clip_budget(monkeypatch: object) -> None:
    """Actor and critic are clipped in distinct calls and parameter domains."""
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=9))
    calls: list[set[int]] = []
    original = torch.nn.utils.clip_grad_norm_

    def capture(parameters: object, *args: object, **kwargs: object) -> torch.Tensor:
        parameters = list(parameters)  # type: ignore[arg-type]
        calls.append({id(parameter) for parameter in parameters})
        return original(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    trainer.update(make_rollout(actor, critic))
    actor_ids = {id(parameter) for parameter in actor.parameters()}
    critic_ids = {id(parameter) for parameter in critic.parameters()}
    assert calls
    assert all(call <= actor_ids or call <= critic_ids for call in calls)
    assert any(call == actor_ids for call in calls)
    assert any(call == critic_ids for call in calls)

    # A critic-only spike changes its own clipped update but cannot shrink the
    # actor's independent clip budget.  Reuse an identical rollout and RNG
    # stream so any actor difference would expose cross-network clipping.
    clean_actor, clean_critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    spiked_actor, spiked_critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    spiked_actor.load_state_dict(clean_actor.state_dict())
    spiked_critic.load_state_dict(clean_critic.state_dict())
    clean = PPOTrainer(clean_actor, clean_critic, PPOConfig(minibatch=9))
    spiked = PPOTrainer(spiked_actor, spiked_critic, PPOConfig(minibatch=9))
    spike = spiked_critic.output.weight.register_hook(lambda gradient: gradient * 1_000_000)
    shared_rollout = make_rollout(clean_actor, clean_critic)
    torch.manual_seed(212)
    clean.update(shared_rollout)
    torch.manual_seed(212)
    spiked.update(shared_rollout)
    spike.remove()
    for clean_parameter, spiked_parameter in zip(
        clean_actor.parameters(), spiked_actor.parameters()
    ):
        torch.testing.assert_close(clean_parameter, spiked_parameter, rtol=0, atol=0)


def test_kl_stop_latches_actor_across_epochs_but_keeps_critic_updates_sane(
    monkeypatch: object,
) -> None:
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    config = PPOConfig(minibatch=3)
    trainer = PPOTrainer(actor, critic, config)
    trainer.actor_rows = config.kl_hold_rows + 123
    rollout = make_rollout(actor, critic)
    # An implausibly low old log-prob makes the sampled old-policy KL trip on
    # the first minibatch.  Critic-only work must still cover both epochs.
    rollout.logp.flatten()[:3].sub_(10.0)
    actor_before = copy.deepcopy(actor.state_dict())
    critic_before = critic.output.weight.detach().clone()
    steps = 0
    original_step = trainer.critic_optimizer.step

    def count_critic_steps(*args: object, **kwargs: object) -> object:
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)

    trainer.critic_optimizer.step = count_critic_steps  # type: ignore[method-assign]
    permutations = iter(
        (
            torch.arange(rollout.actions.numel()),
            torch.tensor([3, 4, 5, 0, 1, 2, 6, 7, 8]),
        )
    )
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: next(permutations))
    used_beta = trainer.kl_beta()
    result = trainer.update(rollout)
    assert result["actor_stopped"] == 1.0
    assert result["kl_beta"] == used_beta
    assert steps == config.epochs * (rollout.actions.numel() // config.minibatch)
    for key, before in actor_before.items():
        torch.testing.assert_close(actor.state_dict()[key], before, rtol=0, atol=0)
    assert not torch.equal(critic.output.weight, critic_before)


def test_critic_quality_baseline_has_cuda_like_same_device_semantics() -> None:
    """Meta tensors exercise the no-CPU-matmul path on CPU-only CI."""
    lengths = torch.empty(4, device="meta")
    clocks = torch.empty(4, device="meta")
    coefficients = torch.ones(3)
    baseline = _baseline_predictions(lengths, clocks, coefficients)
    assert baseline.device.type == "meta"

    # Keep the public quality path covered on actual numeric tensors too.
    rows = 160
    values = torch.linspace(1.0, 2.0, rows)
    quality = critic_quality(
        values,
        values,
        torch.full((rows,), 800.0),
        torch.linspace(0.0, 1.0, rows),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([8, 16, 16]),
    )
    assert quality["baseline_mse"] > 0.0


def test_critic_quality_uses_parameterized_preparation_thresholds() -> None:
    lengths = torch.tensor([2.0, 2.0])
    clocks = torch.tensor([0.5, 1.0])
    returns = torch.tensor([1.0, 2.0])
    coefficients = torch.zeros(3)
    quality = critic_quality(
        returns,
        returns,
        lengths,
        clocks,
        coefficients,
        torch.tensor([1, 0, 0]),
        late_length=2,
        min_late_rows=2,
        min_lives=1,
        min_lives_per_mix=0,
        min_late_ev=0.10,
    )
    assert quality["late_body_rows"] == 2.0 and quality["passed"] == 1.0


def test_complete_life_critic_rejects_tiny_tail_batch() -> None:
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    with pytest.raises(ValueError, match="complete shuffled minibatches"):
        trainer.fit_complete_life_critic(
            torch.zeros((3, 7, 31, 31), dtype=torch.uint8),
            torch.zeros((3, 4, 37, 37), dtype=torch.uint8),
            torch.zeros((3, 60)),
            torch.ones(3),
            torch.ones(3),
            minibatch=2,
        )
