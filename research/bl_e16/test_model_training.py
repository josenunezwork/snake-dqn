"""Focused E16 actor identity, horizon, critic refit, and PPO smoke checks."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from research.bl_e16.model import E16Policy
from research.bl_e16.training import E16Batch, E16Trainer, compute_gae
from research.bl_ppo.model import PPOPolicy
from src.model.ego2s_network import Ego2sNet


def _parent() -> PPOPolicy:
    torch.manual_seed(17)
    return PPOPolicy("F", None, 0.7, b8_model=Ego2sNet(7, 60)).eval()


def _inputs(time: int = 2, batch: int = 2):
    return (
        torch.randint(0, 256, (time, batch, 7, 31, 31), dtype=torch.uint8),
        torch.randint(0, 256, (time, batch, 4, 37, 37), dtype=torch.uint8),
        torch.randn(time, batch, 60),
        torch.arange(time).view(time, 1).expand(time, batch),
        torch.randn(time, batch, 7, 83, 145),
        torch.randn(time, batch, 128, 43),
    )


def _batch(policy: E16Policy, time: int = 2, batch: int = 2) -> E16Batch:
    local, global_, scalars, frame, geometry, history = _inputs(time, batch)
    with torch.no_grad():
        # Each old log probability belongs to its own collected observation.
        # Repeating step-zero logits can trip PPO's KL guard before any update.
        logits = torch.stack(
            [
                policy.actor_logits(
                    local[step],
                    global_[step],
                    scalars[step],
                    geometry=geometry[step],
                    history=history[step],
                )
                for step in range(time)
            ]
        )
        values = policy.value(
            local.reshape(time * batch, *local.shape[2:]),
            global_.reshape(time * batch, *global_.shape[2:]),
            scalars.reshape(time * batch, 60),
            frame.reshape(-1),
            geometry=geometry.reshape(time * batch, *geometry.shape[2:]),
            history=history.reshape(time * batch, 128, 43),
        ).reshape(time, batch)
    actions = Categorical(logits=logits).sample()
    return E16Batch(
        local,
        global_,
        scalars,
        frame,
        actions,
        Categorical(logits=logits).log_prob(actions),
        torch.ones(time, batch),
        torch.zeros(time, batch, dtype=torch.bool),
        values,
        torch.zeros(batch),
        torch.ones(time, batch, 6, dtype=torch.bool),
        geometry=geometry,
        history=history,
        critic_targets=torch.full((time, batch), 17.0),
    )


def test_a_and_g_start_at_exact_parent_actor_logits() -> None:
    parent = _parent()
    local, global_, scalars, _, geometry, history = _inputs(time=1)
    with torch.no_grad():
        expected, _, _ = parent(local[0], global_[0], scalars[0])
        assert torch.equal(
            E16Policy("A", parent).actor_logits(local[0], global_[0], scalars[0]), expected
        )
        assert torch.equal(
            E16Policy("G", parent).actor_logits(
                local[0], global_[0], scalars[0], geometry=geometry[0]
            ),
            expected,
        )
        assert torch.equal(
            E16Policy("G", parent, use_history=True).actor_logits(
                local[0], global_[0], scalars[0], geometry=geometry[0], history=history[0]
            ),
            expected,
        )


def test_gae_zeroes_bootstrap_on_death_and_horizon_cutoff() -> None:
    policy = E16Policy("A", _parent())
    batch = _batch(policy, time=1, batch=1)
    batch.rewards[:] = 3.0
    batch.values[:] = 5.0
    batch.last_value[:] = 999.0
    batch.dones[:] = True
    advantages, returns = compute_gae(batch)
    assert advantages.item() == -2.0 and returns.item() == 3.0
    batch.dones[:] = False
    batch.horizon_cutoff = torch.ones((1, 1), dtype=torch.bool)
    advantages, returns = compute_gae(batch)
    assert advantages.item() == -2.0 and returns.item() == 3.0


def test_critic_refit_preserves_actor_then_e12_gates_real_ppo() -> None:
    policy = E16Policy("G", _parent())
    trainer = E16Trainer(policy, critic_refit_transitions=4)
    batch = _batch(policy)
    before = [parameter.detach().clone() for parameter in policy.actor_parameters()]
    metrics = trainer.trainer_step(batch)
    assert metrics["stage"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, policy.actor_parameters()))
    assert trainer.actor_frozen is False
    critic_before = [parameter.detach().clone() for parameter in policy.critic.parameters()]
    try:
        trainer.trainer_step(batch)
    except RuntimeError as error:
        assert "E12 PASS" in str(error)
    else:
        raise AssertionError("actor update bypassed E12")
    assert all(torch.equal(a, b) for a, b in zip(critic_before, policy.critic.parameters()))
    trainer.mark_e12_passed()
    actor_before = [parameter.detach().clone() for parameter in policy.actor_parameters()]
    critic_before = [parameter.detach().clone() for parameter in policy.critic.parameters()]
    metrics = trainer.trainer_step(batch)
    assert metrics["stage"] == 1.0 and torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, policy.actor_parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, policy.critic.parameters()))


def test_refit_requires_complete_return_targets_and_uses_them() -> None:
    policy = E16Policy("A", _parent())
    trainer = E16Trainer(policy, critic_refit_transitions=4)
    batch = _batch(policy)
    batch.critic_targets = None
    try:
        trainer.trainer_step(batch)
    except ValueError as error:
        assert "complete-return" in str(error)
    else:
        raise AssertionError("critic refit accepted bootstrap-only targets")


def test_masked_legal_row_performs_an_actor_update() -> None:
    policy = E16Policy("A", _parent())
    trainer = E16Trainer(policy, critic_refit_transitions=4)
    batch = _batch(policy)
    batch.action_masks[:, 0, 1:] = False
    batch.actions[:, 0] = 0
    with torch.no_grad():
        for step in range(batch.actions.shape[0]):
            logits = policy.actor_logits(
                batch.local[step],
                batch.global_[step],
                batch.scalars[step],
                action_mask=batch.action_masks[step],
            )
            batch.logp[step] = Categorical(logits=logits).log_prob(batch.actions[step])
    trainer.trainer_step(batch)
    trainer.mark_e12_passed()
    before = [parameter.detach().clone() for parameter in policy.actor_parameters()]
    metrics = trainer.trainer_step(batch)
    assert metrics["stage"] == 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.actor_parameters()))


def test_history_variant_requires_its_own_e3_e14_qualification() -> None:
    policy = E16Policy("G", _parent(), use_history=True)
    try:
        E16Trainer(policy)
    except ValueError as error:
        assert "E3/E14" in str(error)
    else:
        raise AssertionError("unqualified G history variant was trainable")
    E16Trainer(policy, history_qualified=True)


def test_portable_export_contains_cpu_state() -> None:
    policy = E16Policy("A", _parent()).eval()
    payload = policy.portable_state_dict()
    assert payload["arch"] == "bl-e16/v1"
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())
    restored = E16Policy.from_portable_state(payload).eval()
    local, global_, scalars, _, _, _ = _inputs(time=1)
    with torch.no_grad():
        assert torch.equal(
            policy.actor_logits(local[0], global_[0], scalars[0]),
            restored.actor_logits(local[0], global_[0], scalars[0]),
        )


def test_final_optimizer_parameter_poison_is_a_tripwire(monkeypatch) -> None:
    import pytest

    from research.bl_e16.training import E16TripwireError

    policy = E16Policy("A", _parent())
    trainer = E16Trainer(policy, critic_refit_transitions=1)
    batch = _batch(policy)
    trainer.trainer_step(batch)
    trainer.mark_e12_passed()
    real_step = trainer.actor_optimizer.step

    def poisoned_step(*args, **kwargs):
        result = real_step(*args, **kwargs)
        with torch.no_grad():
            next(iter(policy.actor_parameters())).flatten()[0] = float("nan")
        return result

    monkeypatch.setattr(trainer.actor_optimizer, "step", poisoned_step)
    with pytest.raises(E16TripwireError, match="updated actor parameter"):
        trainer.trainer_step(_batch(policy, time=1, batch=2))
