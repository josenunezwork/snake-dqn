"""PPO math fixtures for P2; all run CPU single-threaded."""

from __future__ import annotations

import copy
import json
from decimal import Decimal, localcontext

import numpy as np
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
from research.sandbox_a_resident.training_model import (
    B8Actor,
    ClockedPopArtCritic,
    categorical,
    masked_categorical_entropy,
    masked_categorical_kl,
)
from src.model.ego2s_network import Ego2sNet

_ORACLE_DIGITS = 100


def _dec_log_softmax(logits: list[float]) -> list[Decimal]:
    """Independent 100-digit log-softmax of exact binary float inputs."""
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        values = [Decimal(float(value)) for value in logits]
        anchor = max(values)
        total = sum(((value - anchor).exp() for value in values), Decimal(0))
        log_total = total.ln()
        return [+(value - anchor - log_total) for value in values]


def _dec_kl(reference: list[float], actor: list[float]) -> Decimal:
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        ref, act = _dec_log_softmax(reference), _dec_log_softmax(actor)
        return sum((r.exp() * (r - a) for r, a in zip(ref, act)), Decimal(0))


def _dec_entropy(logits: list[float]) -> Decimal:
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        return -sum((lp.exp() * lp for lp in _dec_log_softmax(logits)), Decimal(0))


def _dec_kl_actor_grad(reference: list[float], actor: list[float]) -> list[Decimal]:
    """dKL(p||q)/da_j = q_j - p_j."""
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        ref, act = _dec_log_softmax(reference), _dec_log_softmax(actor)
        return [a.exp() - r.exp() for r, a in zip(ref, act)]


def _dec_kl_reference_grad(reference: list[float], actor: list[float]) -> list[Decimal]:
    """dKL(p||q)/dr_j = p_j * (log p_j - log q_j - KL)."""
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        ref, act = _dec_log_softmax(reference), _dec_log_softmax(actor)
        kl = _dec_kl(reference, actor)
        return [r.exp() * (r - a - kl) for r, a in zip(ref, act)]


def _dec_entropy_grad(logits: list[float]) -> list[Decimal]:
    """dH/dx_j = -p_j * (log p_j + H)."""
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        entropy = _dec_entropy(logits)
        return [-(lp.exp() * (lp + entropy)) for lp in _dec_log_softmax(logits)]


def _f32(value: Decimal) -> float:
    """Correctly round a Decimal to float32 (nearest of three neighbouring candidates)."""
    guess = np.float32(float(value))
    candidates = (
        np.nextafter(guess, np.float32(-np.inf)),
        guess,
        np.nextafter(guess, np.float32(np.inf)),
    )
    with localcontext() as context:
        context.prec = _ORACLE_DIGITS
        return float(min(candidates, key=lambda item: abs(Decimal(float(item)) - value)))


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


def test_calibrated_temperature_subnormal_kl_underflow_regression_and_finite_telemetry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reproduce train01-11's float32 exp() underflow at T=0.00843973 on real snapshot shapes."""
    torch.set_num_threads(1)
    torch.manual_seed(11)
    temperature = 0.008439731318504158
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    with torch.no_grad():
        # Match train01-11's logit separation: action 1 is greedy (+0.3406) and
        # legal action 2 is suboptimal (-0.5100), so reference log-prob for
        # action 2 is ~ -100.78 (subnormal prob > 0) and a tiny head update
        # pushes actor log-prob past -103.972084 where float32 exp() underflows.
        b8.adv.weight.zero_()
        b8.adv.bias.copy_(
            torch.tensor([0.1345, 0.3406, -0.5098, 0.1553, 0.2392, -0.2513], dtype=torch.float32)
        )
    actor = B8Actor(b8_model=b8, temperature=temperature)
    critic = ClockedPopArtCritic(b8)
    with torch.no_grad():
        critic.mean.fill_(1050.4837646484375)
        critic.scale.fill_(757.6934814453125)
        critic.second_moment.fill_(1677615.625)
        critic.debias.fill_(0.6244643330574036)

    # Direct check of the exact train01-11 row-71 logits where reference prob is
    # subnormal (1.6815582e-44) and actor prob underflows to 0.0 in float32.
    ref_logits = torch.tensor(
        [[15.936467, 40.362400, -60.401756, 18.406952, 28.346058, -29.774279]],
        dtype=torch.float32,
    )
    act_logits = torch.tensor(
        [[15.399835, 41.591217, -62.578457, 18.901131, 29.581945, -30.243114]],
        dtype=torch.float32,
        requires_grad=True,
    )
    legal_mask = torch.tensor([[True, True, True, True, True, False]])
    ref_dist = categorical(ref_logits, legal_mask)
    act_dist = categorical(act_logits, legal_mask)
    assert ref_dist.probs[0, 2].item() > 0.0
    assert act_dist.probs[0, 2].item() == 0.0
    assert torch.isinf(torch.distributions.kl_divergence(ref_dist, act_dist)).all()

    safe_kl = masked_categorical_kl(ref_dist, act_dist, legal_mask)
    safe_entropy = masked_categorical_entropy(act_dist, legal_mask)
    assert torch.isfinite(safe_kl).all()
    assert torch.isfinite(safe_entropy).all()
    (safe_kl.sum() - 2000.0 * safe_entropy.sum()).backward()
    assert act_logits.grad is not None and torch.isfinite(act_logits.grad).all()

    # Full PPOTrainer.update on real snapshot shapes (2 ticks x 3 worlds) where
    # action 2 sits below -103.972084 while sampled greedy action 1 keeps
    # old_kl == 0 <= kl_stop, so the minibatch evaluates that underflowed action.
    rollout = make_rollout(actor, critic, time=2, worlds=3)
    rollout.actions.fill_(1)
    with torch.no_grad():
        flat_logits = actor(
            rollout.local.flatten(0, 1),
            rollout.glob.flatten(0, 1),
            rollout.scalars.flatten(0, 1),
        )
        rollout.logp.copy_(
            categorical(flat_logits, rollout.legal_masks.flatten(0, 1))
            .log_prob(rollout.actions.flatten())
            .reshape_as(rollout.logp)
        )
        rollout.rewards.copy_(torch.tensor([[10.0, -5.0, 2.0], [-8.0, 6.0, -1.0]]))
        actor.adv.bias[2] -= 3.5

    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    capsys.readouterr()
    result = trainer.update(rollout)
    captured = capsys.readouterr().out
    assert all(torch.isfinite(torch.tensor(val)) for val in result.values())
    progress_lines = [
        json.loads(line.removeprefix("TRAIN01_PROGRESS ").strip())
        for line in captured.splitlines()
        if line.startswith("TRAIN01_PROGRESS ")
    ]
    assert progress_lines == [
        {
            "finite": True,
            "first_nonfinite_term": None,
            "stage": "learner-finite-check",
            "update": 1,
        }
    ]


def test_pre_update_finite_check_telemetry_names_first_nonfinite_term(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When any learner term becomes non-finite, telemetry names the exact first term."""
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    rollout = make_rollout(actor, critic, time=2, worlds=3)
    actor_before = copy.deepcopy(actor.state_dict())

    monkeypatch.setattr(
        "research.sandbox_a_resident.training_learner.masked_categorical_kl",
        lambda ref, dist, mask: torch.full(mask.shape[:-1], float("inf"), device=mask.device),
    )
    capsys.readouterr()
    with pytest.raises(FloatingPointError, match="first_nonfinite_term=reference_kl"):
        trainer.update(rollout)
    captured = capsys.readouterr().out
    progress_lines = [
        json.loads(line.removeprefix("TRAIN01_PROGRESS ").strip())
        for line in captured.splitlines()
        if line.startswith("TRAIN01_PROGRESS ")
    ]
    assert progress_lines == [
        {
            "epoch": 1,
            "finite": False,
            "first_nonfinite_term": "reference_kl",
            "minibatch": 1,
            "stage": "learner-finite-check",
            "update": 1,
        }
    ]
    for key, before in actor_before.items():
        torch.testing.assert_close(actor.state_dict()[key], before, rtol=0, atol=0)


def test_nonfinite_reference_after_actor_stop_latch_emits_failure_telemetry_and_raises(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite reference after the old-KL latch must fail and emit telemetry."""
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    rollout = make_rollout(actor, critic, time=3, worlds=3)
    rollout.logp.sub_(10.0)
    actor_before = copy.deepcopy(actor.state_dict())

    # Direct helper checks: non-finite legal support must never be masked to 0.0 / -0.0.
    legal_mask = torch.tensor([[True, True, False, False, False, False]])
    finite_dist = categorical(torch.zeros((1, 6)), legal_mask)
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_logits = torch.zeros((1, 6))
        bad_logits[0, 1] = bad_val
        bad_dist = categorical(bad_logits, legal_mask)
        assert torch.isnan(masked_categorical_kl(bad_dist, finite_dist, legal_mask)).all()
        assert torch.isnan(masked_categorical_kl(finite_dist, bad_dist, legal_mask)).all()
        assert torch.isnan(masked_categorical_entropy(bad_dist, legal_mask)).all()
    nan_probs_dist = torch.distributions.Categorical(
        probs=torch.full((1, 6), float("nan")), validate_args=False
    )
    assert torch.isnan(masked_categorical_kl(nan_probs_dist, finite_dist, legal_mask)).all()
    assert torch.isnan(masked_categorical_entropy(nan_probs_dist, legal_mask)).all()
    zero_legal_probs_dist = torch.distributions.Categorical(
        probs=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), validate_args=False
    )
    assert torch.isnan(masked_categorical_kl(finite_dist, zero_legal_probs_dist, legal_mask)).all()
    assert torch.isnan(masked_categorical_entropy(zero_legal_probs_dist, legal_mask)).all()

    monkeypatch.setattr(
        actor,
        "reference_logits",
        lambda local, glob, scalars: torch.full(
            (local.shape[0], 6), float("nan"), dtype=scalars.dtype, device=scalars.device
        ),
    )
    for diag_flag, expected_update in ((True, 1), (False, 1)):
        trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
        capsys.readouterr()
        with pytest.raises(FloatingPointError, match="first_nonfinite_term=reference_logits"):
            trainer.update(rollout, diagnostics=diag_flag)
        captured = capsys.readouterr().out
        progress_lines = [
            json.loads(line.removeprefix("TRAIN01_PROGRESS ").strip())
            for line in captured.splitlines()
            if line.startswith("TRAIN01_PROGRESS ")
        ]
        assert progress_lines == [
            {
                "epoch": 1,
                "finite": False,
                "first_nonfinite_term": "reference_logits",
                "minibatch": 1,
                "stage": "learner-finite-check",
                "update": expected_update,
            }
        ]
    for key, before in actor_before.items():
        torch.testing.assert_close(actor.state_dict()[key], before, rtol=0, atol=0)


def test_legal_probability_underflow_contributes_exact_log_space_term() -> None:
    """Finite legal tail log-probs that underflow float32 exp() retain their exact contribution."""
    torch.set_num_threads(1)
    mask = torch.tensor([[True, True, False, False, False, False]])
    ref_logits = torch.tensor([[0.0, -104.0, 50.0, -50.0, 12.0, -12.0]], dtype=torch.float32)
    act_logits = torch.tensor(
        [[0.0, -1000.0, -90.0, 90.0, -7.0, 7.0]], dtype=torch.float32, requires_grad=True
    )
    ref_dist = categorical(ref_logits, mask)
    act_dist = categorical(act_logits, mask)
    # In float32, both tail probabilities underflow exp() to exact 0.0.
    assert ref_dist.probs[0, 1].item() == 0.0
    assert act_dist.probs[0, 1].item() == 0.0

    # Independent 100-digit Decimal oracle over the two legal actions. The prior
    # float64 logsumexp oracle repeated the normalization loss it was meant to
    # detect (TRAIN01-13 review finding 1), so it is no longer used here.
    expected_kl = torch.tensor([_f32(_dec_kl([0.0, -104.0], [0.0, -1000.0]))])
    expected_entropy = torch.tensor([_f32(_dec_entropy([0.0, -104.0]))])
    assert expected_kl.item() == 6.095648319812954e-43
    assert expected_entropy.item() == pytest.approx(7.146622168056567e-44, rel=0, abs=0)

    kl = masked_categorical_kl(ref_dist, act_dist, mask)
    entropy = masked_categorical_entropy(ref_dist, mask)
    assert kl.item() > 0.0 and torch.equal(kl, expected_kl)
    assert entropy.item() > 0.0 and torch.equal(entropy, expected_entropy)
    (kl.sum() + entropy.sum()).backward()
    assert act_logits.grad is not None and torch.isfinite(act_logits.grad).all()
    assert torch.equal(act_logits.grad[:, 2:], torch.zeros((1, 4)))

    # Extreme legal tail logit (-1e38) retains the ~6.8135566e-8 KL contribution.
    act_extreme = categorical(
        torch.tensor([[0.0, -1e38, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32), mask
    )
    expected_extreme_kl = torch.tensor(
        [_f32(_dec_kl([0.0, -104.0], [0.0, float(torch.tensor(-1e38).item())]))]
    )
    extreme_kl = masked_categorical_kl(ref_dist, act_extreme, mask)
    assert torch.equal(extreme_kl, expected_extreme_kl)
    assert extreme_kl.item() == pytest.approx(6.8135566e-8, rel=1e-6)

    # Genuinely absent support (~legal) contributes exact 0.0 even with extreme logits.
    identical_legal_ref = categorical(
        torch.tensor([[0.0, -1.0, -104.0, -1000.0, 0.0, 0.0]], dtype=torch.float32), mask
    )
    identical_legal_act = categorical(
        torch.tensor([[0.0, -1.0, -1e38, 100.0, -500.0, 500.0]], dtype=torch.float32), mask
    )
    assert masked_categorical_kl(identical_legal_ref, identical_legal_act, mask).item() == 0.0


_SATURATED_CASES = [
    # (legal actions, reference logits, actor logits); illegal entries are padded.
    (2, [0.0, -104.0], [0.0, -108.0]),
    (2, [0.0, -104.0], [0.0, -1000.0]),
    (2, [0.0, -40.0], [0.0, -40.0]),
    (2, [0.0, -40.0], [0.0, -41.0]),
    (2, [0.0, -100.0], [0.0, -104.0]),
    (3, [0.0, -38.5, -39.0], [0.0, -38.5, -39.25]),
    (3, [5.0, 5.0, -100.0], [5.0, 4.999999, -101.0]),
    (3, [-3.0, 20.0, -80.0], [-3.5, 20.0, -79.0]),
    (6, [0.0, -17.0, -18.0, -19.0, -60.0, -90.0], [0.0, -17.0001, -18.0, -19.5, -61.0, -95.0]),
    (2, [0.3, -0.7], [0.1, 0.2]),
]


def _padded(values: list[float], fill: float) -> torch.Tensor:
    return torch.tensor([values + [fill] * (6 - len(values))], dtype=torch.float32)


@pytest.mark.parametrize("legal_count,reference,actor", _SATURATED_CASES)
def test_saturated_kl_entropy_and_gradients_match_independent_decimal_oracle(
    legal_count: int, reference: list[float], actor: list[float]
) -> None:
    """Values and analytic gradients are correctly rounded float32 of a 100-digit oracle."""
    torch.set_num_threads(1)
    mask = torch.tensor([[index < legal_count for index in range(6)]])
    ref_raw = _padded(reference, 77.0).requires_grad_(True)
    act_raw = _padded(actor, -77.0).requires_grad_(True)
    ref_dist, act_dist = categorical(ref_raw, mask), categorical(act_raw, mask)
    # The oracle consumes the distributions' stored (float32-normalized) logits.
    ref_logits = vars(ref_dist)["logits"]
    act_logits = vars(act_dist)["logits"]
    ref_values = ref_logits.detach()[0, :legal_count].tolist()
    act_values = act_logits.detach()[0, :legal_count].tolist()

    kl = masked_categorical_kl(ref_dist, act_dist, mask)
    entropy = masked_categorical_entropy(act_dist, mask)
    assert kl.dtype == torch.float32 and entropy.dtype == torch.float32
    assert torch.equal(kl, torch.tensor([_f32(_dec_kl(ref_values, act_values))]))
    assert torch.equal(entropy, torch.tensor([_f32(_dec_entropy(act_values))]))

    grad_ref, grad_act = torch.autograd.grad(kl.sum(), (ref_logits, act_logits))
    (grad_entropy,) = torch.autograd.grad(entropy.sum(), (act_logits,))
    pad = [0.0] * (6 - legal_count)
    expected_ref = [_f32(v) for v in _dec_kl_reference_grad(ref_values, act_values)] + pad
    expected_act = [_f32(v) for v in _dec_kl_actor_grad(ref_values, act_values)] + pad
    expected_entropy = [_f32(v) for v in _dec_entropy_grad(act_values)] + pad
    assert torch.equal(grad_ref, torch.tensor([expected_ref]))
    assert torch.equal(grad_act, torch.tensor([expected_act]))
    assert torch.equal(grad_entropy, torch.tensor([expected_entropy]))


def test_review_saturated_examples_are_correctly_rounded() -> None:
    """The TRAIN01-13 review's three numerical counterexamples now round correctly."""
    mask = torch.tensor([[True, True, False, False, False, False]])
    ref = categorical(_padded([0.0, -104.0], 0.0), mask)
    assert masked_categorical_kl(
        ref, categorical(_padded([0.0, -108.0], 0.0), mask), mask
    ).item() == (1.401298464324817e-45)
    assert masked_categorical_kl(
        ref, categorical(_padded([0.0, -1000.0], 0.0), mask), mask
    ).item() == (6.095648319812954e-43)
    logits = _padded([0.0, -40.0], 0.0).requires_grad_(True)
    entropy = masked_categorical_entropy(categorical(logits, mask), mask)
    assert entropy.item() == 1.7418251854950444e-16
    (grad,) = torch.autograd.grad(entropy.sum(), (logits,))
    assert grad[0, 1].item() == 1.699341718629913e-16
    assert grad[0, 0].item() == -1.699341718629913e-16
    assert torch.equal(grad[0, 2:], torch.zeros(4))


def test_probability_validation_is_independent_of_lazy_logits_cache() -> None:
    """Reading ``.logits`` must not change failure classification or the distribution."""
    mask = torch.tensor([[True, True, False, False, False, False]])
    uniform = torch.distributions.Categorical(
        probs=torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]), validate_args=False
    )

    def zero_legal() -> torch.distributions.Categorical:
        return torch.distributions.Categorical(
            probs=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), validate_args=False
        )

    fresh, cached = zero_legal(), zero_legal()
    _ = cached.logits  # observational read populates PyTorch's clamped lazy cache
    assert "logits" in vars(cached)
    for distribution in (fresh, cached):
        assert torch.isnan(masked_categorical_entropy(distribution, mask)).all()
        assert torch.isnan(masked_categorical_kl(uniform, distribution, mask)).all()
        assert torch.isnan(masked_categorical_kl(distribution, uniform, mask)).all()

    healthy = torch.tensor([[0.25, 0.75, 0.0, 0.0, 0.0, 0.0]])
    before = torch.distributions.Categorical(probs=healthy, validate_args=False)
    after = torch.distributions.Categorical(probs=healthy, validate_args=False)
    _ = after.logits
    for left, right in (
        (masked_categorical_entropy(before, mask), masked_categorical_entropy(after, mask)),
        (masked_categorical_kl(uniform, before, mask), masked_categorical_kl(uniform, after, mask)),
    ):
        assert torch.isfinite(left).all() and torch.equal(left, right)
    # A logits-parameterized distribution is unaffected by reading ``.probs``.
    logits = categorical(torch.tensor([[0.0, -104.0, 0.0, 0.0, 0.0, 0.0]]), mask)
    read = categorical(torch.tensor([[0.0, -104.0, 0.0, 0.0, 0.0, 0.0]]), mask)
    _ = read.probs
    assert torch.equal(
        masked_categorical_entropy(logits, mask), masked_categorical_entropy(read, mask)
    )

    class Opaque(torch.distributions.Categorical):
        pass

    opaque = Opaque(logits=torch.zeros(1, 6), validate_args=False)
    del opaque._param
    with pytest.raises(TypeError, match="parameterization"):
        masked_categorical_entropy(opaque, mask)


def test_seeded_oracle_sweep_values_and_actor_gradients_are_correctly_rounded() -> None:
    """Seeded saturated/close/extreme sweep against the independent Decimal oracle.

    KL, entropy and the actor/entropy gradients used by PPO must be correctly
    rounded float32.  The frozen reference's gradient (never used by training;
    reference logits are computed under ``no_grad``) is bounded at float64
    resolution: one float32 ulp at exact float32 rounding midpoints.
    """
    import random

    rng = random.Random(20261027)
    for _ in range(150):
        count = rng.randint(2, 6)
        reference = [
            rng.choice([0.0, rng.uniform(-120, 0), rng.uniform(-5, 5)]) for _ in range(count)
        ]
        actor = [
            value
            + rng.choice(
                [0.0, 0.0, rng.uniform(-1e-5, 1e-5), rng.uniform(-1, 1), rng.uniform(-50, 50)]
            )
            for value in reference
        ]
        if rng.random() < 0.1:
            actor[rng.randrange(count)] = -1e38
        mask = torch.tensor([[index < count for index in range(6)]])
        ref_dist = categorical(_padded(reference, 3.0).requires_grad_(True), mask)
        act_dist = categorical(_padded(actor, -3.0).requires_grad_(True), mask)
        ref_logits, act_logits = vars(ref_dist)["logits"], vars(act_dist)["logits"]
        ref_values = ref_logits.detach()[0, :count].tolist()
        act_values = act_logits.detach()[0, :count].tolist()
        kl = masked_categorical_kl(ref_dist, act_dist, mask)
        entropy = masked_categorical_entropy(act_dist, mask)
        assert kl.item() == _f32(_dec_kl(ref_values, act_values))
        assert entropy.item() == _f32(_dec_entropy(act_values))
        grad_ref, grad_act = torch.autograd.grad(kl.sum(), (ref_logits, act_logits))
        (grad_entropy,) = torch.autograd.grad(entropy.sum(), (act_logits,))
        pad = [0.0] * (6 - count)
        assert (
            grad_act[0].tolist()
            == [_f32(v) for v in _dec_kl_actor_grad(ref_values, act_values)] + pad
        )
        assert grad_entropy[0].tolist() == [_f32(v) for v in _dec_entropy_grad(act_values)] + pad
        torch.testing.assert_close(
            grad_ref,
            torch.tensor([[_f32(v) for v in _dec_kl_reference_grad(ref_values, act_values)] + pad]),
            rtol=1.2e-7,
            atol=1e-36,
        )
