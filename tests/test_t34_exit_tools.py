"""Independent numerical oracle for the A4-preserving supervised loss."""

import pytest
import torch

from research.t34_exit_b8_fit.loss import LossConfig, exit_fit_loss


def _batch(previous):
    return {
        "q_prev": previous,
        "action_star": previous.argmax(1),
        "forbidden_mask": torch.zeros_like(previous, dtype=torch.bool),
        "sample_weight": torch.ones(len(previous)),
    }


@pytest.mark.parametrize("tau", [0.02, 0.05, 0.1])
def test_kl_matches_double_precision_oracle_away_from_equality(tau):
    previous = torch.tensor(
        [[21.1, 21.2, 21.0, 20.7, 21.4, 21.3], [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]]
    )
    q = (previous + torch.tensor([[0.01, -0.03, 0.02, 0.02, -0.01, 0.03]])).requires_grad_()
    loss, _ = exit_fit_loss(q, _batch(previous), LossConfig(tau=tau, advantage_weight=0.0))
    reference_q = q.detach().double().requires_grad_()
    reference = torch.nn.functional.kl_div(
        (reference_q / tau).log_softmax(1),
        (previous.double() / tau).softmax(1),
        reduction="batchmean",
    )
    loss.backward()
    reference.backward()
    assert float(loss.detach()) == pytest.approx(float(reference.detach()), abs=1e-6)
    torch.testing.assert_close(q.grad.double(), reference_q.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("tau", [0.02, 0.05, 0.1])
def test_self_distillation_has_exact_zero_gradient_even_for_large_raw_q(tau):
    previous = torch.tensor([[63.1, 63.2, 63.0, 62.7, 63.4, 63.3], [1000.0] * 6])
    q = previous.clone().requires_grad_()
    loss, _ = exit_fit_loss(q, _batch(previous), LossConfig(tau=tau))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.equal(q.grad, torch.zeros_like(q))


def test_kl_value_and_gradient_respect_independent_q_offsets():
    previous = torch.tensor([[0.0, 0.125, 0.25, -0.125, 0.5, -0.5]])
    q = torch.tensor([[0.125, 0.25, 0.0, -0.25, 0.375, -0.5]], requires_grad=True)
    loss, _ = exit_fit_loss(q, _batch(previous), LossConfig(advantage_weight=0.0))
    loss.backward()
    shifted = (q.detach() + 128).requires_grad_()
    shifted_loss, _ = exit_fit_loss(
        shifted, _batch(previous + 64), LossConfig(advantage_weight=0.0)
    )
    shifted_loss.backward()
    torch.testing.assert_close(shifted_loss, loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shifted.grad, q.grad, atol=1e-6, rtol=1e-6)
