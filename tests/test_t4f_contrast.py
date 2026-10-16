"""Focused RETURN_C contrast-loss and tripwire tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from research.redesign_m3c_20261008.learner_m3c import return_regression_loss
from research.redesign_m3c_20261008.train_m3c import boost_halt_reason


class _StateQ(torch.nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(values.clone())

    def forward(self, local, global_, scalars):
        return self.values[scalars[:, 0].long()]


def _contrast_rows() -> dict[str, torch.Tensor | list[int]]:
    n = 6
    return {
        "local": torch.zeros((n, 7, 31, 31), dtype=torch.uint8),
        "global": torch.zeros((n, 4, 37, 37), dtype=torch.uint8),
        "scalars": torch.zeros((n, 1)),
        "action": torch.arange(n),
        "T": torch.ones(n, dtype=torch.long),
        "died": torch.ones(n, dtype=torch.bool),
        "realised": torch.tensor([-5.0, -3.0, -1.0, 0.0, 8.0, 2.0]),
        "terminal_local": torch.zeros((n, 7, 31, 31), dtype=torch.uint8),
        "terminal_global": torch.zeros((n, 4, 37, 37), dtype=torch.uint8),
        "terminal_scalars": torch.ones((n, 1)),
        "terminal_mask": torch.zeros((n, 6), dtype=torch.bool),
        "weight": torch.full((n,), 99.0),  # contrast deliberately ignores legacy row weights
        "group_sizes": [6],
        "selected": torch.ones(n, dtype=torch.bool),
    }


def test_contrast_uses_dynamic_max_target_reference_and_epsilon_huber() -> None:
    online = _StateQ(torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 6]))
    target = _StateQ(torch.zeros((2, 6)))
    loss, y, telemetry = return_regression_loss(
        torch,
        online,
        target,
        _contrast_rows(),
        gamma=0.9,
        delta=100.0,
        mode="contrast",
        contrast_epsilon=2.0,
        details=True,
    )
    torch.testing.assert_close(y, torch.tensor([-5.0, -3.0, -1.0, 0.0, 8.0, 2.0]))
    # a_ref is action 4 from y, not B8/current-greedy action 5.  The five non-reference
    # errors have magnitudes 9, 8, 7, 7, 7; epsilon=2 leaves 7, 6, 5, 5, 5.
    torch.testing.assert_close(loss, torch.tensor(16.0))
    assert telemetry["contrast_rows"] == 5
    assert telemetry["contrast_active_fraction"] == 1.0
    assert telemetry["mean_abs_target_contrast"] == pytest.approx(9.4)
    assert telemetry["mean_abs_contrast_error"] == pytest.approx(7.6)
    loss.backward()
    # The reference Q is detached: it receives no indirect contrast gradient.
    torch.testing.assert_close(online.values.grad[0, 4], torch.tensor(0.0))
    assert target.values.grad is None


def test_contrast_epsilon_can_make_every_eligible_row_inactive() -> None:
    online = _StateQ(torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 6]))
    target = _StateQ(torch.zeros((2, 6)))
    loss, _, telemetry = return_regression_loss(
        torch,
        online,
        target,
        _contrast_rows(),
        gamma=0.9,
        delta=100.0,
        mode="contrast",
        contrast_epsilon=20.0,
        details=True,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert telemetry["contrast_rows"] == 0
    assert telemetry["contrast_active_fraction"] == 0.0


def test_low_length_tripwire_uses_combined_below_700_rate() -> None:
    args = SimpleNamespace(boost_halt_rate=0.02, boost_halt_low_length_rate=0.01)
    row = {
        "greedy_boost_rate": 0.0,
        "greedy_boost_rate_by_length": {"lt_400": 0.0, "400_700": 0.02},
        "greedy_boost_length_decisions": {"lt_400": 50, "400_700": 50},
    }
    reason = boost_halt_reason(args, row, 100_000)
    assert reason is None
    assert row["greedy_boost_rate_lt_700"] == 0.01
    row["greedy_boost_rate_by_length"]["400_700"] = 0.04
    reason = boost_halt_reason(args, row, 100_000)
    assert reason is not None and "below length 700 0.020000" in reason
