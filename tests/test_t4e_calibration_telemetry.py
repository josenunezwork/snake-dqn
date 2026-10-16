"""Focused contracts for optional T4e return-weight calibration telemetry."""

import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008.learner_m3c import Learner
from research.redesign_m3c_20261008.train_m3c import build_parser


def test_return_gradient_telemetry_is_default_off() -> None:
    args = build_parser().parse_args(
        ["--seed", "1", "--init", "unused", "--demo-data", "unused", "--out", "unused"]
    )
    assert args.return_gradient_telemetry is False


def test_return_gradient_measurement_uses_weighted_shared_gradients_and_exact_rows() -> None:
    learner = object.__new__(Learner)
    learner.torch = torch
    learner.device = torch.device("cpu")
    learner.online = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        learner.online.weight.fill_(2.0)
    x = torch.tensor([[3.0]])
    # d(td)/dw = 3 and d(weight * ret)/dw = 0.2 * 3: the ratio is exactly 0.2.
    td = learner.online(x).sum()
    ret = learner.online(x).sum()
    got = learner._return_gradient_measurement(
        td, ret, 0.2, n=768, m=256, n_return=128, qmax=td.abs()
    )
    assert got == {
        "weighted_return_grad_norm": pytest.approx(0.6),
        "td_grad_norm": pytest.approx(3.0),
        "return_to_td_grad_ratio": pytest.approx(0.2),
        "max_abs_q": pytest.approx(6.0),
        "anchor_rows": 256,
        "return_rows": 128,
        "agent_td_rows": 768,
    }


def test_return_gradient_measurement_is_recorded_for_calibration_harness() -> None:
    learner = object.__new__(Learner)
    learner.last_return_gradient_telemetry = None
    learner.st = {"return_gradient_telemetry": []}
    measurement = {"return_to_td_grad_ratio": 0.5}
    learner._record_return_gradient_telemetry(measurement)
    assert learner.last_return_gradient_telemetry is measurement
    assert learner.st["return_gradient_telemetry"] == [measurement]
