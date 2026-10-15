"""Contract tests for the M3-C TD Huber-delta training option."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from research.redesign_m3c_20261008 import train_m3c  # noqa: E402


def _required_args() -> list[str]:
    """Minimum required CLI arguments; their paths are never opened by parsing."""
    return [
        "--seed",
        "1",
        "--init",
        "init.pth",
        "--demo-data",
        "demos",
        "--out",
        "out",
    ]


def test_td_huber_delta_parser_default_and_override() -> None:
    """The explicit flag remains opt-in and defaults to the legacy delta one."""
    parser = train_m3c.build_parser()
    assert parser.parse_args(_required_args()).td_huber_delta == 1.0
    assert parser.parse_args(_required_args() + ["--td-huber-delta", "100"]).td_huber_delta == 100.0


def test_huber_delta_one_is_bitwise_smooth_l1_beta_one() -> None:
    """Delta one preserves the prior Smooth L1 reduction, including large errors."""
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(20261015)
    qa = torch.randn(97, generator=generator)
    y = qa + torch.randn(97, generator=generator) * 20.0

    huber = F.huber_loss(qa, y, delta=1.0)
    smooth_l1 = F.smooth_l1_loss(qa, y, beta=1.0)

    assert torch.equal(huber, smooth_l1)


def test_huber_delta_one_hundred_gradients_are_quadratic_then_clipped() -> None:
    """Mean-reduced Huber has e/N gradients below delta and +/-delta/N above it."""
    torch.set_num_threads(2)
    residual = torch.tensor([50.0, 150.0, -50.0, -150.0])
    qa = residual.clone().requires_grad_()
    y = torch.zeros_like(qa)

    F.huber_loss(qa, y, delta=100.0).backward()

    expected = torch.tensor([12.5, 25.0, -12.5, -25.0])
    assert qa.grad is not None
    assert torch.equal(qa.grad, expected)
