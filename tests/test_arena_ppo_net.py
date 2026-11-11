"""Tests for `research.arena_ppo.net` (`ArenaNet` S/M presets, masking, symlog two-hot)."""

from __future__ import annotations

import math

import pytest
import torch

from research.arena_ppo import ARENA_GRID_H, ARENA_GRID_W, EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.net import (
    AUX_DEATH_HORIZONS,
    MASK_FILL_VALUE,
    ArenaNet,
    ArenaNetConfig,
    symexp,
    symlog,
    symlog_two_hot_loss,
    symlog_two_hot_value,
    two_hot_encode,
)
from research.arena_ppo.obs import NUM_OBS_CHANNELS
from research.arena_ppo.runtime import normalize_device


def test_arenanet_param_counts_and_token_grid_geometry() -> None:
    """Verify S (~1M) and M (~5M) parameter counts and 11x19 arena token grid."""
    net_s = ArenaNet(ArenaNetConfig.from_size("S"))
    net_m = ArenaNet(ArenaNetConfig.from_size("M"))

    params_s = net_s.parameter_count()
    params_m = net_m.parameter_count()
    assert 900_000 <= params_s <= 1_150_000, f"Expected S ~1M params, got {params_s}"
    assert 4_700_000 <= params_m <= 5_300_000, f"Expected M ~5M params, got {params_m}"
    assert ArenaNet.TOKEN_H == 11 and ArenaNet.TOKEN_W == 19 and ArenaNet.NUM_TOKENS == 209

    arena = torch.zeros(2, NUM_OBS_CHANNELS, ARENA_GRID_H, ARENA_GRID_W)
    feat = net_s.arena_stage2(net_s.arena_stage1(net_s.arena_stem(arena)))
    assert feat.shape == (2, net_s.cfg.arena_dim, 11, 19)


def test_symlog_two_hot_roundtrip_and_loss() -> None:
    """Verify symlog/symexp identity and exact two-hot expectation over return range."""
    support = torch.linspace(-15.0, 15.0, 255, dtype=torch.float32)
    targets = torch.tensor([0.0, 0.005, 0.25, 1.5, 25.0, 400.0, 1950.0], dtype=torch.float32)
    dist = two_hot_encode(symlog(targets), support)
    assert dist.shape == (targets.shape[0], 255)
    assert torch.allclose(dist.sum(dim=-1), torch.ones_like(targets), atol=1e-6)
    decoded = symexp(dist @ support)
    assert torch.allclose(decoded, targets, rtol=1e-5, atol=1e-5)

    logits = torch.randn((targets.shape[0], 255), requires_grad=True)
    val_pred = symlog_two_hot_value(logits, support)
    loss = symlog_two_hot_loss(logits, targets, support).mean()
    assert torch.isfinite(val_pred).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_arenanet_masking_and_backward_cpu_and_mps(device_name: str) -> None:
    """Verify illegal boost actions are never sampled and gradients are finite on CPU and MPS."""
    dev = normalize_device(device_name)
    net = ArenaNet(ArenaNetConfig.from_size("S")).to(dev)
    bsz = 12
    arena = torch.randn((bsz, NUM_OBS_CHANNELS, ARENA_GRID_H, ARENA_GRID_W), device=dev)
    ego = torch.randn((bsz, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE), device=dev)
    scalars = torch.randn((bsz, SCALAR_DIM), device=dev)

    legal_mask = torch.ones((bsz, 6), dtype=torch.bool, device=dev)
    legal_mask[:8, 3:] = False  # First 8 rows disallow boost

    act_out = net.act(arena, ego, scalars, legal_mask=legal_mask, greedy=False)
    assert act_out.actions.shape == (bsz,)
    assert (act_out.actions[:8] < 3).all(), "Masked boost actions 3..5 must never be sampled"
    assert (act_out.masked_logits[:8, 3:] == MASK_FILL_VALUE).all()
    assert act_out.aux_logits.shape == (bsz, len(AUX_DEATH_HORIZONS))

    logp, entropy, val_logits, aux_logits = net.evaluate_actions(
        arena, ego, scalars, legal_mask, act_out.actions
    )
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy).all()
    assert (entropy[:8] <= math.log(3.0) + 1e-4).all()

    total_loss = (
        -logp.mean()
        - 0.01 * entropy.mean()
        + val_logits.square().mean()
        + aux_logits.square().mean()
    )
    total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(math.isfinite(g) for g in grad_norms)
