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
    default_actor_chunk_envs,
    default_minibatch_size,
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


def test_minor2_symlog_two_hot_loss_performs_zero_host_item_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression MINOR 2: symlog_two_hot_loss must never call Tensor.item() on support bounds."""
    net = ArenaNet(ArenaNetConfig.from_size("S"))
    logits = torch.randn((16, 255), requires_grad=True)
    targets = torch.linspace(0.0, 25.0, 16)

    def _forbid_item(self: torch.Tensor) -> float:
        raise RuntimeError("Tensor.item() host sync forbidden in symlog_two_hot_loss")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    loss_a = symlog_two_hot_loss(logits, targets, net.value_support)
    loss_b = symlog_two_hot_loss(
        logits,
        targets,
        torch.linspace(-15.0, 15.0, 255),
        vmin=-15.0,
        vmax=15.0,
    )
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_m1_per_net_defaults_and_minor10_bf16_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify M1 per-net defaults for M fit memory bounds and MINOR 10 bf16 fallback works."""
    assert default_actor_chunk_envs("S") == 1024
    assert default_actor_chunk_envs("M") <= 512
    assert default_minibatch_size("S") == 1536
    assert default_minibatch_size("M") <= 1024

    net = ArenaNet(ArenaNetConfig.from_size("S"))
    fake_cuda = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    ctx_fp16 = net._autocast_ctx(fake_cuda)
    assert getattr(ctx_fp16, "fast_dtype", None) == torch.float16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    ctx_bf16 = net._autocast_ctx(fake_cuda)
    assert getattr(ctx_bf16, "fast_dtype", None) == torch.bfloat16


def test_parameterized_board_resolutions_in_arenanet() -> None:
    """Coordinator #7: ArenaNet adaptive token pooling supports arbitrary arena resolutions."""
    for size in ("S", "M"):
        net = ArenaNet(ArenaNetConfig.from_size(size))  # type: ignore[arg-type]
        for gh, gw in ((48, 64), (112, 176)):
            arena = torch.randn((4, NUM_OBS_CHANNELS, gh, gw))
            ego = torch.randn((4, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE))
            scalars = torch.randn((4, SCALAR_DIM))
            mask = torch.ones((4, 6), dtype=torch.bool)
            out = net.act(arena, ego, scalars, legal_mask=mask)
            assert out.actions.shape == (4,)
            assert torch.isfinite(out.values).all()
            loss = out.masked_logits.square().mean() + out.values.square().mean()
            loss.backward()
            net.zero_grad(set_to_none=True)
