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
    EgoOnlyArenaNet,
    EgoOnlyNetConfig,
    SharedTrunkArenaNet,
    SharedTrunkNetConfig,
    _pad_shared_world_arena,
    build_arena_model,
    default_actor_chunk_envs,
    default_minibatch_size,
    measure_model_flops_per_row,
    symexp,
    symlog,
    symlog_two_hot_loss,
    symlog_two_hot_value,
    two_hot_encode,
)
from research.arena_ppo.obs import NUM_OBS_CHANNELS, NUM_WORLD_CHANNELS, WCH_WALL, SharedWorldObs
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


def test_m1_per_net_defaults_and_minor10_bf16_fallback_and_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify M1 per-variant L4 defaults, MINOR 10 bf16 fallback, and MINOR 11b caching."""
    assert default_actor_chunk_envs("S", net_arch="shared_trunk") == 2048
    assert default_actor_chunk_envs("M", net_arch="shared_trunk") == 1024
    assert default_minibatch_size("S", net_arch="shared_trunk") == 3072
    assert default_minibatch_size("M", net_arch="shared_trunk") == 1536
    assert default_actor_chunk_envs("S", net_arch="ego_only") == 2048
    assert default_actor_chunk_envs("M", net_arch="ego_only") == 1024
    assert default_minibatch_size("S", net_arch="ego_only") == 3072
    assert default_minibatch_size("M", net_arch="ego_only") == 1536
    assert default_actor_chunk_envs("S", net_arch="per_seat") == 512
    assert default_actor_chunk_envs("M", net_arch="per_seat") == 512
    assert default_minibatch_size("S", net_arch="per_seat") == 768
    assert default_minibatch_size("M", net_arch="per_seat") == 768

    fake_cuda = torch.device("cuda", 0)
    # Default is opt-in OFF (`use_bf16_autocast=False`) across all 3 network configs
    net_default = ArenaNet(ArenaNetConfig.from_size("S"))
    assert net_default.cfg.use_bf16_autocast is False
    assert SharedTrunkNetConfig.from_size("S").use_bf16_autocast is False
    assert EgoOnlyNetConfig.from_size("S").use_bf16_autocast is False
    assert getattr(net_default._autocast_ctx(fake_cuda), "fast_dtype", None) is None
    assert net_default.uses_fp16_grad_scaler(fake_cuda) is False

    calls_fp16 = {"n": 0}

    def _no_bf16() -> bool:
        calls_fp16["n"] += 1
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", _no_bf16)
    net_fp16 = ArenaNet(ArenaNetConfig.from_size("S", use_bf16_autocast=True))
    for _ in range(8):
        ctx_fp16 = net_fp16._autocast_ctx(fake_cuda)
        assert getattr(ctx_fp16, "fast_dtype", None) == torch.float16
    assert net_fp16.uses_fp16_grad_scaler(fake_cuda) is True
    assert net_fp16.uses_fp16_grad_scaler(torch.device("cpu")) is False
    assert calls_fp16["n"] == 1, "is_bf16_supported() must be cached per ArenaNet instance"

    calls_bf16 = {"n": 0}

    def _yes_bf16() -> bool:
        calls_bf16["n"] += 1
        return True

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", _yes_bf16)
    net_bf16 = ArenaNet(ArenaNetConfig.from_size("S", use_bf16_autocast=True))
    for _ in range(8):
        ctx_bf16 = net_bf16._autocast_ctx(fake_cuda)
        assert getattr(ctx_bf16, "fast_dtype", None) == torch.bfloat16
    assert net_bf16.uses_fp16_grad_scaler(fake_cuda) is False
    assert calls_bf16["n"] == 1


def test_minor4_r3_from_size_derives_ceil_div8_token_grid_from_board_size() -> None:
    """MINOR 4: ArenaNetConfig.from_size derives ceil(grid_h/8) x ceil(grid_w/8) token grid."""
    cfg_default = ArenaNetConfig.from_size("S", grid_h=83, grid_w=145)
    assert (cfg_default.token_grid_h, cfg_default.token_grid_w) == (11, 19)

    cfg_custom = ArenaNetConfig.from_size("M", grid_h=48, grid_w=64)
    assert (cfg_custom.token_grid_h, cfg_custom.token_grid_w) == (6, 8)
    net_custom = ArenaNet(cfg_custom)
    assert (net_custom.token_h, net_custom.token_w, net_custom.num_tokens) == (6, 8, 48)
    assert net_custom.arena_pos_embed.shape == (1, 48, net_custom.cfg.arena_dim)
    arena = torch.zeros((2, NUM_OBS_CHANNELS, 48, 64))
    a_feat = net_custom.arena_stage2(net_custom.arena_stage1(net_custom.arena_stem(arena)))
    assert a_feat.shape == (2, net_custom.cfg.arena_dim, 6, 8)


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
def test_shared_trunk_and_ego_only_variants_params_flops_roi_and_backward(
    device_name: str,
) -> None:
    """Verify shared_trunk (~1M/~5M) & ego_only variants, FLOP ordering, ROI gather, CPU & MPS."""
    st_s = SharedTrunkArenaNet(SharedTrunkNetConfig.from_size("S"))
    st_m = SharedTrunkArenaNet(SharedTrunkNetConfig.from_size("M"))
    eo_s = EgoOnlyArenaNet(EgoOnlyNetConfig.from_size("S"))
    eo_m = EgoOnlyArenaNet(EgoOnlyNetConfig.from_size("M"))

    assert 900_000 <= st_s.parameter_count() <= 1_200_000
    assert 4_400_000 <= st_m.parameter_count() <= 5_300_000
    assert 150_000 <= eo_s.parameter_count() <= 500_000
    assert 900_000 <= eo_m.parameter_count() <= 2_000_000

    # Exact FlopCounterMode ordering on CPU (run once on cpu parametrize)
    if device_name == "cpu":
        f_eo_s = measure_model_flops_per_row("ego_only", "S")
        f_st_s = measure_model_flops_per_row("shared_trunk", "S")
        f_ps_s = measure_model_flops_per_row("per_seat", "S")
        assert (
            0
            < f_eo_s["forward_flops_per_row"]
            < f_st_s["forward_flops_per_row"]
            < f_ps_s["forward_flops_per_row"]
        )
        assert (
            0
            < f_eo_s["learner_flops_per_row"]
            < f_st_s["learner_flops_per_row"]
            < f_ps_s["learner_flops_per_row"]
        )

    dev = normalize_device(device_name)
    w_bsz, s_cnt = 2, 6
    rows = w_bsz * s_cnt
    for arch in ("shared_trunk", "ego_only"):
        model = build_arena_model(arch, "S", grid_h=ARENA_GRID_H, grid_w=ARENA_GRID_W).to(dev)
        wa = torch.randn((w_bsz, NUM_WORLD_CHANNELS, ARENA_GRID_H, ARENA_GRID_W), device=dev)
        ego = torch.randn((rows, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE), device=dev)
        scal = torch.randn((rows, SCALAR_DIM), device=dev)
        hxy = torch.zeros((w_bsz, s_cnt, 2), dtype=torch.long, device=dev)
        hxy[..., 0] = 40
        hxy[..., 1] = 30
        direction = torch.arange(s_cnt, dtype=torch.long, device=dev).remainder(4).expand(w_bsz, -1)
        alive = torch.ones((w_bsz, s_cnt), dtype=torch.bool, device=dev)
        alive[0, 5] = False  # Verify dead-seat masking inside shared models
        obs = SharedWorldObs(
            world_arena=wa,
            ego=ego,
            scalars=scal,
            head_xy=hxy,
            direction=direction,
            alive=alive,
        )
        legal_mask = torch.ones((rows, 6), dtype=torch.bool, device=dev)
        legal_mask[:6, 3:] = False

        act_out = model.act(obs, legal_mask=legal_mask, greedy=False)
        assert act_out.actions.shape == (rows,)
        assert (act_out.actions[:6] < 3).all()
        assert (act_out.masked_logits[:6, 3:] == MASK_FILL_VALUE).all()

        logp, ent, v_logits, a_logits = model.evaluate_actions(obs, legal_mask, act_out.actions)
        loss = (
            -logp.mean() - 0.01 * ent.mean() + v_logits.square().mean() + a_logits.square().mean()
        )
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) == sum(1 for p in model.parameters() if p.requires_grad)
        assert all(math.isfinite(g) for g in grad_norms)

        # Verify 5x5 heading-rotated ROI gather responds to heading rotation on shared_trunk
        if arch == "shared_trunk":
            model.eval()
            with torch.no_grad():
                out_dir0 = model.forward(obs, legal_mask=legal_mask)
                obs_rot = SharedWorldObs(
                    world_arena=wa,
                    ego=ego,
                    scalars=scal,
                    head_xy=hxy,
                    direction=(direction + 1).remainder(4),
                    alive=alive,
                )
                out_dir1 = model.forward(obs_rot, legal_mask=legal_mask)
                assert not torch.allclose(
                    out_dir0.masked_logits[:5, :3], out_dir1.masked_logits[:5, :3]
                )
                assert not torch.allclose(out_dir0.aux_logits[:5], out_dir1.aux_logits[:5])

        # Custom board resolution (48 x 64)
        custom_m = build_arena_model(arch, "S", grid_h=48, grid_w=64).to(dev)
        obs_c = SharedWorldObs(
            world_arena=torch.randn((2, NUM_WORLD_CHANNELS, 48, 64), device=dev),
            ego=torch.randn((8, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE), device=dev),
            scalars=torch.randn((8, SCALAR_DIM), device=dev),
            head_xy=torch.zeros((2, 4, 2), dtype=torch.long, device=dev),
            direction=torch.zeros((2, 4), dtype=torch.long, device=dev),
            alive=torch.ones((2, 4), dtype=torch.bool, device=dev),
        )
        out_c = custom_m.act(obs_c, legal_mask=torch.ones((8, 6), dtype=torch.bool, device=dev))
        assert out_c.actions.shape == (8,) and torch.isfinite(out_c.values).all()


def test_shared_trunk_wall_channel_padded_with_ones() -> None:
    """Verify `SharedTrunkArenaNet` pads `83x145 -> 84x148` with `1.0` in `WCH_WALL` (`ch 8`)."""
    wa = torch.zeros((2, NUM_WORLD_CHANNELS, ARENA_GRID_H, ARENA_GRID_W), dtype=torch.float32)
    padded = _pad_shared_world_arena(wa)
    assert padded.shape == (2, NUM_WORLD_CHANNELS, 84, 148)
    assert torch.all(padded[:, WCH_WALL, ARENA_GRID_H:, :] == 1.0)
    assert torch.all(padded[:, WCH_WALL, :, ARENA_GRID_W:] == 1.0)
    assert torch.all(padded[:, WCH_WALL, :ARENA_GRID_H, :ARENA_GRID_W] == 0.0)
    non_wall = [c for c in range(NUM_WORLD_CHANNELS) if c != WCH_WALL]
    assert torch.all(padded[:, non_wall, :, :] == 0.0)

    # Also verify `SharedTrunkArenaNet._forward_backbone_impl` actually feeds the 1-padded wall
    # strip into `arena_patch_stem`: zero the stem except for the bottom-right `(3, 3)` tap on
    # `WCH_WALL`, which reads `(y=83, x=147)` in patch `(20, 36)`.
    model = SharedTrunkArenaNet(SharedTrunkNetConfig.from_size("S")).eval()
    stem_conv = model.arena_patch_stem[0]
    assert isinstance(stem_conv, torch.nn.Conv2d)
    captured: list[torch.Tensor] = []
    handle = stem_conv.register_forward_hook(lambda _m, inp, _out: captured.append(inp[0].detach()))
    try:
        obs = SharedWorldObs(
            world_arena=wa,
            ego=torch.zeros((12, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE)),
            scalars=torch.zeros((12, SCALAR_DIM)),
            head_xy=torch.zeros((2, 6, 2), dtype=torch.long),
            direction=torch.zeros((2, 6), dtype=torch.long),
            alive=torch.ones((2, 6), dtype=torch.bool),
        )
        with torch.no_grad():
            model.forward(obs, legal_mask=torch.ones((12, 6), dtype=torch.bool))
    finally:
        handle.remove()

    assert len(captured) == 1
    stem_in = captured[0]
    assert stem_in.shape == (2, NUM_WORLD_CHANNELS, 84, 148)
    assert torch.all(stem_in[:, WCH_WALL, 83, :] == 1.0)
    assert torch.all(stem_in[:, WCH_WALL, :, 145:] == 1.0)
    assert torch.all(stem_in[:, non_wall, 83, :] == 0.0)
    assert torch.all(stem_in[:, non_wall, :, 145:] == 0.0)


@pytest.mark.parametrize("net_arch", ["shared_trunk", "ego_only"])
@pytest.mark.parametrize("net_size", ["S", "M"])
def test_minor5_shared_trunk_and_ego_only_seat_isolation_on_ego_and_scalars(
    net_arch: str,
    net_size: str,
) -> None:
    """Minor (5b): perturbing seat j's ego/scalars leaves every seat i != j unchanged."""
    torch.manual_seed(20261111)
    model = build_arena_model(
        net_arch,  # type: ignore[arg-type]
        net_size,  # type: ignore[arg-type]
        grid_h=ARENA_GRID_H,
        grid_w=ARENA_GRID_W,
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.policy_head.weight, std=0.5)
        torch.nn.init.normal_(model.value_head.weight, std=0.5)
        torch.nn.init.normal_(model.aux_head.weight, std=0.5)

    w_bsz, s_cnt = 2, 6
    rows = w_bsz * s_cnt
    wa = torch.randn((w_bsz, NUM_WORLD_CHANNELS, ARENA_GRID_H, ARENA_GRID_W))
    ego = torch.randn((rows, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE))
    scalars = torch.randn((rows, SCALAR_DIM))
    hxy = torch.tensor(
        [
            [[10, 12], [25, 30], [45, 20], [70, 50], [95, 40], [120, 65]],
            [[15, 60], [35, 18], [55, 42], [80, 22], [105, 58], [130, 33]],
        ],
        dtype=torch.long,
    )
    direction = torch.arange(rows, dtype=torch.long).remainder(4).reshape(w_bsz, s_cnt)
    alive = torch.ones((w_bsz, s_cnt), dtype=torch.bool)
    legal_mask = torch.ones((rows, 6), dtype=torch.bool)
    actions = torch.arange(rows, dtype=torch.long).remainder(6)

    obs_base = SharedWorldObs(
        world_arena=wa,
        ego=ego,
        scalars=scalars,
        head_xy=hxy,
        direction=direction,
        alive=alive,
    )

    # Perturb seat j=2 in world 0 (row 2) and seat j=4 in world 1 (row 10) strongly
    perturbed_rows = [2, 10]
    other_rows = [r for r in range(rows) if r not in perturbed_rows]
    ego_pert = ego.clone()
    scalars_pert = scalars.clone()
    ego_pert[perturbed_rows] = torch.randn_like(ego_pert[perturbed_rows]) * 5.0 + 3.0
    scalars_pert[perturbed_rows] = torch.randn_like(scalars_pert[perturbed_rows]) * 5.0 - 2.0
    obs_pert = SharedWorldObs(
        world_arena=wa,
        ego=ego_pert,
        scalars=scalars_pert,
        head_xy=hxy,
        direction=direction,
        alive=alive,
    )

    with torch.no_grad():
        raw_base = model.forward(obs_base, legal_mask=None)
        raw_pert = model.forward(obs_pert, legal_mask=None)
        out_base = model.forward(obs_base, legal_mask=legal_mask)
        out_pert = model.forward(obs_pert, legal_mask=legal_mask)
        lp_base, ent_base, vl_base, al_base = model.evaluate_actions(obs_base, legal_mask, actions)
        lp_pert, ent_pert, vl_pert, al_pert = model.evaluate_actions(obs_pert, legal_mask, actions)

    # Perturbed seats j MUST change
    assert not torch.allclose(
        raw_base.masked_logits[perturbed_rows], raw_pert.masked_logits[perturbed_rows], atol=1e-4
    )
    assert not torch.allclose(
        out_base.value_logits[perturbed_rows], out_pert.value_logits[perturbed_rows], atol=1e-4
    )

    # Every other seat i != j MUST remain unchanged
    assert torch.allclose(
        raw_base.masked_logits[other_rows],
        raw_pert.masked_logits[other_rows],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        out_base.masked_logits[other_rows],
        out_pert.masked_logits[other_rows],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        out_base.value[other_rows],
        out_pert.value[other_rows],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        out_base.value_logits[other_rows],
        out_pert.value_logits[other_rows],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        out_base.aux_logits[other_rows],
        out_pert.aux_logits[other_rows],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(lp_base[other_rows], lp_pert[other_rows], atol=1e-6, rtol=0.0)
    assert torch.allclose(ent_base[other_rows], ent_pert[other_rows], atol=1e-6, rtol=0.0)
    assert torch.allclose(vl_base[other_rows], vl_pert[other_rows], atol=1e-6, rtol=0.0)
    assert torch.allclose(al_base[other_rows], al_pert[other_rows], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("net_arch", ["ego_only", "shared_trunk", "per_seat"])
def test_amp_cpu_fallback_channels_last_and_evaluate_actions_skip_value_bit_identical(
    net_arch: str,
) -> None:
    """Task A: amp=True falls back cleanly on CPU; evaluate_actions(compute_value=False) exact."""
    torch.manual_seed(20261112)
    m_off = build_arena_model(net_arch, "S", amp=False).eval()  # type: ignore[arg-type]
    m_on = build_arena_model(net_arch, "S", amp=True).eval()  # type: ignore[arg-type]
    m_on.load_state_dict(m_off.state_dict())

    assert m_off.cfg.use_bf16_autocast is False
    assert m_on.cfg.use_bf16_autocast is True
    assert m_on.uses_fp16_grad_scaler(torch.device("cpu")) is False

    w_bsz, s_cnt = 2, 6
    rows = w_bsz * s_cnt
    ego = torch.randn((rows, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE))
    scalars = torch.randn((rows, SCALAR_DIM))
    legal_mask = torch.ones((rows, 6), dtype=torch.bool)
    legal_mask[:4, 3:] = False
    actions = torch.zeros((rows,), dtype=torch.long)

    with torch.no_grad():
        if isinstance(m_off, ArenaNet):
            assert isinstance(m_on, ArenaNet)
            arena = torch.randn((rows, NUM_OBS_CHANNELS, ARENA_GRID_H, ARENA_GRID_W))
            f_off = m_off.forward(arena, ego, scalars, legal_mask=legal_mask, compute_value=True)
            f_on = m_on.forward(arena, ego, scalars, legal_mask=legal_mask, compute_value=True)
            f_noval = m_off.forward(arena, ego, scalars, legal_mask=legal_mask, compute_value=False)
            lp_off, ent_off, vl_off, al_off = m_off.evaluate_actions(
                arena, ego, scalars, legal_mask, actions
            )
            lp_on, ent_on, vl_on, al_on = m_on.evaluate_actions(
                arena, ego, scalars, legal_mask, actions
            )
        else:
            obs = SharedWorldObs(
                world_arena=torch.randn((w_bsz, NUM_WORLD_CHANNELS, ARENA_GRID_H, ARENA_GRID_W)),
                ego=ego,
                scalars=scalars,
                head_xy=torch.zeros((w_bsz, s_cnt, 2), dtype=torch.long),
                direction=torch.zeros((w_bsz, s_cnt), dtype=torch.long),
                alive=torch.ones((w_bsz, s_cnt), dtype=torch.bool),
            )
            f_off = m_off.forward(  # type: ignore[arg-type]
                obs, legal_mask=legal_mask, compute_value=True
            )
            f_on = m_on.forward(  # type: ignore[arg-type]
                obs, legal_mask=legal_mask, compute_value=True
            )
            f_noval = m_off.forward(  # type: ignore[arg-type]
                obs, legal_mask=legal_mask, compute_value=False
            )
            lp_off, ent_off, vl_off, al_off = m_off.evaluate_actions(  # type: ignore[arg-type]
                obs, legal_mask, actions
            )
            lp_on, ent_on, vl_on, al_on = m_on.evaluate_actions(  # type: ignore[arg-type]
                obs, legal_mask, actions
            )

    assert torch.equal(f_off.masked_logits, f_on.masked_logits)
    assert torch.equal(f_off.value, f_on.value)
    assert torch.equal(f_off.value_logits, f_noval.value_logits)
    assert torch.equal(f_noval.value, torch.zeros_like(f_off.value))
    assert torch.equal(lp_off, lp_on) and lp_on.dtype == torch.float32
    assert torch.equal(ent_off, ent_on) and ent_on.dtype == torch.float32
    assert torch.equal(vl_off, vl_on) and vl_on.dtype == torch.float32
    assert torch.equal(al_off, al_on) and al_on.dtype == torch.float32
