"""Unit tests for `research/v3_ppo`: buffer indexing, GAE, action masking, net, and rewards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import research.v3_ppo.trainer as trainer_mod
from research.game_v3.config import GameV3Config
from research.game_v3.obs import LearnerIndex, build_obs
from research.game_v3.sim import GameV3Sim, StepInfo
from research.v3_ppo.buffer import Uint8RolloutBuffer, compute_gae
from research.v3_ppo.config import (
    BYTES_PER_LEARNER_STEP,
    DEFAULT_OBS_LAYOUT,
    NET_PRESETS,
    OBS_RASTER_BYTES_PER_STEP,
    RECIPE_OVERRIDE_KEYS,
    RESUME_MUTABLE_CONFIG_KEYS,
    ConfigTypeError,
    NetPreset,
    V3PPOConfig,
    assert_resume_config_compatible,
    estimate_rollout_buffer_bytes,
    estimate_training_memory,
    format_recipe_overrides,
    get_net_preset,
    immutable_config_diff,
    obs_layout_from_game_config,
    parse_recipe_overrides,
    resolve_full_export_settings,
)
from research.v3_ppo.eval import compute_step_reward
from research.v3_ppo.job import (
    check_eval_wall_budget,
    check_periodic_export_budget,
    decide_checkpoint_actions,
    decide_eval_action,
)
from research.v3_ppo.net import (
    MASK_FILL_VALUE,
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    build_ego_multiscale_net,
    symlog_two_hot_loss,
)
from research.v3_ppo.ppo import (
    V3PPOTrainer,
    _build_adamw_optimizer,
    _move_optimizer_state_to_device,
)
from research.v3_ppo.ppo import build_arg_parser as ppo_arg_parser
from research.v3_ppo.ppo import (
    resolve_effective_train_config,
    verify_full_checkpoint_set,
    verify_light_checkpoint,
    write_create_only_artifact,
    write_create_only_json,
)
from research.v3_ppo.prepare import build_arg_parser as prepare_arg_parser
from research.v3_ppo.prepare import prepare_v3_ppo_packet
from research.v3_ppo.ruler import _parse_nets_csv
from research.v3_ppo.ruler import build_arg_parser as ruler_arg_parser
from research.v3_ppo.ruler import resolve_effective_ruler_config


def test_network_sizes_flags_and_recurrent_seam() -> None:
    """Verify S (~1M) and M (~5M) parameter counts, uint8 conversion, flags, and RNN seam."""
    net_s = build_ego_multiscale_net("S")
    net_m = build_ego_multiscale_net("M")
    s_params = net_s.parameter_count()
    m_params = net_m.parameter_count()

    assert 900_000 <= s_params <= 1_150_000, f"Expected S ~1M params, got {s_params}"
    assert 4_500_000 <= m_params <= 5_500_000, f"Expected M ~5M params, got {m_params}"

    # Disable channels_last and bf16 autocast via flags
    net_plain = build_ego_multiscale_net("S", use_channels_last=False, use_bf16_autocast=False)
    assert net_plain.cfg.use_channels_last is False
    assert net_plain.cfg.use_bf16_autocast is False
    assert net_plain.recurrent_core is None

    b = 6
    r_s1 = torch.randint(0, 256, (b, 13, 64, 64), dtype=torch.uint8)
    r_s4 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    r_s16 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    scalars = torch.randn(b, 8, dtype=torch.float32)
    mask = torch.ones((b, 10), dtype=torch.bool)

    out = net_plain(r_s1, r_s4, r_s16, scalars, action_mask=mask)
    assert out.policy_logits.shape == (b, 10)
    assert out.value_logits.shape == (b, 255)
    assert out.value.shape == (b,)
    assert out.next_hidden is None

    # Verify clean recurrent seam (`recurrent_core` hook) works when attached in future rounds
    class _DummyGRUCore(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.cell = nn.GRUCell(dim, dim)

        def forward(
            self,
            features: torch.Tensor,
            hidden_state: torch.Tensor | None = None,
            episode_starts: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h0 = torch.zeros_like(features) if hidden_state is None else hidden_state
            if episode_starts is not None:
                h0 = torch.where(episode_starts[:, None], torch.zeros_like(h0), h0)
            h1 = self.cell(features, h0)
            return h1, h1

    net_plain.recurrent_core = _DummyGRUCore(net_plain.cfg.trunk_dim)
    h_init = torch.zeros((b, net_plain.cfg.trunk_dim), dtype=torch.float32)
    out_rnn = net_plain(r_s1, r_s4, r_s16, scalars, action_mask=mask, hidden_state=h_init)
    assert out_rnn.next_hidden is not None
    assert out_rnn.next_hidden.shape == (b, net_plain.cfg.trunk_dim)


def test_no_host_sync_in_net_forward_act_evaluate_and_two_hot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure `net.forward`, `act`, `evaluate_actions`, & `symlog_two_hot_loss` never sync host."""
    net = build_ego_multiscale_net("S", use_channels_last=False, use_bf16_autocast=False)
    b = 8
    r_s1 = torch.randint(0, 256, (b, 13, 64, 64), dtype=torch.uint8)
    r_s4 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    r_s16 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    scalars = torch.randn(b, 8, dtype=torch.float32)
    mixed_mask = torch.zeros((b, 10), dtype=torch.bool)
    mixed_mask[:4, :5] = True  # rows 4..7 are all-False fallback rows
    targets = torch.linspace(-50.0, 250.0, steps=b, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(77)

    def _forbid_sync(self: torch.Tensor) -> object:
        raise AssertionError("Unexpected host synchronization (.item() or bool(Tensor)) in net.py!")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "__bool__", _forbid_sync)

    fwd = net(r_s1, r_s4, r_s16, scalars, action_mask=mixed_mask)
    act_out = net.act(r_s1, r_s4, r_s16, scalars, action_mask=mixed_mask, generator=gen)
    greedy_out = net.act(r_s1, r_s4, r_s16, scalars, action_mask=mixed_mask, greedy=True)
    eval_out = net.evaluate_actions(
        r_s1, r_s4, r_s16, scalars, act_out.actions, action_mask=mixed_mask
    )
    loss = symlog_two_hot_loss(eval_out.value_logits, targets, net.bin_centers)

    assert fwd.policy_logits.shape == (b, 10)
    assert greedy_out.actions.shape == (b,)
    assert loss.ndim == 0


def test_action_masking_never_samples_masked_boost() -> None:
    """Boost actions (5..9) must NEVER be sampled when masked out by `action_mask`."""
    torch.manual_seed(12345)
    net = EgoMultiScaleNet(
        EgoMultiScaleNetConfig.from_size("S", use_channels_last=False, use_bf16_autocast=False)
    )
    # Deliberately bias policy_head strongly toward boost actions 5..9 to stress-test the mask
    with torch.no_grad():
        final_linear = net.policy_head[-1]
        assert isinstance(final_linear, nn.Linear)
        final_linear.bias[:5].fill_(-10.0)
        final_linear.bias[5:].fill_(25.0)

    n_draws = 10_000
    r_s1 = torch.randint(0, 256, (64, 13, 64, 64), dtype=torch.uint8)
    r_s4 = torch.randint(0, 256, (64, 13, 32, 32), dtype=torch.uint8)
    r_s16 = torch.randint(0, 256, (64, 13, 32, 32), dtype=torch.uint8)
    scalars = torch.randn(64, 8, dtype=torch.float32)

    # Mask: actions 0..4 legal, boost actions 5..9 illegal
    mask = torch.zeros((64, 10), dtype=torch.bool)
    mask[:, :5] = True

    fwd = net(r_s1, r_s4, r_s16, scalars, action_mask=mask)
    assert torch.all(fwd.policy_logits[:, 5:] == MASK_FILL_VALUE)
    probs = torch.softmax(fwd.policy_logits, dim=-1)
    assert float(probs[:, 5:].max().item()) == 0.0

    gen = torch.Generator(device="cpu").manual_seed(999)
    sampled_chunks = []
    for _ in range(n_draws // 64 + 1):
        act_out = net.act(r_s1, r_s4, r_s16, scalars, action_mask=mask, generator=gen)
        sampled_chunks.append(act_out.actions)
    sampled = torch.cat(sampled_chunks, dim=0)[:n_draws]
    assert int((sampled >= 5).sum().item()) == 0, "Masked boost action was sampled!"
    assert int((sampled < 0).sum().item()) == 0

    greedy_out = net.act(r_s1, r_s4, r_s16, scalars, action_mask=mask, greedy=True)
    assert int((greedy_out.actions >= 5).sum().item()) == 0

    eval_out = net.evaluate_actions(
        r_s1, r_s4, r_s16, scalars, greedy_out.actions, action_mask=mask
    )
    assert torch.isfinite(eval_out.log_probs).all()
    assert torch.isfinite(eval_out.entropy).all()
    assert (eval_out.entropy >= 0.0).all()

    # All-False mask fallback (dead/inactive row safety): falls back to normal non-boost 0..4
    empty_mask = torch.zeros((4, 10), dtype=torch.bool)
    safe_act = net.act(
        r_s1[:4], r_s4[:4], r_s16[:4], scalars[:4], action_mask=empty_mask, generator=gen
    )
    assert int((safe_act.actions >= 5).sum().item()) == 0
    assert int((safe_act.actions < 0).sum().item()) == 0
    assert torch.isfinite(safe_act.log_probs).all()


def test_uint8_rollout_buffer_zero_copy_views_and_scattered_minibatch_gather() -> None:
    """Verify zero-copy uint8 slice views and scattered LearnerIndex minibatch gathering."""
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=8, ambient_food_density=0.02)
    sim = GameV3Sim(gcfg, num_envs=3, device="cpu", seeds=[11, 22, 33])

    # Custom non-contiguous M-of-N mask (seats 1, 4, 6 in each world -> M = 9)
    mask = torch.zeros((3, 8), dtype=torch.bool)
    mask[:, [1, 4, 6]] = True
    learner_idx = LearnerIndex(sim, mask)
    assert learner_idx.count == 9

    buf = Uint8RolloutBuffer(
        rollout_len=5,
        learner_index=learner_idx,
        sim_shape=(sim.E, sim.N),
        device=torch.device("cpu"),
    )
    assert buf.raster_nbytes() == 5 * 9 * OBS_RASTER_BYTES_PER_STEP
    assert buf.nbytes_total() == 5 * 9 * BYTES_PER_LEARNER_STEP
    assert buf.raster_s1.dtype == torch.uint8
    assert buf.raster_s4.dtype == torch.uint8
    assert buf.raster_s16.dtype == torch.uint8

    for t in range(5):
        view = buf.obs_view(t)
        assert view.raster_s1.data_ptr() == buf.raster_s1[t].data_ptr()
        assert view.raster_s4.data_ptr() == buf.raster_s4[t].data_ptr()
        assert view.raster_s16.data_ptr() == buf.raster_s16[t].data_ptr()
        assert view.scalars.data_ptr() == buf.scalars[t].data_ptr()
        assert view.action_mask.data_ptr() == buf.action_mask[t].data_ptr()
        build_obs(sim, learner_index=learner_idx, out=view)
        ref = build_obs(sim, learner_index=learner_idx)
        assert torch.equal(buf.raster_s1[t], ref.raster_s1)
        assert torch.equal(buf.raster_s4[t], ref.raster_s4)
        assert torch.equal(buf.raster_s16[t], ref.raster_s16)
        assert torch.equal(buf.scalars[t], ref.scalars)
        assert torch.equal(buf.action_mask[t], ref.action_mask)
        sim.step(torch.full((3, 8), 2, dtype=torch.long))

    # Mark specific `(t, m)` entries active and verify `valid_flat_indices` + `gather_minibatch`
    buf.acted.zero_()
    active_pairs = [(0, 2), (1, 7), (3, 0), (4, 8)]
    for idx_i, (t, m) in enumerate(active_pairs):
        buf.acted[t, m] = True
        buf.actions[t, m] = idx_i + 1
        buf.log_probs[t, m] = -0.5 * (idx_i + 1)
        buf.values[t, m] = 1.25 * (idx_i + 1)
        buf.advantages[t, m] = 0.75 * (idx_i + 1)
        buf.returns[t, m] = 2.0 * (idx_i + 1)

    flat_valid = buf.valid_flat_indices()
    expected_flat = torch.tensor([t * 9 + m for t, m in active_pairs], dtype=torch.long)
    assert torch.equal(flat_valid, expected_flat)

    mb = buf.gather_minibatch(flat_valid)
    assert mb.raster_s1.dtype == torch.uint8
    for idx_i, (t, m) in enumerate(active_pairs):
        assert torch.equal(mb.raster_s1[idx_i], buf.raster_s1[t, m])
        assert int(mb.actions[idx_i].item()) == idx_i + 1
        assert pytest.approx(float(mb.advantages[idx_i].item())) == 0.75 * (idx_i + 1)
        assert pytest.approx(float(mb.returns[idx_i].item())) == 2.0 * (idx_i + 1)


def test_trainer_builds_at_most_one_obs_per_simulation_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify bootstrap caching keeps steady-state rollout collection at 1 build_obs per tick."""
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=8, ambient_food_density=0.02)
    real_build_obs = trainer_mod.build_obs
    call_counter = {"count": 0}

    def _counted_build_obs(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_counter["count"] += 1
        return real_build_obs(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "build_obs", _counted_build_obs)

    cfg = V3PPOConfig(
        game_config=gcfg,
        num_envs=2,
        learner_slots=4,
        rollout_len=6,
        ppo_epochs=2,
        minibatch_size=8,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    trainer = V3PPOTrainer(cfg, device="cpu")
    assert call_counter["count"] == 0
    trainer.train_step()
    assert call_counter["count"] == 1 + 6  # initial s_0 + 6 post-step ticks (s_6 cached!)
    trainer.train_step()
    assert call_counter["count"] == 1 + 6 + 6  # s_6 reused from cache -> exactly 6 calls!
    trainer.train_step()
    assert call_counter["count"] == 1 + 6 + 6 + 6  # strictly 1 build_obs per tick!
    assert trainer.obs_builds_executed == call_counter["count"]


def test_ppo_update_enforces_uniform_minibatch_size_without_host_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify uniform minibatch sizing (drop/pad) and zero per-batch host syncs in `ppo_update`."""
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=6, ambient_food_density=0.02)
    cfg = V3PPOConfig(
        game_config=gcfg,
        num_envs=2,
        learner_slots=4,
        rollout_len=4,
        ppo_epochs=2,
        minibatch_size=8,
        amp=False,
        use_channels_last=False,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.collect_rollout()

    # Force exact 20 valid rows (`20 = 2 * 8 + 4` -> drops 4-row remainder => 2 batches/epoch)
    trainer.buffer.acted.zero_()
    flat_acted = trainer.buffer.acted.view(-1)
    flat_acted[:20] = True
    trainer.buffer.actions.view(-1)[:20] = 2

    batch_sizes_seen: list[int] = []
    real_eval_actions = trainer.raw_model.evaluate_actions

    def _spy_eval_actions(*args, **kwargs):  # type: ignore[no-untyped-def]
        batch_sizes_seen.append(int(args[0].shape[0]))
        return real_eval_actions(*args, **kwargs)

    monkeypatch.setattr(trainer.raw_model, "evaluate_actions", _spy_eval_actions)

    real_item = torch.Tensor.item
    real_bool = torch.Tensor.__bool__
    v3_ppo_syncs: list[str] = []

    def _counted_item(self: torch.Tensor) -> object:
        caller_file = sys._getframe(1).f_code.co_filename
        if "research/v3_ppo" in caller_file.replace("\\", "/"):
            v3_ppo_syncs.append(f"item:{caller_file}:{sys._getframe(1).f_lineno}")
        return real_item(self)

    def _counted_bool(self: torch.Tensor) -> bool:
        caller_file = sys._getframe(1).f_code.co_filename
        if "research/v3_ppo" in caller_file.replace("\\", "/"):
            v3_ppo_syncs.append(f"bool:{caller_file}:{sys._getframe(1).f_lineno}")
        return real_bool(self)

    monkeypatch.setattr(torch.Tensor, "item", _counted_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", _counted_bool)

    stats_drop, learn_sec_drop = trainer.ppo_update()
    # 2 epochs * 2 full minibatches of exact size 8 (4-row remainder dropped)
    assert batch_sizes_seen == [8, 8, 8, 8]
    assert stats_drop["minibatches"] == 4.0
    assert learn_sec_drop >= 0.0
    # Zero `.item()` / `bool(Tensor)` syncs anywhere in `research/v3_ppo` (single `tolist()` at end)
    assert v3_ppo_syncs == []

    # Now test `n_valid = 5 < minibatch_size = 8`: pads to uniform size 8 with masked zero weights
    batch_sizes_seen.clear()
    flat_acted.zero_()
    flat_acted[:5] = True
    stats_pad, learn_sec_pad = trainer.ppo_update()
    assert batch_sizes_seen == [8, 8]
    assert stats_pad["minibatches"] == 2.0
    assert learn_sec_pad >= 0.0
    assert v3_ppo_syncs == []


def test_adamw_optimizer_builder_and_state_device_transfer() -> None:
    """Verify AdamW builder and fused step-tensor device/dtype transfer helper."""
    lin = nn.Linear(4, 2)
    opt = _build_adamw_optimizer(
        lin.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4, device=torch.device("cpu")
    )
    loss = lin(torch.ones((2, 4))).sum()
    loss.backward()
    opt.step()
    opt.defaults["fused"] = True
    _move_optimizer_state_to_device(opt, torch.device("cpu"))
    for st in opt.state.values():
        assert isinstance(st["step"], torch.Tensor)
        assert st["step"].dtype == torch.float32


def test_memory_estimate_includes_actor_chunk_and_eval_overhead() -> None:
    """Verify `estimate_training_memory` accounts for actor chunk float copy and eval worlds."""
    cfg_with_eval = V3PPOConfig(
        num_envs=128,
        net_size="S",
        rollout_len=32,
        eval_every_minutes=60.0,
    )
    cfg_no_eval = V3PPOConfig(
        num_envs=128,
        net_size="S",
        rollout_len=32,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    est_eval = estimate_training_memory(cfg_with_eval, gpu="L4")
    est_no_eval = estimate_training_memory(cfg_no_eval, gpu="L4")

    assert est_eval["actor_chunk_temp_bytes"] > 0
    assert est_eval["eval_overhead_bytes"] > 0
    assert est_no_eval["eval_overhead_bytes"] == 0
    assert est_eval["eval_phase_peak_bytes"] > est_eval["train_phase_peak_bytes"]
    assert est_eval["estimated_peak_bytes"] == est_eval["eval_phase_peak_bytes"]
    assert est_no_eval["estimated_peak_bytes"] == est_no_eval["train_phase_peak_bytes"]
    assert est_eval["estimated_peak_bytes"] > est_no_eval["estimated_peak_bytes"]
    assert est_eval["fits_gpu_budget"] is True

    # Raising `actor_chunk_learners` above learner activation size increases estimated peak VRAM
    cfg_big_chunk = V3PPOConfig(
        num_envs=256,
        net_size="S",
        rollout_len=32,
        minibatch_size=512,
        actor_chunk_learners=6144,
        eval_every_minutes=0.0,
    )
    cfg_small_chunk = V3PPOConfig(
        num_envs=256,
        net_size="S",
        rollout_len=32,
        minibatch_size=512,
        actor_chunk_learners=512,
        eval_every_minutes=0.0,
    )
    assert (
        estimate_training_memory(cfg_big_chunk)["estimated_peak_bytes"]
        > estimate_training_memory(cfg_small_chunk)["estimated_peak_bytes"]
    )


def test_gae_death_vs_truncation_and_respawn_gaps() -> None:
    """Verify death zeros bootstrap, truncation bootstraps V(s'), and dead ticks stay isolated."""
    gamma = 0.99
    lam = 0.95
    rewards = torch.tensor(
        [
            [1.0, 2.0, 0.5],
            [-1.0, 3.0, 0.5],
            [99.0, 1.5, 0.5],  # col 0 inactive at t=2 (reward must be ignored)
            [0.2, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [10.0, 5.0, 2.0],
            [8.0, 6.0, 3.0],
            [50.0, 7.0, 4.0],  # col 0 inactive at t=2 (value must NEVER be bootstrapped into t=1!)
            [1.0, 8.0, 5.0],
        ],
        dtype=torch.float32,
    )
    died = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    truncated = torch.tensor(
        [
            [False, False, False],
            [False, True, False],  # mid-rollout truncation on col 1
            [False, False, False],
            [True, True, True],  # rollout-boundary truncation
        ]
    )
    acted = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [False, True, True],  # col 0 waiting for respawn at t=2
            [True, True, True],
        ]
    )
    last_values = torch.tensor([3.0, 20.0, 10.0], dtype=torch.float32)

    adv, ret = compute_gae(
        rewards=rewards,
        values=values,
        died=died,
        truncated=truncated,
        acted=acted,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=lam,
    )

    assert pytest.approx(float(adv[3, 0].item()), abs=1e-5) == 0.2 + gamma * 3.0 - 1.0
    assert float(adv[2, 0].item()) == 0.0
    assert float(ret[2, 0].item()) == 0.0
    assert pytest.approx(float(adv[1, 0].item()), abs=1e-5) == -9.0
    assert pytest.approx(float(ret[1, 0].item()), abs=1e-5) == -1.0
    expected_adv_0_0 = (1.0 + gamma * 8.0 - 10.0) + gamma * lam * (-9.0)
    assert pytest.approx(float(adv[0, 0].item()), abs=1e-5) == expected_adv_0_0

    adv_3_1 = 4.0 + gamma * 20.0 - 8.0
    adv_2_1 = (1.5 + gamma * 8.0 - 7.0) + gamma * lam * adv_3_1
    adv_1_1 = 3.0 + gamma * 7.0 - 6.0
    adv_0_1 = (2.0 + gamma * 6.0 - 5.0) + gamma * lam * adv_1_1
    assert pytest.approx(float(adv[3, 1].item()), abs=1e-5) == adv_3_1
    assert pytest.approx(float(adv[2, 1].item()), abs=1e-5) == adv_2_1
    assert pytest.approx(float(adv[1, 1].item()), abs=1e-5) == adv_1_1
    assert pytest.approx(float(adv[0, 1].item()), abs=1e-5) == adv_0_1

    # Verify rollout-boundary truncation in V3PPOTrainer keeps surviving snakes alive in GameV3Sim
    gcfg = GameV3Config(arena_radius=80.0, num_snakes=6, ambient_food_density=0.02)
    cfg = V3PPOConfig(
        game_config=gcfg, num_envs=2, rollout_len=4, eval_every_minutes=0.0, eval_every_updates=0
    )
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.collect_rollout()
    assert int(trainer.buffer.truncated[-1].sum().item()) > 0
    assert int(trainer.sim.alive.sum().item()) > 0  # surviving snakes remain in the world!


def test_step_reward_coefficients_and_simultaneous_kill_death_rule() -> None:
    """Verify survivor reward, dying-snake-only penalty, and no kill bonus on simultaneous death."""
    cfg = V3PPOConfig(
        reward_alive_length_coef=0.002,
        reward_alive_const=0.05,
        reward_length_delta_coef=0.25,
        reward_kill_coef=0.75,
        reward_death_penalty=1.5,
    )
    # 1 world, 5 snakes:
    # Slot 0: survives, grows from 10 -> 13 (+3 delta), gets 1 kill (killed slot 1)
    # Slot 1: was length 20, DIES on this tick (info.length becomes 0 in GameV3Sim!)
    # Slot 2: survives, shrinks from 18 -> 17 (-1 boost shed), 0 kills
    # Slot 3: already dead waiting to respawn (`acted = False`)
    # Slot 4: gets 1 `kill_credit` in GameV3Sim AND simultaneously dies on the same tick ->
    #         trainer deliberately pays ONLY `-reward_death_penalty` (never rewards kamikaze trades)
    acted = torch.tensor([[True, True, True, False, True]])
    info = StepInfo(
        alive=torch.tensor([[True, False, True, False, False]]),
        life_start=torch.tensor([[False, False, False, False, False]]),
        died=torch.tensor([[False, True, False, False, True]]),
        truncated=torch.tensor([[False, False, False, False, False]]),
        in_respawn_delay=torch.tensor([[False, False, False, True, False]]),
        length=torch.tensor([[13, 0, 17, 0, 0]], dtype=torch.int32),
        prev_length=torch.tensor([[10, 20, 18, 0, 15]], dtype=torch.int32),
        mass_eaten=torch.tensor([[3, 0, 0, 0, 0]], dtype=torch.int32),
        killer_id=torch.tensor([[-1, 0, -1, -1, 2]], dtype=torch.int32),
        kill_credit=torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.int32),
        life_age=torch.tensor([[12, 0, 9, 0, 0]], dtype=torch.int32),
    )

    rew = compute_step_reward(info, acted, cfg)
    assert rew.shape == (1, 5)

    assert pytest.approx(float(rew[0, 0].item()), abs=1e-6) == 0.05 + 0.026 + 0.75 + 0.75
    assert pytest.approx(float(rew[0, 1].item()), abs=1e-6) == -1.5
    assert pytest.approx(float(rew[0, 2].item()), abs=1e-6) == 0.05 + 0.034 - 0.25
    assert float(rew[0, 3].item()) == 0.0
    assert pytest.approx(float(rew[0, 4].item()), abs=1e-6) == -1.5


def test_fast_learner_ops_match_reference_within_float32_rounding() -> None:
    """Verify `fast_learner_ops` matches dense reference preprocessing, loss, and step updates."""
    torch.manual_seed(20261111)
    net_fast = build_ego_multiscale_net(
        "S", use_channels_last=True, use_bf16_autocast=False, fast_preprocess=True
    )
    net_slow = build_ego_multiscale_net(
        "S", use_channels_last=True, use_bf16_autocast=False, fast_preprocess=False
    )
    net_slow.load_state_dict(net_fast.state_dict())

    b = 12
    r_s1 = torch.randint(0, 256, (b, 13, 64, 64), dtype=torch.uint8)
    r_s4 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    r_s16 = torch.randint(0, 256, (b, 13, 32, 32), dtype=torch.uint8)
    scalars = torch.randn(b, 8, dtype=torch.float32)
    mask = torch.ones((b, 10), dtype=torch.bool)
    mask[:, 7:] = False
    actions = torch.randint(0, 7, (b,), dtype=torch.long)

    # 1. Single-call `_prepare_raster` matches reference `uint8 -> float32 / 255` conversion
    prep_fast = net_fast._prepare_raster(r_s1, expected_hw=64)
    prep_slow = net_slow._prepare_raster(r_s1, expected_hw=64)
    assert torch.equal(prep_fast, prep_slow)
    assert prep_fast.is_contiguous(memory_format=torch.channels_last)

    # 2. `evaluate_actions(compute_values=False)` matches reference within float32 rounding
    ev_fast = net_fast.evaluate_actions(
        r_s1, r_s4, r_s16, scalars, actions, action_mask=mask, compute_values=False
    )
    ev_slow = net_slow.evaluate_actions(
        r_s1, r_s4, r_s16, scalars, actions, action_mask=mask, compute_values=True
    )
    assert torch.allclose(ev_fast.log_probs, ev_slow.log_probs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ev_fast.entropy, ev_slow.entropy, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ev_fast.value_logits, ev_slow.value_logits, atol=1e-6, rtol=1e-6)

    # 3. `symlog_two_hot_loss(fast=True)` matches `fast=False` values & gradients within ~1e-6
    logits_a = torch.randn(b, 255, dtype=torch.float32, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    targets = torch.tensor(
        [-1e12, -250.0, -3.5, -0.1, 0.0, 0.1, 2.75, 19.0, 400.0, 1e12, -12.3, 88.8],
        dtype=torch.float32,
    )
    weights = torch.linspace(0.2, 1.0, steps=b, dtype=torch.float32)
    loss_fast = (
        symlog_two_hot_loss(logits_a, targets, net_fast.bin_centers, fast=True) * weights
    ).mean()
    loss_slow = (
        symlog_two_hot_loss(logits_b, targets, net_slow.bin_centers, fast=False) * weights
    ).mean()
    assert torch.allclose(loss_fast, loss_slow, atol=1e-6, rtol=1e-6)
    loss_fast.backward()
    loss_slow.backward()
    assert logits_a.grad is not None and logits_b.grad is not None
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-6, rtol=1e-6)

    # 4. Multi-update `V3PPOTrainer.train_step()` rounding-level equivalence (`~1e-7..1e-6` over 6
    #    updates; NOT bit-for-bit identical because `log_probs.exp()` vs `softmax` and two-bin
    #    gather vs 255-bin `scatter_add_` accumulate float32 ULP rounding differences).
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=6, ambient_food_density=0.02)
    base_kw = dict(
        game_config=gcfg,
        num_envs=2,
        learner_slots=4,
        rollout_len=4,
        ppo_epochs=2,
        minibatch_size=8,
        amp=False,
        use_channels_last=False,
        eval_every_minutes=0.0,
        eval_every_updates=0,
        seed=5150,
    )
    assert V3PPOConfig().compile_model is False
    tr_fast = V3PPOTrainer(V3PPOConfig(**base_kw, fast_learner_ops=True), device="cpu")
    assert tr_fast.raw_model.is_compiled is False
    for _ in range(6):
        out_fast = tr_fast.train_step()
    assert tr_fast.raw_model.compiled_trunk_calls == 0

    tr_slow = V3PPOTrainer(V3PPOConfig(**base_kw, fast_learner_ops=False), device="cpu")
    for _ in range(6):
        out_slow = tr_slow.train_step()

    for metric_key in ("policy_loss", "value_loss", "entropy", "grad_norm"):
        assert pytest.approx(out_fast[metric_key], abs=1e-4) == out_slow[metric_key]
    max_weight_diff = max(
        float((p_fast - tr_slow.raw_model.state_dict()[p_name]).abs().max().item())
        for p_name, p_fast in tr_fast.raw_model.state_dict().items()
    )
    assert max_weight_diff <= 2e-6
    assert torch.equal(tr_fast.sim.length, tr_slow.sim.length)


def test_compiled_trunk_executes_during_rollout_and_learner_update() -> None:
    """Verify `compile_model=True` invokes the compiled trunk in both rollout and learner steps."""
    gcfg = GameV3Config(arena_radius=60.0, num_snakes=6, ambient_food_density=0.02)
    base_kw = dict(
        game_config=gcfg,
        num_envs=2,
        learner_slots=4,
        rollout_len=4,
        ppo_epochs=2,
        minibatch_size=8,
        amp=False,
        use_channels_last=False,
        eval_every_minutes=0.0,
        eval_every_updates=0,
        seed=5150,
    )
    tr_uncompiled = V3PPOTrainer(V3PPOConfig(**base_kw, compile_model=False), device="cpu")
    tr_compiled = V3PPOTrainer(
        V3PPOConfig(**base_kw, compile_model=True, fast_learner_ops=True),
        device="cpu",
    )
    assert tr_compiled.raw_model.is_compiled is True
    orig_compiled_fn = tr_compiled.raw_model._compiled_encode_trunk
    assert orig_compiled_fn is not None and callable(orig_compiled_fn)
    assert set(tr_compiled.raw_model.state_dict().keys()) == set(
        tr_uncompiled.raw_model.state_dict()
    )

    grad_flags_seen: list[bool] = []

    def _spy_compiled_trunk(*args, **kwargs):  # type: ignore[no-untyped-def]
        grad_flags_seen.append(bool(torch.is_grad_enabled()))
        return orig_compiled_fn(*args, **kwargs)

    tr_compiled.raw_model._compiled_encode_trunk = _spy_compiled_trunk
    tr_compiled.collect_rollout()
    rollout_calls = len(grad_flags_seen)
    # 4 rollout ticks (`act`) + 1 bootstrap `s_T` (`forward`) = 5 compiled calls, all no_grad!
    assert rollout_calls >= 5
    assert all(flag is False for flag in grad_flags_seen)

    upd_stats, learn_sec = tr_compiled.ppo_update()
    learner_flags = grad_flags_seen[rollout_calls:]
    # 2 PPO epochs * (32 learner steps / minibatch_size 8 = 4 minibatches) = 8 compiled grad calls!
    assert len(learner_flags) >= 4
    assert all(flag is True for flag in learner_flags)
    assert tr_compiled.raw_model.compiled_trunk_calls == len(grad_flags_seen)
    assert upd_stats["grad_norm"] > 0.0
    assert learn_sec >= 0.0


def test_compile_and_fast_learner_flags_parse_across_all_clis() -> None:
    """Verify `--compile` and `--no-fast-learner-ops` parse identically on ppo, ruler, & prepare."""
    ppo_args = ppo_arg_parser().parse_args(
        ["--output-dir", "/tmp/o", "--compile", "--no-fast-learner-ops"]
    )
    assert ppo_args.compile is True
    assert ppo_args.fast_learner_ops is False

    ruler_args = ruler_arg_parser().parse_args(
        ["--output-dir", "/tmp/o", "--compile", "--no-fast-learner-ops"]
    )
    assert ruler_args.compile is True
    assert ruler_args.fast_learner_ops is False

    prep_args = prepare_arg_parser().parse_args(
        ["--packet", "/tmp/p", "--compile", "--no-fast-learner-ops"]
    )
    assert prep_args.compile is True
    assert prep_args.fast_learner_ops is False


def test_net_presets_registry_and_unknown_size_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify every registered preset matches built parameter count and unknown sizes fail fast."""
    assert tuple(NET_PRESETS.keys()) == ("S", "M")
    for name, preset in NET_PRESETS.items():
        assert preset.name == name
        net = build_ego_multiscale_net(preset.name)
        actual_params = net.parameter_count()
        assert actual_params == preset.exact_param_count
        rel_err = abs(actual_params - preset.memory_model_param_count) / float(actual_params)
        assert rel_err <= 0.10

    for bad_size in ("L", "XL", "s", ""):
        with pytest.raises(ValueError, match="Unknown net_size"):
            get_net_preset(bad_size)
        with pytest.raises(ValueError, match="Unknown net_size"):
            EgoMultiScaleNetConfig.from_size(bad_size)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown net_size"):
            V3PPOConfig(net_size=bad_size)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown net_size"):
        _parse_nets_csv("S,L")

    for mode in ("canary", "ruler", "train"):
        with pytest.raises(ValueError, match="Unknown net_size"):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_{mode}",
                mode=mode,
                gpu="L4",
                net_size="L",  # type: ignore[arg-type]
                output_dir=tmp_path / f"out_{mode}",
                allow_missing_canary_receipt=True,
                allow_unmerged_commit=True,
            )

    with pytest.raises(ValueError, match="Unknown net_size"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_ruler_nets",
            mode="ruler",
            gpu="L4",
            ruler_nets="S,XL",
            output_dir=tmp_path / "out_ruler_nets",
            allow_missing_canary_receipt=True,
            allow_unmerged_commit=True,
        )

    # Prove adding a preset is a single-entry registry addition in `NET_PRESETS`
    xs_preset = NetPreset(
        name="XS",  # type: ignore[arg-type]
        ruler_order=-1,
        s1_c1=8,
        s1_c2=16,
        s32_c1=8,
        stage8_ch=24,
        stem_out_ch=32,
        scalar_embed_dim=16,
        trunk_dim=64,
        trunk_blocks=1,
        head_dim=32,
        exact_param_count=0,
        memory_model_param_count=150_000,
        learner_act_bytes_per_row_bf16=256_000,
        actor_fwd_act_bytes_per_row_bf16=64_000,
        default_minibatch_size=2048,
        default_actor_chunk_learners=4096,
        est_light_checkpoint_bytes=4 * 1024 * 1024,
    )
    monkeypatch.setitem(NET_PRESETS, "XS", xs_preset)
    xs_net = build_ego_multiscale_net("XS")  # type: ignore[arg-type]
    xs_cfg = V3PPOConfig(net_size="XS", num_envs=8, rollout_len=4)  # type: ignore[arg-type]
    assert xs_cfg.resolved_minibatch_size == 2048
    assert estimate_training_memory(xs_cfg)["estimated_peak_bytes"] > 0
    assert xs_net.parameter_count() > 0


def test_obs_layout_single_sourced_trainer_agreement_and_checkpoint_field(
    tmp_path: Path,
) -> None:
    """Verify raster sizes derive from GameV3Config, assert at trainer init, and save in ckpts."""
    custom_game = GameV3Config(
        arena_radius=120.0,
        num_snakes=6,
        obs_fine_size=32,
        obs_coarse_size=16,
    )
    custom_layout = obs_layout_from_game_config(custom_game)
    assert custom_layout.raster_sizes == (32, 16, 16)
    assert custom_layout.raster_bytes_per_step == 13 * (32 * 32 + 16 * 16 + 16 * 16)

    sim = GameV3Sim(custom_game, num_envs=2, device="cpu", seeds=[11, 22])
    mask = torch.ones((2, 6), dtype=torch.bool)
    idx = LearnerIndex(sim, mask)
    buf = Uint8RolloutBuffer(
        rollout_len=3,
        learner_index=idx,
        sim_shape=(2, 6),
        device=torch.device("cpu"),
        obs_layout=custom_layout,
    )
    assert buf.raster_s1.shape == (3, 12, 13, 32, 32)
    assert buf.raster_s4.shape == (3, 12, 13, 16, 16)
    assert buf.nbytes_total() == (
        estimate_rollout_buffer_bytes(
            num_envs=2,
            learner_slots=6,
            rollout_len=3,
            game_cfg=custom_game,
        )["rollout_buffer_bytes"]
    )

    bad_cfg = V3PPOConfig(game_config=custom_game, num_envs=2, rollout_len=2, minibatch_size=8)
    with pytest.raises(ValueError, match="Observation raster sizes"):
        V3PPOTrainer(bad_cfg, device="cpu")

    ok_game = GameV3Config(arena_radius=120.0, num_snakes=6, max_length=64)
    ok_cfg = V3PPOConfig(
        game_config=ok_game,
        num_envs=2,
        learner_slots=2,
        rollout_len=2,
        minibatch_size=4,
        ppo_epochs=1,
        amp=False,
        use_channels_last=False,
    )
    trainer = V3PPOTrainer(ok_cfg, device="cpu")
    assert trainer.obs_layout == DEFAULT_OBS_LAYOUT
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    full_idx = trainer.save_full_checkpoint(ckpt_dir, tag="u0000000")
    light_path = ckpt_dir / "light_u0000000.pt"

    light_meta = verify_light_checkpoint(light_path)
    full_meta = verify_full_checkpoint_set(full_idx)
    assert light_meta["obs_layout"] == DEFAULT_OBS_LAYOUT.to_dict()
    assert full_meta["obs_layout"] == DEFAULT_OBS_LAYOUT.to_dict()

    # Checkpoints lacking `obs_layout` (legacy e2865f93 schema) still verify and load cleanly.
    legacy_payload = torch.load(light_path, map_location="cpu", weights_only=False)
    assert "obs_layout" in legacy_payload
    del legacy_payload["obs_layout"]
    legacy_light = tmp_path / "legacy_light.pt"
    write_create_only_artifact(legacy_light, lambda tmp: torch.save(legacy_payload, tmp))
    assert verify_light_checkpoint(legacy_light)["obs_layout"] is None
    restored = V3PPOTrainer(ok_cfg, device="cpu")
    restored.load_checkpoint(legacy_light)

    # Tampered `obs_layout` is rejected cleanly.
    bad_payload = dict(legacy_payload)
    bad_payload["obs_layout"] = {**DEFAULT_OBS_LAYOUT.to_dict(), "fine_size": 32}
    bad_light = tmp_path / "bad_light.pt"
    write_create_only_artifact(bad_light, lambda tmp: torch.save(bad_payload, tmp))
    with pytest.raises(ValueError, match="obs_layout"):
        verify_light_checkpoint(bad_light)

    bad_idx_obj = json.loads(full_idx.read_text(encoding="utf-8"))
    bad_idx_obj["obs_layout"] = {**DEFAULT_OBS_LAYOUT.to_dict(), "mid_size": 16}
    bad_idx = ckpt_dir / "full_bad.index.json"
    write_create_only_json(bad_idx, bad_idx_obj)
    with pytest.raises(ValueError, match="obs_layout"):
        verify_full_checkpoint_set(bad_idx)


def test_train_and_ruler_cli_config_json_and_env_precedence() -> None:
    """Verify CLI > --config-json > env > defaults precedence across ppo and ruler entrypoints."""
    parser = ppo_arg_parser()

    # 1. Pure defaults when neither CLI overrides, --config-json, nor env vars are given
    eff_default = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/out"]),
        environ={},
    )
    assert eff_default.cfg == V3PPOConfig()
    assert eff_default.device == "cpu"
    assert eff_default.max_wall_seconds is None
    assert eff_default.max_session_export_bytes is None

    # 2. Full 4-layer precedence: CLI flag > --config-json > env > defaults
    env_vars = {
        "SNAKE_DQN_DEVICE": "cuda:0",
        "V3_PPO_CONFIG_JSON": json.dumps(
            {"num_envs": 32, "lr": 1.0e-4, "rollout_len": 16, "amp": True}
        ),
        "V3_PPO_AMP": "0",
        "V3_PPO_COMPILE": "1",
        "V3_PPO_MAX_WALL_SECONDS": "1200",
        "V3_PPO_MAX_SESSION_EXPORT_BYTES": "999999",
    }
    cli_json = json.dumps({"lr": 2.0e-4, "ppo_epochs": 4, "compile_model": False})
    parsed = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--config-json",
            cli_json,
            "--ppo-epochs",
            "6",
            "--total-updates",
            "42",
            "--compile",
            "--net-size",
            "M",
        ]
    )
    eff = resolve_effective_train_config(parsed, environ=env_vars)
    assert eff.cfg.num_envs == 32  # from env V3_PPO_CONFIG_JSON
    assert eff.cfg.rollout_len == 16  # from env V3_PPO_CONFIG_JSON
    assert eff.cfg.amp is False  # env V3_PPO_AMP=0 overrode env JSON amp=True
    assert eff.cfg.lr == pytest.approx(2.0e-4)  # --config-json overrode env JSON lr=1e-4
    assert eff.cfg.ppo_epochs == 6  # CLI --ppo-epochs=6 overrode --config-json ppo_epochs=4
    assert eff.cfg.total_updates == 42  # CLI --total-updates=42 overrode default 100
    assert eff.cfg.compile_model is True  # CLI --compile overrode --config-json compile_model=False
    assert eff.cfg.net_size == "M"  # CLI --net-size=M overrode default S
    assert eff.cfg.resolved_minibatch_size == NET_PRESETS["M"].default_minibatch_size
    assert eff.device == "cuda:0"
    assert eff.max_wall_seconds == pytest.approx(1200.0)
    assert eff.max_session_export_bytes == 999999

    # 3. Unknown JSON keys and invalid boolean env vars fail loudly
    with pytest.raises(ValueError, match="Unknown keys in --config-json"):
        resolve_effective_train_config(
            parser.parse_args(["--output-dir", "/tmp/o", "--config-json", '{"bogus_key": 1}']),
            environ={},
        )
    with pytest.raises(ValueError, match="Invalid boolean environment variable V3_PPO_AMP"):
        resolve_effective_train_config(
            parser.parse_args(["--output-dir", "/tmp/o"]),
            environ={"V3_PPO_AMP": "maybe"},
        )

    # 4. Ruler CLI > env > defaults
    r_eff = resolve_effective_ruler_config(
        ruler_arg_parser().parse_args(["--output-dir", "/tmp/r", "--envs", "64,128", "--no-amp"]),
        environ={
            "V3_PPO_RULER_ENVS": "256",
            "V3_PPO_AMP": "1",
            "V3_PPO_RULER_NETS": "M",
            "SNAKE_DQN_DEVICE": "cpu",
        },
    )
    assert r_eff["envs_list"] == [64, 128]
    assert r_eff["amp"] is False
    assert r_eff["net_sizes"] == ["M"]
    assert r_eff["device"] == "cpu"


def test_resume_config_diff_allows_mutable_keys_and_rejects_immutable_changes() -> None:
    """Single-sourced resume diff permits runtime/schedule keys and rejects arch/MDP changes."""
    base = V3PPOConfig()
    mutated = V3PPOConfig(
        total_updates=500,
        eval_every_minutes=15.0,
        compile_model=True,
        fast_learner_ops=False,
        full_exports="periodic",
    )
    assert (
        set(
            [
                "total_updates",
                "eval_every_minutes",
                "compile_model",
                "fast_learner_ops",
                "full_exports",
            ]
        )
        <= RESUME_MUTABLE_CONFIG_KEYS
    )
    assert immutable_config_diff(base, mutated) == {}
    assert_resume_config_compatible(base, mutated)

    incompat = V3PPOConfig(num_envs=64, lr=1.0e-3)
    diffs = immutable_config_diff(base, incompat)
    assert set(diffs.keys()) == {"lr", "num_envs"}
    with pytest.raises(ValueError, match="Continuation checkpoint config mismatch on 'lr'"):
        assert_resume_config_compatible(
            base,
            incompat,
            context="Continuation checkpoint config mismatch",
            reference_label="ckpt",
            candidate_label="requested",
        )


def test_pure_checkpoint_and_eval_schedulers() -> None:
    """Verify pure checkpoint and evaluation scheduling functions across edge cases."""
    # 1. Idle step: nothing due
    idle = decide_checkpoint_actions(
        updates_completed=5,
        checkpoint_every_updates=0,
        full_checkpoint_every_updates=0,
        light_checkpoint_interval_sec=1800.0,
        full_checkpoint_interval_sec=7200.0,
        now_mono=1100.0,
        last_light_work_mono=1000.0,
        last_light_export_mono=1000.0,
        last_full_work_mono=1000.0,
        published_periodic_fulls=0,
        max_periodic_full_exports=3,
    )
    assert not idle.save_work_light
    assert not idle.save_work_full
    assert not idle.want_periodic_full_export
    assert not idle.want_periodic_light_export

    # 2. Capped periodic FULL: saves FULL in work/, blocks FULL export, preserves LIGHT interval
    capped_full = decide_checkpoint_actions(
        updates_completed=16,
        checkpoint_every_updates=0,
        full_checkpoint_every_updates=0,
        light_checkpoint_interval_sec=30.0,
        full_checkpoint_interval_sec=75.0,
        now_mono=1160.0,
        last_light_work_mono=1140.0,
        last_light_export_mono=1140.0,
        last_full_work_mono=1080.0,
        published_periodic_fulls=1,
        max_periodic_full_exports=1,
    )
    assert capped_full.save_work_light is True
    assert capped_full.save_work_full is True
    assert capped_full.want_periodic_full_export is False
    assert capped_full.want_periodic_light_export is False

    # Next step (`now_mono=1170.0`): `last_light_work_mono` advanced to 1160, but
    # `last_light_export_mono` stayed 1140 -> LIGHT export triggers right on time (+30s)!
    next_light = decide_checkpoint_actions(
        updates_completed=17,
        checkpoint_every_updates=0,
        full_checkpoint_every_updates=0,
        light_checkpoint_interval_sec=30.0,
        full_checkpoint_interval_sec=75.0,
        now_mono=1170.0,
        last_light_work_mono=1160.0,
        last_light_export_mono=1140.0,
        last_full_work_mono=1160.0,
        published_periodic_fulls=1,
        max_periodic_full_exports=1,
    )
    assert next_light.due_light_work_time is False
    assert next_light.due_light_export_time is True
    assert next_light.save_work_light is True
    assert next_light.want_periodic_light_export is True

    # 3. Export byte-budget reserve calculation
    ok_b, proj_b = check_periodic_export_budget(
        current_exported_bytes=200,
        new_files_bytes=100,
        max_session_export_bytes=500,
        last_measured_full_bytes=150,
        est_light_bytes=50,
        num_envs=4,
    )
    assert ok_b is True and proj_b == 450
    no_b, proj_over = check_periodic_export_budget(
        current_exported_bytes=260,
        new_files_bytes=100,
        max_session_export_bytes=500,
        last_measured_full_bytes=150,
        est_light_bytes=50,
        num_envs=4,
    )
    assert no_b is False and proj_over == 510

    # 4. Evaluation scheduling: update/minute triggers and 2x wall-clock budget guard
    ev_run = decide_eval_action(
        updates_completed=10,
        total_updates=100,
        eval_every_updates=5,
        eval_every_minutes=0.0,
        now_mono=1050.0,
        last_eval_mono=1000.0,
        wall_t0=1000.0,
        max_wall_seconds=300.0,
        checkpoint_reserve_seconds=20.0,
        eval_estimated_seconds=45.0,
        last_eval_wall_seconds=None,
    )
    assert ev_run.due_by_updates is True
    assert ev_run.should_run is True
    assert ev_run.should_skip_for_wall is False
    assert ev_run.required_wall_seconds == pytest.approx(90.0)
    assert ev_run.remaining_wall_seconds == pytest.approx(230.0)

    ev_skip = decide_eval_action(
        updates_completed=11,
        total_updates=100,
        eval_every_updates=0,
        eval_every_minutes=2.0,
        now_mono=1250.0,
        last_eval_mono=1100.0,
        wall_t0=1000.0,
        max_wall_seconds=300.0,
        checkpoint_reserve_seconds=20.0,
        eval_estimated_seconds=45.0,
        last_eval_wall_seconds=20.0,
    )
    # Measured `last_eval_wall_seconds=20.0` -> requires 40.0s, only 30.0s remain!
    assert ev_skip.due_by_minutes is True
    assert ev_skip.should_run is False
    assert ev_skip.should_skip_for_wall is True
    assert ev_skip.required_wall_seconds == pytest.approx(40.0)
    assert ev_skip.remaining_wall_seconds == pytest.approx(30.0)

    can_unl, rem_unl, req_unl = check_eval_wall_budget(
        elapsed_wall_seconds=9999.0,
        max_wall_seconds=None,
        checkpoint_reserve_seconds=0.0,
        eval_estimated_seconds=30.0,
        last_eval_wall_seconds=None,
    )
    assert can_unl is True and rem_unl == float("inf") and req_unl == pytest.approx(60.0)

    # 5. `none` & `final` modes still save `work/` FULLs when due, never request periodic FULL
    #    export, and `include_final_full_reserve=False` reserves only `est_light_bytes`!
    for mode_no_per in ("none", "final"):
        act_no_per = decide_checkpoint_actions(
            updates_completed=8,
            checkpoint_every_updates=0,
            full_checkpoint_every_updates=0,
            light_checkpoint_interval_sec=30.0,
            full_checkpoint_interval_sec=75.0,
            now_mono=1080.0,
            last_light_work_mono=1060.0,
            last_light_export_mono=1060.0,
            last_full_work_mono=1000.0,
            published_periodic_fulls=0,
            max_periodic_full_exports=3,
            full_exports=mode_no_per,  # type: ignore[arg-type]
        )
        assert act_no_per.save_work_full is True
        assert act_no_per.want_periodic_full_export is False
        assert act_no_per.want_periodic_light_export is False

    ok_none_b, proj_none_b = check_periodic_export_budget(
        current_exported_bytes=200,
        new_files_bytes=50,
        max_session_export_bytes=320,
        last_measured_full_bytes=10_000,
        est_light_bytes=50,
        num_envs=4,
        include_final_full_reserve=False,
    )
    assert ok_none_b is True and proj_none_b == 300


def test_full_exports_flag_precedence_legacy_mapping_and_conflict_refusals(
    tmp_path: Path,
) -> None:
    """Verify `--full-exports`, legacy `--max-periodic-full-exports`, and conflict refusals."""
    assert V3PPOConfig().full_exports == "none"
    assert resolve_full_export_settings(None, None) == ("none", 3)
    assert resolve_full_export_settings("none", None) == ("none", 3)
    assert resolve_full_export_settings("final", None) == ("final", 0)
    assert resolve_full_export_settings("periodic", None) == ("periodic", 3)
    assert resolve_full_export_settings(None, 0) == ("final", 0)
    assert resolve_full_export_settings(None, 2) == ("periodic", 2)
    assert resolve_full_export_settings("final", 0) == ("final", 0)
    assert resolve_full_export_settings("periodic", 2) == ("periodic", 2)

    with pytest.raises(ValueError, match="full_exports='periodic' requires"):
        V3PPOConfig(full_exports="periodic", max_periodic_full_exports=0)

    parser = ppo_arg_parser()

    # Legacy `--max-periodic-full-exports` without `--full-exports` keeps working
    eff_mp0 = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--max-periodic-full-exports", "0"]),
        environ={},
    )
    assert eff_mp0.cfg.full_exports == "final"
    assert eff_mp0.cfg.max_periodic_full_exports == 0

    eff_mp2 = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--max-periodic-full-exports", "2"]),
        environ={},
    )
    assert eff_mp2.cfg.full_exports == "periodic"
    assert eff_mp2.cfg.max_periodic_full_exports == 2

    # CLI `--full-exports none` overrides `V3_PPO_CONFIG_JSON` `"periodic"` cleanly
    eff_override = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--full-exports", "none"]),
        environ={
            "V3_PPO_CONFIG_JSON": json.dumps(
                {"full_exports": "periodic", "max_periodic_full_exports": 2}
            )
        },
    )
    assert eff_override.cfg.full_exports == "none"

    # A `--config-json` / `--ppo-config-json` dump that merely carries `max_periodic_full_exports`
    # (or even `full_exports: "periodic"` in `prepare --ppo-config-json`) MUST NOT silently select
    # `periodic`: only an explicit `--full-exports` or `--max-periodic-full-exports` CLI flag does!
    old_dump_json = json.dumps({"num_envs": 128, "max_periodic_full_exports": 3})
    eff_old_dump = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--config-json", old_dump_json]),
        environ={},
    )
    assert eff_old_dump.cfg.full_exports == "none"

    from research.v3_ppo.prepare import build_arg_parser as prep_arg_parser
    from research.v3_ppo.prepare import options_from_cli_args, prepare_packet_from_options

    for idx_dump, dump_payload in enumerate(
        (
            {"num_envs": 128, "max_periodic_full_exports": 3},
            {"num_envs": 128, "max_periodic_full_exports": 0},
        )
    ):
        p_parser = prep_arg_parser()
        p_args = p_parser.parse_args(
            [
                "--mode",
                "train",
                "--gpu",
                "L4",
                "--hours",
                "4.0",
                "--ppo-config-json",
                json.dumps(dump_payload),
                "--allow-missing-canary-receipt",
                "--allow-unmerged-commit",
                "--output-dir",
                str(tmp_path / f"out_dump_none_{idx_dump}"),
                "--packet",
                str(tmp_path / f"pkt_dump_none_{idx_dump}"),
            ]
        )
        s_dump = prepare_packet_from_options(options_from_cli_args(p_args, p_parser))
        assert s_dump["wall_headroom"]["full_exports"] == "none"
        mf_dump = json.loads(Path(s_dump["manifest_path"]).read_text(encoding="utf-8"))
        cfg_dump = json.loads(mf_dump["env"]["V3_PPO_CONFIG_JSON"])
        assert cfg_dump["full_exports"] == "none"
        assert "max_periodic_full_exports" not in cfg_dump

    # Explicit `"full_exports"` inside `prepare --ppo-config-json` is refused with a clear message
    # stating that export mode is selected only on the command line.
    for fe_json_val in ("periodic", "final", "none"):
        p_parser = prep_arg_parser()
        p_args = p_parser.parse_args(
            [
                "--mode",
                "train",
                "--gpu",
                "L4",
                "--hours",
                "4.0",
                "--ppo-config-json",
                json.dumps({"num_envs": 128, "full_exports": fe_json_val}),
                "--allow-missing-canary-receipt",
                "--allow-unmerged-commit",
                "--output-dir",
                str(tmp_path / f"out_refuse_fe_{fe_json_val}"),
                "--packet",
                str(tmp_path / f"pkt_refuse_fe_{fe_json_val}"),
            ]
        )
        with pytest.raises(
            ValueError,
            match="selected only on the command line via --full-exports",
        ):
            options_from_cli_args(p_args, p_parser)

    # Conflicting `--full-exports` + `--max-periodic-full-exports` refused on `ppo` and `prepare`
    conflict_pairs: list[tuple[str, int]] = [
        ("none", 0),
        ("none", 3),
        ("final", 1),
        ("periodic", 0),
    ]
    for fe_mode, mp_val in conflict_pairs:
        with pytest.raises(ValueError, match="Conflicting FULL export options"):
            resolve_full_export_settings(fe_mode, mp_val)
        with pytest.raises(ValueError, match="Conflicting FULL export options"):
            resolve_effective_train_config(
                parser.parse_args(
                    [
                        "--output-dir",
                        "/tmp/o",
                        "--full-exports",
                        fe_mode,
                        "--max-periodic-full-exports",
                        str(mp_val),
                    ]
                ),
                environ={},
            )
        with pytest.raises(ValueError, match="Conflicting FULL export options"):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_conflict_{fe_mode}_{mp_val}",
                commit="HEAD",
                mode="train",
                gpu="L4",
                full_exports=fe_mode,  # type: ignore[arg-type]
                max_periodic_full_exports=mp_val,
                allow_missing_canary_receipt=True,
                output_dir=tmp_path / f"out_conflict_{fe_mode}_{mp_val}",
                allow_unmerged_commit=True,
            )


def test_recipe_override_allowlist_and_config_type_range_validation(tmp_path: Path) -> None:
    """Verify exact allow-list, inline/file parsing, and strict type & range validation."""
    expected_allowlist = (
        "reward_alive_length_coef",
        "reward_alive_const",
        "reward_length_delta_coef",
        "reward_kill_coef",
        "reward_death_penalty",
        "gamma",
        "gae_lambda",
        "entropy_coef",
        "lr",
        "ppo_epochs",
        "minibatch_size",
        "rollout_len",
        "clip_ratio",
    )
    assert RECIPE_OVERRIDE_KEYS == expected_allowlist
    assert format_recipe_overrides({}) == "none"

    all_recipe_payload = {
        "reward_alive_length_coef": 0.0,
        "reward_alive_const": -0.002,
        "reward_length_delta_coef": 0.25,
        "reward_kill_coef": 0.8,
        "reward_death_penalty": 2.0,
        "gamma": 0.995,
        "gae_lambda": 0.92,
        "entropy_coef": 0.005,
        "lr": 1.5e-4,
        "ppo_epochs": 3,
        "minibatch_size": 512,
        "rollout_len": 48,
        "clip_ratio": 0.15,
    }
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(all_recipe_payload), encoding="utf-8")

    parsed_inline = parse_recipe_overrides(json.dumps(all_recipe_payload))
    parsed_file = parse_recipe_overrides(str(recipe_file))
    assert parsed_inline == all_recipe_payload
    assert parsed_file == all_recipe_payload

    parser = ppo_arg_parser()
    eff = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--recipe-json", str(recipe_file)]),
        environ={},
    )
    for key, val in all_recipe_payload.items():
        assert getattr(eff.cfg, key) == val
    assert eff.recipe_overrides == all_recipe_payload

    # Unknown keys (including real non-allowlisted V3PPOConfig fields) refused
    for bad_key in ("unknown_key", "value_coef", "num_envs", "adam_eps", "net_size"):
        with pytest.raises(ValueError, match="Unsupported recipe override keys"):
            parse_recipe_overrides({bad_key: 1})

    # Non-object JSON and missing file refused
    for non_obj in ("[1, 2]", "42", "true", "null", str(tmp_path / "missing.json")):
        with pytest.raises(ValueError):
            parse_recipe_overrides(non_obj)

    # Wrong types refused through V3PPOConfig validation (satisfies both TypeError and ValueError)
    wrong_type_payloads: list[dict[str, object]] = [
        {"ppo_epochs": True},
        {"ppo_epochs": 2.5},
        {"ppo_epochs": "2"},
        {"rollout_len": False},
        {"rollout_len": 16.0},
        {"rollout_len": "16"},
        {"minibatch_size": True},
        {"minibatch_size": 64.5},
        {"minibatch_size": "64"},
        {"gamma": True},
        {"gamma": "0.99"},
        {"gamma": None},
        {"lr": False},
        {"lr": "3e-4"},
        {"reward_death_penalty": "1.0"},
        {"reward_alive_length_coef": [0.001]},
    ]
    for bad_type_ov in wrong_type_payloads:
        with pytest.raises(ConfigTypeError):
            resolve_effective_train_config(
                parser.parse_args(
                    ["--output-dir", "/tmp/o", "--recipe-json", json.dumps(bad_type_ov)]
                ),
                environ={},
            )

    # Out-of-range values refused through V3PPOConfig validation
    out_of_range_payloads: list[dict[str, object]] = [
        {"gamma": -0.01},
        {"gamma": 1.01},
        {"gae_lambda": -0.1},
        {"gae_lambda": 1.2},
        {"entropy_coef": -0.001},
        {"lr": 0.0},
        {"lr": -1.0e-4},
        {"ppo_epochs": 0},
        {"minibatch_size": 0},
        {"rollout_len": 0},
        {"clip_ratio": 0.0},
        {"clip_ratio": -0.2},
        {"reward_death_penalty": -0.5},
    ]
    for bad_range_ov in out_of_range_payloads:
        with pytest.raises(ValueError):
            resolve_effective_train_config(
                parser.parse_args(
                    ["--output-dir", "/tmp/o", "--recipe-json", json.dumps(bad_range_ov)]
                ),
                environ={},
            )

    # Setting recipe fields equal to code defaults (including preset minibatch_size) reports no overrides
    eff_default_eq_s = resolve_effective_train_config(
        parser.parse_args(
            [
                "--output-dir",
                "/tmp/o",
                "--net-size",
                "S",
                "--recipe-json",
                json.dumps({"gamma": 0.99, "minibatch_size": 2048, "rollout_len": 32}),
            ]
        ),
        environ={},
    )
    assert eff_default_eq_s.recipe_overrides == {}
    assert format_recipe_overrides(eff_default_eq_s.recipe_overrides) == "none"

    eff_default_eq_m = resolve_effective_train_config(
        parser.parse_args(
            [
                "--output-dir",
                "/tmp/o",
                "--net-size",
                "M",
                "--recipe-json",
                json.dumps({"minibatch_size": 1024}),
            ]
        ),
        environ={},
    )
    assert eff_default_eq_m.recipe_overrides == {}


def test_config_overlay_validates_recipe_change_history_and_minibatch_divisibility() -> None:
    """`recipe_change_history` passes through env/CLI overlays with strict validation; mb remainder checked."""
    from research.v3_ppo.config import (
        check_minibatch_divisibility,
        format_minibatch_remainder_warning,
    )

    parser = ppo_arg_parser()
    valid_hist = [
        {
            "update": 12,
            "changed_fields": {
                "gamma": {"old": 0.99, "new": 0.995},
                "minibatch_size": {"old": None, "new": 512},
            },
        }
    ]
    base_saved = V3PPOConfig(gamma=0.995, minibatch_size=512, recipe_change_history=valid_hist)
    saved_json = json.dumps(base_saved.to_dict())

    eff_env = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o"]),
        environ={"V3_PPO_CONFIG_JSON": saved_json},
    )
    assert list(eff_env.cfg.recipe_change_history) == valid_hist
    assert eff_env.recipe_overrides == {"gamma": 0.995, "minibatch_size": 512}

    eff_cli = resolve_effective_train_config(
        parser.parse_args(["--output-dir", "/tmp/o", "--config-json", saved_json]),
        environ={},
    )
    assert list(eff_cli.cfg.recipe_change_history) == valid_hist

    # Malformed `recipe_change_history` payloads in `V3_PPO_CONFIG_JSON` are rejected
    bad_histories: list[object] = [
        "not-a-list",
        [{"update": -1, "changed_fields": {"gamma": {"old": 0.99, "new": 0.995}}}],
        [{"update": True, "changed_fields": {"gamma": {"old": 0.99, "new": 0.995}}}],
        [{"update": 2, "extra": 1, "changed_fields": {"gamma": {"old": 0.99, "new": 0.995}}}],
        [{"update": 2, "changed_fields": {}}],
        [{"update": 2, "changed_fields": {"num_envs": {"old": 4, "new": 8}}}],
        [{"update": 2, "changed_fields": {"gamma": {"old": 0.99, "new": 0.995, "extra": 0}}}],
        [{"update": 2, "changed_fields": {"gamma": {"old": 0.99, "new": 1.5}}}],
        [{"update": 2, "changed_fields": {"ppo_epochs": {"old": 2, "new": "3"}}}],
        [{"update": 2, "changed_fields": {"gamma": {"old": 0.99, "new": 0.99}}}],
    ]
    for bad_h in bad_histories:
        with pytest.raises((ValueError, ConfigTypeError)):
            resolve_effective_train_config(
                parser.parse_args(["--output-dir", "/tmp/o"]),
                environ={"V3_PPO_CONFIG_JSON": json.dumps({"recipe_change_history": bad_h})},
            )

    # Minibatch divisibility: exact multiple -> None; <= 25% -> warning dict; > 25% -> ValueError
    assert (
        check_minibatch_divisibility(
            V3PPOConfig(num_envs=4, learner_slots=4, rollout_len=4, minibatch_size=16)
        )
        is None
    )
    exact_25 = check_minibatch_divisibility(
        V3PPOConfig(num_envs=4, learner_slots=4, rollout_len=4, minibatch_size=48)
    )
    assert exact_25 is not None
    assert exact_25["dropped_rows_per_epoch"] == 16
    assert exact_25["dropped_percent"] == 25.0
    warn_line = format_minibatch_remainder_warning(exact_25)
    assert "\n" not in warn_line
    assert "16/64 rows (25.00%)" in warn_line

    with pytest.raises(ValueError, match="exceeding the 25% cap"):
        check_minibatch_divisibility(
            V3PPOConfig(num_envs=4, learner_slots=4, rollout_len=4, minibatch_size=47)
        )
