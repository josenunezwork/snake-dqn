"""Unit tests for `research/v3_ppo`: buffer indexing, GAE, action masking, net, and rewards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import research.v3_ppo.ppo as ppo_mod
from research.game_v3.config import GameV3Config
from research.game_v3.obs import LearnerIndex, build_obs
from research.game_v3.sim import GameV3Sim, StepInfo
from research.v3_ppo.buffer import Uint8RolloutBuffer, compute_gae
from research.v3_ppo.config import (
    BYTES_PER_LEARNER_STEP,
    NET_PRESETS,
    OBS_RASTER_BYTES_PER_STEP,
    V3PPOConfig,
    estimate_training_memory,
    get_net_preset,
)
from research.v3_ppo.eval import compute_step_reward
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
from research.v3_ppo.prepare import build_arg_parser as prepare_arg_parser
from research.v3_ppo.prepare import prepare_v3_ppo_packet
from research.v3_ppo.ruler import _parse_nets_csv
from research.v3_ppo.ruler import build_arg_parser as ruler_arg_parser


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


def test_uint8_rollout_buffer_indexing_and_single_obs_per_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify zero-copy uint8 slice views, scattered LearnerIndex, and 1 build_obs/tick."""
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

    # Spy on `build_obs` inside `ppo` to prove strictly 1 `build_obs` per tick in steady state
    real_build_obs = ppo_mod.build_obs
    call_counter = {"count": 0}

    def _counted_build_obs(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_counter["count"] += 1
        return real_build_obs(*args, **kwargs)

    monkeypatch.setattr(ppo_mod, "build_obs", _counted_build_obs)

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


def test_ppo_update_single_sync_uniform_minibatch_and_fused_adamw_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify uniform minibatch sizing (drop/pad), <=1 host sync in `ppo_update`, & AdamW helper."""
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

    # Verify AdamW builder & state transfer helper
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


def test_fast_learner_ops_rounding_equivalence_and_compile_execution() -> None:
    """Verify `fast_learner_ops` float32 rounding equivalence & `--compile` trunk execution."""
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

    # 5. `--compile` (`compile_model=True`) compiles the shared trunk and invokes the compiled
    #    callable on BOTH rollout (`collect_rollout` -> `act` + bootstrap `forward`, no_grad)
    #    AND PPO learner updates (`ppo_update` -> `evaluate_actions`, grad enabled).
    tr_compiled = V3PPOTrainer(
        V3PPOConfig(**base_kw, compile_model=True, fast_learner_ops=True),
        device="cpu",
    )
    assert tr_compiled.raw_model.is_compiled is True
    orig_compiled_fn = tr_compiled.raw_model._compiled_encode_trunk
    assert orig_compiled_fn is not None and callable(orig_compiled_fn)
    assert set(tr_compiled.raw_model.state_dict().keys()) == set(tr_fast.raw_model.state_dict())

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

    # 6. `--compile` and `--no-fast-learner-ops` parse cleanly on all 3 CLIs
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


def test_net_presets_registry_and_unknown_size_rejected(tmp_path: Path) -> None:
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
