"""Unit tests for `research/v3_ppo`: buffer indexing, GAE, action masking, net, and rewards."""

from __future__ import annotations

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
    OBS_RASTER_BYTES_PER_STEP,
    V3PPOConfig,
)
from research.v3_ppo.eval import compute_step_reward
from research.v3_ppo.net import (
    MASK_FILL_VALUE,
    EgoMultiScaleNet,
    EgoMultiScaleNetConfig,
    build_ego_multiscale_net,
)
from research.v3_ppo.ppo import V3PPOTrainer


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


def test_gae_death_vs_truncation_and_respawn_gaps() -> None:
    """Verify death zeros bootstrap, truncation bootstraps V(s'), and dead ticks stay isolated."""
    gamma = 0.99
    lam = 0.95
    # 3 learner columns across T = 4 ticks:
    # Col 0: normal step t=0 -> DIES at t=1 -> inactive respawn wait at t=2 -> respawns at t=3
    # Col 1: normal step t=0 -> TRUNCATED at t=1 -> continues at t=2 -> TRUNCATED at t=3 (boundary)
    # Col 2: survives uninterrupted t=0..2 -> TRUNCATED at t=3 (rollout boundary)
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

    # Col 0 checks:
    # t=3: truncated -> delta_3 = 0.2 + 0.99 * 3.0 - 1.0 = 2.17
    assert pytest.approx(float(adv[3, 0].item()), abs=1e-5) == 0.2 + gamma * 3.0 - 1.0
    # t=2: inactive -> adv=0, ret=0
    assert float(adv[2, 0].item()) == 0.0
    assert float(ret[2, 0].item()) == 0.0
    # t=1: DIED -> next_val = 0.0 AND trace cut: adv_1 = -1.0 + 0 - 8.0 = -9.0; ret_1 = -1.0
    assert pytest.approx(float(adv[1, 0].item()), abs=1e-5) == -9.0
    assert pytest.approx(float(ret[1, 0].item()), abs=1e-5) == -1.0
    # t=0: survives into t=1 -> delta_0 = 1.0 + 0.99 * 8.0 - 10.0 = -1.08;
    # adv_0 = delta_0 + gamma*lam*(-9.0)
    expected_adv_0_0 = (1.0 + gamma * 8.0 - 10.0) + gamma * lam * (-9.0)
    assert pytest.approx(float(adv[0, 0].item()), abs=1e-5) == expected_adv_0_0

    # Col 1 checks (mid-rollout truncation at t=1 bootstraps values[2, 1]=7.0 AND cuts GAE trace!):
    # t=3: truncated -> adv_3 = 4.0 + 0.99 * 20.0 - 8.0 = 15.8
    adv_3_1 = 4.0 + gamma * 20.0 - 8.0
    # t=2: normal -> delta_2 = 1.5 + 0.99 * 8.0 - 7.0 = 2.42; adv_2 = delta_2 + gamma*lam*adv_3_1
    adv_2_1 = (1.5 + gamma * 8.0 - 7.0) + gamma * lam * adv_3_1
    # t=1: TRUNCATED -> bootstraps values[2, 1]=7.0, trace cut (does NOT include gamma*lam*adv_2_1!)
    adv_1_1 = 3.0 + gamma * 7.0 - 6.0
    adv_0_1 = (2.0 + gamma * 6.0 - 5.0) + gamma * lam * adv_1_1
    assert pytest.approx(float(adv[3, 1].item()), abs=1e-5) == adv_3_1
    assert pytest.approx(float(adv[2, 1].item()), abs=1e-5) == adv_2_1
    assert pytest.approx(float(adv[1, 1].item()), abs=1e-5) == adv_1_1
    assert pytest.approx(float(adv[0, 1].item()), abs=1e-5) == adv_0_1

    # Verify rollout-boundary truncation in V3PPOTrainer keeps surviving snakes alive in GameV3Sim
    gcfg = GameV3Config(arena_radius=80.0, num_snakes=6, ambient_food_density=0.02)
    cfg = V3PPOConfig(game_config=gcfg, num_envs=2, rollout_len=4, eval_every_updates=0)
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.collect_rollout()
    assert int(trainer.buffer.truncated[-1].sum().item()) > 0
    assert int(trainer.sim.alive.sum().item()) > 0  # surviving snakes remain in the world!


def test_step_reward_coefficients_and_dying_snake_only_penalty() -> None:
    """Verify alive length, length delta without death-zeroing leak, kills, and death penalty."""
    cfg = V3PPOConfig(
        reward_alive_length_coef=0.002,
        reward_alive_const=0.05,
        reward_length_delta_coef=0.25,
        reward_kill_coef=0.75,
        reward_death_penalty=1.5,
    )
    # 1 world, 4 snakes:
    # Slot 0: survives, grows from 10 -> 13 (+3 delta), gets 1 kill (killed slot 1)
    # Slot 1: was length 20, DIES on this tick (info.length becomes 0 in GameV3Sim!)
    # Slot 2: survives, shrinks from 18 -> 17 (-1 boost shed), 0 kills
    # Slot 3: already dead waiting to respawn (`acted = False`)
    acted = torch.tensor([[True, True, True, False]])
    info = StepInfo(
        alive=torch.tensor([[True, False, True, False]]),
        life_start=torch.tensor([[False, False, False, False]]),
        died=torch.tensor([[False, True, False, False]]),
        truncated=torch.tensor([[False, False, False, False]]),
        in_respawn_delay=torch.tensor([[False, False, False, True]]),
        length=torch.tensor([[13, 0, 17, 0]], dtype=torch.int32),
        prev_length=torch.tensor([[10, 20, 18, 0]], dtype=torch.int32),
        mass_eaten=torch.tensor([[3, 0, 0, 0]], dtype=torch.int32),
        killer_id=torch.tensor([[-1, 0, -1, -1]], dtype=torch.int32),
        kill_credit=torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
        life_age=torch.tensor([[12, 0, 9, 0]], dtype=torch.int32),
    )

    rew = compute_step_reward(info, acted, cfg)
    assert rew.shape == (1, 4)

    # Slot 0 (survivor + killer): 0.05 + 0.002*13 + 0.25*(13-10) + 0.75*1 = 1.576
    assert pytest.approx(float(rew[0, 0].item()), abs=1e-6) == 0.05 + 0.026 + 0.75 + 0.75
    # Slot 1 (dying snake ONLY pays -1.5; NO -20 * 0.25 leak from length zeroing on death!)
    assert pytest.approx(float(rew[0, 1].item()), abs=1e-6) == -1.5
    # Slot 2 (survivor with -1 boost shed): 0.05 + 0.002*17 + 0.25*(-1) = -0.166
    assert pytest.approx(float(rew[0, 2].item()), abs=1e-6) == 0.05 + 0.034 - 0.25
    # Slot 3 (inactive dead slot): exact 0.0
    assert float(rew[0, 3].item()) == 0.0
