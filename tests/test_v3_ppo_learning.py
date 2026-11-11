"""End-to-end tiny-arena CPU PPO self-play learning proof for `research/v3_ppo`."""

from __future__ import annotations

import pytest

from research.game_v3.config import GameV3Config
from research.v3_ppo.config import V3PPOConfig
from research.v3_ppo.eval import evaluate_policy, evaluate_scripted_baseline
from research.v3_ppo.ppo import V3PPOTrainer


@pytest.mark.slow
def test_tiny_arena_selfplay_beats_random_safe_on_fixed_eval() -> None:
    """Train 200 pure self-play updates on a small R=80, N=6 arena and beat `random_safe`."""
    game_cfg = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        spawn_wall_margin=10.0,
        ambient_food_density=0.008,
        ambient_pellet_value=3,
        respawn_delay=8,
    )
    cfg = V3PPOConfig(
        game_config=game_cfg,
        num_envs=2,
        learner_slots=None,  # pure 6/6 self-play (12 learners)
        net_size="S",
        amp=False,
        use_channels_last=False,
        rollout_len=16,
        gamma=0.98,
        gae_lambda=0.95,
        lr=4.0e-4,
        ppo_epochs=2,
        minibatch_size=64,
        entropy_coef=0.01,
        total_updates=200,
        eval_every_updates=0,
        eval_envs=8,
        eval_ticks=240,
        eval_learner_slots=2,
        eval_seed=20261112,
        eval_greedy=True,
        seed=20261111,
    )

    random_safe_eval = evaluate_scripted_baseline("random_safe", cfg, device="cpu")
    rs_comb = random_safe_eval["combined"]

    trainer = V3PPOTrainer(cfg, device="cpu")
    init_eval = evaluate_policy(trainer.raw_model, cfg, device="cpu")
    init_comb = init_eval["combined"]

    # Sanity-check that the untrained policy starts well below `random_safe`.
    assert init_comb["mean_episode_return"] < rs_comb["mean_episode_return"] - 1.5
    assert init_comb["mean_final_length"] < rs_comb["mean_final_length"] - 5.0

    for _ in range(cfg.total_updates):
        trainer.train_step()

    assert trainer.updates_completed == 200
    trained_eval = trainer.run_evaluation()
    tr_comb = trained_eval["combined"]

    assert tr_comb["mean_episode_return"] >= rs_comb["mean_episode_return"] + 0.50, (
        f"Expected trained mean_episode_return ({tr_comb['mean_episode_return']}) clearly above "
        f"random_safe ({rs_comb['mean_episode_return']})"
    )
    assert tr_comb["mean_final_length"] >= rs_comb["mean_final_length"] + 2.0, (
        f"Expected trained mean_final_length ({tr_comb['mean_final_length']}) clearly above "
        f"random_safe ({rs_comb['mean_final_length']})"
    )
    assert tr_comb["p95_final_length"] > rs_comb["p95_final_length"]
    assert tr_comb["deaths_per_1k_agent_steps"] < rs_comb["deaths_per_1k_agent_steps"]
