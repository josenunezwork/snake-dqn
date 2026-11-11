"""Multi-seed CPU PPO self-play learning proof for `research/v3_ppo` on redesigned M1 eval."""

from __future__ import annotations

import pytest

from research.game_v3.config import GameV3Config
from research.v3_ppo.config import V3PPOConfig
from research.v3_ppo.eval import evaluate_policy, evaluate_scripted_baseline
from research.v3_ppo.ppo import V3PPOTrainer

FRESH_EVAL_SEED = 27182818
TRAIN_SEEDS = (1001, 2002)


@pytest.mark.slow
def test_selfplay_improves_reproducibly_over_untrained_across_two_seeds() -> None:
    """Prove reproducible multi-seed self-play improvement over untrained on M1 evaluation.

    Uses a non-tiny rollout batch (`E=6, N=6, T=32` => `1,152` agent-steps/update, `mb=192`)
    with default PPO optimization hyperparameters (`lr=3e-4, gamma=0.99, entropy_coef=0.01`),
    a matured measurement window (`burnin=120, window=600` where most `random_safe` learner
    lives complete inside the window), two independent training seeds (`1001`, `2002`), and
    an untouched evaluation seed (`27182818`).
    """
    game_cfg = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        spawn_wall_margin=10.0,
        ambient_food_density=0.008,
        ambient_pellet_value=3,
        respawn_delay=8,
    )

    def _make_cfg(train_seed: int) -> V3PPOConfig:
        return V3PPOConfig(
            game_config=game_cfg,
            num_envs=6,
            learner_slots=None,  # pure 6/6 self-play (36 learners, up to 1,152 rows/update)
            net_size="S",
            amp=False,
            use_channels_last=False,
            rollout_len=32,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3.0e-4,
            ppo_epochs=2,
            minibatch_size=192,
            entropy_coef=0.01,
            total_updates=80,
            eval_every_minutes=0.0,
            eval_every_updates=0,
            eval_envs=6,
            eval_burnin_ticks=120,
            eval_ticks=600,
            eval_learner_slots=2,
            eval_seed=FRESH_EVAL_SEED,
            eval_greedy=False,
            seed=train_seed,
        )

    ref_cfg = _make_cfg(TRAIN_SEEDS[0])
    rs_comb = evaluate_scripted_baseline("random_safe", ref_cfg, device="cpu")["combined"]
    fs_comb = evaluate_scripted_baseline("food_seeking", ref_cfg, device="cpu")["combined"]
    print(
        f"\n[M2 Eval Seed={FRESH_EVAL_SEED}] " f"random_safe={rs_comb} | food_seeking={fs_comb}",
        flush=True,
    )
    # Confirm the evaluation window is long enough that most `random_safe` lives complete inside it.
    assert rs_comb["completed_lives"] > rs_comb["censored_lives"]

    for train_seed in TRAIN_SEEDS:
        cfg = _make_cfg(train_seed)
        trainer = V3PPOTrainer(cfg, device="cpu")
        init_comb = evaluate_policy(trainer.raw_model, cfg, device="cpu")["combined"]

        for _ in range(cfg.total_updates):
            trainer.train_step()

        assert trainer.updates_completed == cfg.total_updates
        tr_comb = trainer.run_evaluation()["combined"]
        print(
            f"[M2 Seed={train_seed}] init={init_comb} -> trained_u{cfg.total_updates}={tr_comb}",
            flush=True,
        )

        # Assert clear, reproducible improvement over the untrained network across both seeds:
        tr_ret = tr_comb["mean_completed_episode_return"]
        in_ret = init_comb["mean_completed_episode_return"]
        assert (
            tr_ret > in_ret + 1.0
        ), f"seed={train_seed}: expected trained return ({tr_ret}) > untrained ({in_ret}) + 1.0"
        tr_len = tr_comb["time_averaged_length"]
        in_len = init_comb["time_averaged_length"]
        assert tr_len > in_len + 3.0, (
            f"seed={train_seed}: expected trained time_averaged_length ({tr_len}) > "
            f"untrained ({in_len}) + 3.0"
        )
        tr_dlen = tr_comb["mean_death_length"]
        in_dlen = init_comb["mean_death_length"]
        assert tr_dlen > in_dlen + 3.0, (
            f"seed={train_seed}: expected trained mean_death_length ({tr_dlen}) > "
            f"untrained ({in_dlen}) + 3.0"
        )
        tr_life = tr_comb["mean_lifetime_ticks"]
        in_life = init_comb["mean_lifetime_ticks"]
        assert tr_life > in_life * 1.5, (
            f"seed={train_seed}: expected trained mean_lifetime_ticks ({tr_life}) > "
            f"1.5 * untrained ({in_life})"
        )
        tr_d1k = tr_comb["deaths_per_1k_agent_steps"]
        in_d1k = init_comb["deaths_per_1k_agent_steps"]
        assert (
            tr_d1k < in_d1k * 0.60
        ), f"seed={train_seed}: expected trained deaths/1k ({tr_d1k}) < 0.60 * untrained ({in_d1k})"
