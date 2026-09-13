"""Boundary tests for the non-promotion one-hero Watch evaluator."""

from __future__ import annotations

import pytest

from src.core.runtime_contract import EffectiveWorldConfig, RuntimeModeContract
from src.core.world_runtime import WorldRuntimeSpec
from src.evaluation.protocol import solo_watch_diagnostic
from src.simd_env import eval_engine


def _solo_profile():
    world = EffectiveWorldConfig(
        width=240,
        height=180,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=1,
        max_frames=5000,
        initial_food=1,
        max_food=2,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        max_length=100,
        starvation_max_frames=500,
        max_capacity=400,
        kill_scale=0.3,
        death_value=-3.0,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 100.0},
    )
    return solo_watch_diagnostic(world)


def test_solo_adapter_binds_empty_roster_runtime_and_diagnostic_authority(monkeypatch) -> None:
    profile = _solo_profile()
    runtime = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)
    observed = {}

    def fake_simd(hero, opponents, frames, seeds, **kwargs):
        observed.update(
            hero=hero,
            opponents=opponents,
            frames=frames,
            seeds=seeds,
            profile=kwargs["profile"],
            mix_id=kwargs["mix_id"],
            runtime=kwargs["world_runtime_spec"],
        )
        return [{"seed": 17, "mass_integral": 1.0, "survival_fraction": 1.0, "probes": {}}]

    monkeypatch.setattr(eval_engine, "run_simd_eval", fake_simd)
    rows = eval_engine.run_solo_watch_diagnostic(
        ("scripted", "random_safe"),
        5000,
        [17],
        profile=profile,
        world_runtime_spec=runtime,
    )

    assert observed["opponents"] == ()
    assert observed["profile"] == profile
    assert observed["runtime"] == runtime
    assert observed["mix_id"] == "solo-no-opponents-v1"
    assert rows[0]["opponent_roster"] == []
    assert rows[0]["evaluation_authority"] == "diagnostic-only"
    assert rows[0]["promotion_eligible"] is False
    assert rows[0]["strict_authority"] is False


def test_solo_adapter_rejects_source_exact_or_wrong_profile_before_simulation(monkeypatch) -> None:
    profile = _solo_profile()
    monkeypatch.setattr(
        eval_engine, "run_simd_eval", lambda *args, **kwargs: pytest.fail("ran sim")
    )

    with pytest.raises(ValueError, match="fresh-reset horizon-bound"):
        eval_engine.run_solo_watch_diagnostic(
            ("scripted", "random_safe"),
            5000,
            [17],
            profile=profile,
            world_runtime_spec=WorldRuntimeSpec.source_exact(profile),
        )


def test_direct_solo_profile_path_rejects_a_noncanonical_mix_before_simulation(monkeypatch) -> None:
    profile = _solo_profile()
    monkeypatch.setattr(
        eval_engine, "_TerminalHeroBatchSim", lambda *args, **kwargs: pytest.fail("ran")
    )

    with pytest.raises(ValueError, match="fixed empty-roster mix"):
        eval_engine.run_simd_eval(
            ("scripted", "random_safe"),
            [],
            profile.scored_horizon,
            [17],
            profile=profile,
            mix_id="any-other-mix",
        )


def test_native_six_snake_checkpoint_cannot_cross_into_solo_profile(tmp_path) -> None:
    """The native v3 world-digest validator keeps S6 and S1 populations apart."""
    import torch

    from src.model.obs_spec import RASTER31V3_CONTRACT

    profile = _solo_profile()
    source_world = EffectiveWorldConfig(**{**profile.world.__dict__, "num_snakes": 6})
    source_runtime = RuntimeModeContract(
        mode="pqn_train",
        training=True,
        respawn=False,
        hero_terminal=True,
        population_floor=True,
        reset_strategy="per_env_autoreset_v1",
    )
    checkpoint = tmp_path / "six-snake-native.pth"
    torch.save(
        {
            "obs_contract_digest": RASTER31V3_CONTRACT.digest,
            "effective_world": dict(source_world.__dict__),
            "effective_world_digest": source_world.digest,
            "runtime_contract": dict(source_runtime.__dict__),
            "runtime_contract_digest": source_runtime.digest,
            "model_head": {"algorithm": "pqn", "head": "dueling_q", "critic_outputs": 6},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="world does not match"):
        eval_engine.validate_v3_checkpoint_for_profile(str(checkpoint), profile)


def _native_solo_checkpoint_state(*, completion_mode: str | None) -> dict:
    """Materialize the combined trainer's native v2 solo checkpoint metadata."""
    from src.model.obs_spec import RASTER31V3
    from src.training.pqn_trainer import PQNConfig, PQNTrainer

    config = {
        "num_envs": 1,
        "num_snakes": 1,
        "rollout_len": 1,
        "max_frames": 5000,
        "game_width": 240,
        "game_height": 180,
        "initial_food": 1,
        "max_food": 2,
        "max_length": 100,
        "starvation_max": 500,
        "max_capacity": 400,
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "flip_augment": False,
        "mechanics_version": 2,
        "reward_version": 2,
        "episode_reset_mode": "per_env_autoreset_v1",
        "episode_seed_mode": "derived_env_episode_v1",
        "episode_completion_mode": completion_mode,
        "hero_frac": 1.0,
        "pool_capacity": 0,
        "pool_admission_mode": "disabled_v1",
        "rollout_policy_mode": "snapshot_pool",
        "decision_phase_mode": "watch_pre_move_v1",
        "source_revision": "solo-native-validator-test",
    }
    if completion_mode is None:
        config.pop("episode_completion_mode")
        config["pool_admission_mode"] = "scheduled_v1"
    trainer = PQNTrainer(PQNConfig(**config))
    try:
        return trainer.checkpoint_state()
    finally:
        trainer.close()


def test_native_v2_solo_lifecycle_checkpoint_is_accepted_by_solo_profile(tmp_path) -> None:
    """A completed-on-death checkpoint must prove its native v2 lifecycle."""
    import torch

    from src.training.pqn_lifecycle import SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1

    checkpoint = tmp_path / "native-solo-v2.pth"
    torch.save(
        _native_solo_checkpoint_state(completion_mode=SOLE_SNAKE_DEATH_OR_FRAME_CAP_V1),
        checkpoint,
    )
    eval_engine.validate_v3_checkpoint_for_profile(str(checkpoint), _solo_profile())


@pytest.mark.parametrize("completion_mode", [None, "population_floor_or_frame_cap_v1"])
def test_default_or_missing_solo_completion_lifecycle_is_rejected(
    tmp_path, completion_mode: str | None
) -> None:
    """An S1 checkpoint cannot claim death-complete evaluation without v2 proof."""
    import torch

    checkpoint = tmp_path / f"invalid-solo-{completion_mode}.pth"
    torch.save(_native_solo_checkpoint_state(completion_mode=completion_mode), checkpoint)

    with pytest.raises(ValueError, match="solo|lifecycle|completion"):
        eval_engine.validate_v3_checkpoint_for_profile(str(checkpoint), _solo_profile())
