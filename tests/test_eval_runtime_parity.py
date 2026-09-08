"""Small runtime oracles for the explicit E1 evaluation profile."""

from dataclasses import replace

import pytest

from src.evaluation.protocol import promotion_v2_watch_rect
from src.scripts.tournament_eval import _evaluation_world_from_config, rollout
from src.simd_env.eval_engine import run_simd_eval

pytestmark = pytest.mark.usefixtures("setup_config")


def _short_watch_profile():
    """Keep the deployment identity while making a deterministic unit-scale run."""
    from src.core.game_config import AppConfig, GameSettings, initialize_config

    initialize_config(
        AppConfig(
            game=GameSettings(
                width=400,
                height=300,
                num_snakes=3,
                initial_food=30,
                max_food=35,
                mechanics_version=2,
                frame_rate=1,
            )
        )
    )
    return replace(
        promotion_v2_watch_rect(_evaluation_world_from_config()),
        scored_horizon=3,
        observation_progress_horizon=17,
    )


def test_actual_live_and_simd_profiled_runs_emit_named_denominators():
    """Both adapters execute their real update loop and retain profile identity."""
    from src.core.game_config import AppConfig, GameSettings, initialize_config

    initialize_config(
        AppConfig(
            game=GameSettings(
                width=400,
                height=300,
                num_snakes=3,
                initial_food=30,
                max_food=35,
                mechanics_version=2,
                frame_rate=1,
            )
        )
    )
    profile = _short_watch_profile()
    hero = ("scripted", "greedy_food")
    opponents = [("scripted", "random_safe")] * (profile.world.num_snakes - 1)

    live = rollout(hero, opponents, frames=3, seed=17, profile=profile)
    simd = run_simd_eval(hero, opponents, frames=3, seeds=[17], profile=profile)[0]

    for record in (live, simd):
        assert record["evaluation_profile_digest"] == profile.digest
        assert record["evaluation_profile"]["scored_horizon"] == 3
        assert record["evaluation_profile"]["observation_progress_horizon"] == 17
        assert record["denominators"]["scored_frames"] == 3
        assert record["denominators"]["decision_frames"] == 3
        assert record["probes"]["kill_opportunity_count"] is None
        assert record["probes"]["entrapment_event"] is None
        assert 0.0 <= record["survival_fraction"] <= 1.0


def test_run_simd_eval_zero_pads_terminal_hero_and_excludes_padding_from_decisions(monkeypatch):
    """The public SIMD wrapper scores the death transition and pads later frames."""
    import src.simd_env.eval_engine as eval_engine
    from src.simd_env.batch_sim import DEATH_WALL

    profile = _short_watch_profile()
    original_step = eval_engine._TerminalHeroBatchSim.step
    calls = 0

    def terminal_first_step(self, actions, active_env_mask=None):
        nonlocal calls
        original_step(self, actions, active_env_mask)
        if calls == 0:
            self.alive[:, 0] = False
            self._last_done[:, 0] = True
            self._last_death_cause[:, 0] = DEATH_WALL
            self._boosted_this_step[:, 0] = True
        calls += 1

    monkeypatch.setattr(eval_engine._TerminalHeroBatchSim, "step", terminal_first_step)
    record = run_simd_eval(
        ("scripted", "greedy_food"),
        [("scripted", "random_safe")] * (profile.world.num_snakes - 1),
        frames=3,
        seeds=[23],
        profile=profile,
    )[0]

    assert record["denominators"]["scored_frames"] == 3
    assert record["denominators"]["decision_frames"] == 1
    assert record["deaths"] == 1.0
    assert record["survival_fraction"] == 0.0
    assert record["probes"]["boost_frame_fraction"] == 1.0


def test_live_rollout_rejects_profile_for_another_world():
    profile = _short_watch_profile()
    wrong_world = replace(profile.world, width=profile.world.width + profile.world.segment_size)
    with pytest.raises(ValueError, match="does not match"):
        rollout(
            ("scripted", "greedy_food"),
            [("scripted", "random_safe")] * (profile.world.num_snakes - 1),
            frames=3,
            seed=29,
            profile=replace(profile, world=wrong_world),
        )


def test_profiled_roster_identity_is_derived_and_rejects_tampering():
    profile = _short_watch_profile()
    hero = ("scripted", "greedy_food")
    opponents = [("scripted", "random_safe")] * (profile.world.num_snakes - 1)
    live = rollout(hero, opponents, frames=3, seed=41, profile=profile)
    simd = run_simd_eval(hero, opponents, frames=3, seeds=[41], profile=profile)[0]

    assert live["world_identity"] == simd["world_identity"]
    assert live["world_identity"]["mix_id"] == "unspecified"
    forged = {**live["world_identity"], "roster_id": "forged"}
    with pytest.raises(ValueError, match="world_identity"):
        rollout(hero, opponents, frames=3, seed=41, profile=profile, world_identity=forged)
    with pytest.raises(ValueError, match="world_identities"):
        run_simd_eval(
            hero,
            opponents,
            frames=3,
            seeds=[41],
            profile=profile,
            world_identities={41: forged},
        )


def test_profile_capacity_boundary_fails_closed_in_both_public_wrappers(monkeypatch):
    """Reaching the declared capacity on the final scored frame never returns a score."""
    import src.scripts.tournament_eval as tournament_eval
    import src.simd_env.eval_engine as eval_engine

    profile = replace(
        _short_watch_profile(), world=replace(_short_watch_profile().world, max_capacity=2)
    )
    hero = ("scripted", "greedy_food")
    opponents = [("scripted", "random_safe")] * (profile.world.num_snakes - 1)

    original_update = tournament_eval.create_training_game_state

    def live_at_cap(*args, **kwargs):
        game_state = original_update(*args, **kwargs)
        original_step = game_state.update

        def step(*step_args, **step_kwargs):
            original_step(*step_args, **step_kwargs)
            snake = game_state.snakes[0]
            snake.length = 2

        game_state.update = step
        return game_state

    original_sim_step = eval_engine._TerminalHeroBatchSim.step

    def sim_at_cap(self, actions, active_env_mask=None):
        original_sim_step(self, actions, active_env_mask)
        self.length[:, 0] = 2

    monkeypatch.setattr(tournament_eval, "create_training_game_state", live_at_cap)
    monkeypatch.setattr(tournament_eval, "_evaluation_world_from_config", lambda: profile.world)
    monkeypatch.setattr(eval_engine._TerminalHeroBatchSim, "step", sim_at_cap)
    with pytest.raises(RuntimeError, match="max_capacity"):
        rollout(hero, opponents, frames=3, seed=43, profile=profile)
    with pytest.raises(RuntimeError, match="max_capacity"):
        run_simd_eval(hero, opponents, frames=3, seeds=[43], profile=profile)


def test_actual_serialized_v3_checkpoint_attaches_to_live_profile(tmp_path):
    """A producer-format v3 checkpoint reaches RasterServingPolicy without a mock."""
    import torch

    from src.core.game_config import AppConfig, GameSettings, initialize_config
    from src.core.runtime_contract import EffectiveWorldConfig
    from src.core.seeding import initialize_run_seed
    from src.evaluation.protocol import promotion_v2_watch_rect
    from src.game.game_state_factory import create_training_game_state
    from src.scripts.tournament_eval import _attach_agent
    from src.training.pqn_trainer import PQNConfig, PQNTrainer
    from web.backend.raster_policy import RasterServingPolicy

    seed = initialize_run_seed(9017)
    config = PQNConfig(
        num_envs=1,
        num_snakes=6,
        rollout_len=2,
        obs_spec="raster31v3",
        recipe="corrected-v3",
        flip_augment=False,
        max_frames=1234,
        starvation_max=77,
        max_length=66,
        source_revision="e1-runtime-test",
        seed=seed.effective_seed,
        requested_device="cpu",
        effective_device="cpu",
    )
    trainer = PQNTrainer(config, device=torch.device("cpu"))
    checkpoint = tmp_path / "actual-v3.pth"
    trainer.save_checkpoint(str(checkpoint))
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    profile = replace(
        promotion_v2_watch_rect(EffectiveWorldConfig(**blob["effective_world"])),
        scored_horizon=1,
        observation_progress_horizon=1234,
    )
    world = profile.world
    initialize_config(
        AppConfig(
            game=GameSettings(
                width=world.width,
                height=world.height,
                segment_size=world.segment_size,
                wall_thickness=world.wall_thickness,
                num_snakes=world.num_snakes,
                initial_food=world.initial_food,
                max_food=world.max_food,
                mechanics_version=world.mechanics_version,
                frame_rate=world.frame_rate,
                max_length=world.max_length,
                min_boost_length=world.min_boost_length,
                boost_length_cost_frames=world.boost_length_cost_frames,
            )
        )
    )
    game = create_training_game_state(eval_mode=False)
    try:
        cache = {}
        _attach_agent(game, 0, ("checkpoint", str(checkpoint)), 59, cache, profile)
        assert isinstance(game.snakes[0].policy, RasterServingPolicy)
    finally:
        game.full_cleanup()
