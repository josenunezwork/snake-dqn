"""Small runtime oracles for the explicit E1 evaluation profile."""

from dataclasses import replace

import pytest

from src.evaluation.protocol import promotion_v2_watch_rect
from src.scripts.tournament_eval import _evaluation_world_from_config, rollout
from src.simd_env.eval_engine import run_simd_eval

pytestmark = pytest.mark.usefixtures("setup_config")


def _short_watch_profile():
    """Keep the deployment identity while making a deterministic unit-scale run."""
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
