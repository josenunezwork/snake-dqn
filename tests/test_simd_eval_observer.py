"""Contract tests for the optional profiled SIMD frame observer."""

from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.protocol import promotion_v2_watch_rect
from src.scripts.tournament_eval import _evaluation_world_from_config
from src.simd_env.eval_engine import run_simd_eval

pytestmark = pytest.mark.usefixtures("setup_config")


def _short_watch_profile():
    """Return a real profile with a unit-test horizon and a small roster."""
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
        scored_horizon=4,
        observation_progress_horizon=17,
    )


def _run_args(profile):
    return (
        ("scripted", "greedy_food"),
        [("scripted", "random_safe")] * (profile.world.num_snakes - 1),
    )


def test_observer_is_detached_and_preserves_profiled_records():
    """Mutating callback-owned copies cannot alter actions, state, or metrics."""
    profile = _short_watch_profile()
    hero, opponents = _run_args(profile)
    expected = run_simd_eval(hero, opponents, 4, [17, 23], profile=profile)
    seen = []

    def mutating_observer(snapshot):
        seen.append(
            (
                snapshot["frame"],
                tuple(snapshot["seeds"]),
                snapshot["pre"]["actions"].copy(),
                snapshot["pre"]["resolved_masks"].copy(),
                snapshot["post"]["transition_valid"].copy(),
            )
        )
        for section in ("pre", "post"):
            for value in snapshot[section].values():
                if isinstance(value, np.ndarray):
                    assert not value.flags.writeable
                    value.setflags(write=True)
                    value[...] = 0
        snapshot["pre"]["food_cells"] = ()
        snapshot["seeds"] = ()

    observed = run_simd_eval(
        hero, opponents, 4, [17, 23], profile=profile, frame_observer=mutating_observer
    )

    assert observed == expected
    assert [frame for frame, *_ in seen] == [0, 1, 2, 3]
    assert all(seeds == (17, 23) for _, seeds, *_ in seen)
    assert all(actions.shape == (2,) for *_, actions, _, _ in seen)
    assert all(masks.shape == (2, 6) for *_, masks, _ in seen)
    assert all(valid.shape == (2,) for *_, valid in seen)


def test_observer_reports_valid_death_then_zero_event_padding(monkeypatch):
    """The death step is observable; subsequent terminal-hero padding is explicit."""
    import src.simd_env.eval_engine as eval_engine
    from src.simd_env.batch_sim import DEATH_WALL

    profile = _short_watch_profile()
    hero, opponents = _run_args(profile)
    original_step = eval_engine._TerminalHeroBatchSim.step
    calls = 0

    def terminal_first_step(self, actions, active_env_mask=None):
        nonlocal calls
        original_step(self, actions, active_env_mask)
        if calls == 0:
            self.alive[:, 0] = False
            self._last_done[:, 0] = True
            self._last_death_cause[:, 0] = DEATH_WALL
        calls += 1

    monkeypatch.setattr(eval_engine._TerminalHeroBatchSim, "step", terminal_first_step)
    seen = []
    run_simd_eval(
        hero,
        opponents,
        4,
        [29],
        profile=profile,
        frame_observer=seen.append,
    )

    assert len(seen) == 4
    first_pre = seen[0]["pre"]
    first_post = seen[0]["post"]
    assert bool(first_pre["alive"][0])
    assert bool(first_pre["resolved_masks"][0, first_pre["actions"][0]])
    assert bool(first_post["transition_valid"][0])
    assert not bool(first_post["alive"][0])
    assert bool(first_post["done"][0])
    assert int(first_post["death_cause"][0]) == DEATH_WALL
    for snapshot in seen[1:]:
        pre = snapshot["pre"]
        post = snapshot["post"]
        assert not bool(pre["alive"][0])
        assert int(pre["resolved_masks"][0].sum()) == 0
        assert not bool(post["transition_valid"][0])
        assert not bool(post["alive"][0])
        assert not bool(post["food_ate"][0])
        assert not bool(post["boosted"][0])
        assert not bool(post["done"][0])
        assert int(post["death_cause"][0]) == 0
        assert int(post["kills"][0]) == 0


def test_observer_errors_abort_the_evaluation():
    """A capture failure cannot be mistaken for a completed evaluation."""
    profile = _short_watch_profile()
    hero, opponents = _run_args(profile)

    def fail(_snapshot):
        raise RuntimeError("observer write failed")

    with pytest.raises(RuntimeError, match="observer write failed"):
        run_simd_eval(hero, opponents, 4, [31], profile=profile, frame_observer=fail)


def test_observer_requires_an_explicit_profile():
    """The seam is intentionally unavailable to the legacy diagnostic path."""
    with pytest.raises(ValueError, match="frame_observer requires"):
        run_simd_eval(
            ("scripted", "greedy_food"),
            [],
            1,
            [37],
            frame_observer=lambda _snapshot: None,
        )
