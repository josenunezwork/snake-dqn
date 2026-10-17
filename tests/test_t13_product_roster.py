"""Focused contract tests for the descriptive T13 Watch harness."""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from research.t13_product_roster_20261017 import episodes


@contextlib.contextmanager
def _t12_watch12_loader():
    """Apply T12's constructor-local Watch12 patch to its original harness."""
    from web.backend import session as session_module

    original = session_module.load_config

    def loader(path):
        config = original(path)
        return replace(config, game=replace(config.game, num_snakes=12))

    with patch("web.backend.session.load_config", side_effect=loader):
        yield


@pytest.fixture(autouse=True)
def restore_global_config(monkeypatch):
    """Keep the historical Watch12 construction posture inside these tests."""
    from src.core import game_config

    monkeypatch.setattr(game_config, "_current_config", game_config._current_config)
    # T13's evidence harness predates the released mixed product default. Its
    # fixed all-B8 construction remains a historical comparison fixture.
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "all_b8")


def test_selection_restores_environment(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")
    with episodes._selection("frp3-s12", veto=True):
        assert episodes.os.environ["SNAKE_SERVE_CHECKPOINT"] == "frp3-s12"
        assert episodes.os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v8"
    assert episodes.os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert episodes.os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_counter_schema_preserves_raw_integer_histograms():
    row = episodes._counter()
    row["length_histogram"].update([1, 3, 3])
    row["death_causes"].update(["body"])
    assert episodes._plain(row)["length_histogram"] == {"1": 1, "3": 2}
    assert episodes._plain(row)["death_causes"] == {"body": 1}


def test_invalid_roster_is_rejected_before_session_construction():
    try:
        episodes.run_watch("not-a-roster", 1, horizon=1)
    except ValueError as exc:
        assert "roster must be" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid roster was accepted")


@pytest.mark.skipif(
    not Path("saved_snakes/b8_s3_t7_20261016.pth").is_file(),
    reason="checkout lacks the locally pinned B8 serving artifact",
)
def test_mixed_watch_keeps_default_b8_hero_and_wraps_only_harness_opponents(monkeypatch):
    """The default session stays unmodified while T13's explicit mix is applied."""
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    record = episodes.run_watch("mixed", 2624486211, horizon=1)
    assert record["served"]["registry"]["requested"] is None
    assert record["served"]["registry"]["name"] == "b8-s3"
    assert record["served"]["veto"]["wrapped_snake_ids"] == []
    assert record["mixed_policy_identities"]["opponent_ids"] == list(range(1, 12))
    assert record["hero"]["decisions"] == 1
    assert record["arena"]["decisions"] == 12
    assert episodes.os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"


@pytest.mark.skipif(
    not Path("saved_snakes/b8_s3_t7_20261016.pth").is_file(),
    reason="checkout lacks the locally pinned B8 serving artifact",
)
def test_all_b8_trace_matches_original_t12_watch12_at_h100():
    """Instrumentation must not change the served B8 decision trace or deaths."""
    from research.t12_b8_serving_20261017 import episodes as t12_episodes

    seed = 2624486211
    with contextlib.redirect_stderr(io.StringIO()), _t12_watch12_loader():
        baseline = t12_episodes.run_watch(seed, horizon=100)
    with contextlib.redirect_stderr(io.StringIO()):
        observed = episodes.run_watch("all_b8", seed, horizon=100)
    assert baseline["hero"]["decisions"] == observed["hero"]["decisions"]
    assert baseline["hero"]["deaths"] == observed["hero"]["deaths"]
    assert baseline["hero"]["decision_length_histogram"] == {
        int(length): count for length, count in observed["hero"]["length_histogram"].items()
    }


def test_decision_observer_samples_pre_move_length_and_restores_after_exception():
    class FakeSnake:
        id = 7
        length = 5
        is_boosting = False
        _pre_collision_action = None

        def update(self):
            self.length -= 1
            self.is_boosting = True
            self._pre_collision_action = 3
            raise RuntimeError("simulated collision path failure")

    snake = FakeSnake()
    original = snake.update
    counters = {snake.id: episodes._counter()}
    with pytest.raises(RuntimeError, match="simulated"):
        with episodes._DecisionObserver([snake], counters):
            snake.update()
    assert counters[snake.id]["decisions"] == 1
    assert counters[snake.id]["length_histogram"] == {5: 1}
    # The call raised before its return, so boost accounting is deliberately not
    # reported; wrapper restoration remains mandatory for subsequent frames.
    assert counters[snake.id]["boost_decisions"] == 0
    assert snake.update.__func__ is original.__func__
