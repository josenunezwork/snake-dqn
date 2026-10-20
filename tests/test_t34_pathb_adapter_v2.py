"""Real adapter construction and Watch identity checks for Path B v2."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from research.t34_pathb.adapter_v2 import B8_SHA, open_terminal, open_watch
from research.t34_shadow.identity import _physical, _value_digest

pytestmark = pytest.mark.skipif(
    not Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
    ).is_file(),
    reason="released B8 gameplay artifacts are not installed",
)


def test_terminal_adapter_runs_real_bare_b8_world_without_controller() -> None:
    row = {
        "seed": 2026101934,
        "mix": "scripted",
        "specs": [["scripted", "greedy_food"]] * 5,
    }
    session = open_terminal(row, 24, "bare")
    try:
        assert session.controller is None
        assert session.world.mix == "scripted"
        assert session.identity().checkpoint == B8_SHA
        session.step()
        assert session.frame == 1
        assert session.controller_state() == {"active": False, "label": "bare B8 s3"}
    finally:
        session.close()


def test_watch_adapter_constructs_actual_mixed_frp3_v8_and_preserves_pathb_off() -> None:
    """A real Watch tick must pass the adapter's physical/carry/v8 identity bridge."""

    session = open_watch({"seed": 2026101934}, 24, "bare")
    try:
        live = session.live
        assert live is not None
        assert getattr(live, "pathb", None) is None
        assert getattr(live, "watch_roster", None) == "mixed"
        assert len(live.game.snakes) == 6
        assert live.game.snakes[0].id == 0
        for opponent in live.game.snakes[1:]:
            assert opponent.safety_veto is not None
            assert opponent.safety_veto.lam == pytest.approx(8.0)
        initial_live = _physical(
            live.game,
            session.world,
            session.specs,
            session._live_lives,
            live=True,
            rng=random.getstate(),
        )
        initial_grid = _physical(
            None,
            session.world,
            session.specs,
            session._grid_lives,
            live=False,
        )
        assert _value_digest(initial_live) == _value_digest(initial_grid)
        session.step()
        assert session.frame == 1
        assert live.last_error is None
    finally:
        session.close()
