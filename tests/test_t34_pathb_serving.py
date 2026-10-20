"""Path B uses the actual live transition and preserves default-OFF operation."""

from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import pytest

from research.t3_rollout_labels_20261015.engine import verify_expand
from research.t34_pathb.live import evaluate, prepared_fingerprint, snapshot
from tests.test_web_watch_roster import (  # noqa: F401
    _clean_roster_env,
    _released_checkpoint_fixture,
)
from web.backend.pathb import eligible
from web.backend.session import GameSession


class Branch:
    def __init__(self, deaths: dict[int, int], events: list, action: int | None = None) -> None:
        self.deaths, self.events, self.action = deaths, events, action
        self.frame = 0
        self.alive = True

    def clone(self) -> "Branch":
        return Branch(self.deaths, self.events)

    def step(self, first_action: int | None = None) -> None:
        if first_action is not None:
            self.action = first_action
        self.frame += 1
        self.events.append((self.action, self.frame))
        self.alive = self.frame < self.deaths.get(self.action, 999)


def test_verify_bare_then_expand_lockstep_and_q_ties() -> None:
    events: list = []
    world = Branch({0: 1}, events)
    result = verify_expand(world, 0, [9, 4, 4, 0, 0, 0], [1, 1, 1, 0, 0, 0], 3)
    assert result["proposed_action"] == 1
    assert events == [(0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]


def test_keep_verified_bare_and_longest_surviving_fallback() -> None:
    result = verify_expand(Branch({}, []), 2, [9] * 6, [1] * 6, 2)
    assert result["proposed_action"] == 2
    assert set(result["outcomes"]) == {2}
    world = Branch({0: 1, 1: 2, 2: 2}, [])
    assert (
        verify_expand(world, 0, [9, 4, 4, 0, 0, 0], [1, 1, 1, 0, 0, 0], 3)["proposed_action"] == 1
    )
    assert (
        verify_expand(Branch({0: 1, 1: 1}, []), 1, [4] * 6, [1, 1, 0, 0, 0, 0], 3)[
            "proposed_action"
        ]
        == 1
    )


def test_off_creates_no_hook_or_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAKE_SERVE_PATHB", raising=False)
    session = GameSession()
    assert session.pathb is None
    assert not hasattr(session.game.snakes[0], "action_transform")
    assert "pathb" not in session.control_state()
    session.step()
    session.close(True)


def test_live_clone_exact_next_decision_and_rng_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAKE_SERVE_PATHB", raising=False)
    session = GameSession()
    session.step()
    payload = snapshot(session.game, depth=2)
    py_state, np_state = random.getstate(), deepcopy(np.random.get_state())
    result = evaluate(payload)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    captured = {}

    def observed(snake, others, q, mask, bare):
        captured.update(
            prepared_fingerprint=prepared_fingerprint(session.game),
            bare_action=bare,
            mask=mask.cpu().numpy().astype(bool).tolist(),
        )
        return bare

    session.game.snakes[0].action_transform = observed
    session.step()
    assert captured
    for key in captured:
        assert captured[key] == result[key], key
    session.close(True)


def test_scope_is_exact_mixed_watch6_and_fixed_slot_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAKE_SERVE_PATHB", raising=False)
    session = GameSession()
    assert eligible(session)
    session.hero_id = 3  # Inspector focus must not redirect assistance.
    assert eligible(session)
    for attr, value in [("mode", "play"), ("watch_roster", "all_b8")]:
        original = getattr(session, attr)
        setattr(session, attr, value)
        assert not eligible(session)
        setattr(session, attr, original)
    session.game.snakes.append(session.game.snakes[-1])
    assert not eligible(session)
    session.close(True)
