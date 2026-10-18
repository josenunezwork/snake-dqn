"""Deterministic unit tests for the Q0 profiled-evaluation metric adapter."""

from __future__ import annotations

import numpy as np
import pytest

from research.q0q1_20261017.audit_episode_metrics import FrameObserver


def _snapshot(frame: int, *, valid=(True, True), alive=(True, False), done=(False, True)):
    return {
        "frame": frame,
        "seeds": (11, 22),
        "pre": {
            "logical_mass": np.array([650, 810]),
            "resolved_masks": np.array(
                [[True, True, False, True, False, False], [True, False, False, False, False, False]]
            ),
            "actions": np.array([3, 1]),
        },
        "post": {
            "alive": np.array(alive),
            "logical_mass": np.array([651, 811]),
            "transition_valid": np.array(valid),
            "food_ate": np.array([True, True]),
            "boosted": np.array([True, False]),
            "done": np.array(done),
            "death_cause": np.array([0, 4]),
            "reward": np.array([2.5, -3.0]),
            "kills": np.array([1, 0]),
        },
    }


def test_frame_observer_uses_exact_post_events_and_strata() -> None:
    observer = FrameObserver(horizon=3)
    observer(_snapshot(0))
    observer(_snapshot(1, valid=(True, False), alive=(True, False), done=(False, False)))
    result = observer.result()

    first, second = result["episodes"]["11"], result["episodes"]["22"]
    assert result["complete_horizon"] is False
    assert first["decision_frames"] == 2
    assert first["food_mass_collected"] == 2
    assert first["boost_eligible_decisions"] == 2
    assert first["boost_executed_frames"] == 2
    assert first["reward_on_surviving_frames"] == 5.0
    assert first["mass_sum"] == 1302.0
    assert first["by_length"]["short_lt700"]["food_event_frames"] == 2
    assert first["by_space"]["open_3plus_legal_actions"]["decision_frames"] == 2
    assert second["decision_frames"] == 1
    assert second["terminal_padding_frames"] == 1
    assert second["food_mass_collected"] == 1  # death-step event remains visible
    assert second["mass_sum"] == 0.0  # terminal death step has zero mass
    assert second["reward_on_death_frames"] == -3.0
    assert second["death_causes"] == {"4": 1}
    assert second["by_length"]["long_ge800"]["death_events"] == 1


def test_frame_observer_fails_closed_on_bad_sequence_or_shape() -> None:
    observer = FrameObserver(horizon=2)
    with pytest.raises(ValueError, match="non-contiguous"):
        observer(_snapshot(1))
    broken = _snapshot(0)
    broken["pre"]["resolved_masks"] = np.ones((2, 5), dtype=bool)
    with pytest.raises(ValueError, match="resolved_masks"):
        observer(broken)
