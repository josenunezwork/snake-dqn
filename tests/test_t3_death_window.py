"""Unit contracts for T3's post-death collection worker.

These tests are intentionally pure: numerical world validation is performed by
the five completed-key receipt path, not by pytest replaying costly episodes.
"""

from __future__ import annotations

import numpy as np

from research.t3_rollout_labels_20261015 import death_window as worker


def test_strict_death_window_excludes_exactly_300_decisions() -> None:
    assert worker.inside_death_window(701, 1000)
    assert worker.inside_death_window(1000, 1000)
    assert not worker.inside_death_window(700, 1000)
    assert not worker.inside_death_window(1001, 1000)


def test_counter_reservoir_is_bounded_and_deterministic() -> None:
    def collect(seed: int) -> tuple[int, ...]:
        rng = np.random.default_rng(seed)
        reservoir: list[dict[str, int]] = []
        seen = 0
        for decision in range(100):
            seen = worker._reservoir_add(reservoir, {"decision": decision}, seen, rng)
        assert seen == 100
        return tuple(item["decision"] for item in reservoir)

    first, second = collect(9), collect(9)
    assert len(first) == worker.COUNTER_RESERVOIR
    assert first == second
    assert first != tuple(range(worker.COUNTER_RESERVOIR))


def test_alternative_outcomes_adds_trajectory_proven_b8_fatal_outcome(
    monkeypatch,
) -> None:
    snapshot = {"branch": object(), "legal": [0, 1, 2], "action": 1, "decision": 44}
    called: list[tuple[object, list[int], int, int]] = []

    def fake_rollout(branch, legal, depth, decision):
        called.append((branch, list(legal), depth, decision))
        return {
            0: {"survives": True, "frames_executed": depth},
            2: {"survives": False, "frames_executed": 6},
        }

    monkeypatch.setattr(worker, "rollout_all", fake_rollout)

    outcomes = worker._alternative_outcomes(snapshot)

    assert called == [(snapshot["branch"], [0, 2], worker.DEPTH, 44)]
    assert outcomes[1] == {"survives": False, "frames_executed": 0}


def test_all_alternatives_dying_is_not_a_decisive_label() -> None:
    outcomes = {
        0: {"survives": False, "frames_executed": 8},
        1: {"survives": False, "frames_executed": 9},
    }

    decisive, surviving = worker.classify(outcomes, 0)

    assert not decisive
    assert not surviving.any()


def test_single_legal_action_never_invokes_an_empty_rollout(monkeypatch) -> None:
    snapshot = {"branch": object(), "legal": [4], "action": 4, "decision": 5}
    monkeypatch.setattr(
        worker, "rollout_all", lambda *args: (_ for _ in ()).throw(AssertionError())
    )

    outcomes = worker._alternative_outcomes(snapshot)

    assert outcomes == {4: {"survives": False, "frames_executed": 0}}
