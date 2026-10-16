"""Focused contracts for the T4f macro-boost bank builder."""

from types import SimpleNamespace

import numpy as np

from research.t4f_return_20261016 import macro_collect as macro


def test_macro_repeats_relative_boost_through_new_selector_phases(monkeypatch) -> None:
    """Only frame one consumes the captured selector; later frames re-enter it."""
    from research.t3_rollout_labels_20261015 import bank_engine

    branch = SimpleNamespace(
        sim=SimpleNamespace(alive=np.array([[True]]), length=np.array([[900]]))
    )
    snapshot = {
        "legal": np.array([0, 3]),
        "length": 900,
        "branch": SimpleNamespace(clone=lambda: branch),
    }
    calls: list[str] = []

    def first_step(world, action) -> None:
        assert world is branch and action == 3
        calls.append("prepared")

    def forced(world, action) -> None:
        assert world is branch and action == 3
        calls.append("selector")

    def ordinary(world, **_kwargs) -> None:
        calls.append("b8")
        world.sim.alive[0, 0] = False

    monkeypatch.setattr(bank_engine, "step_prepared", first_step)
    monkeypatch.setattr(bank_engine, "choose_and_step", ordinary)
    monkeypatch.setattr(macro, "_force_next_selector_action", forced)
    result = macro.rollout_macro(snapshot, 3, 3)

    assert calls == ["prepared", "selector", "selector", "b8"]
    assert result["T"] == 4
    assert result["died"] is True


def test_grouped_states_requires_consistent_branch_identity() -> None:
    archive = {
        "source_decision": np.array([1, 1]),
        "source_seed": np.array([7, 7]),
        "action": np.array([0, 3]),
        "state_packed": np.array([[1], [2]], np.uint8),
        "state_scalars": np.array([[1.0], [1.0]], np.float32),
    }
    try:
        macro.grouped_states(archive)
    except ValueError as error:
        assert "inconsistent state identity" in str(error)
    else:  # pragma: no cover
        raise AssertionError("inconsistent branch observations were accepted")
