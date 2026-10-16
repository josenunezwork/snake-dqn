"""Focused, no-gameplay contracts for T5 restored-bank labels."""

import numpy as np

from research.t5_separability_20261016 import restore
from research.t5_separability_20261016.restore import state_labels


def test_state_labels_keep_d300_survivor_as_explicit_truncation() -> None:
    rows = {
        "source_decision": np.array([7, 7, 7, 7]),
        "action": np.array([0, 1, 3, 4]),
        "died": np.array([True, True, False, True]),
        "T": np.array([12, 300, 300, 300]),
        "state_scalars": np.zeros((4, 60), np.float32),
        "state_packed": np.zeros((4, 8), np.uint8),
    }
    state = state_labels(rows)[7]
    assert state["died_by_action"].tolist() == [True, True, False, False, True, False]
    assert state["truncated_by_action"].tolist() == [False, False, False, True, False, False]
    assert state["boost_only_escape"] is True


def test_state_labels_rejects_nonidentical_observation_rows() -> None:
    rows = {
        "source_decision": np.array([7, 7]),
        "action": np.array([0, 3]),
        "died": np.array([True, False]),
        "T": np.array([2, 300]),
        "state_scalars": np.array([[0.0] * 60, [1.0] * 60], np.float32),
        "state_packed": np.zeros((2, 8), np.uint8),
    }
    try:
        state_labels(rows)
    except ValueError as error:
        assert "non-identical state rows" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected invalid state identity to be rejected")


def test_capture_computes_features_before_selector_transition(monkeypatch) -> None:
    """The mutable simulator advances after the selector callback returns."""
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import ego_raster, t5_features

    class Runtime:
        def prepare(self, sim):
            sim.prepared = True

    class World:
        runtime = Runtime()
        sim = type("Sim", (), {"prepared": False, "frame": 0})()

    def choose_and_step(world, *, capture_obs):
        world.runtime.prepare(world.sim)
        world.sim.frame = 1

    monkeypatch.setattr(bank_engine, "choose_and_step", choose_and_step)
    monkeypatch.setattr(
        ego_raster,
        "build_ego_raster",
        lambda sim, *_args, **_kwargs: {"scalars": np.array([[sim.frame]], dtype=np.float32)},
    )
    monkeypatch.setattr(
        t5_features, "compute_slot_features", lambda sim, _e, _s: np.full(24, sim.frame, np.float32)
    )
    obs, features, _elapsed = restore._capture_state(World(), 0, wanted=True)
    assert obs is not None and features is not None
    assert obs["scalars"].item() == 0
    assert np.all(features == 0)
