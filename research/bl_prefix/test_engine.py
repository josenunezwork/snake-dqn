"""Focused no-gameplay checks for the E6/E7 root transport contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from research.bl_prefix import engine


def _record() -> engine.PrefixRecord:
    return engine.PrefixRecord(
        {
            "local": np.zeros((7, 31, 31), np.uint8),
            "global": np.zeros((4, 37, 37), np.uint8),
            "scalars": np.zeros(60, np.float32),
        },
        0,
        np.array([True, False, False, False, False, False]),
        np.array([True, False, False, False, False, False]),
    )


def test_relocate_requires_bound_bytes_even_when_source_is_absent(tmp_path: Path) -> None:
    shipped = tmp_path / "b8.pth"
    shipped.write_bytes(b"bound-b8")
    root = engine.Root(
        {"specs": []},
        tuple(_record() for _ in range(128)),
        11,
        "scripted",
        99,
        "/absent/B8.pth",
        {"/absent/B8.pth": hashlib.sha256(b"bound-b8").hexdigest()},
    )
    moved = engine.relocate_root(root, {"/absent/B8.pth": shipped})
    assert moved.b8_path == str(shipped)
    shipped.write_bytes(b"changed")
    with pytest.raises(engine.IncompleteRoot, match="identity"):
        engine.relocate_root(root, {"/absent/B8.pth": shipped})


def test_load_root_requires_all_export_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "root.npz"
    path.write_bytes(b"not a root")
    with pytest.raises(ValueError, match="missing or unsafe export"):
        engine.load_root(path)


def test_capture_rejects_short_actual_history_without_touching_sim() -> None:
    world = object()
    with pytest.raises(engine.IncompleteRoot, match="128"):
        engine.capture_root(world, [_record()] * 127)  # type: ignore[arg-type]


def test_step_keeps_raw_legal_distinct_from_advisory_and_forced_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sim:
        S = 2
        alive = np.array([[True, True]])
        frame = np.array([12])
        length = np.array([[20, 5]])

        def get_resolved_action_mask(self):
            return np.ones((1, 2, 6), bool)

        def get_legal_action_mask(self):
            return np.array([[[True, True, True, False, False, False], [True] * 6]], bool)

        def get_advisory_action_mask(self):
            return np.array([[[False, True, False, False, False, False], [True] * 6]], bool)

        def get_alive(self):
            return self.alive

        def step_with_policy(self, select):
            self.selected = select(self)
            self.frame[0] += 1

    class Runtime:
        def prepare(self, sim):
            pass

        def observe_step(self, sim):
            pass

    class Opponent:
        def actions(self, mask, sim, slots):
            return np.array([0])

    class B8:
        mask_mode = "advisory"

        def q_values(self, observation):
            return np.array([99.0, 2.0, 1.0, 0.0, 0.0, 0.0])

        def choose(self, observation, mask):
            self.mask = mask.copy()
            return int(np.where(mask, self.q_values(observation), -np.inf).argmax())

    monkeypatch.setattr(engine, "_observation", lambda sim: _record().observation)
    b8 = B8()
    world = engine.EngineWorld(Sim(), Runtime(), [Opponent()], 1, "scripted", [], "fixture", b8)
    result = engine.step(world, b8, forced_action=3)
    assert result.legal.tolist() == [True, True, True, False, False, False]
    assert result.advisory.tolist() == [False, True, False, False, False, False]
    assert result.action == 1 and result.forced_fallback is True
    assert b8.mask.tolist() == [False, True, False, False, False, False]


def test_parity_refuses_incomplete_prefix_before_reconstructing_source() -> None:
    root = engine.Root(
        {"specs": []}, tuple(_record() for _ in range(127)), 1, "scripted", 1, "missing", {}
    )
    with pytest.raises(engine.IncompleteRoot, match="128"):
        engine.verify_root_parity(root)
