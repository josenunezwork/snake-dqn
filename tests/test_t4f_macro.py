"""Focused contracts for the T4f macro-boost bank builder."""

from types import SimpleNamespace

import numpy as np

from research.t4f_return_20261016 import macro_collect as macro


def test_boost_left_macro_does_not_repeat_three_turns(monkeypatch) -> None:
    """Only frame one consumes boost-left; later frames use fresh B8 directions."""
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

    def b8_direction_boost(world) -> int:
        assert world is branch
        calls.append("b8-direction-boost")
        return 4

    def ordinary(world, **_kwargs) -> None:
        calls.append("b8")
        world.sim.alive[0, 0] = False

    monkeypatch.setattr(bank_engine, "step_prepared", first_step)
    monkeypatch.setattr(bank_engine, "choose_and_step", ordinary)
    monkeypatch.setattr(macro, "_force_b8_direction_boost", b8_direction_boost)
    result = macro.rollout_macro(snapshot, 3, 3)

    assert calls == ["prepared", "b8-direction-boost", "b8-direction-boost", "b8"]
    assert result["T"] == 4
    assert result["died"] is True


class _MacroSim:
    """Minimal simulator seam that records action and relative-heading changes."""

    E = 1
    S = 1

    def __init__(self) -> None:
        self.alive = np.array([[True]])
        self.length = np.array([[900]])
        self.actions: list[int] = []
        self.headings: list[int] = []
        self.heading = 0

    def get_resolved_action_mask(self) -> np.ndarray:
        return np.ones((1, 1, 6), dtype=bool)

    def step_with_policy(self, selector) -> None:
        action = int(selector(self)[0, 0])
        self.actions.append(action)
        self.heading += {-1: -1, 0: 0, 1: 1}[action % 3 - 1]
        self.headings.append(self.heading)


class _MacroWorld:
    def __init__(self) -> None:
        self.sim = _MacroSim()
        self.runtime = SimpleNamespace(prepare=lambda _sim: None, observe_step=lambda _sim: None)
        self.hero = object()
        self.opponents = object()

    def clone(self):
        return self


def test_rollout_macro_uses_real_direction_boost_selector(monkeypatch) -> None:
    """The real macro helper turns left once, then follows fresh straight B8 directions."""
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import eval_engine

    branch = _MacroWorld()
    snapshot = {
        "legal": np.array([0, 3]),
        "length": 900,
        "branch": branch,
    }

    def first_step(world, action) -> None:
        assert world is branch and action == 3
        world.sim.actions.append(action)
        world.sim.heading -= 1
        world.sim.headings.append(world.sim.heading)

    def ordinary(world, **_kwargs) -> None:
        world.sim.alive[0, 0] = False

    def dispatch(_sim, _masks, actions, _heroes, _opponents) -> None:
        # B8's greedy direction is straight on both refreshed selector frames.
        actions[0, 0] = 1

    monkeypatch.setattr(bank_engine, "step_prepared", first_step)
    monkeypatch.setattr(bank_engine, "choose_and_step", ordinary)
    monkeypatch.setattr(eval_engine, "_dispatch_actions", dispatch)

    result = macro.rollout_macro(snapshot, 3, 3)

    assert result["died"] is True
    assert branch.sim.actions == [3, 4, 4]
    assert branch.sim.headings == [-1, -1, -1]
    assert [branch.sim.headings[0]] + [
        current - previous
        for previous, current in zip(branch.sim.headings, branch.sim.headings[1:])
    ] == [-1, 0, 0]


class _SelectorSim:
    """Small Watch-phase stand-in that records the action selected per frame."""

    E = 1
    S = 1

    def __init__(self, masks: list[np.ndarray]) -> None:
        self._masks = iter(masks)
        self.selected: list[int] = []

    def get_resolved_action_mask(self) -> np.ndarray:
        return next(self._masks)

    def step_with_policy(self, selector) -> None:
        self.selected.append(int(selector(self)[0, 0]))


def _selector_world(masks: list[np.ndarray]) -> SimpleNamespace:
    prepared = _SelectorSim(masks)
    runtime = SimpleNamespace(prepare=lambda _sim: None, observe_step=lambda _sim: None)
    return SimpleNamespace(
        sim=prepared,
        runtime=runtime,
        hero=object(),
        opponents=object(),
    )


def test_macro_honors_changed_b8_directions_with_boost(monkeypatch) -> None:
    """Each macro frame gets B8's current direction rather than repeating left."""
    from src.simd_env import eval_engine

    masks = [np.array([[[True, True, True, True, True, True]]], dtype=bool)] * 2
    world = _selector_world(masks)
    b8_actions = iter([1, 2])

    def dispatch(_sim, _masks, actions, _heroes, _opponents) -> None:
        actions[0, 0] = next(b8_actions)

    monkeypatch.setattr(eval_engine, "_dispatch_actions", dispatch)

    assert macro._force_b8_direction_boost(world) == 4
    assert macro._force_b8_direction_boost(world) == 5
    assert world.sim.selected == [4, 5]


def test_macro_falls_back_to_b8_normal_direction_when_boost_is_illegal(monkeypatch) -> None:
    """A refreshed mask can forbid the forced boost variant without changing direction."""
    from src.simd_env import eval_engine

    # B8 chooses right (2), but its boost-right variant (5) is unavailable.
    masks = [np.array([[[True, True, True, True, True, False]]], dtype=bool)]
    world = _selector_world(masks)

    def dispatch(_sim, _masks, actions, _heroes, _opponents) -> None:
        actions[0, 0] = 2

    monkeypatch.setattr(eval_engine, "_dispatch_actions", dispatch)

    assert macro._force_b8_direction_boost(world) == 2
    assert world.sim.selected == [2]


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
