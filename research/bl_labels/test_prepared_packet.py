"""Focused unit contracts for the F0 prepared-decision packet."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_served_search import b2_engine


class _Tensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def numpy(self) -> np.ndarray:
        return self.value


class _Hero:
    def __init__(self) -> None:
        self.net = object()
        self.calls = 0

    def actions(self, _masks, _sim, _rows) -> np.ndarray:
        self.calls += 1
        return np.array([4], dtype=np.int64)


class _Sim:
    def __init__(self) -> None:
        self._policy_phase_prepared = True
        self.alive = np.array([[True] + [False] * 5], dtype=bool)
        self.length = np.full((1, 6), 9, dtype=np.int64)
        self.frame = np.array([12], dtype=np.int64)

    def get_advisory_action_mask(self) -> np.ndarray:
        return np.array([[[False, True, True, False, False, False]]], dtype=bool)

    def get_legal_action_mask(self) -> np.ndarray:
        return np.array([[[True, True, True, False, False, False]]], dtype=bool)

    def step(self, _actions: np.ndarray) -> None:
        self._policy_phase_prepared = False

    def get_step_events(self) -> dict[str, np.ndarray]:
        return {"death_cause": np.zeros((1, 6), dtype=np.int64)}


class _Frozen:
    def actions(self, _sim, _actions) -> None:
        return None

    def observe_step(self, _sim) -> None:
        return None


def _driver() -> b2_engine.WatchDriver:
    driver = b2_engine.WatchDriver.__new__(b2_engine.WatchDriver)
    driver.sim = _Sim()
    driver.hero = _Hero()
    driver.frozen = _Frozen()
    driver.stats = SimpleNamespace()
    driver.eligible_opponent_rows = 0
    driver.decision = 7
    driver.life = 2
    driver.awaiting_respawn = False
    driver._prepare_generation = 11
    return driver


def test_packet_reuses_one_forward_and_forced_or_packet_step_skips_hero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"raster": 0, "forward": 0}

    def raster(_sim, _config, *, backend, rows):
        assert backend == "numba" and rows.tolist() == [[0, 0]]
        calls["raster"] += 1
        return {
            "local": np.ones((1, 2, 2), dtype=np.uint8),
            "global": np.ones((1, 1, 2), dtype=np.uint8),
            "scalars": np.ones((1, 3), dtype=np.float32),
        }

    def forward(_net, _obs, _mode):
        calls["forward"] += 1
        return _Tensor(np.array([[1, 7, 7, 0, 0, 0]], dtype=np.float32))

    monkeypatch.setattr("src.simd_env.ego2s_policy.ego_config_for", lambda _net: object())
    monkeypatch.setattr("src.simd_env.ego2s_policy.forward_q", forward)
    monkeypatch.setattr("src.simd_env.ego_raster.build_ego_raster", raster)
    monkeypatch.setattr(b2_engine, "configure", lambda *_args: nullcontext())
    monkeypatch.setattr(b2_engine, "branch_scope", lambda *_args: nullcontext())
    driver = _driver()

    packet = driver.prepared_decision()
    assert calls == {"raster": 1, "forward": 1}
    assert packet.bare == 1  # equal-Q tie keeps the first advisory action
    assert packet.q_bytes == np.array([1, 7, 7, 0, 0, 0], dtype=np.float32).tobytes()
    with pytest.raises(ValueError):
        packet.observation["local"].setflags(write=True)

    packet_event = driver.step(packet=packet)
    assert packet_event["hero_action"] == 1
    assert driver.hero.calls == 0

    driver.sim._policy_phase_prepared = True
    with pytest.raises(RuntimeError, match="stale B2 prepared decision"):
        driver.step(packet=packet)

    forced_event = driver.step(forced=2)
    assert forced_event["hero_action"] == 2
    assert driver.hero.calls == 0


def test_packet_rejects_prepare_and_sim_staleness_even_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.simd_env.ego2s_policy.ego_config_for", lambda _net: object())
    monkeypatch.setattr(
        "src.simd_env.ego2s_policy.forward_q",
        lambda *_args: _Tensor(np.zeros((1, 6), dtype=np.float32)),
    )
    monkeypatch.setattr(
        "src.simd_env.ego_raster.build_ego_raster",
        lambda *_args, **_kwargs: {
            "local": np.zeros((1, 1), dtype=np.uint8),
            "global": np.zeros((1, 1), dtype=np.uint8),
            "scalars": np.zeros((1, 1), dtype=np.float32),
        },
    )
    driver = _driver()
    packet = driver.prepared_decision()
    driver._prepare_generation += 1
    with pytest.raises(RuntimeError, match="stale B2 prepared decision"):
        driver.step(forced=0, packet=packet)

    driver._prepare_generation -= 1
    driver.sim = _Sim()  # restore uses a different simulator object.
    with pytest.raises(RuntimeError, match="stale B2 prepared decision"):
        driver.step(packet=packet)


def test_prepare_invalidates_packet_and_advances_respawn_life(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver()
    packet = b2_engine.PreparedDecision(
        driver.sim, driver.decision, driver._prepare_generation, {}, b"" * 24, (), 0
    )
    driver.awaiting_respawn = True
    monkeypatch.setattr(
        "research.bl_ppo.collector._prepare_sim",
        lambda sim: setattr(sim, "_policy_phase_prepared", True),
    )

    driver.prepare()

    assert driver.life == 3
    with pytest.raises(RuntimeError, match="stale B2 prepared decision"):
        driver.step(forced=0, packet=packet)


def test_empty_advisory_uses_normal_fallback_and_shadow_mismatch_fails() -> None:
    from research.bl_labels import factory

    q = np.array([1, 5, 5, 100, 0, 0], dtype=np.float32)
    _mask, bare = factory.ceiling._legal(q, np.zeros(6, dtype=bool))
    assert bare == 1

    sim = object()
    packet = b2_engine.PreparedDecision(sim, 4, 2, {}, q.tobytes(), (0, 1, 2), 1)
    bad = SimpleNamespace(shadow_action_and_q=lambda _packet: (2, q.tobytes()))
    with pytest.raises(factory.FactoryInvalid, match="prepared decision shadow mismatch"):
        factory._assert_prepared_shadow(bad, packet)

    selected = next(
        decision
        for decision in range(256)
        if factory._prepared_shadow_selected("f0-production-d1024", 0, decision)
    )
    assert factory._prepared_shadow_selected("f0-production-d1024", 0, selected)


def test_f1_legacy_driver_observation_seam_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1's frozen ruler still asks the factory helper for a driver raster."""
    from research.bl_labels import factory

    seen = []
    driver = SimpleNamespace(sim=object())
    monkeypatch.setattr(
        "src.simd_env.ego_raster.build_ego_raster",
        lambda sim, config, *, backend, rows: (
            seen.append((sim, config.version, backend, rows.tolist()))
            or {
                "local": np.array([[1]], dtype=np.uint8),
                "global": np.array([[2]], dtype=np.uint8),
                "scalars": np.array([[3]], dtype=np.float32),
            }
        ),
    )

    observed = factory._raw_observation(driver)

    assert seen == [(driver.sim, "bv8", "numba", [[0, 0]])]
    assert observed["local"].tolist() == [1]
