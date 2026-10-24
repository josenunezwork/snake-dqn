"""Offline, mocked behavioral tests for PLAN_v3 B2 trajectory collection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from research.bl_served_search import b2_engine as engine


def _stats(*, stock: int = 0, fallback: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        as_dict=lambda: {
            "kernel_rows": 7,
            "stock_rows": stock,
            "fallback_rows": fallback,
            "noncandidate_rows": 0,
        }
    )


class BranchDriver:
    """Minimal restored branch whose calls make continuation identity observable."""

    def __init__(self, dies_at: int | None) -> None:
        self.dies_at = dies_at
        self.steps: list[int | None] = []
        self.prepares = 0
        self.stats = _stats()

    def prepare(self) -> None:
        self.prepares += 1

    def step(self, forced: int | None = None) -> dict[str, Any]:
        self.steps.append(forced)
        alive = self.dies_at is None or len(self.steps) < self.dies_at
        return {
            "alive": alive,
            "death_cause": "body" if not alive else "none",
            "life": 5,
            "decision": len(self.steps) - 1,
        }


def test_h300_failure_short_circuits_without_continuation() -> None:
    restored: list[bytes] = []
    driver = BranchDriver(dies_at=3)

    def restore(snapshot: bytes) -> BranchDriver:
        restored.append(snapshot)
        return driver

    result = engine.continue_branch(b"state", 4, 1440, restore=restore)
    assert restored == [b"state"]
    assert result["H300"] == {"survives": False, "frames_executed": 3}
    assert result["continuation"] is None
    assert result["death"]["death_cause"] == "body"
    assert driver.steps == [4, None, None]
    assert driver.prepares == 2


def test_h300_survivor_continues_same_restored_driver_for_1440_frames() -> None:
    restored: list[bytes] = []
    driver = BranchDriver(dies_at=None)

    def restore(snapshot: bytes) -> BranchDriver:
        restored.append(snapshot)
        return driver

    result = engine.continue_branch(b"state", 2, 1440, restore=restore)
    assert restored == [b"state"]
    assert result["H300"] == {"survives": True, "frames_executed": 300}
    assert result["continuation"]["survives"] is True
    assert result["continuation"]["frames_executed"] == 1440
    assert len(driver.steps) == 1440
    assert driver.steps[0] == 2
    assert set(driver.steps[1:]) == {None}
    assert driver.prepares == 1439


class CollectDriver:
    """Small Watch-shaped fake that exposes collection order without game execution."""

    def __init__(self, lengths: list[int], *, stock: int = 0, fallback: int = 0) -> None:
        self.lengths = lengths
        self.decision = 0
        self.life = 9
        self.stats = _stats(stock=stock, fallback=fallback)
        self.sim = SimpleNamespace(
            alive=np.array([[True]]), length=np.array([[0]]), frame=np.array([[0]])
        )
        self.root_queries: list[int] = []

    def prepare(self) -> None:
        self.sim.length[0, 0] = self.lengths[self.decision]
        self.sim.frame[0, 0] = 100 + self.decision

    def legal_and_bare(self) -> tuple[list[int], int]:
        self.root_queries.append(self.decision)
        return [1, 2], 1

    def snapshot(self) -> bytes:
        return f"snapshot-{self.decision}".encode()

    def step(self) -> dict[str, Any]:
        current = self.decision
        self.decision += 1
        causes = ["none", "body"] if current == 4 else ["none"] * 6
        return {"decision": current, "all_death_causes": causes, "death_cause": causes[1]}


def _spec(frames: int) -> dict[str, Any]:
    return {
        "namespace": "bl-c15-b2-natural-roots/v1",
        "allocation_digest": "a" * 64,
        "seeds": list(range(32)),
        "trajectory_frames": frames,
        "root_stride": 1,
        "escape_total_frames": 1440,
        "unit_timeout_seconds": 60,
    }


def test_collect_hash_selects_only_eligible_roots_before_any_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = CollectDriver([799, 800, 801, 802, 803, 804, 805])
    branched: list[tuple[bytes, int]] = []

    def branch(snapshot: bytes, action: int, depth: int, **_kwargs: Any) -> dict[str, Any]:
        assert driver.decision == 7  # Every eligibility coordinate preceded any outcome work.
        branched.append((snapshot, action))
        return {"action": action, "H300": {"survives": True, "frames_executed": 300}}

    monkeypatch.setattr(engine, "continue_branch", branch)
    result = engine.collect(_spec(7), 3, driver, restore=lambda payload: payload)

    expected = sorted(
        (
            engine.root_hash(_spec(7)["namespace"], 3, 9, 100 + decision),
            decision,
        )
        for decision in range(1, 7)
    )[:4]
    assert [root["selection_hash"] for root in result["roots"]] == [key for key, _ in expected]
    assert len(result["roots"]) == 4
    assert all(root["length"] >= 800 for root in result["roots"])
    assert driver.root_queries == list(range(1, 7))
    assert len(branched) == 8  # two legal/B8 actions per selected root
    assert result["deaths"] == [
        {"decision": 4, "all_death_causes": ["none", "body"], "death_cause": "body"}
    ]


def test_collect_cap_is_incomplete_without_a_branch() -> None:
    driver = CollectDriver([800])
    values = iter((0.0, 1.0))
    with pytest.raises(RuntimeError, match="INCOMPLETE B2 unit wall cap"):
        engine.collect(
            {**_spec(1), "unit_timeout_seconds": 1},
            0,
            driver,
            clock=lambda: next(values),
        )
    assert driver.root_queries == []


@pytest.mark.parametrize("counters", [{"stock": 1}, {"fallback": 1}])
def test_collect_refuses_stock_or_fallback_native_rows(counters: dict[str, int]) -> None:
    driver = CollectDriver([799], **counters)
    with pytest.raises(ValueError, match="native execution counters"):
        engine.collect(_spec(1), 0, driver)


@pytest.mark.parametrize("safe", [True, False])
def test_labels_all_domain_legal_actions_not_only_advisory(monkeypatch, safe: bool) -> None:
    from src.simd_env import ego2s_policy, ego_raster

    driver = engine.WatchDriver.__new__(engine.WatchDriver)
    driver.hero = SimpleNamespace(net=object())
    driver.sim = SimpleNamespace(
        _policy_phase_prepared=True,
        alive=np.array([[True]]),
        get_advisory_action_mask=lambda: np.array([[[False, safe, False, False, False, False]]]),
        get_legal_action_mask=lambda: np.ones((1, 1, 6), dtype=bool),
    )
    driver.decision = 0
    driver._prepare_generation = 0
    monkeypatch.setattr(
        ego2s_policy,
        "forward_q",
        lambda *args: SimpleNamespace(
            numpy=lambda: np.array([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        ),
    )
    monkeypatch.setattr(ego2s_policy, "ego_config_for", lambda _net: object())
    monkeypatch.setattr(
        ego_raster,
        "build_ego_raster",
        lambda *_args, **_kwargs: {
            "local": np.zeros((1, 1), dtype=np.uint8),
            "global": np.zeros((1, 1), dtype=np.uint8),
            "scalars": np.zeros((1, 1), dtype=np.float32),
        },
    )
    legal, bare = driver.legal_and_bare()
    assert legal == list(range(6))
    assert bare == 1


def _pool_fake_branch(task: tuple[bytes, int, int, float]) -> dict:
    # Test-only immutable task data, deliberately independent of completion order.
    snapshot, action, endpoint, _seconds = task
    return {
        "action": action,
        "endpoint": endpoint,
        "state_sha256": __import__("hashlib").sha256(snapshot).hexdigest(),
    }


def test_spawned_pool_preserves_serial_branch_bytes() -> None:
    from research.bl_served_search.ceiling import canonical

    tasks = [(b"mock complete state", action, 1440, 10.0) for action in (4, 1, 5, 0)]
    serial = engine.ordered_branches(tasks, 1, worker_fn=_pool_fake_branch)
    parallel = engine.ordered_branches(tasks, 2, worker_fn=_pool_fake_branch)
    assert canonical(serial).encode() == canonical(parallel).encode()


def test_collector_passes_absolute_unit_deadline_to_spawned_tasks(monkeypatch) -> None:
    captured = []

    def pool(tasks, workers, **kwargs):
        captured.extend(tasks)
        return [{"action": task[1]} for task in tasks]

    monkeypatch.setattr(engine, "ordered_branches", pool)
    engine.collect({**_spec(1), "workers": 6}, 0, CollectDriver([800]), clock=lambda: 10000.0)
    assert captured
    assert all(task[3] == 10060.0 for task in captured)


def test_branch_worker_uses_shared_absolute_deadline(monkeypatch) -> None:
    from research.t1_commitment_20261015 import replay

    monkeypatch.setattr(replay, "configure_cpu", lambda: None)
    monkeypatch.setattr(engine.time, "monotonic", lambda: 10000.0)
    monkeypatch.setattr(engine.ceiling, "worker_threads", lambda: {"torch": 1, "numba": 1})

    def branch(snapshot, action, endpoint, *, pulse):
        pulse()
        return {"action": action}

    monkeypatch.setattr(engine, "continue_branch", branch)
    assert engine._branch_task((b"mock", 1, 1440, 10060.0))["action"] == 1
    with pytest.raises(RuntimeError, match="branch wall cap"):
        engine._branch_task((b"mock", 1, 1440, 9999.0))


def test_restore_replaces_spawn_defaults_with_snapshot_game_config() -> None:
    from dataclasses import replace

    from src.core.game_config import AppConfig, GameConfig, get_config, initialize_config
    from src.simd_env.vector61_policy import check_live_rule_config

    previous = get_config()
    defaults = AppConfig.from_defaults()
    deployed = replace(defaults, game=replace(defaults.game, mechanics_version=2))
    try:
        initialize_config(deployed)
        driver = engine.WatchDriver.__new__(engine.WatchDriver)
        sim = SimpleNamespace(
            _grid_pad=1,
            grid_h=2,
            grid_w=2,
            _kernels=None,
            cfg=SimpleNamespace(
                min_boost_length=deployed.game.min_boost_length,
                boost_length_cost_frames=deployed.game.boost_length_cost_frames,
                mechanics_version=2,
                arena_type=deployed.game.arena_type,
            ),
        )
        for name in ("_owner", "_slot", "_food"):
            padded = np.zeros((1, 4, 4), dtype=np.int64)
            setattr(sim, name + "_pad", padded)
            setattr(sim, name, padded[:, 1:3, 1:3])
        driver.sim, driver.frozen, driver.hero = sim, SimpleNamespace(), SimpleNamespace()
        driver.decision, driver.life, driver.awaiting_respawn = 7, 2, False
        payload = driver.snapshot()
        # This is the initial configuration state of a fresh spawn worker.
        initialize_config(defaults)
        assert GameConfig.MECHANICS_VERSION == 1
        with pytest.raises(ValueError, match="differs from GameConfig"):
            check_live_rule_config(sim)
        restored = engine.WatchDriver.restore(payload)
        assert GameConfig.MECHANICS_VERSION == 2
        check_live_rule_config(restored.sim)
        assert np.shares_memory(restored.sim._owner, restored.sim._owner_pad)
    finally:
        initialize_config(previous)
