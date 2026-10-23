"""Tiny same-platform exact-driver restore checks for the F0 seams."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_labels.factory import GateDriver, WatchDriver, _digest_driver

B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
FRP3 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth")

pytestmark = pytest.mark.skipif(
    not B8.is_file() or not FRP3.is_file(),
    reason="real-driver checks require the pinned local B8/FRP3 checkpoints",
)


def _run_three_steps(driver, restore):
    before_after = []
    for _ in range(3):
        driver.prepare()
        _legal, bare = driver.legal_and_bare()
        payload = driver.snapshot()
        before = _digest_driver(driver, "f0-real-test")
        event = driver.step(bare)
        after = _digest_driver(driver, "f0-real-test")

        restored = restore(payload)
        assert before == _digest_driver(restored, "f0-real-test")
        restored.step(bare)
        assert after == _digest_driver(restored, "f0-real-test")
        before_after.append((bare, event["death_cause"]))
    return before_after


def test_watch6_three_step_snapshot_restore_digest_identity() -> None:
    driver = WatchDriver(0, B8, FRP3)
    assert _run_three_steps(driver, WatchDriver.restore) == [(1, "none"), (0, "none"), (2, "none")]


def test_gate_three_step_snapshot_restore_digest_identity() -> None:
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    plan = planned_rosters(gid._context(1), [0], ["frozen"])[0]
    driver = GateDriver(plan, B8, FRP3)
    assert _run_three_steps(driver, GateDriver.restore) == [(1, "none"), (0, "none"), (2, "none")]


def _rollout_metrics(payload: bytes, restore, first_action: int | None) -> dict[str, object]:
    driver = restore(payload)
    lengths: list[int] = []
    death = None
    for offset in range(300):
        if offset:
            driver.prepare()
        event = driver.step(first_action if offset == 0 else None)
        lengths.append(int(event["hero_length"]))
        if not event["alive"]:
            death = event
            break
    return {
        "survives": death is None and len(lengths) == 300,
        "frames": len(lengths),
        "death": None if death is None else int(death["decision"]),
        "length": lengths[-1],
        "mass": sum(lengths) / 300,
    }


def test_bare_h300_matches_uninterrupted_natural_rollout_for_all_four_populations() -> None:
    """The same bare action path must match exact branch replay on every roster."""
    from research.bl_served_search.b2_engine import continue_branch
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    factories = [
        ("watch6", lambda: WatchDriver(0, B8, FRP3)),
        *[
            (
                f"gate-{roster}",
                lambda roster=roster: GateDriver(
                    planned_rosters(gid._context(1), [0], [roster])[0], B8, FRP3
                ),
            )
            for roster in ("frozen", "mixed", "scripted")
        ],
    ]
    for _name, make in factories:
        driver = make()
        driver.prepare()
        _legal, bare = driver.legal_and_bare()
        payload = driver.snapshot()
        restore = type(driver).restore
        branch = continue_branch(payload, bare, 300, restore=restore)
        branch_metrics = _rollout_metrics(payload, restore, bare)
        natural_metrics = _rollout_metrics(payload, restore, None)
        assert branch_metrics == natural_metrics
        assert branch["H300"]["survives"] == natural_metrics["survives"]
        endpoint = branch["continuation"] if branch["H300"]["survives"] else branch["death"]
        assert int(endpoint["hero_length"]) == natural_metrics["length"]


def test_gate_rosters_preserve_disabled_v8_and_watch_enables_native_rows() -> None:
    """L2 gate opponents are intentionally no-v8; Watch6 carries the v8 roster."""
    from research.bl_served_search.b2_engine import driver_stats
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    for roster in ("frozen", "mixed"):
        driver = GateDriver(planned_rosters(gid._context(1), [0], [roster])[0], B8, FRP3)
        for _ in range(5):
            driver.prepare()
            driver.step()
        stats = driver_stats(driver)
        assert stats["kernel_rows"] == 0
        assert stats["eligible_opponent_rows"] == 0

    driver = WatchDriver(0, B8, FRP3)
    for _ in range(5):
        driver.prepare()
        driver.step()
    stats = driver_stats(driver)
    assert stats["kernel_rows"] == stats["eligible_opponent_rows"]
    assert stats["kernel_rows"] > 0
