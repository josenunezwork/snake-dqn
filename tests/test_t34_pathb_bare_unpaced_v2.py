"""Pre-data amendment: the Mac bare arm runs unpaced; outcomes must not change.

Spec: docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md

The file name matches the trace-receipt glob ``tests/test_t34_pathb_*v2.py``. In
receipt mode (``T34_TRACE_RECEIPT=1``) a missing B8 artifact FAILS instead of
skipping, so a receipt can only report these identity tests as passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from research.t34_pathb import retention_v2

REPO = Path(retention_v2.__file__).resolve().parents[2]

# Fields whose values are wall-clock or pacing bookkeeping by construction, not
# gameplay outcomes.  Everything else returned by ``_collect`` must be equal.
PACING_FIELDS = {"overruns", "unpaced_clock", "host_contended_ticks"}

B8_ARTIFACT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
)
RECEIPT_MODE = os.environ.get("T34_TRACE_RECEIPT") == "1"

# Unregistered development seeds (asserted disjoint from every registered bank
# row below), chosen from an unpaced bare scan so that the runs pass the early
# game and include hero deaths: terminal scripted dies at tick 504 and frozen at
# tick 1272 (early-break path); mixed survives 1300 ticks (no mixed dev seed of 50
# died within 1300); Watch6 hero dies and respawns inside the horizon.
TERMINAL_SEEDS = {"scripted": 777001013, "frozen": 777001011, "mixed": 777001000}
TERMINAL_DEATHS = {"scripted": 504, "frozen": 1272}
TERMINAL_HORIZON = 1300
WATCH_SEED = 777002011  # hero dies at tick 705, respawns
WATCH_HORIZON = 760


def _require_b8() -> None:
    if B8_ARTIFACT.is_file():
        return
    if RECEIPT_MODE:
        pytest.fail("trace-receipt mode requires the released B8 artifact; skipping is not allowed")
    pytest.skip("released B8 gameplay artifacts are not installed")


def _registered_seeds() -> set[int]:
    bank = json.loads((REPO / "research/t34_pathb/RETENTION_BANK_V2.json").read_text())
    seeds = {row["seed"] for row in bank["terminal_rows"] + bank["product_rows"]}
    return seeds | {bank["timing_row"]["seed"]}


def _canonical(metrics: dict[str, Any]) -> bytes:
    kept = {key: value for key, value in metrics.items() if key not in PACING_FIELDS}
    return json.dumps(kept, sort_keys=True, separators=(",", ":"), default=str).encode()


def _world_digest(session: Any) -> str:
    from research.t34_pathb.engine_v2 import _digest

    return _digest(session.world, session.checkpoint_sha)


def _run(opener: Any, row: dict[str, Any], fps: int, ticks: int, jitter: bool = False) -> tuple:
    session = opener(row, fps, "bare")
    try:
        if jitter:
            # Adversarial wall-clock perturbation: stall irregularly between ticks.
            original = session.step
            count = {"n": 0}

            def stalled_step() -> None:
                count["n"] += 1
                time.sleep(0.003 * (count["n"] % 7))
                original()

            session.step = stalled_step  # type: ignore[method-assign]
        started, cpu = time.monotonic(), time.process_time()
        metrics = retention_v2._collect(
            session,
            horizon=ticks,
            fps=fps,
            started=started,
            cpu_started=cpu,
            check_cap=lambda: None,
        )
        wall = time.monotonic() - started
        digest = _world_digest(session)
    finally:
        session.close(wait=True)
    return metrics, digest, wall


def _assert_identical(opener: Any, row: dict[str, Any], ticks: int) -> dict[str, Any]:
    paced, paced_digest, paced_wall = _run(opener, row, retention_v2.ARM_FPS["paced"], ticks)
    free, free_digest, free_wall = _run(opener, row, retention_v2.ARM_FPS["bare"], ticks)
    stalled, stalled_digest, _ = _run(opener, row, retention_v2.ARM_FPS["bare"], ticks, True)
    assert (
        hashlib.sha256(_canonical(paced)).hexdigest()
        == hashlib.sha256(_canonical(free)).hexdigest()
    )
    assert _canonical(free) == _canonical(stalled)
    assert paced_digest == free_digest == stalled_digest
    assert paced["bare_prefix_trace"]["steps"] and (
        paced["bare_prefix_trace"] == free["bare_prefix_trace"]
    )
    assert free["overruns"] == 0
    assert free["unpaced_clock"] == "bare_free_running_no_controller"
    assert paced["unpaced_clock"] is None
    # The paced run really slept: at least the 24 fps period per executed tick.
    assert paced_wall >= paced["decisions"] / 24 * 0.95
    print(
        f"\n[bare-unpaced] decisions={paced['decisions']} deaths={paced['deaths']} "
        f"paced={paced_wall:.2f}s free={free_wall:.2f}s"
    )
    return paced


def test_dev_seeds_are_unregistered() -> None:
    used = set(TERMINAL_SEEDS.values()) | {WATCH_SEED}
    assert not used & _registered_seeds()


@pytest.mark.parametrize("mix", sorted(TERMINAL_SEEDS))
def test_terminal_bare_outcomes_identical_paced_vs_unpaced(mix: str) -> None:
    _require_b8()
    from research.t34_pathb.adapter_v2 import open_terminal

    bank = json.loads((REPO / "research/t34_pathb/RETENTION_BANK_V2.json").read_text())
    roster = next(row["roster"] for row in bank["terminal_rows"] if row["mix"] == mix)
    seed = TERMINAL_SEEDS[mix]
    row = {"seed": seed, "mix": mix, "roster": {**roster, "seed": seed}}
    paced = _assert_identical(open_terminal, row, ticks=TERMINAL_HORIZON)
    assert paced["decisions"] >= 480
    if mix in TERMINAL_DEATHS:
        # Exercises the terminal early-break path (dead remainder scored zero).
        assert paced["deaths"] == 1 and paced["decisions"] == TERMINAL_DEATHS[mix]
    else:
        assert paced["deaths"] == 0 and paced["decisions"] == TERMINAL_HORIZON


def test_watch6_bare_outcomes_identical_paced_vs_unpaced_through_a_death() -> None:
    """Product rows drive the real Watch session; fps 0 still constructs speed 24."""
    _require_b8()
    from research.t34_pathb.adapter_v2 import open_watch

    paced = _assert_identical(open_watch, {"seed": WATCH_SEED}, ticks=WATCH_HORIZON)
    assert paced["deaths"] >= 1


class _FakeBare:
    arm = "bare"
    live = None
    hero_alive = True
    hero_length = 10
    hero_death = False
    decisions_delta = 1
    boost_decisions_delta = 0
    executed_action = 1

    def step(self) -> None:
        pass

    def controller_state(self) -> dict[str, Any]:
        return {"active": False, "label": "bare B8 s3"}

    def settle_between_ticks(self, _check: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("bare arm must never hold an unpaced clock")


def test_bare_unpaced_never_sleeps_or_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retention_v2.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError("slept"))
    )
    metrics = retention_v2._collect(
        _FakeBare(),
        horizon=5,
        fps=retention_v2.ARM_FPS["bare"],
        started=time.monotonic(),
        cpu_started=time.process_time(),
        check_cap=lambda: None,
    )
    assert metrics["overruns"] == 0 and metrics["decisions"] == 5


def test_paced_arm_cannot_run_unpaced() -> None:
    fake = _FakeBare()
    fake.arm = "paced"
    with pytest.raises(retention_v2.RegistrationError):
        retention_v2._collect(
            fake,
            horizon=1,
            fps=0,
            started=time.monotonic(),
            cpu_started=time.process_time(),
            check_cap=lambda: None,
        )


def test_arm_rates_are_the_amendment() -> None:
    assert retention_v2.ARM_FPS == {"bare": 0, "paced": 24}
    assert tuple(retention_v2.ARM_FPS) == retention_v2.MAC_ARMS


def _amended() -> dict[str, Any]:
    return {
        "paced_fps": 24,
        "bare_fps": 0,
        "amendments": [retention_v2.bare_unpaced_amendment_binding()],
    }


def test_phase_start_arm_rate_check(monkeypatch: pytest.MonkeyPatch) -> None:
    retention_v2.require_arm_rates(_amended())
    for key in ("bare_fps", "amendments"):
        registration = _amended()
        del registration[key]
        with pytest.raises(retention_v2.RegistrationError):
            retention_v2.require_arm_rates(registration)
    stale = _amended()
    stale["amendments"] = [{**stale["amendments"][0], "sha256": "0" * 64}]
    with pytest.raises(retention_v2.RegistrationError):
        retention_v2.require_arm_rates(stale)
    monkeypatch.setitem(retention_v2.ARM_FPS, "bare", 24)
    with pytest.raises(retention_v2.RegistrationError, match="runner arm rates"):
        retention_v2.require_arm_rates(_amended())


def test_registration_without_bare_fps_is_refused() -> None:
    from research.t34_pathb import registration_retention_v2 as builder

    ratified = json.loads(
        (REPO / "research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json").read_text()
    )
    registration = copy.deepcopy(ratified)
    registration.pop("bare_fps", None)
    with pytest.raises(builder.RegistrationError, match="bare arm must run unpaced"):
        builder._check_common(registration, root=REPO)
    with pytest.raises(retention_v2.RegistrationError, match="bare-unpaced"):
        retention_v2.require_arm_rates(registration)


def test_every_mac_arm_call_site_uses_arm_fps() -> None:
    source = Path(retention_v2.__file__).read_text()
    assert "open_terminal(row, 24, arm)" not in source
    assert "arm_fps = 24" not in source
    assert "opener(row, 24, arm)" not in source
    assert source.count("ARM_FPS[arm]") >= 5
