"""Pre-data amendment: the Mac bare arm runs unpaced; outcomes must not change.

Spec: docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from research.t34_pathb import retention_v2

# Fields whose values are wall-clock or pacing bookkeeping by construction, not
# gameplay outcomes.  Everything else returned by ``_collect`` must be equal.
PACING_FIELDS = {"overruns", "unpaced_clock", "host_contended_ticks"}

B8_ARTIFACT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth"
)
real_b8 = pytest.mark.skipif(
    not B8_ARTIFACT.is_file(), reason="released B8 gameplay artifacts are not installed"
)


def _canonical(metrics: dict[str, Any]) -> bytes:
    kept = {key: value for key, value in metrics.items() if key not in PACING_FIELDS}
    return json.dumps(kept, sort_keys=True, separators=(",", ":"), default=str).encode()


def _world_digest(session: Any) -> str:
    from research.t34_pathb.engine_v2 import _digest

    return _digest(session.world, session.checkpoint_sha)


def _run(opener: Any, row: dict[str, Any], fps: int, ticks: int, jitter: bool = False) -> tuple:
    session = opener(row, fps, "bare")
    started, cpu = time.monotonic(), time.process_time()
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
        metrics = retention_v2._collect(
            session,
            horizon=ticks,
            fps=fps,
            started=started,
            cpu_started=cpu,
            check_cap=lambda: None,
        )
        digest = _world_digest(session)
    finally:
        session.close(wait=True)
    return metrics, digest, time.monotonic() - started


def _assert_identical(opener: Any, row: dict[str, Any], ticks: int) -> None:
    paced, paced_digest, paced_wall = _run(opener, row, retention_v2.PACED_FPS, ticks)
    free, free_digest, free_wall = _run(opener, row, retention_v2.BARE_FPS, ticks)
    stalled, stalled_digest, _ = _run(opener, row, retention_v2.BARE_FPS, ticks, jitter=True)
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
        f"\n[bare-unpaced] ticks={ticks} paced={paced_wall:.2f}s free={free_wall:.2f}s "
        f"free_ms_per_tick={free_wall * 1000 / max(1, free['decisions']):.2f}"
    )


@real_b8
def test_terminal_bare_outcomes_identical_paced_vs_unpaced() -> None:
    from research.t34_pathb.adapter_v2 import open_terminal

    row = {
        "seed": 2026101934,
        "mix": "scripted",
        "specs": [["scripted", "greedy_food"]] * 5,
    }
    _assert_identical(open_terminal, row, ticks=72)


@real_b8
def test_watch6_bare_outcomes_identical_paced_vs_unpaced() -> None:
    """Product rows drive the real Watch session; fps 0 still constructs speed 24."""
    from research.t34_pathb.adapter_v2 import open_watch

    _assert_identical(open_watch, {"seed": 2026101934}, ticks=48)


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
    assert set(retention_v2.ARM_FPS) == set(retention_v2.MAC_ARMS)


def test_registration_without_bare_fps_is_refused() -> None:
    ratified = json.loads(
        Path("research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json").read_text()
    )
    assert "bare_fps" not in ratified
    from research.t34_pathb import registration_retention_v2 as builder

    with pytest.raises(builder.RegistrationError, match="bare arm must run unpaced"):
        builder._check_common(ratified, root=Path.cwd())


def test_precheck_and_efficacy_open_bare_unpaced() -> None:
    """Source-level guard: every Mac arm call site routes through ARM_FPS."""
    source = Path(retention_v2.__file__).read_text()
    assert "open_terminal(row, 24, arm)" not in source
    assert "arm_fps = 24" not in source
    assert "opener(row, 24, arm)" not in source
    assert source.count("ARM_FPS[arm]") >= 5
