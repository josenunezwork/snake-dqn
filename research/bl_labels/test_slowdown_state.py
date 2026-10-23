"""Light immutable baseline migration and resume checks, with no factory run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_labels import slowdown_state as state
from research.bl_labels.exports import _verify


def control(namespace="d256", pin="a" * 40):
    return {
        "namespace": namespace,
        "source_commit": pin,
        "mac_gate": {"stage": "d256"},
        "allocation": [
            {"allocation_id": "p", "stream": "watch6"},
            {"allocation_id": "g", "stream": "gate"},
        ],
    }


def test_original_hour_freezes_once_and_resume_never_extends_or_mixes(tmp_path, monkeypatch):
    first = {"workers": 6, "elapsed_seconds": 4463, "first_hour_life_seconds": [147.9] * 136}
    resumed_legacy = {"workers": 6, "elapsed_seconds": 402, "first_hour_life_seconds": [999] * 200}
    guards = state.build_guards(control(), 12, 0, [first, resumed_legacy], tmp_path)
    assert guards["watch6"].snapshot(0)["baselines"]["6"]["median"] == 147.9
    state.persist_baselines(guards, tmp_path, control(), 0)
    frozen = state._path(tmp_path, "watch6", 6)
    before = frozen.read_bytes()
    digest = _verify(frozen)
    saved = {s: g.snapshot(20) for s, g in guards.items()}
    terminal = {"workers": 12, "elapsed_seconds": 20, "slowdown_state_by_stream": saved}

    def forbid_recompute(_values):
        raise AssertionError("frozen baseline was recomputed")

    monkeypatch.setattr(state.statistics, "median", forbid_recompute)
    restored = state.build_guards(control(), 6, 100, [first, resumed_legacy, terminal], tmp_path)
    state.persist_baselines(restored, tmp_path, control(), 100)
    assert frozen.read_bytes() == before and _verify(frozen) == digest
    assert "12" not in restored["watch6"].snapshot(100)["baselines"]
    assert json.loads(before)["baseline"]["sample_count"] == 136


def test_foreign_namespace_and_tampered_baseline_fail_closed(tmp_path):
    first = {"workers": 6, "elapsed_seconds": 4000, "first_hour_life_seconds": [100] * 60}
    guards = state.build_guards(control(), 6, 0, [first], tmp_path)
    state.persist_baselines(guards, tmp_path, control(), 0)
    with pytest.raises(ValueError, match="namespace"):
        state.build_guards(control("foreign"), 6, 1, [first], tmp_path)
    path = state._path(tmp_path, "watch6", 6)
    path.write_text("{}")
    with pytest.raises(ValueError):
        state.build_guards(control(), 6, 1, [first], tmp_path)


def test_existing_same_pin_stream_baseline_limits_new_stream(tmp_path):
    first = {"workers": 6, "elapsed_seconds": 4000, "first_hour_life_seconds": [100] * 60}
    one = tmp_path / "one-state"
    guards = state.build_guards(control(), 6, 0, [first], one)
    state.persist_baselines(guards, one, control(), 0)
    fresh = control("new")
    fresh["mac_gate"]["stage"] = "d1024"
    later = state.build_guards(fresh, 6, 0, [], tmp_path / "two-state")
    guard = later["gate"]
    for i in range(60):
        guard.admit(str(i), False, 0)
    for i in range(60):
        assert guard.boundary(str(i), 131) is None
    event = guard.tick(3600)
    assert event["stop_reason"] == "slowdown baseline sanity"
    assert "6" not in guard.snapshot(3600)["baselines"]


def test_new_same_attempt_baseline_immediately_constrains_next_stream(tmp_path):
    fresh = control("new", "d" * 40)
    fresh["mac_gate"]["stage"] = "d1024"
    guards = state.build_guards(fresh, 6, 0, [], tmp_path)
    for stream, duration in (("watch6", 100), ("gate", 131)):
        guard = guards[stream]
        for i in range(60):
            guard.admit(str(i), False, 0)
        for i in range(60):
            assert guard.boundary(str(i), duration) is None
    assert guards["watch6"].tick(3600) is None
    state.persist_baselines(guards, tmp_path, fresh, 3600)
    assert guards["gate"].tick(3600)["stop_reason"] == "slowdown baseline sanity"


def test_baseline_validation_failure_precedes_attempt_start(tmp_path, monkeypatch):
    from research.bl_labels import mac_driver
    from research.bl_labels.exports import empty_ledger, publish_json
    from research.bl_labels.test_mac_integration import _patch_inline_runtime

    _patch_inline_runtime(monkeypatch)
    ctl = control()
    ctl.pop("mac_gate")
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(ctl["namespace"], ctl["allocation"]))

    def refuse(*_args):
        raise ValueError("frozen slowdown baseline differs; HOLD")

    monkeypatch.setattr(state, "build_guards", refuse)
    with pytest.raises(ValueError, match="baseline differs"):
        mac_driver.run(
            ctl,
            ledger,
            Path("b8"),
            Path("frp3"),
            tmp_path / "output",
            workers=1,
            state_dir=tmp_path / "state",
        )
    assert not list((tmp_path / "state").glob("attempt-*.start.json"))


def test_driver_window_trip_seals_exact_reason_telemetry_and_charge(tmp_path, monkeypatch):
    from research.bl_labels import mac_driver
    from research.bl_labels.exports import empty_ledger, publish_json
    from research.bl_labels.slowdown_guard import SCHEMA, SlowdownGuard
    from research.bl_labels.test_mac_integration import (
        _allocations,
        _install_fake_factory,
        _patch_inline_runtime,
    )

    _patch_inline_runtime(monkeypatch)
    _install_fake_factory(monkeypatch)
    ctl = {"namespace": "trip-fixture", "source_commit": "a" * 40, "allocation": _allocations(12)}
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(ctl["namespace"], ctl["allocation"]))
    observed = {}

    def guards(_control, workers, started, _terminals, _state):
        guard = SlowdownGuard(
            "watch6",
            workers,
            started,
            persisted={
                "schema": SCHEMA,
                "stream": "watch6",
                "width_wall_seconds": {"6": 3600},
                "baselines": {"6": {"median": 0, "sample_count": 60, "frozen": True}},
                "candidates": {},
            },
        )
        observed["guard"] = guard
        return {"watch6": guard}

    monkeypatch.setattr(state, "build_guards", guards)

    def snapshot(_output):
        guard = observed.get("guard")
        if guard is not None and len(guard._window) == 24:
            raise mac_driver.MacGuardError("synthetic later host warning")
        return {"fixture": True}

    monkeypatch.setattr(mac_driver, "_guard_snapshot", snapshot)
    result = mac_driver.run(
        ctl,
        ledger,
        Path("b8"),
        Path("frp3"),
        tmp_path / "output",
        workers=6,
        state_dir=tmp_path / "state",
    )
    assert result["status"] == "PREEMPTED"
    assert result["stop_reason"] == "slowdown guard: window median > 1.30 x baseline"
    assert result["post_slowdown_stop_reason"].startswith("operational guard:")
    assert result["slowdown_trips"] == 1
    assert result["slowdown_trip_records"][0]["width"] == 6
    event = next(
        e for e in result["telemetry_sealed_no_outcomes"] if e["event"] == "slowdown_guard"
    )
    assert event["statistic"] == "window_median" and event["window_k"] == 24
    assert result["worker_seconds_denominator"] == 6 * result["elapsed_seconds"]
