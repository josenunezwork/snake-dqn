"""Light unit tests for W12 operational-only runtime policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_labels import w12_runtime as runtime
from research.bl_labels.exports import publish_json


def _attempt(
    state: Path, name: str, *, workers: int = 12, fallback=None, thermal=(), clean_intervals=()
):
    start = state / f"attempt-{name}.start.json"
    publish_json(start, {"workers": workers})
    terminal = state / f"attempt-{name}.terminal.json"
    value = {
        "workers": workers,
        "elapsed_seconds": 1.0,
        "start_sha256": runtime._verify(start),
        "thermal_trip_epochs": list(thermal),
        "thermal_clean_intervals": [list(value) for value in clean_intervals],
    }
    if fallback is not None:
        value["operational_width_fallback"] = fallback
    publish_json(terminal, value)


def test_resume_w12_without_attempts_and_sticky_rss_fallback(tmp_path: Path) -> None:
    assert runtime.resolve_resume_workers(12, tmp_path) == (12, [])
    _attempt(tmp_path, "one", fallback={"reason": "rss"})
    assert runtime.resolve_resume_workers(12, tmp_path) == (6, ["operational_width_fallback:rss"])


def test_resume_thermal_requires_24_clean_hours(tmp_path: Path) -> None:
    _attempt(tmp_path, "one", thermal=(1000.0,))
    assert runtime.resolve_resume_workers(12, tmp_path, now=1000 + 86399)[0] == 6
    assert runtime.resolve_resume_workers(12, tmp_path, now=1000 + 86400)[0] == 6
    _attempt(
        tmp_path,
        "two",
        thermal=(1000.0,),
        clean_intervals=((1000.0, 1000.0 + 86400),),
    )
    assert runtime.resolve_resume_workers(12, tmp_path, now=1000 + 86400) == (12, [])


@pytest.mark.parametrize("from_speed_root", [False, True])
def test_cross_dose_thermal_trip_blocks_w12_from_either_canonical_root(
    tmp_path: Path, from_speed_root: bool
) -> None:
    artifacts = tmp_path / "artifacts"
    d256_state = artifacts / "bl-labels-f0" / "mac-pack-02" / "d256-state-mac-01"
    future_state = artifacts / "bl-labels-f0-speed" / "d1024-d4096-pack-01" / "d1024-state-mac-01"
    _attempt(d256_state, "trip", thermal=(1000.0,))
    _attempt(future_state, "other")
    state = future_state if from_speed_root else d256_state
    assert runtime.resolve_resume_workers(12, state, now=1000 + 86400)[0] == 6


def test_resume_rejects_unsealed_attempt_hole(tmp_path: Path) -> None:
    publish_json(tmp_path / "attempt-one.start.json", {"workers": 12})
    with pytest.raises(ValueError, match="hole"):
        runtime.resolve_resume_workers(12, tmp_path)


def test_sanity_is_per_stream_at_three_hours() -> None:
    assert runtime.w12_sanity_decision(
        {"gate": 23, "watch": 46}, 3600, {"gate": 10, "watch": 20}
    ) == (
        True,
        {},
    )
    passed, actual = runtime.w12_sanity_decision(
        {"gate": 35, "watch": 69}, 3 * 3600, {"gate": 10, "watch": 20}
    )
    assert passed and actual == {"gate": pytest.approx(35 / 3), "watch": pytest.approx(23)}
    assert not runtime.w12_sanity_decision(
        {"gate": 34, "watch": 69}, 3 * 3600, {"gate": 10, "watch": 20}
    )[0]


def test_rate_drift_baseline_is_ruling_constant() -> None:
    assert runtime.rate_drift_baseline(10) == pytest.approx(13.6)
    with pytest.raises(ValueError):
        runtime.rate_drift_baseline(0)


def test_same_pin_future_stage_uses_original_d256_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.bl_labels import d256_resume, w12_calibration

    control = {"source_commit": d256_resume.OLD_PIN, "mac_gate": {"stage": "d1024"}}
    original = {"namespace": "original"}
    expected = {"rates_per_wall_hour": {"all": 1.0}}
    monkeypatch.setattr(w12_calibration, "PACK", Path("/canonical-pack"))
    monkeypatch.setattr(d256_resume, "original_control", lambda pack: original)
    monkeypatch.setattr(runtime, "d256_w6_parent_rates", lambda control, pack: expected)
    assert runtime.resolve_w6_parent_rates(control) == expected


def test_d256_aggregate_baseline_charges_zero_completion_w6_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_labels import d256_resume

    control = {
        "allocation": [
            {"allocation_id": "gate", "stream": "gate"},
            {"allocation_id": "watch", "stream": "watch"},
        ]
    }
    first, second = tmp_path / "first", tmp_path / "second"
    publish_json(
        first / "runtime-result.json",
        {"workers": 6, "elapsed_seconds": 10, "completed_parents": ["gate"]},
    )
    publish_json(
        second / "runtime-result.json",
        {"workers": 6, "elapsed_seconds": 90, "completed_parents": []},
    )
    monkeypatch.setattr(
        d256_resume, "attempt_chain", lambda _pack, _control: ([first, second], 100)
    )
    baseline = runtime.d256_w6_parent_rates(control, tmp_path)
    assert baseline["rates_per_wall_hour"] == {"all": pytest.approx(36.0)}
    assert baseline["stream_diagnostics"] == {"completed_parents": {"gate": 1, "watch": 0}}


def test_teq_phase_timing_mismatch_refuses_baseline(tmp_path: Path) -> None:
    pin = "a" * 40
    bank = tmp_path / "beq.json"
    publish_json(bank, {"allocation": [{"allocation_id": "p0", "stream": "gate"}]})
    phase = tmp_path / "timing-0" / "phase-ledger.json"
    publish_json(phase, {"pin": pin, "workers": 6, "parent_indexes": [1], "elapsed_seconds": 10})
    table = tmp_path / "timing-table.json"
    publish_json(
        table,
        {"cells": [{"order": 0, "pin": pin, "workers": 6, "parents": 1, "elapsed_seconds": 10}]},
    )
    receipt = tmp_path / "receipt.json"
    publish_json(
        receipt,
        {
            "new": {"commit": pin},
            "bank": {"path": str(bank), "sha256": runtime._verify(bank)},
            "timing_table_sha256": runtime._verify(table),
        },
    )
    with pytest.raises(ValueError, match="phase ledger"):
        runtime.future_w6_parent_rates({"source_commit": pin}, receipt)


def test_sanity_threshold_exactly_passes_and_below_fails() -> None:
    assert runtime.w12_sanity_decision({"all": 345}, 3 * 3600, {"all": 100})[0]
    assert not runtime.w12_sanity_decision({"all": 344}, 3 * 3600, {"all": 100})[0]
