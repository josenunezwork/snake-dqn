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


def test_d4096_sanity_skips_unadmitted_streams_with_matched_exposure() -> None:
    passed, actual = runtime.w12_sanity_decision(
        {"gate": 0, "watch": 93},
        3 * 3600,
        {"gate": 10, "watch": 20},
        stream_elapsed_seconds={"gate": 0, "watch": 3 * 3600},
    )
    assert passed is True
    assert actual == {"watch": pytest.approx(31)}
    with pytest.raises(ValueError, match="no matching stream exposure"):
        runtime.w12_sanity_decision(
            {"gate": 1, "watch": 93},
            3 * 3600,
            {"gate": 10, "watch": 20},
            stream_elapsed_seconds={"gate": 0, "watch": 3 * 3600},
        )


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


def _d4096_control(tmp_path: Path, metadata: Path, readiness: Path) -> dict:
    return {
        "source_commit": "1" * 40,
        "allocation": [
            {"allocation_id": "watch-000", "stream": "watch"},
            {"allocation_id": "gate-000", "stream": "gate"},
        ],
        "mac_gate": {
            "stage": "d4096",
            "bindings": {
                "sealed_collection": {
                    "readiness": {"path": str(readiness), "sha256": runtime._verify(readiness)}
                }
            },
        },
    }


def _d4096_readiness(tmp_path: Path, *, streams: dict | None = None) -> tuple[Path, Path]:
    terminal = tmp_path / "d1024-terminal.json"
    publish_json(terminal, {"schema": "fixture-terminal/v1", "status": "COMPLETE"})
    metadata = tmp_path / "d1024-w6-metadata.json"
    publish_json(
        metadata,
        {
            "schema": "bl-labels-f0-d1024-old-w6-throughput/v1",
            "stage": "d1024",
            "workers": 6,
            "source_commit": "1" * 40,
            "d1024_terminal": {"path": str(terminal), "sha256": runtime._verify(terminal)},
            "streams": streams
            or {
                "watch": {"completed_parents": 308, "elapsed_seconds": 36000},
                "gate": {"completed_parents": 154, "elapsed_seconds": 18000},
            },
        },
    )
    readiness = tmp_path / "readiness.json"
    publish_json(
        readiness,
        {
            "schema": "fixture-readiness/v1",
            "d1024_w6_metadata": {"path": str(metadata), "sha256": runtime._verify(metadata)},
        },
    )
    return metadata, readiness


def test_d4096_uses_only_sealed_same_stream_d1024_metadata(tmp_path: Path) -> None:
    metadata, readiness = _d4096_readiness(tmp_path)
    baseline = runtime.resolve_w6_parent_rates(_d4096_control(tmp_path, metadata, readiness))
    assert baseline["source"] == "d1024-old-w6-sealed-metadata"
    assert baseline["rates_per_wall_hour"] == {
        "gate": pytest.approx(30.8),
        "watch": pytest.approx(30.8),
    }
    assert len(baseline["provenance"]) == 3


def test_d4096_refuses_missing_stream_even_if_a_watch_rate_is_present(tmp_path: Path) -> None:
    metadata, readiness = _d4096_readiness(
        tmp_path, streams={"watch": {"completed_parents": 308, "elapsed_seconds": 36000}}
    )
    with pytest.raises(ValueError, match="exactly its streams"):
        runtime.resolve_w6_parent_rates(_d4096_control(tmp_path, metadata, readiness))


def test_d4096_refuses_unbound_or_nonterminal_metadata(tmp_path: Path) -> None:
    metadata, readiness = _d4096_readiness(tmp_path)
    control = _d4096_control(tmp_path, metadata, readiness)
    control["mac_gate"]["bindings"]["sealed_collection"]["readiness"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="readiness binding changed"):
        runtime.resolve_w6_parent_rates(control)
    # A terminal reference is mandatory; a rate claim alone cannot prove the
    # comparator comes from completed D1024 OLD W6 parent metadata.
    terminal = tmp_path / "terminal-2.json"
    publish_json(terminal, {"status": "COMPLETE"})
    bad_metadata = tmp_path / "bad-metadata.json"
    publish_json(
        bad_metadata,
        {
            "schema": "bl-labels-f0-d1024-old-w6-throughput/v1",
            "stage": "d1024",
            "workers": 6,
            "source_commit": "1" * 40,
            "streams": {
                "watch": {"completed_parents": 308, "elapsed_seconds": 36000},
                "gate": {"completed_parents": 154, "elapsed_seconds": 18000},
            },
        },
    )
    bad_readiness = tmp_path / "bad-readiness.json"
    publish_json(
        bad_readiness,
        {"d1024_w6_metadata": {"path": str(bad_metadata), "sha256": runtime._verify(bad_metadata)}},
    )
    with pytest.raises(ValueError, match="terminal binding"):
        runtime.resolve_w6_parent_rates(_d4096_control(tmp_path, bad_metadata, bad_readiness))


def _d1024_terminal_chain(tmp_path: Path) -> tuple[Path, list[Path]]:
    control = tmp_path / "d1024-control.json"
    value = {
        "namespace": "d1024-fixture",
        "source_commit": "1" * 40,
        "allocation": [
            {"allocation_id": "watch-000", "stream": "watch"},
            {"allocation_id": "gate-000", "stream": "gate"},
        ],
        "mac_gate": {"stage": "d1024"},
    }
    publish_json(control, value)
    terminals = []
    for index, (status, completed) in enumerate(
        (("PREEMPTED", ["watch-000"]), ("COMPLETE", ["gate-000"]))
    ):
        start = tmp_path / f"attempt-{index}.start.json"
        publish_json(
            start,
            {
                "namespace": value["namespace"],
                "source_commit": value["source_commit"],
                "workers": 6,
            },
        )
        terminal = tmp_path / f"attempt-{index}.terminal.json"
        publish_json(
            terminal,
            {
                "schema": "bl-labels-f0-mac-driver/v1",
                "namespace": value["namespace"],
                "workers": 6,
                "collector_source_commit": "61900af28932e47e22e332840fc4406559adb062",
                "w12_calibration_source_commit": "61900af28932e47e22e332840fc4406559adb062",
                "status": status,
                "start_receipt": start.name,
                "start_sha256": runtime._verify(start),
                "attempts_ledger": [path.name for path in [*terminals, terminal]],
                "completed_parents": completed,
                "slowdown_state_by_stream": {
                    "watch": {"width_wall_seconds": {"6": 36000}},
                    "gate": {"width_wall_seconds": {"6": 18000}},
                },
            },
        )
        terminals.append(terminal)
    return control, terminals


def test_seal_d1024_w6_throughput_uses_only_terminal_metadata(tmp_path: Path) -> None:
    control, terminals = _d1024_terminal_chain(tmp_path)
    out = tmp_path / "throughput.json"
    receipt = runtime.seal_d1024_old_w6_throughput(control, terminals, out)
    assert receipt["streams"] == {
        "gate": {"completed_parents": 1, "elapsed_seconds": 18000.0},
        "watch": {"completed_parents": 1, "elapsed_seconds": 36000.0},
    }
    assert receipt["d1024_terminal"]["path"] == str(terminals[-1].resolve())
    assert len(receipt["terminal_chain"]) == 2
    assert runtime._verify(out)


def test_seal_d1024_w6_throughput_refuses_unordered_or_incomplete_chain(tmp_path: Path) -> None:
    control, terminals = _d1024_terminal_chain(tmp_path)
    with pytest.raises(ValueError, match="completed in order"):
        runtime.seal_d1024_old_w6_throughput(
            control, list(reversed(terminals)), tmp_path / "bad.json"
        )
    with pytest.raises(ValueError, match="completed in order"):
        runtime.seal_d1024_old_w6_throughput(control, terminals[:1], tmp_path / "incomplete.json")


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
