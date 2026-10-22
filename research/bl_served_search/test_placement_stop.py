"""Synthetic tests for the paired-native L2' placement forecast."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from research.bl_served_search import placement_stop


def _receipt(label: str, seconds: float) -> dict[str, Any]:
    fixtures = []
    records = {mode: [] for mode in placement_stop.MODES}
    for mix, length in (
        ("frozen", 800),
        ("frozen", 1200),
        ("frozen", 1500),
        ("mixed", 800),
        ("mixed", 1200),
        ("mixed", 1500),
        ("scripted", 800),
        ("scripted", 1200),
        ("scripted", 1500),
    ):
        fixture_id = f"{mix}-{length}"
        fixtures.append({"fixture_id": fixture_id, "mix": mix})
        for mode in placement_stop.MODES:
            records[mode].append(
                {
                    "fixture_id": fixture_id,
                    "batches": [
                        {"worker_wall_seconds": seconds, "semantic": {"chosen": 0}}
                        for _ in range(30)
                    ],
                }
            )
    cells = {
        mode: {
            "fixture_sha256": "fixture",
            "source_identity": {"source": "same"},
            "completed_batches": 270,
            "aggregate_batches_per_second": 3.0,
            "records": records[mode],
        }
        for mode in placement_stop.MODES
    }
    return {
        "platform": {"label": label, "system": "Darwin" if label == "mac" else "Linux"},
        "fixtures": {"fixtures": fixtures},
        "cells": cells,
        "decision_identity": {"status": "PASS", "compared_batches": 270, "mismatches": []},
    }


def _stub_native_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native-ruler module owns the exhaustive branch/counter attestation."""
    monkeypatch.setattr(placement_stop, "_paired_validator", lambda receipt: None)


def test_forecast_prices_each_mode_from_its_actual_worker_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_native_validator(monkeypatch)
    mac = _receipt("mac", 1.0)
    l4 = _receipt("l4", 0.1)
    value = placement_stop.forecast(mac, l4)
    assert value["status"] == "L4_LAWFUL"
    assert value["modes"]["all"]["l4"]["per_mix"]["frozen"][
        "actual_worker_seconds"
    ] == pytest.approx(9.0)
    assert value["modes"]["exact_stop"]["mac"]["whole_workload_mix_counts"] == {
        "frozen": 64,
        "mixed": 64,
        "scripted": 32,
    }
    assert value["modes"]["all"]["l4"]["workers_per_chunk"] == 12
    assert value["timing_transfer"] == "FORBIDDEN: Mac timings never price L4"


def test_missing_l4_only_allows_prospective_mac_amendment(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_native_validator(monkeypatch)
    value = placement_stop.forecast(_receipt("mac", 1.0))
    assert value["status"] == "MAC_AMENDMENT_ELIGIBLE"
    assert value["l4_receipt_present"] is False
    assert value["modes"]["exact_stop"]["mac"]["paced_displacement_requires_judge_input"] is True
    assert "no automatic dispatch" in value["authority"]


def test_exact_stop_mac_amendment_can_qualify_when_full_evaluation_exceeds_60h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_native_validator(monkeypatch)
    mac = _receipt("mac", 1.0)
    for record in mac["cells"]["all"]["records"]:
        for batch in record["batches"]:
            batch["worker_wall_seconds"] = 500.0
    value = placement_stop.forecast(mac)
    assert value["modes"]["all"]["mac"]["eligible"] is False
    assert value["modes"]["exact_stop"]["mac"]["eligible"] is True
    assert value["status"] == "MAC_AMENDMENT_ELIGIBLE"


def test_identity_or_counter_failure_refuses_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_native_validator(monkeypatch)
    bad = _receipt("mac", 1.0)
    bad["decision_identity"] = {"status": "PASS", "compared_batches": 269, "mismatches": []}
    with pytest.raises(ValueError, match="chosen-action identity"):
        placement_stop.forecast(bad)
    bad = _receipt("mac", 1.0)
    bad["cells"]["all"]["completed_batches"] = 269
    with pytest.raises(ValueError, match="counter"):
        placement_stop.forecast(bad)


def test_provenance_mismatch_and_slow_mac_are_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_native_validator(monkeypatch)
    mac = _receipt("mac", 500.0)
    l4 = _receipt("l4", 0.1)
    value = placement_stop.forecast(mac, l4)
    assert value["status"] == "L4_LAWFUL"
    mismatch = copy.deepcopy(l4)
    mismatch["cells"]["all"]["fixture_sha256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        placement_stop.forecast(mac, mismatch)
    value = placement_stop.forecast(mac)
    assert value["status"] == "NOT_READY"


def test_cross_platform_semantic_mismatch_refuses_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_native_validator(monkeypatch)
    mac = _receipt("mac", 1.0)
    l4 = _receipt("l4", 0.1)
    l4["cells"]["exact_stop"]["records"][0]["batches"][0]["semantic"]["chosen"] = 1
    with pytest.raises(ValueError, match="semantic mismatch"):
        placement_stop.forecast(mac, l4)


def test_real_native_receipt_contract_is_consumed_without_validator_stub() -> None:
    from research.bl_served_search.test_native_ruler import paired_receipt

    mac = paired_receipt("mac")
    l4 = paired_receipt("l4")
    value = placement_stop.forecast(mac, l4)
    assert value["status"] == "L4_LAWFUL"
    assert value["decision_identity"] == {
        "status": "PASS",
        "compared_batches": 270,
        "mismatches": [],
    }
