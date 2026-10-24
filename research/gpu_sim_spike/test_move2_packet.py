"""Offline packet contracts for MOVE2; never allocates an A100."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.gpu_sim_spike import move2_analyze, move2_events, move2_pack, move2_units

PIN = "a" * 40


def test_move2_allocations_are_fresh_and_mode_separated() -> None:
    assert len(move2_units.allocation()) == 9
    assert len(move2_units.allocation(canary=True)) == 3
    assert move2_units.namespace(PIN) != move2_units.namespace(PIN, canary=True)
    assert "gpu-move-a100-20261023-02" in move2_units.namespace(PIN)


def test_move2_ledger_resumes_only_verified_same_namespace_units(tmp_path: Path) -> None:
    cells, ns = move2_units.allocation(smoke=True), move2_units.namespace(PIN, smoke=True)
    move2_units.publish_unit(tmp_path, ns, cells, cells[0], {"descriptor": {"sha256": "b" * 64}})
    ledger = tmp_path / "recovery" / "verified-ledger.json"
    move2_units.build_ledger(tmp_path, ledger, commit=PIN, smoke=True)
    assert move2_units.load_ledger(ledger, ns, cells) == {
        cells[0]: {"descriptor": {"sha256": "b" * 64}}
    }
    with pytest.raises(ValueError, match="foreign"):
        move2_units.load_ledger(ledger, move2_units.namespace("b" * 40, smoke=True), cells)


def test_move2_rejects_large_trace_inventory(tmp_path: Path) -> None:
    (tmp_path / "recorded-states").mkdir()
    with pytest.raises(ValueError, match="trace"):
        move2_units.build_ledger(tmp_path, tmp_path / "verified-ledger.json", commit=PIN)


def test_move2_rejects_metadata_only_foreign_unit(tmp_path: Path) -> None:
    cells, ns = move2_units.allocation(), move2_units.namespace(PIN)
    payload = move2_units.publish_unit(tmp_path, ns, cells, cells[0], {})
    body = json.loads(payload.read_text())
    body["namespace"] = "foreign"
    payload.unlink()
    Path(str(payload) + ".sha256").unlink()
    Path(str(payload) + ".complete").unlink()
    move2_units.publish_json(payload, body)
    with pytest.raises(ValueError, match="foreign"):
        move2_units.build_ledger(tmp_path, tmp_path / "verified-ledger.json", commit=PIN)


def _native_report() -> dict:
    cases = []
    for index, case in enumerate(move2_events.CASES):
        calls = {"respawn": 0, "collisions": 0}
        calls["respawn" if index < 4 else "collisions"] = 1
        cases.append({"case": case, "identity": {"match": True}, "native_calls": calls})
    return {
        "status": "PASS",
        "portability_identity": "PASS",
        "fallback": False,
        "extension_sha256": "c" * 64,
        "native_calls": {"respawn": 4, "collisions": 6},
        "cases": cases,
    }


def test_native_report_accepts_producer_shaped_identity_dicts() -> None:
    assert move2_pack.native_report_valid(_native_report())


@pytest.mark.parametrize(
    "mutate", ["failed_identity", "missing_case", "missing_extension", "wrong_native_call"]
)
def test_native_report_rejects_incomplete_or_false_native_proof(mutate: str) -> None:
    report = _native_report()
    if mutate == "failed_identity":
        report["cases"][0]["identity"] = {"match": False}
    elif mutate == "missing_case":
        report["cases"].pop()
    elif mutate == "missing_extension":
        report.pop("extension_sha256")
    else:
        report["cases"][0]["native_calls"]["respawn"] = 0
    assert not move2_pack.native_report_valid(report)


def _cell(unit_id: str) -> dict:
    mix, worlds = unit_id.split("-")[1], int(unit_id.rsplit("e", 1)[1])
    reference, indexed = [3.0, 6.0, 9.0], [1.0, 2.0, 3.0]
    proofs = [
        {
            "repeat": repeat,
            "candidate_vs_reference": {"match": True},
            "candidate_vs_frozen": {"match": True},
        }
        for repeat in range(3)
    ]
    return {
        "mix": mix,
        "worlds": worlds,
        "warmed_interleaved_repeats": 3,
        "warmups": 3,
        "capacity": 10400,
        "lengths": [800, 900, 1000, 1100, 1300, 1500],
        "reference_seconds": reference,
        "indexed_seconds": indexed,
        "speedup": 3.0,
        "frozen_device_identity": proofs,
        "recorded_inputs": [{"repeat": repeat, "input_sha256": "d" * 64} for repeat in range(3)],
    }


def test_cell_contract_rejects_forged_matrix_identity_and_samples() -> None:
    unit = "cell-frozen-e1024"
    assert move2_analyze.valid_cell(unit, _cell(unit))
    forged = _cell(unit)
    forged["worlds"] = 64
    assert not move2_analyze.valid_cell(unit, forged)
    missing = _cell(unit)
    missing["frozen_device_identity"].pop()
    assert not move2_analyze.valid_cell(unit, missing)
    nonfinite = _cell(unit)
    nonfinite["reference_seconds"][1] = float("nan")
    assert not move2_analyze.valid_cell(unit, nonfinite)


@pytest.mark.parametrize("status,expected", [("INVALID", "INVALID"), ("NOT_RUN", "INCOMPLETE")])
def test_reducer_preserves_failed_or_unrun_native_status(
    tmp_path: Path, status: str, expected: str
) -> None:
    from research.gpu_sim_spike import move2_analyze

    move2_units.publish_json(
        tmp_path / "cpu-discrepancies.json", {"commit": PIN, "status": "DIAGNOSTIC"}
    )
    move2_units.publish_json(tmp_path / "native-parity.json", {"commit": PIN, "status": status})
    result = move2_analyze.aggregate([tmp_path], PIN, canary=True)
    assert result["verdict"] == expected
    assert result["native_a100_parity"] == status
    assert result["optimization_identity"] == "INCOMPLETE"
