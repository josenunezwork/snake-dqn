"""Mechanical tests for the independent W12 throughput admission only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_labels import d1024_w12, mac_gates, w12_admission, w12_calibration
from research.bl_labels.exports import publish_json


@pytest.fixture
def ruled_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide outcome-free pinned inputs so ordinary CI needs no artifact volume."""
    calibration = tmp_path / "w12-calibration-01" / "receipt.json"
    calibration.parent.mkdir(parents=True)
    samples = calibration.parent / "host-samples.jsonl"
    samples.write_text('{"thermal_not_ok":0}\n')
    judgment = tmp_path / "JUDGMENT_W12_THROUGHPUT_20261023.md"
    judgment.write_text("independent ruling\n")
    publish_json(
        calibration,
        {
            "schema": "bl-labels-f0-w12-calibration/v1",
            "status": "FAIL",
            "code_sha": w12_admission.F0_COMMIT,
            "platform": "darwin-arm64",
            "bank_sha256": w12_admission.F1_BANK_SHA256,
            "duration_seconds": 1800,
            "w12_w6_wall_ratio": 1.4,
            "thermal_not_ok": 0,
            "slowdown_trips": 0,
            "outcome_exports": False,
            "host_samples_sha256": w12_admission.sha(samples),
            "host_samples_bytes": samples.stat().st_size,
            "cells": [
                {"workers": 6, "units": 192, "seconds": 100},
                {"workers": 12, "units": 192, "seconds": 70},
            ],
        },
    )
    monkeypatch.setattr(w12_admission, "CALIBRATION_RECEIPT", calibration)
    monkeypatch.setattr(w12_admission, "JUDGMENT", judgment)
    monkeypatch.setattr(w12_admission, "CALIBRATION_SHA256", w12_admission.sha(calibration))
    monkeypatch.setattr(w12_admission, "JUDGMENT_SHA256", w12_admission.sha(judgment))
    return calibration


def test_build_recomputes_the_sealed_calibration_without_mutating_it(ruled_inputs: Path) -> None:
    before = w12_admission.sha(ruled_inputs)
    value = w12_admission.build()
    assert before == w12_admission.CALIBRATION_SHA256
    assert value["status"] == "PASS"
    assert value["calibration_receipt_sha256"] == before
    assert value["judgment_sha256"] == w12_admission.JUDGMENT_SHA256
    assert value["computed"]["w12_units_per_wall_second_ratio"] >= 1.20
    assert value["computed"]["w12_w6_wall_ratio"] == pytest.approx(1.4)
    assert w12_admission.sha(ruled_inputs) == before


def test_create_only_record_rechecks_the_full_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruled_inputs: Path
) -> None:
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", receipt)
    value = w12_admission.create(receipt)
    assert w12_admission.verify(receipt, expected_code_sha=w12_admission.F0_COMMIT)
    assert json.loads(receipt.read_text()) == value
    with pytest.raises(ValueError, match="create-only"):
        w12_admission.create(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["computed"].update(w12_units_per_wall_second_ratio=1.21),
        lambda value: value["criterion"].update(minimum=1.19),
        lambda value: value.update(judgment_sha256="0" * 64),
        lambda value: value["provenance"].update(thermal_not_ok=1),
    ],
)
def test_record_tampering_is_refused_even_when_resealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruled_inputs: Path, mutate
) -> None:
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", path)
    value = copy.deepcopy(w12_admission.build())
    mutate(value)
    publish_json(path, value)
    with pytest.raises(ValueError, match="differs"):
        w12_admission.verify(path, expected_code_sha=w12_admission.F0_COMMIT)


def test_wrong_code_pin_cannot_repurpose_calibration_01(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruled_inputs: Path
) -> None:
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", path)
    publish_json(path, w12_admission.build())
    with pytest.raises(ValueError, match="bound only"):
        w12_admission.verify(path, expected_code_sha="a" * 40)


def test_future_gate_treats_only_the_ruled_record_as_w12_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruled_inputs: Path
) -> None:
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", path)
    w12_admission.create(path)
    assert (
        mac_gates._future_calibration(path, w12_admission.F0_COMMIT, "unused-for-admission")
        == "PASS"
    )


def test_d1024_unwraps_admission_for_checkpoint_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruled_inputs: Path
) -> None:
    """The admission is operational; checkpoint hashes stay in calibration-01."""
    admission = tmp_path / "w12-admission-01" / "receipt.json"
    admission.parent.mkdir()
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", admission)
    b8, frp3 = tmp_path / "b8", tmp_path / "frp3"
    b8.write_bytes(b"b8")
    frp3.write_bytes(b"frp3")
    calibration = json.loads(ruled_inputs.read_text())
    calibration.update(b8_sha256=w12_calibration.sha(b8), frp3_sha256=w12_calibration.sha(frp3))
    bound_calibration = tmp_path / "checkpoint-bound-calibration" / "receipt.json"
    bound_calibration.parent.mkdir()
    (bound_calibration.parent / "host-samples.jsonl").write_text(
        (ruled_inputs.parent / "host-samples.jsonl").read_text()
    )
    publish_json(bound_calibration, calibration)
    monkeypatch.setattr(w12_admission, "CALIBRATION_RECEIPT", bound_calibration)
    monkeypatch.setattr(w12_admission, "CALIBRATION_SHA256", w12_admission.sha(bound_calibration))
    w12_admission.create(admission)
    control = tmp_path / "d1024-control.json"
    pin = "a" * 40
    publish_json(
        control,
        {
            "source_commit": pin,
            "mac_gate": {
                "stage": "d1024",
                "status": "HOLD",
                "w12_calibration_source_commit": w12_admission.F0_COMMIT,
            },
        },
    )
    speed = tmp_path / "speed.json"
    publish_json(speed, {"status": "PASS"})
    monkeypatch.setenv("F0_COMMIT", pin)
    monkeypatch.setattr(mac_gates, "_future_source_pin", lambda _: pin)
    monkeypatch.setattr(mac_gates, "_future_readiness", lambda *_: {})
    monkeypatch.setattr(d1024_w12.speed_equiv, "verify_receipt", lambda *_a, **_k: None)
    monkeypatch.setattr(
        d1024_w12,
        "_calibration_workers",
        lambda *_: (12, ["--owner-w12-ok", "--w12-calibration-receipt", str(admission)]),
    )
    _, workers, flags, old_fallback = d1024_w12._validate_inputs(
        "d1024", control, speed, admission, b8, frp3, None, tmp_path / "f4", tmp_path / "inputs"
    )
    assert (workers, flags, old_fallback) == (
        12,
        ["--owner-w12-ok", "--w12-calibration-receipt", str(admission)],
        False,
    )
