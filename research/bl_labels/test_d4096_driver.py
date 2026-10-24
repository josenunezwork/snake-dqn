"""Synthetic coordinator seams for sealed D4096; no real collector or host lease."""

from pathlib import Path

import pytest

from research.bl_labels import d4096_pack, mac_driver, mac_gates, w12_admission
from research.bl_labels.exports import _verify, empty_ledger, publish_json
from research.bl_labels.test_mac_integration import (
    _allocations,
    _install_fake_factory,
    _patch_inline_runtime,
)


def test_direct_driver_w12_requires_exact_admitted_calibration(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    publish_json(receipt, {"schema": w12_admission.SCHEMA, "status": "PASS"})
    monkeypatch.setattr(w12_admission, "ADMISSION_RECEIPT", receipt)
    monkeypatch.setattr(mac_driver, "_verify_w12_calibration", lambda *_a, **_k: "verified")
    control = {"mac_gate": {"stage": "d4096", "bindings": {}}}
    with pytest.raises(ValueError, match="bound w12-admission-01"):
        mac_driver._require_w12_authorization(control, True, receipt)
    control["mac_gate"]["bindings"]["calibration"] = {
        "path": str(receipt.resolve()),
        "sha256": _verify(receipt),
        "status": "PASS",
    }
    assert mac_driver._require_w12_authorization(control, True, receipt) == "verified"
    control["mac_gate"]["bindings"]["calibration"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound w12-admission-01"):
        mac_driver._require_w12_authorization(control, True, receipt)


def test_d4096_attempt_seals_pending_and_exposure_before_three_hours(tmp_path, monkeypatch):
    _patch_inline_runtime(monkeypatch)
    monkeypatch.setattr(mac_driver.os, "getpriority", lambda *_: 15)
    _install_fake_factory(monkeypatch)
    monkeypatch.setattr(mac_gates, "require_admission", lambda *_: None)
    monkeypatch.setattr(mac_driver, "_require_d4096_operation", lambda *_: None)
    monkeypatch.setattr(mac_driver, "_collector_fallback", lambda *_: (None, None))
    monkeypatch.setattr(mac_driver, "_require_w12_authorization", lambda *_: "a" * 64)
    monkeypatch.setenv("F0_COMMIT", d4096_pack.SCIENCE)
    state = tmp_path / "state"
    control = {
        "namespace": "synthetic-d4096-driver",
        "source_commit": d4096_pack.SCIENCE,
        "allocation": _allocations(2),
        "mac_gate": {
            "stage": "d4096",
            "package_root": str(tmp_path),
            "deadline_seconds": 468000,
            "state_dir": str(state),
        },
    }
    ledger = tmp_path / "empty.json"
    publish_json(ledger, empty_ledger(control["namespace"], control["allocation"]))
    result = mac_driver.run(
        control,
        ledger,
        Path("b8"),
        Path("frp3"),
        tmp_path / "output",
        workers=12,
        owner_w12_ok=True,
        w12_calibration_receipt=tmp_path / "admission",
        deadline_seconds=468000,
        timing_armed=tmp_path / "unarmed",
        state_dir=state,
    )
    start = mac_gates._read(state / result["start_receipt"])
    assert start["funding_state"] == "PENDING_SEALED"
    assert start["judgment_sha256"] == d4096_pack.JUDGMENT_SHA
    assert result["w12_sanity_checked"] is False
    assert result["w12_sanity_stream_elapsed_seconds"]["watch6"] > 0
    assert result["worker_seconds_denominator"] == 12 * result["elapsed_seconds"]


@pytest.mark.parametrize("inherited", [15, 19, 20])
def test_niced_d4096_worker_never_attempts_to_raise_priority(monkeypatch, inherited):
    current = [inherited]
    requested = []
    monkeypatch.setattr(mac_driver.os, "getpriority", lambda *_: current[0])

    def setpriority(_which, _pid, value):
        assert value >= current[0], "unprivileged priority escalation would fail"
        requested.append(value)
        current[0] = value

    monkeypatch.setattr(mac_driver.os, "setpriority", setpriority)
    mac_driver._set_worker_priority(15)
    assert requested == [inherited]


def test_direct_driver_checks_discard_before_attempt_or_output_processing(tmp_path, monkeypatch):
    monkeypatch.setattr(mac_gates, "require_admission", lambda *_: None)
    publish_json(tmp_path / "d4096-not-run-by-rule.json", {"foreign": True})
    control = {"mac_gate": {"stage": "d4096", "package_root": str(tmp_path)}}
    with pytest.raises(ValueError, match="discard seal"):
        mac_driver._require_d4096_operation(control, tmp_path / "output", None, None)
    assert not (tmp_path / "output").exists()
