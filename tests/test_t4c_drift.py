"""Focused contracts for the T4c same-init drift report."""

import json
from pathlib import Path

import numpy as np
import pytest

from research.t4c_20261016 import drift_report
from research.t4c_20261016.drift import raw_rates


def test_consequential_uses_ctrl_best_minus_other_selected_action() -> None:
    ctrl = np.asarray([[1.0, 0.8, 0, 0, 0, 0], [1.0, 0.69, 0, 0, 0, 0]])
    other = np.asarray([[0.0, 1.0, 0, 0, 0, 0], [0.0, 1.0, 0, 0, 0, 0]])
    result = raw_rates(ctrl, other)
    assert result["argmax_changes"] == 2
    assert result["consequential_changes"] == 1


def test_saved_round62_exactly_reproduces_historical_raw_c() -> None:
    artifacts = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
    new_path = artifacts / "t4c-drift-20261016" / "drift-summary-verified.json"
    old_path = artifacts / "t4a-posthoc-20261016" / "posthoc-summary.json"
    if not new_path.is_file() or not old_path.is_file():
        pytest.skip("local archived diagnostic artifacts are not available")
    new = json.loads(new_path.read_text())
    old = json.loads(old_path.read_text())
    for arm in ("TWO_SIDED", "LABELS", "LABELS_KEEPV8"):
        actual = new["rounds"]["62"]["arms"][arm]["raw"]["argmax_change_rate"]
        old_c = old["arms"][arm]["standardized"]["overall"]["sets"]
        expected = old_c["c_ctrl_trajectory_outside_bank_rows"]["argmax_change_rate"]
        assert actual == expected


def test_strict_gap_uses_selected_action_not_runner_up() -> None:
    ctrl = np.asarray([[0.3, 0, 0, 0, 0, 0], [1, 0.9, 0, 0, 0, 0]])
    candidate = np.asarray([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    assert raw_rates(ctrl, candidate)["consequential_changes"] == 1


def test_seed_rates_keep_pooled_weights_and_reject_missing_support() -> None:
    capture = {
        "length": np.asarray([700, 701, 1000]),
        "fingerprint": np.asarray(["a", "b", "c"]),
        "ctrl_q": np.asarray([[1, 0, 0, 0, 0, 0]] * 3),
        "B8_q": np.asarray([[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]),
    }
    weights = {"700_999:D1": 0.8, "ge1000:D1": 0.2}
    result = drift_report.seed_rates(capture, "B8", np.zeros(9), weights, np.asarray([]))
    assert result["raw_change_rate"] == pytest.approx(0.4)
    assert result["consequential_change_rate"] == pytest.approx(0.4)
    missing = drift_report.seed_rates(capture, "B8", np.zeros(9), weights, np.asarray(["c"]))
    assert missing["raw_change_rate"] is None
    assert missing["missing_strata"] == ["ge1000:D1"]


def test_receipt_rejects_changed_input(tmp_path) -> None:
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"original")
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps({"schema": "test", "inputs_sha256": {str(source): drift_report.sha256(source)}})
    )
    assert drift_report.receipt(path, "test", {})["schema"] == "test"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        drift_report.receipt(path, "test", {})


def test_capture_validator_rejects_wrong_seed_and_changed_payload(tmp_path, monkeypatch) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    manifest = bank / "manifest.json"
    manifest.write_text("{}")
    historical = tmp_path / "historical.json"
    historical.write_text("{}")
    monkeypatch.setattr(drift_report, "BANK", bank)
    monkeypatch.setattr(drift_report, "expected_row", lambda seed, round_index: historical)
    monkeypatch.setattr(drift_report, "checkpoint", lambda arm, seed: tmp_path / f"{arm}.pth")
    inputs = [historical, manifest]
    for arm in ("CTRL", *drift_report.ARMS):
        path = tmp_path / f"{arm}.pth"
        path.write_bytes(arm.encode())
        inputs.append(path)
    capture = tmp_path / "captures/CTRL_s10__r65__scripted.npz"
    capture.parent.mkdir()
    np.savez(
        capture,
        length=np.asarray([700]),
        fingerprint=np.asarray(["a"]),
        ctrl_q=np.ones((1, 6)),
        B8_q=np.ones((1, 6)),
    )
    receipt_path = capture.with_suffix(".receipt.json")
    item = {
        "schema": "t4c-drift-capture/v1",
        "seed": 10,
        "round_index": 65,
        "mix": "scripted",
        "capture": str(capture),
        "rows": 1,
        "capture_sha256": drift_report.sha256(capture),
        "historical_row": str(historical),
        "historical_row_sha256": drift_report.sha256(historical),
        "inputs_sha256": {str(p): drift_report.sha256(p) for p in inputs},
    }
    receipt_path.write_text(json.dumps(item))
    assert len(drift_report.validated_capture(tmp_path, 10, 65, {})["length"]) == 1
    item["seed"] = 11
    receipt_path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="identity mismatch"):
        drift_report.validated_capture(tmp_path, 10, 65, {})
    item["seed"] = 10
    receipt_path.write_text(json.dumps(item))
    capture.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        drift_report.validated_capture(tmp_path, 10, 65, {})
