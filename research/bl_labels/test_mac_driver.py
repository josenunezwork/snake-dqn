"""Mac driver operational contracts without starting an exact-simulator job."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_labels import mac_driver
from research.bl_labels.exports import empty_ledger, publish_json, publish_life


def _allocation() -> list[dict[str, object]]:
    return [{"allocation_id": "watch6-000", "seed": 1, "stream": "watch6", "horizon": 10}]


def test_worker_ceiling_requires_owner_approval_and_calibration() -> None:
    mac_driver._validate_workers(6, False, False)
    with pytest.raises(ValueError, match="explicit owner approval"):
        mac_driver._validate_workers(12, False, True)
    with pytest.raises(ValueError, match="passing 30-minute"):
        mac_driver._validate_workers(12, True, False)
    mac_driver._validate_workers(12, True, True)


def test_width_resume_is_limited_to_future_stages_and_w6_w12() -> None:
    d1024 = {"mac_gate": {"stage": "d1024"}}
    d4096 = {"mac_gate": {"stage": "d4096"}}
    d256 = {"mac_gate": {"stage": "d256"}}
    assert mac_driver._width_change_allowed(d1024, 6, 12)
    assert mac_driver._width_change_allowed(d1024, 12, 6)
    assert mac_driver._width_change_allowed(d4096, 6, 12)
    assert not mac_driver._width_change_allowed(d256, 6, 12)
    assert not mac_driver._width_change_allowed(d1024, 1, 2)
    assert not mac_driver._width_change_allowed(d1024, 6, 8)


def test_frozen_collector_fallback_is_future_stage_w6_only() -> None:
    control = {
        "mac_gate": {
            "stage": "d1024",
            "collector_root": "/tmp/frozen-factory",
            "collector_source_commit": "c" * 40,
        }
    }
    assert mac_driver._collector_fallback(control, 6) == (Path("/tmp/frozen-factory"), "c" * 40)
    with pytest.raises(ValueError, match="future-stage W6"):
        mac_driver._collector_fallback(control, 12)
    with pytest.raises(ValueError, match="future-stage W6"):
        mac_driver._collector_fallback(
            {"mac_gate": {"stage": "d1024", "collector_root": "/tmp/frozen-factory"}}, 6
        )


def test_w12_authorization_requires_owner_receipt_and_bound_calibration_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "calibration.json"
    receipt.write_text("{}")
    seen: dict[str, object] = {}

    def verify(path: Path, *, expected_code_sha: str | None = None) -> str:
        seen["path"], seen["pin"] = path, expected_code_sha
        return "a" * 64

    monkeypatch.setattr(mac_driver, "_verify_w12_calibration", verify)
    control = {"mac_gate": {"w12_calibration_source_commit": "f" * 40}}
    assert mac_driver._require_w12_authorization(control, True, receipt) == "a" * 64
    assert seen == {"path": receipt, "pin": "f" * 40}
    with pytest.raises(ValueError, match="sealed calibration receipt"):
        mac_driver._require_w12_authorization(control, True, None)
    with pytest.raises(ValueError, match="sealed calibration receipt"):
        mac_driver._require_w12_authorization(control, False, receipt)


def test_w12_calibration_uses_explicit_frozen_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The calibration pin remains the frozen executor, not the speed candidate."""
    receipt = tmp_path / "receipt.json"
    frozen = "1" * 40
    candidate = "2" * 40
    value = {
        "schema": "bl-labels-f0-w12-calibration/v1",
        "status": "PASS",
        "duration_seconds": 1800,
        "w12_w6_wall_ratio": 1.2,
        "thermal_not_ok": 0,
        "slowdown_trips": 0,
        "platform": "darwin-arm64",
        "code_sha": frozen,
    }
    receipt.write_text(json.dumps(value))
    monkeypatch.setattr("research.bl_labels.exports._verify", lambda path: "b" * 64)
    monkeypatch.setenv("F0_COMMIT", candidate)
    assert mac_driver._verify_w12_calibration(receipt, expected_code_sha=frozen) == "b" * 64
    with pytest.raises(ValueError, match="frozen Mac admission"):
        mac_driver._verify_w12_calibration(receipt, expected_code_sha=candidate)


def test_resume_refuses_foreign_leaf_before_worker_admission(tmp_path: Path) -> None:
    allocation = _allocation()
    namespace = "mac-test"
    publish_life(
        tmp_path,
        "foreign",
        allocation,
        {
            "schema": "bl-labels-f0-life/v1",
            "namespace": "foreign",
            "allocation_id": "watch6-000",
            "allocation": allocation[0],
            "life": 0,
            "rows": [],
        },
    )
    payload = tmp_path / "lives/watch6-000--life-000000.json"
    digest = __import__("hashlib").sha256(payload.read_bytes()).hexdigest()
    ledger = tmp_path / "verified-ledger.json"
    value = empty_ledger(namespace, allocation)
    value["units"] = [
        {
            "unit_id": "watch6-000--life-000000",
            "path": "lives/watch6-000--life-000000.json",
            "sha256": digest,
            "size": payload.stat().st_size,
        }
    ]
    publish_json(ledger, value)
    with pytest.raises(ValueError, match="foreign F0 resume leaf"):
        mac_driver._prepare_jobs({"namespace": namespace, "allocation": allocation}, ledger)


def test_timing_arm_is_observed_at_life_boundary(tmp_path: Path) -> None:
    armed = tmp_path / "mac-timing-armed"
    assert not mac_driver._timing_armed(armed)
    armed.write_text("timing\n")
    assert mac_driver._timing_armed(armed)


def test_guard_snapshot_accepts_a_fresh_output_directory(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(mac_driver.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        mac_driver,
        "_command_output",
        lambda command: (
            "Now drawing from 'AC Power'\n"
            if command[-1] == "batt"
            else (
                '"AppleClamshellState" = No\n'
                if command[0].endswith("ioreg")
                else "Note: No thermal warning level has been recorded\n"
                "Note: No performance warning level has been recorded\n"
                "Note: No CPU power status has been recorded\n"
            )
        ),
    )
    output.mkdir()
    assert mac_driver._guard_snapshot(output)["ac_power"] is True


def test_companion_facade_uses_keyword_bound_attempt_identity(monkeypatch) -> None:
    """Production bridges cannot hide positional/keyword API drift behind fixtures."""
    from research.compute import factory_companion

    seen: dict[str, object] = {}

    def bind(**kwargs):
        seen["bind"] = kwargs

    def update(**kwargs):
        seen["update"] = kwargs
        return {"action": "continue", "receipt": {"fixture": True}}

    monkeypatch.setattr(factory_companion, "bind_factory_attempt", bind)
    monkeypatch.setattr(factory_companion, "factory_control_update", update)
    mac_driver._bind_companion_factory_attempt("attempt-1", policy_sha="a" * 64)
    result = mac_driver._companion_control_update(
        "attempt-1",
        policy_sha="a" * 64,
        protected=True,
        streams={"watch6": {"baseline_ready": False}},
        owned=[{"pid": 1, "start": "fixture"}],
    )
    assert seen["bind"] == {"identity": "attempt-1", "policy_sha": "a" * 64}
    assert seen["update"] == {
        "identity": "attempt-1",
        "policy_sha": "a" * 64,
        "protected": True,
        "streams": {"watch6": {"baseline_ready": False}},
        "owned": [{"pid": 1, "start": "fixture"}],
    }
    assert result["action"] == "continue"
