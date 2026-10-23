"""Mac driver operational contracts without starting an exact-simulator job."""

from __future__ import annotations

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
