"""Bounded real-coordinator checks with opaque synthetic collectors only."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from research.bl_labels import mac_driver, mac_gates
from research.bl_labels.exports import empty_ledger, publish_json
from research.bl_labels.test_mac_integration import (
    _allocations,
    _install_fake_factory,
    _patch_inline_runtime,
)


def _run(tmp_path, monkeypatch, *, rss=0, total_rss=None, thermal=False, sanity=False):
    _patch_inline_runtime(monkeypatch)
    _install_fake_factory(monkeypatch)
    monkeypatch.setattr(mac_gates, "require_admission", lambda _: None)
    monkeypatch.setattr(mac_driver, "_require_w12_authorization", lambda *_: "a" * 64)
    monkeypatch.setenv("F0_COMMIT", "f" * 40)
    rss_by_pid = {}

    def rss_many(pids):
        if total_rss is None:
            return {pid: rss for pid in pids}
        if pids and not rss_by_pid:
            per_worker, remainder = divmod(total_rss, len(pids))
            rss_by_pid.update(
                {pid: per_worker + (index < remainder) for index, pid in enumerate(pids)}
            )
        return {pid: rss_by_pid[pid] for pid in pids}

    monkeypatch.setattr(mac_driver, "_rss_many", rss_many)
    calls = 0

    def snapshot(_output):
        nonlocal calls
        calls += 1
        if thermal and calls > 1:
            raise mac_driver.MacGuardError("thermal performance_warning_level=1")
        return {"fixture": True}

    monkeypatch.setattr(mac_driver, "_guard_snapshot", snapshot)
    state = tmp_path / "state"
    control = {
        "namespace": "w12-guards",
        "source_commit": "f" * 40,
        "allocation": _allocations(12),
        "mac_gate": {
            "stage": "d1024",
            "package_root": str(tmp_path),
            "deadline_seconds": 144000,
            "state_dir": str(state),
        },
    }
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(control["namespace"], control["allocation"]))
    return mac_driver.run(
        control,
        ledger,
        Path("b8"),
        Path("frp3"),
        tmp_path / "output",
        workers=12,
        state_dir=state,
        owner_w12_ok=True,
        w12_calibration_receipt=Path("admission"),
        deadline_seconds=144000,
        timing_armed=tmp_path / "unarmed",
        started_monotonic=time.monotonic() - (10801 if sanity else 0),
    )


def test_w12_rss_soft_guard_seals_w6_continuation_and_exact_charge(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, rss=3 * 1024**3)
    assert result["status"] == "PREEMPTED"
    assert result["operational_width_fallback"] == "rss"
    assert result["admission_workers_final"] == 6
    assert result["worker_seconds_denominator"] == 12 * result["elapsed_seconds"]
    assert result["w12_sanity_checked"] is False


def test_first_w12_thermal_warning_seals_w6_and_trip_epoch(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, thermal=True)
    assert result["status"] == "PREEMPTED"
    assert result["operational_width_fallback"] == "thermal"
    assert result["admission_workers_final"] == 6
    assert len(result["thermal_trip_epochs"]) == 1


def test_w12_three_hour_parent_wall_rate_fallback(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, sanity=True)
    assert result["w12_sanity_checked"] is True
    assert result["operational_width_fallback"] == "sanity"
    assert result["admission_workers_final"] == 6
    event = next(
        x
        for x in result["telemetry_sealed_no_outcomes"]
        if x["event"] == "w12_parent_throughput_sanity"
    )
    assert event["required_ratio"] == 1.15
    assert event["wall_hours"] == 3


def test_no_warning_no_fallback(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch)
    assert result["status"] == "COMPLETE"
    assert result["operational_width_fallback"] is None
    assert result["worker_seconds_denominator"] == 12 * result["elapsed_seconds"]


def test_exactly_34_gib_is_allowed_and_one_byte_more_falls_back(tmp_path, monkeypatch):
    clean = _run(tmp_path / "clean", monkeypatch, total_rss=34 * 1024**3)
    assert clean["status"] == "COMPLETE"
    tripped = _run(tmp_path / "tripped", monkeypatch, total_rss=34 * 1024**3 + 1)
    assert tripped["operational_width_fallback"] == "rss"


def test_w6_wall_time_cannot_complete_or_fail_the_w12_sanity_window(tmp_path, monkeypatch):
    first = _run(tmp_path, monkeypatch)
    state = tmp_path / "state"
    control = {
        "namespace": "w12-guards",
        "source_commit": "f" * 40,
        "allocation": _allocations(12),
        "mac_gate": {
            "stage": "d1024",
            "package_root": str(tmp_path),
            "deadline_seconds": 144000,
            "state_dir": str(state),
        },
    }
    second = mac_driver.run(
        control,
        tmp_path / "ledger.json",
        Path("b8"),
        Path("frp3"),
        tmp_path / "w6-output",
        workers=6,
        state_dir=state,
        deadline_seconds=144000,
        timing_armed=tmp_path / "unarmed",
        started_monotonic=time.monotonic() - 10801,
    )
    assert second["status"] == "COMPLETE"
    assert second["w12_sanity_checked"] is False
    assert second["w12_sanity_elapsed_seconds"] == first["w12_sanity_elapsed_seconds"]
    assert second["w12_sanity_completed_parents"] == first["w12_sanity_completed_parents"]
    assert second["operational_width_fallback"] is None


@pytest.mark.parametrize(
    "other",
    [
        "/opt/bin/codex exec task",
        "python -m pytest tests",
        "python research/study/train.py",
    ],
)
def test_exclusive_factory_refuses_other_compute(monkeypatch, other):
    monkeypatch.setattr(mac_driver, "_command_output", lambda _: f"123 1 {other}\n")
    with pytest.raises(mac_driver.MacGuardError, match="concurrent compute pid 123"):
        mac_driver._exclusive_factory_guard(set())


def test_exclusive_factory_allows_owned_descendants_and_idle_os(monkeypatch):
    monkeypatch.setattr(
        mac_driver,
        "_command_output",
        lambda _: (
            f"123 {os.getpid()} python -m research.bl_labels.mac_driver\n"
            "124 123 python research/worker.py\n1 0 /sbin/launchd\n"
        ),
    )
    mac_driver._exclusive_factory_guard(set())


def test_guard_refusal_leaves_no_orphan_output_or_start(tmp_path, monkeypatch):
    _patch_inline_runtime(monkeypatch)
    monkeypatch.setattr(mac_gates, "require_admission", lambda _: None)
    monkeypatch.setenv("F0_COMMIT", "f" * 40)

    def refuse(_owned):
        raise mac_driver.MacGuardError("exclusive factory window")

    monkeypatch.setattr(mac_driver, "_exclusive_factory_guard", refuse)
    state = tmp_path / "state"
    output = tmp_path / "output"
    control = {
        "namespace": "guard-refusal",
        "allocation": _allocations(),
        "source_commit": "f" * 40,
        "mac_gate": {
            "stage": "d1024",
            "package_root": str(tmp_path),
            "deadline_seconds": 1200,
            "state_dir": str(state),
        },
    }
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(control["namespace"], control["allocation"]))
    with pytest.raises(mac_driver.MacGuardError, match="exclusive"):
        mac_driver.run(
            control,
            ledger,
            Path("b8"),
            Path("frp3"),
            output,
            workers=6,
            state_dir=state,
            deadline_seconds=1200,
        )
    assert not output.exists()
    assert not list(state.glob("attempt-*.start.json"))
