"""Unit tests for the P1 -05 quiet-window evidence helpers."""

import json
from pathlib import Path

import pytest

from research.bl_ppo import p1_attempt2_renicer as renicer
from research.bl_ppo import p1_attempt2_window as window


def rows() -> list[dict[str, object]]:
    return [
        {
            "pid": 10,
            "ppid": 1,
            "nice": 0,
            "stat": "S",
            "start": "Thu Oct 23 12:00:00 2026",
            "command": "p1_attempt2_control supervisor",
        },
        {
            "pid": 11,
            "ppid": 10,
            "nice": 0,
            "stat": "S",
            "start": "Thu Oct 23 12:00:01 2026",
            "command": "colab",
        },
        {
            "pid": 12,
            "ppid": 11,
            "nice": 0,
            "stat": "S",
            "start": "Thu Oct 23 12:00:02 2026",
            "command": "python",
        },
        {
            "pid": 20,
            "ppid": 1,
            "nice": 10,
            "stat": "S",
            "start": "Thu Oct 23 12:00:03 2026",
            "command": "research/colab_cli",
        },
    ]


def test_descendants_and_renicer_follow_new_descendants_by_ppid() -> None:
    assert [row["pid"] for row in window.descendants(rows(), 10)] == [10, 11, 12]
    assert renicer.eligible(rows(), 10) == [20]


def test_renicer_pending_root_uses_arm_marker_and_exact_record_path(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    window.arm(arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "arm.json")
    record = tmp_path / "control-root.json"
    pending = rows()
    pending[0]["command"] = f"p1_attempt2_control {record.resolve()} persistent-shell"
    assert renicer.pending_roots(pending, root_records=[record], arm_path=arm) == {10}


def test_control_sample_rejects_nice15_child() -> None:
    bad = rows()
    bad[2]["nice"] = 15
    with pytest.raises(window.WindowError, match="absolute nice 0"):
        window.control_sample(root_pid=10, rows=bad, phase="during-staging")


def test_verify_drain_requires_terminal_result_lease_absent_and_no_tree(tmp_path: Path) -> None:
    result = tmp_path / "runtime-result.json"
    result.write_text(
        json.dumps({"status": "PREEMPTED", "mac_lease": {"released": True}, "cleanup_error": None})
    )
    arm = tmp_path / "arm"
    window.arm(arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "arm.json")
    value = window.verify_drain(
        factory_result=result,
        factory_root_pid=99,
        lease_path=tmp_path / "lease",
        arm_path=arm,
        rows=rows(),
    )
    assert value["factory_status"] == "PREEMPTED"
    with pytest.raises(window.WindowError, match="remains live"):
        window.verify_drain(
            factory_result=result,
            factory_root_pid=10,
            lease_path=tmp_path / "lease",
            arm_path=arm,
            rows=rows(),
        )


def test_verify_drain_rejects_current_factory_even_if_old_root_is_dead(tmp_path: Path) -> None:
    result = tmp_path / "runtime-result.json"
    result.write_text(json.dumps({"status": "PREEMPTED", "cleanup_error": None, "mac_lease": None}))
    arm = tmp_path / "arm"
    window.arm(arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "arm.json")
    current = rows() + [
        {
            "pid": 66,
            "ppid": 1,
            "nice": 10,
            "stat": "S",
            "start": "now",
            "command": "python -m research.bl_labels.mac_driver",
        }
    ]
    with pytest.raises(window.WindowError, match="actual factory"):
        window.verify_drain(
            factory_result=result,
            factory_root_pid=99,
            lease_path=tmp_path / "lease",
            arm_path=arm,
            rows=current,
        )


def test_arm_disarm_is_owner_bound_and_create_only(tmp_path: Path) -> None:
    arm = tmp_path / "mac-timing-armed"
    evidence = tmp_path / "arm.json"
    value = window.arm(arm_path=arm, owner="p1", deadline_seconds=10, evidence=evidence)
    assert arm.exists() and evidence.with_name("arm.json.sha256").exists()
    with pytest.raises(window.WindowError, match="already owned"):
        window.arm(arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "again.json")
    proof = tmp_path / "release.json"
    proof.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "endpoint": "fixture-endpoint",
                "job_id": "p1",
            }
        )
    )
    done = window.disarm(arm_path=arm, owner_token=value["owner_token"], release_proof=proof)
    assert done["kind"] == "disarm" and not arm.exists()


def test_control_registration_binds_marker_start_and_current_arm(tmp_path: Path) -> None:
    arm = tmp_path / "arm"
    armed = window.arm(
        arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "arm.json"
    )
    record = window.register_control(
        arm_path=arm, owner_token=armed["owner_token"], root_pid=10, rows=rows()
    )
    root_record = tmp_path / "root.json"
    root_record.write_text(json.dumps(record))
    assert window.read_registered_control(root_record=root_record, arm_path=arm, rows=rows()) == 10


def test_disarm_requires_actual_allocation_recovery_release_and_no_control_tree(tmp_path):
    arm = tmp_path / "arm"
    armed = window.arm(
        arm_path=arm, owner="p1", deadline_seconds=10, evidence=tmp_path / "armed.json"
    )
    root = tmp_path / "control-root-allocation-retry-1.json"
    root.write_text(json.dumps({"arm_sha256": window.sha256(arm), "root": {"pid": 10}}))
    proof = tmp_path / "release.json"
    proof.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": None,
                "endpoint": "retry",
                "job_id": "p1",
            }
        )
    )
    with pytest.raises(window.WindowError, match="disappear"):
        window.disarm(
            arm_path=arm,
            owner_token=armed["owner_token"],
            release_proof=proof,
            root_records=[root],
            rows=rows(),
        )
    with pytest.raises(window.WindowError, match="verified release"):
        window.disarm(
            arm_path=arm,
            owner_token=armed["owner_token"],
            release_proof=proof,
            root_records=[root],
            rows=[],
        )
    value = json.loads(proof.read_text())
    value["job_id"] = "p1-retry-1"
    proof.write_text(json.dumps(value))
    window.disarm(
        arm_path=arm,
        owner_token=armed["owner_token"],
        release_proof=proof,
        root_records=[root],
        rows=[],
    )
    assert not arm.exists()
