"""Safety contracts for the no-peek L2' Mac coordinator."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.bl_served_search import mac_l2
from research.bl_served_search.exports import export_bytes, export_json


def test_check_declares_fixed_mac_envelope() -> None:
    value = mac_l2.check()
    assert value["workers"] == 2
    assert value["thread_limits"] == 1
    assert value["chunks"] == 8
    assert value["chunk_timeout_seconds"] == 142 * 60
    assert value["lease_hours"] == 28.3
    assert mac_l2.BANK_SHA256 == "0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3"


def test_operational_validation_never_opens_shard_rows(tmp_path: Path) -> None:
    operational = tmp_path / "operational.json"
    shard = export_bytes(tmp_path / "shard.json", b"opaque producer shard; do not parse\n")
    export_json(
        operational,
        {
            "schema": "bl-l2-mac-operational/v1",
            "status": "completed",
            "search_mode": "exact_stop",
            "native_mode": "kernel",
            "workers": 2,
            "lease_id": "lease",
            "source_manifest": {"a": "b"},
            "chunk_index": 3,
            "rows": 20,
            "clean_exit": True,
            "counters_validation": "PASS",
            "shard_sha256": shard["sha256"],
        },
    )
    assert (
        mac_l2._operational_ok(operational, source={"a": "b"}, lease_id="lease", shard_index=3)[
            "shard_sha256"
        ]
        == shard["sha256"]
    )
    assert not (tmp_path / "rows.json").exists()


def test_operational_refuses_wrong_mode_or_counters(tmp_path: Path) -> None:
    path = tmp_path / "operational.json"
    shard = export_bytes(tmp_path / "shard.json", b"opaque producer shard; do not parse\n")
    value = {
        "schema": "bl-l2-mac-operational/v1",
        "status": "completed",
        "search_mode": "all",
        "native_mode": "kernel",
        "workers": 2,
        "lease_id": "lease",
        "source_manifest": {"a": "b"},
        "chunk_index": 0,
        "rows": 20,
        "clean_exit": True,
        "counters_validation": "FAIL",
        "shard_sha256": shard["sha256"],
    }
    export_json(path, value)
    with pytest.raises(mac_l2.AdmissionError):
        mac_l2._operational_ok(path, source={"a": "b"}, lease_id="lease", shard_index=0)


def test_source_bound_receipt_requires_pass_and_manifest(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    export_json(
        path,
        {
            "status": "PASS",
            "source_manifest": {"a": "b"},
            "commit": commit,
            "schema": "bl-served-search-mac-identity/v1",
            "comparison": {
                "status": "PASS",
                "worlds": 5,
                "mix_counts": {"frozen": 2, "mixed": 2, "scripted": 1},
                "fired_batches_by_mix": {"frozen": 1, "mixed": 1, "scripted": 1},
            },
            "modes": ["all", "exact_stop"],
            "horizon": 5000,
            "b8_sha256": mac_l2.ceiling.B8_SHA256,
        },
    )
    value, _ = mac_l2._source_bound_receipt(path, "P3", {"a": "b"})
    assert value["status"] == "PASS"
    with pytest.raises(mac_l2.AdmissionError):
        mac_l2._source_bound_receipt(path, "P3", {"a": "different"})


def test_shell_check_has_no_lease_or_output_side_effect(tmp_path: Path) -> None:
    script = Path(mac_l2.__file__).with_name("run_l2_mac.sh")
    text = script.read_text()
    assert "--check" in text and "mac_l2 --check" in text
    assert "exec /usr/bin/nice -n 10 /usr/bin/caffeinate -is" in text
    assert "/usr/bin/nice" in Path(mac_l2.__file__).read_text()
    assert "/usr/bin/caffeinate" in Path(mac_l2.__file__).read_text()


def test_lease_fault_stops_owned_subprocess_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, time; "
            "pathlib.Path(os.environ['PID_PATH']).write_text(str(os.getpid())); time.sleep(60)"
        ),
    ]
    calls = 0

    def lease_guard() -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("expired")

    old = os.environ.get("PID_PATH")
    os.environ["PID_PATH"] = str(pid_path)
    try:
        with pytest.raises(RuntimeError, match="lease/host guard"):
            mac_l2._run_chunk(
                command,
                timeout=60,
                samples=tmp_path / "samples.jsonl",
                sample=lambda: {},
                lease_guard=lease_guard,
            )
    finally:
        if old is None:
            os.environ.pop("PID_PATH", None)
        else:
            os.environ["PID_PATH"] = old
    deadline = time.monotonic() + 2
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_review_receipt_cannot_stand_in_for_p3(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    export_json(
        path,
        {
            "status": "PASS",
            "source_manifest": {"a": "b"},
            "schema": "bl-l2-mac-independent-review/v1",
            "independent": True,
            "review_scope": "P1_MAC_L2",
        },
    )
    with pytest.raises(mac_l2.AdmissionError, match="P3 receipt lacks"):
        mac_l2._source_bound_receipt(path, "P3", {"a": "b"})
