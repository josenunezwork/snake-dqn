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
    assert value["native_mode_default"] == "stock"
    assert value["modes"]["A"] == {
        "search_mode": "exact_stop",
        "chunk_timeout_seconds": 142 * 60,
        "lease_hours": 28.3,
    }
    assert value["modes"]["B"] == {
        "search_mode": "all",
        "chunk_timeout_seconds": int(6.3 * 60 * 60),
        "lease_hours": 60.0,
    }
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
            "native_mode": "stock",
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
        mac_l2._operational_ok(
            operational,
            source={"a": "b"},
            lease_id="lease",
            shard_index=3,
            mode="A",
            native_mode="stock",
        )["shard_sha256"]
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
        "native_mode": "stock",
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
        mac_l2._operational_ok(
            path,
            source={"a": "b"},
            lease_id="lease",
            shard_index=0,
            mode="A",
            native_mode="stock",
        )


def test_source_bound_receipt_requires_pass_and_manifest(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    export_json(
        path,
        {
            "status": "READY_MODE_A",
            "source_manifest": {"a": "b"},
            "commit": commit,
            "schema": "bl-served-search-mac-identity/v1",
            "platform": "mac",
            "workers": 2,
            "threads": 1,
            "readiness": {
                "mode_a": {"search_mode": "exact_stop", "native_mode": "stock", "status": "READY"},
                "mode_b": {"search_mode": "all", "native_mode": "stock", "status": "READY"},
            },
            "permitted_search_modes": ["all", "exact_stop"],
            "execution_modes": {
                "all": {"search_mode": "all", "native_mode": "stock"},
                "exact_stop": {"search_mode": "exact_stop", "native_mode": "stock"},
            },
            "comparison": {
                "status": "PASS",
                "worlds": 5,
                "mix_counts": {"frozen": 2, "mixed": 2, "scripted": 1},
                "fired_batches_by_mix": {
                    "frozen": {"all": 1, "exact_stop": 1},
                    "mixed": {"all": 1, "exact_stop": 1},
                    "scripted": {"all": 1, "exact_stop": 1},
                },
            },
            "modes": ["all", "exact_stop"],
            "horizon": 5000,
            "b8_sha256": mac_l2.ceiling.B8_SHA256,
        },
    )
    value, _ = mac_l2._source_bound_receipt(path, "P3", {"a": "b"}, mode="A", native_mode="stock")
    assert value["status"] == "READY_MODE_A"
    with pytest.raises(mac_l2.AdmissionError):
        mac_l2._source_bound_receipt(path, "P3", {"a": "different"}, mode="A", native_mode="stock")


def test_p3_mode_b_requires_its_stock_readiness(tmp_path: Path) -> None:
    path = tmp_path / "p3.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    export_json(
        path,
        {
            "status": "READY_MODE_B",
            "source_manifest": {"a": "b"},
            "commit": commit,
            "schema": "bl-served-search-mac-identity/v1",
            "platform": "mac",
            "workers": 2,
            "threads": 1,
            "readiness": {
                "mode_a": {
                    "search_mode": "exact_stop",
                    "native_mode": "stock",
                    "status": "NOT_READY",
                },
                "mode_b": {"search_mode": "all", "native_mode": "stock", "status": "READY"},
            },
            "permitted_search_modes": ["all"],
            "execution_modes": {"all": {"search_mode": "all", "native_mode": "stock"}},
            "comparison": {
                "worlds": 5,
                "mix_counts": {"frozen": 2, "mixed": 2, "scripted": 1},
                "fired_batches_by_mix": {
                    "frozen": {"all": 1, "exact_stop": 0},
                    "mixed": {"all": 1, "exact_stop": 0},
                    "scripted": {"all": 1, "exact_stop": 0},
                },
            },
            "modes": ["all", "exact_stop"],
            "horizon": 5000,
            "b8_sha256": mac_l2.ceiling.B8_SHA256,
        },
    )
    assert mac_l2._source_bound_receipt(path, "P3", {"a": "b"}, mode="B", native_mode="stock")
    with pytest.raises(mac_l2.AdmissionError, match="selected-mode"):
        mac_l2._source_bound_receipt(path, "P3", {"a": "b"}, mode="A", native_mode="stock")
    with pytest.raises(mac_l2.AdmissionError, match="selected-mode"):
        mac_l2._source_bound_receipt(path, "P3", {"a": "b"}, mode="B", native_mode="kernel")


def test_review_receipt_requires_review_04(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    receipt = {
        "status": "PASS",
        "source_manifest": {"a": "b"},
        "commit": commit,
        "schema": "bl-l2-mac-independent-review/v1",
        "independent": True,
        "review_scope": "P1_MAC_L2",
        "review_id": "review-04",
    }
    export_json(path, receipt)
    assert mac_l2._source_bound_receipt(path, "review", {"a": "b"})
    rejected = tmp_path / "review-03.json"
    export_json(rejected, {**receipt, "review_id": "review-03"})
    with pytest.raises(mac_l2.AdmissionError, match="independent P1 review"):
        mac_l2._source_bound_receipt(rejected, "review", {"a": "b"})


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
    with pytest.raises(mac_l2.AdmissionError, match="P3 receipt is not PASS"):
        mac_l2._source_bound_receipt(path, "P3", {"a": "b"})
