"""Mac-exclusive timing lease behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research import mac_lease


def test_acquire_records_owner_purpose_id_and_expiry_then_releases(tmp_path) -> None:
    path = tmp_path / "mac-lease.json"
    lease = mac_lease.acquire("claude", "K-1 timing", seconds=120, path=path, clock=lambda: 100.0)
    value = json.loads(path.read_text())
    assert value["id"] == lease.lease_id
    assert value["owner"] == "claude"
    assert value["purpose"] == "K-1 timing"
    assert value["until_epoch"] == 220.0
    mac_lease.release(lease)
    assert not path.exists()


def test_conflicting_unexpired_lease_refuses_without_replacing_record(tmp_path) -> None:
    path = tmp_path / "mac-lease.json"
    original = mac_lease.acquire("first", "timing", seconds=100, path=path, clock=lambda: 10.0)
    with pytest.raises(mac_lease.LeaseHeld, match="first"):
        mac_lease.acquire("second", "other", seconds=10, path=path, clock=lambda: 11.0)
    assert json.loads(path.read_text())["id"] == original.lease_id


def test_expired_lease_is_reclaimed_and_previous_holder_cannot_release(tmp_path) -> None:
    path = tmp_path / "mac-lease.json"
    old = mac_lease.acquire("first", "timing", seconds=1, path=path, clock=lambda: 10.0)
    fresh = mac_lease.acquire("second", "ruler", seconds=10, path=path, clock=lambda: 12.0)
    assert fresh.lease_id != old.lease_id
    with pytest.raises(mac_lease.LeaseOwnershipError):
        mac_lease.release(old)
    mac_lease.release(fresh)


def test_malformed_lease_fails_closed(tmp_path) -> None:
    path = tmp_path / "mac-lease.json"
    path.write_text("not json")
    with pytest.raises(mac_lease.LeaseError, match="invalid existing"):
        mac_lease.acquire("claude", "timing", seconds=1, path=path)


@pytest.mark.parametrize("seconds", [True, float("nan"), float("inf"), float("-inf")])
def test_acquire_rejects_nonfinite_or_boolean_duration(tmp_path, seconds) -> None:
    with pytest.raises(ValueError, match="positive duration"):
        mac_lease.acquire("claude", "timing", seconds=seconds, path=tmp_path / "lease.json")


@pytest.mark.parametrize("clock", [lambda: True, lambda: float("nan"), lambda: float("inf")])
def test_acquire_rejects_nonfinite_or_boolean_clock(tmp_path, clock) -> None:
    with pytest.raises(ValueError, match="finite time"):
        mac_lease.acquire("claude", "timing", seconds=1, path=tmp_path / "lease.json", clock=clock)


@pytest.mark.parametrize("until", [True, float("nan"), float("inf"), float("-inf")])
def test_existing_nonfinite_or_boolean_expiry_fails_closed(tmp_path, until) -> None:
    path = tmp_path / "lease.json"
    path.write_text(json.dumps({"id": "old", "until_epoch": until}))
    with pytest.raises(mac_lease.LeaseError, match="invalid existing"):
        mac_lease.acquire("claude", "timing", seconds=1, path=path, clock=lambda: 1.0)


def test_held_context_keeps_live_fence_after_receipt_expiry(tmp_path) -> None:
    path = tmp_path / "lease.json"
    with mac_lease.held("first", "timing", seconds=0.02, path=path):
        time.sleep(0.03)
        with pytest.raises(mac_lease.LeaseHeld, match="live"):
            mac_lease.acquire("second", "other timing", seconds=1, path=path)
    recovered = mac_lease.acquire("second", "other timing", seconds=1, path=path)
    mac_lease.release(recovered)


def test_import_does_not_require_git_repository(tmp_path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import research.bl_served_search.ruler, research.bl_served_search.ruler_v2; "
            "print('import-safe')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-safe"
