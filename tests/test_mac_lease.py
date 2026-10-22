"""Mac-exclusive timing lease behavior."""

from __future__ import annotations

import json

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
