"""Focused safety tests for the scheduled Mac exclusive lease."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.bl_served_search import mac_lease


def _future(minutes: int = 5) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def test_acquire_require_and_matching_release(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    lease = mac_lease.acquire(path, "claude", "native ruler", _future())

    assert mac_lease.require(path, "claude", lease["id"]) == lease
    assert mac_lease.release(path, "claude", lease["id"]) == lease
    assert not path.exists()


def test_acquire_is_atomic_and_never_steals_expired_lease(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    original = {
        "owner": "other",
        "purpose": "existing",
        "until": "2000-01-01T00:00:00Z",
        "id": "old-id",
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(mac_lease.MacLeaseError, match="no stale auto-steal"):
        mac_lease.acquire(path, "claude", "native ruler", _future())
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_require_refuses_absent_foreign_and_expired_leases(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    with pytest.raises(mac_lease.MacLeaseError, match="absent"):
        mac_lease.require(path, "claude", "mine")

    lease = mac_lease.acquire(path, "other", "native ruler", _future())
    with pytest.raises(mac_lease.MacLeaseError, match="foreign"):
        mac_lease.require(path, "claude", lease["id"])
    path.unlink()
    path.write_text(
        json.dumps({**lease, "owner": "claude", "until": "2000-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    with pytest.raises(mac_lease.MacLeaseError, match="expired"):
        mac_lease.require(path, "claude", lease["id"])


def test_release_refuses_foreign_lease_without_touching_it(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    lease = mac_lease.acquire(path, "claude", "native ruler", _future())

    with pytest.raises(mac_lease.MacLeaseError, match="foreign"):
        mac_lease.release(path, "other", lease["id"])
    assert mac_lease.require(path, "claude", lease["id"]) == lease


def test_timezone_is_required_and_symlink_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mac-lease.json"
    with pytest.raises(mac_lease.MacLeaseError, match="timezone"):
        mac_lease.acquire(path, "claude", "native ruler", "2030-01-01T00:00:00")

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path.symlink_to(target)
    with pytest.raises(mac_lease.MacLeaseError, match="symlink"):
        mac_lease.require(path, "claude", "mine")
    with pytest.raises(mac_lease.MacLeaseError, match="no stale auto-steal"):
        mac_lease.acquire(path, "claude", "native ruler", _future())
    assert target.read_text(encoding="utf-8") == "{}"
