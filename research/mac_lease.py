"""Create-only, exclusive lease for Mac timing and paced work.

The lease is intentionally a small file rather than a process-local lock: it
also makes an operator's exclusive-window claim inspectable by later receipts.
All mutations are serialized through an adjacent advisory lock, while the lease
itself is always written with ``O_EXCL`` and removed only by its holder.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator

WORKTREE_REPO = Path(__file__).resolve().parents[1]


def _primary_repo() -> Path:
    """Find the shared checkout behind this worktree's common Git directory."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=WORKTREE_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_path = Path(common)
    if common_path.name != ".git":
        raise RuntimeError(f"unexpected Git common directory: {common_path}")
    return common_path.parent


# Deliberately shared by every snake-dqn-* worktree on this Mac.
DEFAULT_LEASE_PATH = _primary_repo() / "exp" / "mac-lease.json"


class LeaseError(RuntimeError):
    """Base class for a lease admission failure."""


class LeaseHeld(LeaseError):
    """Raised when another unexpired lease owns the Mac."""


class LeaseOwnershipError(LeaseError):
    """Raised when a caller tries to release a lease it does not hold."""


@dataclass(frozen=True)
class MacLease:
    """An immutable lease record retained in timing receipts."""

    lease_id: str
    owner: str
    purpose: str
    until_epoch: float
    path: Path

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "id": self.lease_id,
            "owner": self.owner,
            "purpose": self.purpose,
            "until": datetime.fromtimestamp(self.until_epoch, UTC).isoformat(),
            "until_epoch": self.until_epoch,
            "path": str(self.path),
        }


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


@contextmanager
def _serialized(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_path(path).open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LeaseError(f"invalid existing Mac lease: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise LeaseError(f"invalid existing Mac lease: {path}")
    if not isinstance(value.get("until_epoch"), (int, float)):
        raise LeaseError(f"invalid existing Mac lease: {path}")
    return value


def acquire(
    owner: str,
    purpose: str,
    *,
    seconds: float,
    path: Path = DEFAULT_LEASE_PATH,
    clock: Callable[[], float] = time.time,
) -> MacLease:
    """Atomically acquire a lease, reclaiming only an expired immutable record."""
    if not owner.strip() or not purpose.strip() or seconds <= 0:
        raise ValueError("lease owner, purpose, and positive duration are required")
    path = Path(path)
    with _serialized(path):
        now = float(clock())
        if path.exists():
            existing = _read(path)
            if float(existing["until_epoch"]) > now:
                raise LeaseHeld(
                    "Mac lease held by "
                    f"{existing.get('owner', 'unknown')} until {existing.get('until')} "
                    f"for {existing.get('purpose', 'unknown')} (id={existing['id']})"
                )
            # The mutation lock prevents a second helper from replacing an
            # expired record between inspection and this unlink.
            path.unlink()
        until = now + float(seconds)
        lease = MacLease(uuid.uuid4().hex, owner, purpose, until, path)
        payload = {"schema": "mac-exclusive-lease/v1", **lease.receipt}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
    return lease


def release(lease: MacLease) -> None:
    """Release only the exact lease record acquired by this caller."""
    with _serialized(lease.path):
        if not lease.path.exists():
            raise LeaseOwnershipError(f"Mac lease is absent: {lease.path}")
        value = _read(lease.path)
        if value.get("id") != lease.lease_id or value.get("owner") != lease.owner:
            raise LeaseOwnershipError("refusing to release a different Mac lease")
        lease.path.unlink()


@contextmanager
def held(
    owner: str,
    purpose: str,
    *,
    seconds: float,
    path: Path = DEFAULT_LEASE_PATH,
) -> Iterator[MacLease]:
    """Hold one exclusive Mac lease for the complete timing operation."""
    lease = acquire(owner, purpose, seconds=seconds, path=path)
    try:
        yield lease
    finally:
        release(lease)
