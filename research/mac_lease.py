"""Create-only, exclusive lease for Mac timing and paced work.

The lease is intentionally a small file rather than a process-local lock: it
also makes an operator's exclusive-window claim inspectable by later receipts.
All mutations are serialized through an adjacent advisory lock, while the lease
itself is always written with ``O_EXCL`` and removed only by its holder.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import sys
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


# A ``None`` default deliberately defers Git discovery until a Mac lease is
# actually acquired.  Colab/L4 runtime bundles import ruler modules without a
# ``.git`` directory and never need a Mac lease.
DEFAULT_LEASE_PATH: Path | None = None


def default_lease_path() -> Path:
    """Resolve the Mac-wide path only at the point a lease is needed."""
    return _primary_repo() / "exp" / "mac-lease.json"


def _resolve_path(path: Path | None) -> Path:
    return default_lease_path() if path is None else Path(path)


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


def _live_path(path: Path) -> Path:
    """Return the advisory fence retained for an active ``held`` context."""
    return path.with_name(path.name + ".live")


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
    until = value.get("until_epoch")
    if (
        isinstance(until, bool)
        or not isinstance(until, (int, float))
        or not math.isfinite(float(until))
    ):
        raise LeaseError(f"invalid existing Mac lease: {path}")
    return value


def _expired_record_has_live_holder(path: Path) -> bool:
    """Check an expired record's process-live fence without waiting.

    This runs while holding the mutation lock.  A successful nonblocking lock
    means no active ``held`` context retains the fence, so it is immediately
    released and the expired record may be reclaimed.
    """
    with _live_path(path).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def acquire(
    owner: str,
    purpose: str,
    *,
    seconds: float,
    path: Path | None = DEFAULT_LEASE_PATH,
    clock: Callable[[], float] = time.time,
    _live_fence_held: bool = False,
) -> MacLease:
    """Atomically acquire a lease, reclaiming only an expired immutable record."""
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(float(seconds))
        or seconds <= 0
        or not owner.strip()
        or not purpose.strip()
    ):
        raise ValueError("lease owner, purpose, and positive duration are required")
    path = _resolve_path(path)
    with _serialized(path):
        clock_value = clock()
        if isinstance(clock_value, bool) or not isinstance(clock_value, (int, float)):
            raise ValueError("lease clock must return a finite time")
        now = float(clock_value)
        if not math.isfinite(now):
            raise ValueError("lease clock must return a finite time")
        if path.exists():
            existing = _read(path)
            if float(existing["until_epoch"]) > now:
                raise LeaseHeld(
                    "Mac lease held by "
                    f"{existing.get('owner', 'unknown')} until {existing.get('until')} "
                    f"for {existing.get('purpose', 'unknown')} (id={existing['id']})"
                )
            if not _live_fence_held and _expired_record_has_live_holder(path):
                raise LeaseHeld(
                    "Mac lease remains live after its receipt expiry " f"(id={existing['id']})"
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
    path: Path | None = DEFAULT_LEASE_PATH,
) -> Iterator[MacLease]:
    """Hold one exclusive Mac lease for the complete timing operation."""
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _live_path(path).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            try:
                active_owner = _read(path).get("owner", "unknown")
            except LeaseError:
                active_owner = "unknown"
            raise LeaseHeld(f"Mac lease has an active process-live holder: {active_owner}") from exc
        try:
            lease = acquire(owner, purpose, seconds=seconds, path=path, _live_fence_held=True)
            try:
                yield lease
            finally:
                release(lease)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
