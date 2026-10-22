"""Exclusive, create-only lease used by scheduled Mac measurements.

The lease deliberately has no stale-steal path.  An expired lease is evidence
that requires an operator decision; callers must not replace it automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class MacLeaseError(RuntimeError):
    """Raised when the exclusive Mac lease cannot be safely used."""


REQUIRED_FIELDS = frozenset(("owner", "purpose", "until", "id"))


def _parse_until(value: object) -> datetime:
    if not isinstance(value, str):
        raise MacLeaseError("lease until must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MacLeaseError("lease until is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MacLeaseError("lease until must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validated_lease(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != REQUIRED_FIELDS:
        raise MacLeaseError("lease must contain exactly owner, purpose, until, id")
    lease = {key: value[key] for key in REQUIRED_FIELDS}
    if not all(isinstance(item, str) and item for item in lease.values()):
        raise MacLeaseError("lease fields must be non-empty strings")
    _parse_until(lease["until"])
    return lease


def _read(path: Path) -> dict[str, str]:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise MacLeaseError("Mac lease is absent") from error
    if stat.S_ISLNK(info.st_mode):
        raise MacLeaseError("refusing symlink Mac lease")
    if not stat.S_ISREG(info.st_mode):
        raise MacLeaseError("Mac lease is not a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        with os.fdopen(os.open(path, flags), "r", encoding="utf-8") as handle:
            return _validated_lease(json.load(handle))
    except OSError as error:
        raise MacLeaseError("could not safely read Mac lease") from error
    except json.JSONDecodeError as error:
        raise MacLeaseError("Mac lease is not valid JSON") from error


def acquire(path: Path | str, owner: str, purpose: str, until: str) -> dict[str, str]:
    """Atomically create a future exclusive lease and return its identity.

    Existing files, including expired leases, are never replaced.
    """
    if not owner or not purpose:
        raise MacLeaseError("lease owner and purpose must be non-empty")
    expires = _parse_until(until)
    if expires <= datetime.now(timezone.utc):
        raise MacLeaseError("lease until must be in the future")
    target = Path(path)
    payload = {
        "owner": owner,
        "purpose": purpose,
        "until": expires.isoformat().replace("+00:00", "Z"),
        "id": str(uuid.uuid4()),
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise MacLeaseError("Mac lease already exists; no stale auto-steal") from error
    except OSError as error:
        raise MacLeaseError("could not atomically create Mac lease") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The only file this helper may remove is the just-created, incomplete lease.
        target.unlink(missing_ok=True)
        raise
    return payload


def require(path: Path | str, owner: str, id: str) -> dict[str, str]:
    """Require a current lease belonging to ``owner`` and ``id``."""
    lease = _read(Path(path))
    if lease["owner"] != owner or lease["id"] != id:
        raise MacLeaseError("Mac lease is foreign")
    if _parse_until(lease["until"]) <= datetime.now(timezone.utc):
        raise MacLeaseError("Mac lease is expired")
    return lease


def release(path: Path | str, owner: str, id: str) -> dict[str, str]:
    """Remove only the matching lease; a foreign lease is never touched."""
    target = Path(path)
    lease = _read(target)
    if lease["owner"] != owner or lease["id"] != id:
        raise MacLeaseError("Mac lease is foreign")
    try:
        target.unlink()
    except OSError as error:
        raise MacLeaseError("could not release Mac lease") from error
    return lease


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("acquire", "require", "release"):
        item = commands.add_parser(command)
        item.add_argument("--path", required=True)
        item.add_argument("--owner", required=True)
        if command == "acquire":
            item.add_argument("--purpose", required=True)
            item.add_argument("--until", required=True)
        else:
            item.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            result = acquire(args.path, args.owner, args.purpose, args.until)
        elif args.command == "require":
            result = require(args.path, args.owner, args.id)
        else:
            result = release(args.path, args.owner, args.id)
    except MacLeaseError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI entrypoint.
    raise SystemExit(_main())
