"""Shared fail-closed Mac lease; no automatic expiry stealing."""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from research.bl_served_search import mac_lease

DEFAULT_LEASE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/mac-lease.json"
)


@contextmanager
def acquire(path: Path, owner: str, purpose: str, seconds: float) -> Iterator[dict[str, Any]]:
    """Atomically reserve the common lease, refusing even an expired incumbent."""
    now = dt.datetime.now(dt.timezone.utc)
    value = {
        "owner": owner,
        "purpose": purpose,
        "until": (now + dt.timedelta(seconds=seconds)).isoformat(),
        "id": secrets.token_hex(16),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete bytes atomically; no reader can observe partially written JSON.
    temporary = path.with_name(f".{path.name}.{value['id']}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        yield value
    finally:
        if not path.is_symlink() and path.is_file() and json.loads(path.read_text()) == value:
            mac_lease.release(path, owner, value["id"])


def require(path: Path, value: dict[str, Any]) -> None:
    """Attest continued ownership and the deadline before admitting a batch."""
    if path.is_symlink() or json.loads(path.read_text()) != value:
        raise RuntimeError("Mac lease changed or disappeared")
    if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(value["until"]):
        raise TimeoutError("Mac lease expired")
    mac_lease.require(path, value["owner"], value["id"])
