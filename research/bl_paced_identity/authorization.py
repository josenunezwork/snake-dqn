"""Fail-closed evidence for bounded L1' measurement probes.

Probe authorization is deliberately distinct from the owner serving clause.  A
valid file may admit wider worker/pending cells and MPS only for L1' measurement
probes; it never grants serving authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import EventValidationError, canonical_bytes

SCHEMA = "bl-paced-identity-probe-authorization/v1"
SCOPE = "L1_PRIME"
MODE = "PROBE_ONLY"
_DEVICES = frozenset({"cpu", "mps"})
_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "authorized",
        "scope",
        "mode",
        "allowed_workers",
        "allowed_max_pending",
        "allowed_devices",
    }
)


@dataclass(frozen=True)
class ProbeAuthorization:
    """Validated, hash-bound authorization limited to an L1' probe envelope."""

    path: str
    sha256: str
    allowed_workers: int
    allowed_max_pending: int
    allowed_devices: tuple[str, ...]
    scope: str = SCOPE
    mode: str = MODE

    def permits(self, workers: int, max_pending: int, device: str) -> bool:
        """Return whether one concrete probe cell fits the recorded envelope."""
        return (
            type(workers) is int
            and type(max_pending) is int
            and workers >= 1
            and max_pending >= workers
            and workers <= self.allowed_workers
            and max_pending <= self.allowed_max_pending
            and device in self.allowed_devices
        )

    def evidence(self) -> dict[str, Any]:
        """Return the journal-safe identity of the authorization used."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "scope": self.scope,
            "mode": self.mode,
            "allowed_workers": self.allowed_workers,
            "allowed_max_pending": self.allowed_max_pending,
            "allowed_devices": list(self.allowed_devices),
        }


def _fail(message: str) -> None:
    raise EventValidationError(f"invalid probe authorization: {message}")


def _positive_int(value: Any, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(f"{field} must be in 1..{maximum}")
    return value


def load_probe_authorization(path: str | Path) -> ProbeAuthorization:
    """Load one canonical, explicit, probe-only authorization file.

    The file is intentionally a small exact schema so accidental serving
    authorizations, malformed JSON, and ambiguous extra capabilities fail
    closed before a transport can allocate a worker.
    """
    if not isinstance(path, (str, Path)) or not str(path):
        _fail("path is required")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail("path must name a regular non-symlink file")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventValidationError("invalid probe authorization: unreadable JSON") from exc
    if not isinstance(value, dict) or set(value) != _REQUIRED_KEYS:
        _fail("schema keys do not match")
    if canonical_bytes(value) != raw:
        _fail("file must be canonical JSON")
    if value["schema"] != SCHEMA or value["scope"] != SCOPE or value["mode"] != MODE:
        _fail("wrong schema, scope, or mode")
    if value["authorized"] is not True:
        _fail("authorized must be true")
    workers = _positive_int(value["allowed_workers"], "allowed_workers", 4)
    pending = _positive_int(value["allowed_max_pending"], "allowed_max_pending", 8)
    if workers > pending:
        _fail("allowed_workers cannot exceed allowed_max_pending")
    devices = value["allowed_devices"]
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, str) or item not in _DEVICES for item in devices)
    ):
        _fail("allowed_devices must contain cpu and/or mps")
    if len(set(devices)) != len(devices):
        _fail("allowed_devices must not repeat a device")
    return ProbeAuthorization(
        path=str(candidate.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        allowed_workers=workers,
        allowed_max_pending=pending,
        allowed_devices=tuple(devices),
    )
