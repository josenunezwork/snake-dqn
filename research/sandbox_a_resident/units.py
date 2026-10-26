"""Immutable engineering units and outcome-blind resume ledger checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

NAMESPACE = "a100-dev/sandbox-a"
PHASES = ("cached", "collection", "integrated")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unit_id(worlds: int, phase: str) -> str:
    if worlds < 1 or phase not in PHASES:
        raise ValueError("foreign allocation")
    return f"{NAMESPACE}/e{worlds}/{phase}"


def allocation(control: dict[str, Any]) -> str:
    """Freeze recipe/endpoint/input identity, independent of attempt output path."""
    return hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_ledger(path: Path, control: dict[str, Any]) -> dict[str, dict]:
    """Mac-verified descriptors only; the runtime never opens prior numerical outputs."""
    value = json.loads(path.read_text())
    expected = {unit_id(e, p) for e in control["worlds"] for p in PHASES}
    if (
        value.get("schema") != "sandbox-a-ledger/v1"
        or value.get("namespace") != NAMESPACE
        or value.get("allocation") != allocation(control)
    ):
        raise ValueError("foreign ledger namespace/allocation")
    if not isinstance(value.get("units"), list):
        raise ValueError("ledger units must be an array")
    units: dict[str, dict] = {}
    for row in value.get("units", []):
        identity = row.get("id")
        if identity not in expected or identity in units:
            raise ValueError("foreign/duplicate verified unit")
        if type(row.get("size")) is not int or row["size"] < 1 or len(row.get("sha256", "")) != 64:
            raise ValueError("malformed verified unit descriptor")
        timing = row.get("timing")
        if (
            not isinstance(timing, dict)
            or type(timing.get("size")) is not int
            or timing["size"] < 1
            or len(timing.get("sha256", "")) != 64
        ):
            raise ValueError("complete unit requires verified timing and checkpoint descriptors")
        units[identity] = row
    return units


def publish(path: Path, writer: Callable[[Path], None]) -> dict[str, Any]:
    """Publish immutable payload, hash, then completion marker; no overwrite/retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if any(
        p.exists()
        for p in (path, temporary, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    ):
        raise FileExistsError(path)
    writer(temporary)
    checksum = digest(temporary)
    # Hard-link publication refuses racing writers, unlike overwrite-capable rename.
    os.link(temporary, path)
    temporary.unlink()
    Path(str(path) + ".sha256").write_text(checksum + "\n")
    Path(str(path) + ".complete").write_text("complete\n")
    return {"path": str(path), "size": path.stat().st_size, "sha256": checksum}


def verified_descriptor(path: Path, identity: str) -> dict[str, Any]:
    """Build a ledger entry by path/size/hash only, never reading tensor outcomes."""
    expected = Path(str(path) + ".sha256").read_text().strip()
    if not Path(str(path) + ".complete").is_file() or digest(path) != expected:
        raise ValueError("incomplete/corrupt unit")
    return {
        "id": identity,
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": expected,
    }
