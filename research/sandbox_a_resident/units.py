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


def unit_stem(worlds: int, phase: str, episode_limit: int | None = None) -> str:
    """Stable file identity; legacy controls retain their original names."""
    if (
        type(worlds) is not int
        or worlds < 1
        or phase not in PHASES
        or (episode_limit is not None and (type(episode_limit) is not int or episode_limit < 1))
    ):
        raise ValueError("foreign allocation")
    prefix = "" if episode_limit is None else f"ep{episode_limit}-"
    return f"{prefix}e{worlds}-{phase}"


def cell_specs(control: dict[str, Any]) -> list[tuple[int, str, int | None, int]]:
    """Enumerate disjoint fixed cells, preserving pre-A2 ledger identities."""
    limits = control.get("episode_limits", [None])
    if "episode_limits" in control and (
        not isinstance(limits, list) or any(type(limit) is not int or limit < 1 for limit in limits)
    ):
        raise ValueError("positive integer episode limits required")
    if (
        not isinstance(control["worlds"], list)
        or not control["worlds"]
        or any(type(worlds) is not int or worlds < 1 for worlds in control["worlds"])
    ):
        raise ValueError("positive integer worlds required")
    if (
        not limits
        or len(limits) != len(set(limits))
        or len(control["worlds"]) != len(set(control["worlds"]))
    ):
        raise ValueError("duplicate/empty cell sweep")
    base = control.get("seed_offset_base", 0)
    if type(base) is not int or base < 0:
        raise ValueError("nonnegative integer seed base required")
    return [
        (worlds, phase, limit, base + index * 1_000_000)
        for index, (limit, worlds, phase) in enumerate(
            (limit, worlds, phase)
            for limit in limits
            for worlds in control["worlds"]
            for phase in PHASES
        )
    ]


def unit_id(worlds: int, phase: str, episode_limit: int | None = None) -> str:
    if worlds < 1 or phase not in PHASES:
        raise ValueError("foreign allocation")
    unit_stem(worlds, phase, episode_limit)
    prefix = "" if episode_limit is None else f"/ep{episode_limit}"
    return f"{NAMESPACE}{prefix}/e{worlds}/{phase}"


def allocation(control: dict[str, Any]) -> str:
    """Freeze recipe/endpoint/input identity, independent of attempt output path."""
    return hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_ledger(path: Path, control: dict[str, Any]) -> dict[str, dict]:
    """Mac-verified descriptors only; the runtime never opens prior numerical outputs."""
    value = json.loads(path.read_text())
    expected = {unit_id(e, p, limit) for e, p, limit, _ in cell_specs(control)}
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
