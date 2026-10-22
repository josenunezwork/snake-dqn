"""Outcome-blind completed-unit bookkeeping for PLAN_v3 B2.

The Colab worker publishes one immutable trajectory export at a time.  This
module deliberately does not know the result schema: it selects roots only
from frozen coordinates and it builds/verifies ledgers from file identity
(relative path, byte size, and SHA-256) only.  The Mac reducer is the sole
consumer that may open unit payload JSON.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.bl_served_search.exports import export_json, verify_export

SCHEMA = "bl-plan-v3-b2-units/v1"
TRAJECTORIES = 32
MAX_ROOTS_PER_TRAJECTORY = 4
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_UNIT_ID = re.compile(r"^b2-[0-9a-f]{64}-trajectory-(?:[0-2][0-9]|3[0-1])$")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _index(value: object, name: str, *, upper: int | None = None) -> int:
    if type(value) is not int or value < 0 or (upper is not None and value >= upper):
        raise ValueError(f"{name} must be a non-boolean integer in range")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unit_namespace(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def unit_id(namespace: str, trajectory_index: int) -> str:
    """Return the frozen B2 trajectory unit ID for a namespace/index pair."""
    namespace = _nonempty_string(namespace, "namespace")
    trajectory_index = _index(trajectory_index, "trajectory_index", upper=TRAJECTORIES)
    return f"b2-{_unit_namespace(namespace)}-trajectory-{trajectory_index:02d}"


def root_hash(namespace: str, trajectory_index: int, life: int, frame: int) -> str:
    """Hash a root coordinate without consulting model predictions or outcomes."""
    namespace = _nonempty_string(namespace, "namespace")
    trajectory_index = _index(trajectory_index, "trajectory_index", upper=TRAJECTORIES)
    life = _index(life, "life")
    frame = _index(frame, "frame")
    encoded = "\x1f".join(
        ("PLAN_v3/B2/root/v1", namespace, str(trajectory_index), str(life), str(frame))
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _root_coordinates(row: Mapping[str, Any]) -> tuple[int, int, int]:
    if not isinstance(row, Mapping):
        raise ValueError("eligible root must be a mapping")
    return (
        _index(row.get("trajectory_index"), "trajectory_index", upper=TRAJECTORIES),
        _index(row.get("life"), "life"),
        _index(row.get("frame"), "frame"),
    )


def select_roots(
    eligible_rows: Iterable[Mapping[str, Any]],
    namespace: str,
    *,
    limit: int = MAX_ROOTS_PER_TRAJECTORY,
) -> list[dict[str, Any]]:
    """Choose hash-ranked roots from pre-filtered L>=800 coordinates.

    Callers establish eligibility before calling this function.  Only the
    coordinate triple is read; every other field, including predictions and
    outcomes, is intentionally ignored.  Duplicate coordinates are refused so
    stream order cannot bias which snapshot is retained.
    """
    namespace = _nonempty_string(namespace, "namespace")
    if type(limit) is not int or not 0 <= limit <= MAX_ROOTS_PER_TRAJECTORY:
        raise ValueError("limit must be a non-boolean integer from 0 through 4")
    ranked: list[tuple[str, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in eligible_rows:
        trajectory_index, life, frame = _root_coordinates(row)
        coordinate = (trajectory_index, life, frame)
        if coordinate in seen:
            raise ValueError("duplicate eligible root coordinate")
        seen.add(coordinate)
        ranked.append((root_hash(namespace, *coordinate), *coordinate))
    ranked.sort()
    return [
        {
            "trajectory_index": trajectory_index,
            "life": life,
            "frame": frame,
            "root_sha256": digest,
        }
        for digest, trajectory_index, life, frame in ranked[:limit]
    ]


def unit_payload_path(export_dir: Path, unit: str) -> Path:
    """Return the sole portable payload location allowed for a B2 unit."""
    if not isinstance(unit, str) or _UNIT_ID.fullmatch(unit) is None:
        raise ValueError("malformed B2 unit ID")
    return Path(export_dir) / "units" / f"{unit}.json"


def empty_ledger(namespace: str, allocation_digest: str) -> dict[str, Any]:
    """Make an empty, allocation-bound B2 resume ledger without filesystem I/O."""
    return {
        "schema": SCHEMA + "/ledger",
        "binding": {
            "namespace": _nonempty_string(namespace, "namespace"),
            "allocation_sha256": _digest(allocation_digest, "allocation_digest"),
        },
        "units": [],
    }


def _portable_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ledger path must be a nonempty portable relative path")
    path = Path(value)
    if path.is_absolute() or path.name != value.split("/")[-1] or "\\" in value:
        raise ValueError("ledger path must be portable and relative")
    parts = value.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError("ledger path traversal refused")
    return value


def verify_ledger(
    ledger: Mapping[str, Any], export_dir: Path, namespace: str, allocation_digest: str
) -> set[str]:
    """Return exactly the complete, allocation-bound units safe to skip.

    This is intentionally an identity check: it verifies create-only export
    sidecars, file sizes, and hashes but never parses result payload contents.
    """
    verified = validate_ledger(ledger, namespace, allocation_digest)
    base = Path(export_dir).resolve()
    for entry in ledger["units"]:
        unit = entry["unit_id"]
        relative = entry["path"]
        candidate = (base / relative).resolve()
        if candidate.parent != (base / "units").resolve() or candidate != unit_payload_path(
            base, unit
        ):
            raise ValueError("B2 unit path escapes export directory")
        if verify_export(candidate) != entry["sha256"]:
            raise ValueError("B2 unit digest mismatch")
        if candidate.stat().st_size != entry["size"]:
            raise ValueError("B2 unit size mismatch")
    return verified


def validate_ledger(ledger: Mapping[str, Any], namespace: str, allocation_digest: str) -> set[str]:
    """Validate a SHA-bound ledger without opening any completed unit payload.

    The runtime uses this form after the supervisor has uploaded the
    Mac-verified ledger.  It can therefore skip only the exact published unit
    IDs while remaining unable to interpret or inspect prior outcomes.
    """
    expected = empty_ledger(namespace, allocation_digest)["binding"]
    if not isinstance(ledger, Mapping) or ledger.get("schema") != SCHEMA + "/ledger":
        raise ValueError("invalid B2 ledger schema")
    if ledger.get("binding") != expected:
        raise ValueError("foreign B2 ledger namespace or allocation")
    entries = ledger.get("units")
    if not isinstance(entries, list):
        raise ValueError("ledger units must be a list")
    verified: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("ledger unit must be a mapping")
        unit = entry.get("unit_id")
        if (
            not isinstance(unit, str)
            or unit != unit_id(namespace, _unit_index(unit))
            or unit in verified
        ):
            raise ValueError("duplicate, foreign, or malformed B2 unit ID")
        relative = _portable_path(entry.get("path"))
        expected_relative = f"units/{unit}.json"
        if relative != expected_relative:
            raise ValueError("foreign B2 unit path")
        size = entry.get("size")
        sha256 = entry.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError("B2 unit size mismatch")
        _digest(sha256, "unit sha256")
        verified.add(unit)
    return verified


def _unit_index(unit: object) -> int:
    if not isinstance(unit, str) or _UNIT_ID.fullmatch(unit) is None:
        raise ValueError("malformed B2 unit ID")
    return int(unit.rsplit("-", 1)[1])


def make_ledger(
    export_dir: Path, namespace: str, allocation_digest: str, unit_ids: Iterable[str]
) -> dict[str, Any]:
    """Inventory completed unit exports by path/size/hash, without opening JSON."""
    ledger = empty_ledger(namespace, allocation_digest)
    unique = set()
    for unit in unit_ids:
        index = _unit_index(unit)
        if unit != unit_id(namespace, index) or unit in unique:
            raise ValueError("duplicate or foreign B2 unit ID")
        unique.add(unit)
        path = unit_payload_path(export_dir, unit)
        ledger["units"].append(
            {
                "unit_id": unit,
                "path": f"units/{path.name}",
                "size": path.stat().st_size,
                "sha256": verify_export(path),
            }
        )
    ledger["units"].sort(key=lambda row: row["unit_id"])
    verify_ledger(ledger, export_dir, namespace, allocation_digest)
    return ledger


def export_ledger(path: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Publish a previously verified ledger through the create-only export protocol."""
    if not isinstance(ledger, Mapping):
        raise ValueError("ledger must be a mapping")
    return export_json(Path(path), dict(ledger))
