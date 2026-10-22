"""Outcome-blind operational-snapshot ledger helpers for P1.

The ledger deliberately inspects only names, sizes and SHA-256 sidecars.  It
does not deserialize a snapshot, so making a completion decision cannot expose
an actor outcome.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-p1-verified-ledger/v1"


def unit_name(namespace: str, allocation: str, index: int) -> str:
    """Return the immutable operational payload name for one frozen index."""
    if not all(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
        for value in (namespace, allocation)
    ):
        raise ValueError("P1 operational namespace/allocation must be safe nonempty identifiers")
    if type(index) is not int or index < 1:
        raise ValueError("P1 operational snapshot index must be positive")
    return f"operational-{namespace}-{allocation}-{index}.pt"


def unit_id(namespace: str, allocation: str, index: int) -> str:
    """Derive the stable unit identity from the frozen allocation index."""
    unit_name(namespace, allocation, index)
    return f"{namespace}/{allocation}/{index}"


def _index_from_name(name: str, namespace: str, allocation: str) -> int | None:
    """Parse against the already-frozen identifiers, avoiding '-' ambiguity."""
    prefix = f"operational-{namespace}-{allocation}-"
    if not name.startswith(prefix) or not name.endswith(".pt"):
        return None
    tail = name[len(prefix) : -3]
    return int(tail) if re.fullmatch(r"[1-9][0-9]*", tail) else None


def empty_ledger(namespace: str, allocation: str) -> dict[str, Any]:
    """Create the first-attempt ledger without any inspected payloads."""
    unit_name(namespace, allocation, 1)
    return {"schema": SCHEMA, "namespace": namespace, "allocation": allocation, "units": []}


def read_ledger(path: Path, *, namespace: str, allocation: str) -> list[dict[str, Any]]:
    """Validate an uploaded verified ledger and return its ordered units."""
    unit_name(namespace, allocation, 1)
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unrecognized P1 verified ledger")
    if value.get("namespace") != namespace or value.get("allocation") != allocation:
        raise ValueError("P1 ledger namespace/allocation is foreign")
    units = value.get("units")
    if not isinstance(units, list):
        raise ValueError("P1 ledger units must be a list")
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for row in units:
        if not isinstance(row, dict):
            raise ValueError("P1 ledger unit is malformed")
        name, checksum, size, ident = (
            row.get("name"),
            row.get("sha256"),
            row.get("size"),
            row.get("unit_id"),
        )
        index = _index_from_name(name, namespace, allocation) if isinstance(name, str) else None
        if index is None:
            raise ValueError("P1 ledger contains a foreign snapshot allocation")
        if (
            ident != unit_id(namespace, allocation, index)
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise ValueError("P1 ledger unit identity or hash is malformed")
        if ident in seen:
            raise ValueError("P1 ledger has duplicate verified unit ids")
        seen.add(ident)
        checked.append(
            {"name": name, "sha256": checksum, "size": size, "unit_id": ident, "index": index}
        )
    return sorted(checked, key=lambda row: row["index"])


def build_ledger(root: Path, output: Path, *, namespace: str, allocation: str) -> dict[str, Any]:
    """Build a ledger from complete exports using only path, size and hash."""
    units: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("operational-*.pt")):
        index = _index_from_name(path.name, namespace, allocation)
        if index is None:
            raise ValueError("refusing foreign operational snapshot namespace/allocation")
        checksum = verify_export(path)
        units.append(
            {
                "name": path.name,
                "sha256": checksum,
                "size": path.stat().st_size,
                "unit_id": unit_id(namespace, allocation, index),
            }
        )
    if len({row["unit_id"] for row in units}) != len(units):
        raise ValueError("refusing duplicate operational snapshot ids")
    value = empty_ledger(namespace, allocation)
    value["units"] = sorted(units, key=lambda row: row["unit_id"])
    export_json(output, value)
    return value


def latest_verified(ledger: list[dict[str, Any]], directory: Path) -> Path | None:
    """Resolve only the latest Mac-verified operational export by its ledger row."""
    if not ledger:
        return None
    row = max(ledger, key=lambda item: item["index"])
    path = Path(directory) / row["name"]
    if verify_export(path) != row["sha256"] or path.stat().st_size != row["size"]:
        raise ValueError("P1 latest ledger snapshot does not match verified bytes")
    return path


def main() -> None:
    """Build a create-only Mac ledger without deserializing outcome payloads."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--allocation", required=True)
    args = parser.parse_args()
    value = build_ledger(
        args.exports, args.ledger, namespace=args.namespace, allocation=args.allocation
    )
    print(json.dumps({"ledger": str(args.ledger), "verified_units": len(value["units"])}))


if __name__ == "__main__":
    main()
