"""Synthetic, outcome-free lifecycle rehearsal for the R33 package contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.ops.package_spec import (
    PackageRefused,
    canonical_identity,
    empty_ledger,
    frozen_identity,
    publish_json,
    read_ledger,
    unit_id,
    verify_immutable,
)

PIN = "0" * 64
ENVIRONMENT_SHA256 = "1" * 64
PACKET = "r33-linux-contract-pilot"


def _identity() -> dict[str, str]:
    return frozen_identity(pin=PIN, environment_sha256=ENVIRONMENT_SHA256, packet=PACKET)


def _payload(namespace: str, allocation: str, index: int) -> dict[str, Any]:
    return {
        "schema": "r33-synthetic-unit/v1",
        "unit_id": unit_id(namespace, allocation, index),
        "index": index,
        "identity": _identity(),
        "science": {"synthetic": True, "decision": "EXCLUDED"},
        "elapsed_seconds": 0.0,
    }


def export_unit(root: Path, namespace: str, allocation: str, index: int) -> dict[str, Any]:
    """Export one synthetic unit only if its immutable destination is absent."""
    path = Path(root) / "units" / f"unit-{index:06d}.json"
    if path.exists() or Path(f"{path}.sha256").exists() or Path(f"{path}.complete").exists():
        raise FileExistsError(path)
    publish_json(path, _payload(namespace, allocation, index))
    return {
        "index": index,
        "unit_id": unit_id(namespace, allocation, index),
        **verify_immutable(path),
    }


def build_verified_ledger(
    exports: Path, out: Path, namespace: str, allocation: str
) -> dict[str, Any]:
    """Build a Mac ledger from path/size/hash only; payloads remain unopened."""
    units: list[dict[str, Any]] = []
    for path in sorted((Path(exports) / "units").glob("unit-*.json")):
        # A crash before marker publication is an incomplete export, not an error
        # that could prevent a valid prefix from being resumed.
        if not Path(f"{path}.complete").is_file():
            continue
        index = int(path.stem.rsplit("-", 1)[1])
        evidence = verify_immutable(path)
        units.append(
            {
                "index": index,
                "unit_id": unit_id(namespace, allocation, index),
                "sha256": evidence["sha256"],
                "size": evidence["size"],
            }
        )
    if len({row["unit_id"] for row in units}) != len(units):
        raise PackageRefused("duplicate verified unit ids")
    ledger = empty_ledger(namespace, allocation, _identity())
    ledger["units"] = sorted(units, key=lambda row: row["index"])
    publish_json(out, ledger)
    return ledger


def consume(
    namespace: str, allocation: str, ledger_path: Path, count: int, out: Path
) -> dict[str, Any]:
    """Resume exact verified IDs and export only missing allocation shards."""
    if count < 1:
        raise PackageRefused("count must be positive")
    done = {row["index"] for row in read_ledger(ledger_path, namespace, allocation, _identity())}
    created = [
        export_unit(out, namespace, allocation, index)
        for index in range(count)
        if index not in done
    ]
    receipt = {
        "schema": "r33-synthetic-consumer/v1",
        "namespace": namespace,
        "allocation": allocation,
        "skipped": sorted(done),
        "created": [row["index"] for row in created],
        "status": "COMPLETE",
    }
    publish_json(Path(out) / "consumer-receipt.json", receipt)
    return receipt


def terminal(out: Path, namespace: str, allocation: str, count: int) -> dict[str, Any]:
    """Perform consumer/terminal admission from complete immutable exports."""
    ledger = build_verified_ledger(out, Path(out) / "verified-ledger.json", namespace, allocation)
    found = {row["index"] for row in ledger["units"]}
    if found != set(range(count)):
        raise PackageRefused("terminal refusal: missing or foreign verification shards")
    values = [
        json.loads((Path(out) / "units" / f"unit-{index:06d}.json").read_text())
        for index in range(count)
    ]
    for index, value in enumerate(values):
        canonical = canonical_identity(value)
        if canonical != {
            "identity": _identity(),
            "index": index,
            "schema": "r33-synthetic-unit/v1",
            "science": {"synthetic": True, "decision": "EXCLUDED"},
            "unit_id": unit_id(namespace, allocation, index),
        }:
            raise PackageRefused("terminal refusal: foreign or noncanonical synthetic unit")
    result = {
        "schema": "r33-terminal-receipt/v1",
        "status": "PASS",
        "namespace": namespace,
        "allocation": allocation,
        "verified_units": count,
        "next_task": {"status": "HOLD", "reason": "operator admission required"},
    }
    publish_json(Path(out) / "terminal-receipt.json", result)
    return result


def rehearse(
    root: Path, namespace: str = "r33-synthetic", allocation: str = "allocation-0001"
) -> dict[str, Any]:
    """Run a full tiny export→ledger→resume→consumer→terminal rehearsal."""
    root = Path(root)
    first = root / "first"
    export_unit(first, namespace, allocation, 0)
    # Interrupted after sidecar but before marker: it must not enter the ledger.
    partial = first / "units" / "unit-000001.json"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text('{"partial":true}\n')
    partial.with_name(partial.name + ".sha256").write_text("0" * 64 + "  unit-000001.json\n")
    initial_ledger = root / "initial-verified-ledger.json"
    ledger = build_verified_ledger(first, initial_ledger, namespace, allocation)
    if [row["index"] for row in ledger["units"]] != [0]:
        raise PackageRefused("partial export reached verified ledger")
    resumed = root / "resumed"
    receipt = consume(namespace, allocation, initial_ledger, 3, resumed)
    # The resume output receives a verified copy of the old payload before union.
    old = first / "units" / "unit-000000.json"
    destination = resumed / "units" / old.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".sha256", ".complete"):
        (destination.parent / f"{old.name}{suffix}").write_bytes(
            (old.parent / f"{old.name}{suffix}").read_bytes()
        )
    result = terminal(resumed, namespace, allocation, 3)
    return {"initial_ledger_units": len(ledger["units"]), "resume": receipt, "terminal": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-ledger", type=Path)
    args = parser.parse_args()
    if args.input_ledger is not None and not args.input_ledger.is_file():
        raise SystemExit("refuse: COLAB_INPUT_DIR ledger was not staged")
    print(json.dumps(rehearse(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
