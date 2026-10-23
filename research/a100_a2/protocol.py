"""Frozen A2 allocation and immutable unit/ledger identities."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export

NAMESPACE = "a100-a2-20261022-v1"
SEEDS = (20261022, 20261023)
ENCODERS = ("c0", "s_random", "s_b8", "m_random")
TARGETS = ("d300", "hazard")
DOSES = (25, 50, 100)
HORIZONS = (30, 60, 100, 200, 300)
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def allocation(namespace: str = NAMESPACE) -> list[dict[str, Any]]:
    """The only admitted factorial, with deterministic frozen indices."""
    return [
        dict(index=i, table=t, encoder=e, target=h, dose=d, seed=s, fold=f)
        for i, (t, e, h, d, s, f) in enumerate(
            product(("t5", "a5"), ENCODERS, TARGETS, DOSES, SEEDS, range(5))
        )
    ]


def allocation_hash(bindings: dict[str, str], namespace: str = NAMESPACE) -> str:
    """Bind units to input hashes as well as factorial and registration bytes."""
    value = {"namespace": namespace, "units": allocation(), "bindings": bindings}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def unit_name(index: int) -> str:
    if type(index) is not int or not 0 <= index < len(allocation()):
        raise ValueError("foreign allocation index")
    return f"fit-{index:04d}.npz"


def empty_ledger(namespace: str, alloc: str) -> dict[str, Any]:
    return {
        "schema": "a2-verified-ledger/v1",
        "namespace": namespace,
        "allocation": alloc,
        "units": [],
    }


def read_ledger(path: Path, namespace: str, alloc: str) -> dict[int, dict[str, Any]]:
    """Consume a bound Mac ledger; no outcome or prediction is opened."""
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "a2-verified-ledger/v1"
        or value.get("namespace") != namespace
        or value.get("allocation") != alloc
    ):
        raise ValueError("foreign namespace/allocation in ledger")
    checked: dict[int, dict[str, Any]] = {}
    for row in value["units"]:
        i = row["index"]
        if i in checked or row["name"] != unit_name(i):
            raise ValueError("duplicate or foreign ledger unit")
        if (
            row.get("unit_id") != f"{namespace}/{alloc}/{i}"
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("malformed ledger unit")
        checked[i] = row
    return checked


def publish_fit(
    output: Path, index: int, values: dict[str, Any], namespace: str, alloc: str
) -> dict[str, Any]:
    """Publish a complete final-fit prediction atomically; incomplete fits never count."""
    import io

    import numpy as np

    stream = io.BytesIO()
    np.savez_compressed(stream, **values, namespace=namespace, allocation=alloc, index=index)
    path = output / unit_name(index)
    result = export_bytes(path, stream.getvalue())
    return {
        "index": index,
        "unit_id": f"{namespace}/{alloc}/{index}",
        "name": path.name,
        "size": result["size"],
        "sha256": result["sha256"],
    }


def build_ledger(roots: list[Path], output: Path, namespace: str, alloc: str) -> dict[str, Any]:
    """Hash-only verification, refusing duplicates and foreign payload filenames."""
    found: dict[int, dict[str, Any]] = {}
    allowed = {unit_name(i): i for i in range(len(allocation()))}
    for root in roots:
        # Unit namespace is bound by the immutable attempt inventory, not payload reads.
        receipt_path = root / "allocation.json"
        verify_export(receipt_path)
        binding = json.loads(receipt_path.read_text())
        if binding != {"namespace": namespace, "allocation": alloc}:
            raise ValueError("foreign attempt namespace/allocation")
        for path in sorted(root.glob("fit-*.npz")):
            if path.name not in allowed:
                raise ValueError("foreign fit filename")
            if not Path(str(path) + ".complete").exists():
                continue
            i = allowed[path.name]
            if i in found:
                raise ValueError("duplicate unit across attempts")
            sha = verify_export(path)
            found[i] = {
                "index": i,
                "unit_id": f"{namespace}/{alloc}/{i}",
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha,
                "path": str(path.resolve()),
            }
    value = empty_ledger(namespace, alloc)
    value["units"] = [found[i] for i in sorted(found)]
    export_json(output, value)
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--allocation", required=True)
    args = parser.parse_args()
    value = build_ledger(args.exports, args.out, args.namespace, args.allocation)
    print(json.dumps({"verified_units": len(value["units"]), "ledger_sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
