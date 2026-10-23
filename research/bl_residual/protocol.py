"""F3 complete-fit units, source bindings and outcome-blind resume ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export

B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
QUALIFIED_TREE = "553af3f692515b2bf66173356fe4d099acc96942"
QUALIFICATION_RECORD = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json"
)
NAMESPACE = "bl-residual-f3-20261023-v1"
DOSES = (256, 1024, 4096)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def allocation_hash(control: dict) -> str:
    """Bind all scientific controls; only resume metadata varies per attempt."""
    return canonical_sha(
        {k: v for k, v in control.items() if k not in ("allocation", "ledger_sha256")}
    )


def empty_ledger(namespace: str, allocation: str) -> dict[str, Any]:
    return {
        "schema": "bl-residual-ledger/v1",
        "namespace": namespace,
        "allocation": allocation,
        "units": [],
    }


def read_ledger(path: Path, namespace: str, allocation: str) -> dict[int, dict]:
    """Validate all three complete-fit identities before any input labels open."""
    value = json.loads(path.read_text())
    if any(
        value.get(k) != v for k, v in empty_ledger(namespace, allocation).items() if k != "units"
    ):
        raise ValueError("foreign ledger namespace/allocation")
    if not isinstance(value.get("units"), list):
        raise ValueError("ledger lacks units")
    rows = {}
    for row in value["units"]:
        index = row.get("index")
        sha = row.get("sha256")
        if (
            type(index) is not int
            or index not in range(3)
            or index in rows
            or row.get("unit_id") != f"{namespace}/{allocation}/{index}"
            or row.get("name") != f"fit-{index:04d}.pth"
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("malformed/duplicate/foreign fit ledger unit")
        rows[index] = row
    return rows


def build_ledger(roots: list[Path], output: Path, namespace: str, allocation: str) -> dict:
    """Read only identity receipts and path/size/hash; never torch payloads."""
    result = empty_ledger(namespace, allocation)
    seen = set()
    for root in roots:
        verify_export(root / "allocation.json")
        if json.loads((root / "allocation.json").read_text()) != {
            "namespace": namespace,
            "allocation": allocation,
        }:
            raise ValueError("foreign attempt")
        for path in sorted(root.glob("fit-*.pth")):
            if path.name not in {f"fit-{i:04d}.pth" for i in range(3)}:
                raise ValueError("foreign unit filename")
            if not Path(str(path) + ".complete").exists():
                continue
            index = int(path.stem.split("-")[1])
            if index in seen:
                raise ValueError("duplicate fit across attempts")
            seen.add(index)
            result["units"].append(
                {
                    "index": index,
                    "unit_id": f"{namespace}/{allocation}/{index}",
                    "name": path.name,
                    "sha256": verify_export(path),
                    "size": path.stat().st_size,
                    "path": str(path.resolve()),
                }
            )
    export_json(output, result)
    read_ledger(output, namespace, allocation)
    return result


def require_judgment(approval: Path, registration: Path, recipe_sha: str) -> dict:
    """No environment token or draft can authorize a factory-label fit."""
    review = json.loads(approval.read_text())
    if (
        registration.name == "F4_DRAFT.md"
        or "UNRATIFIED" in registration.read_text()
        or review.get("schema") != "bl-residual-f4-judgment/v1"
        or review.get("decision") != "APPROVE"
        or review.get("registration_sha256") != digest(registration)
        or review.get("recipe_sha256") != recipe_sha
    ):
        raise ValueError("F4_UNRATIFIED: factory-label fits forbidden")
    return {"review_sha256": digest(approval), "registration_sha256": digest(registration)}
