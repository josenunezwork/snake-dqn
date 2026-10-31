"""Mac-only outcome-blind union ledger builder from immutable completed unit exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.watch6_alllen_screen import protocol as p
from research.watch6_alllen_screen.run import validate_contract
from tools.ops.package_spec import empty_ledger, publish_json, unit_id, verify_immutable


def build(contract: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    """Build a verified resume ledger using only path, size, SHA256, and contract binding."""
    validate_contract(contract)
    ledger = empty_ledger(contract["namespace"], p.UNIT_KIND, p.contract_identity(contract))
    seen: set[int] = set()
    for root in roots:
        unit_files = sorted(root.glob("unit-*.json"))
        if not unit_files:
            continue
        contract_path = root / "contract.json"
        verify_immutable(contract_path)
        binding = json.loads(contract_path.read_text(encoding="utf-8"))
        if binding != contract:
            raise ValueError("foreign namespace/source/allocation in export root")
        for path in unit_files:
            stem = path.stem
            if not stem.startswith("unit-"):
                continue
            index = int(stem.split("-", 1)[1])
            if index in seen or not 0 <= index < contract["count"]:
                raise ValueError(f"duplicate or out-of-range allocation index: {index}")
            verified = verify_immutable(path)
            ledger["units"].append(
                {
                    "index": index,
                    "unit_id": unit_id(contract["namespace"], p.UNIT_KIND, index),
                    "size": verified["size"],
                    "sha256": verified["sha256"],
                }
            )
            seen.add(index)
    ledger["units"].sort(key=lambda row: row["index"])
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--roots", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    publish_json(args.out.resolve(), build(contract, [r.resolve() for r in args.roots]))


if __name__ == "__main__":
    main()
