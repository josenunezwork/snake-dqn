"""Per-assignment excluded-world identity check before any selection gameplay."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from research.bl_served_search import ceiling
from research.bl_served_search.exports import digest, export_json, verify_export


def semantic(value: Any) -> Any:
    """Remove timing fields and checkpoint directory presentation only."""
    if isinstance(value, list):
        return [semantic(item) for item in value]
    if isinstance(value, dict):
        result = {
            key: semantic(item)
            for key, item in value.items()
            if not key.endswith(("_seconds", "_minutes"))
        }
        if "specs" in result:
            result["specs"] = [
                [kind, Path(name).name if kind == "checkpoint" else name]
                for kind, name in result["specs"]
            ]
        return result
    return value


def verify(reference: Path, actual: Path, bank: Path, out: Path) -> dict:
    """Require all three excluded worlds to match and bind this runtime identity."""
    values = []
    for path in (reference, actual):
        verify_export(path)
        value = json.loads(path.read_text())
        if (
            value.get("schema") != "bl-served-search-l2-fixture-shard/v1"
            or value.get("status") != "completed"
            or value.get("clean_exit") is not True
            or value.get("identity", {}).get("source_manifest") != ceiling.source_manifest()
            or value.get("identity", {}).get("bank_sha256") != digest(bank)
            or value.get("rows_sha256") != ceiling.digest_value(value.get("rows"))
        ):
            raise ValueError("invalid source/bank-bound excluded-world parity input")
        expected = ceiling.validate_bank(json.loads(bank.read_text()), fixture=True)
        if [row["world_id"] for row in value["rows"]] != [row["world_id"] for row in expected]:
            raise ValueError("parity must cover the three excluded worlds in order")
        values.append(semantic(value["rows"]))
    if values[0] != values[1]:
        raise ValueError("INVALID: Mac/runtime excluded-world semantic mismatch")
    result = {
        "schema": "bl-served-search-runtime-parity/v2",
        "status": "VERIFIED",
        "bank_sha256": digest(bank),
        "b8_sha256": ceiling.B8_SHA256,
        "source_manifest": ceiling.source_manifest(),
        "mac_reference_sha256": digest(reference),
        "runtime_verification_sha256": digest(actual),
        "runtime_identity": {
            "node": platform.node(),
            "machine": platform.machine(),
            "colab_cli_root": str(Path(os.environ["COLAB_CLI_ROOT"]).resolve()),
        },
        "selection_worlds": 0,
        "excluded_worlds": 3,
    }
    ceiling.validate_runtime_parity(
        result,
        bank_sha256=digest(bank),
        source=ceiling.source_manifest(),
        require_local_runtime=True,
    )
    export_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "actual", "bank", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.reference, args.actual, args.bank, args.out)))


if __name__ == "__main__":
    main()
