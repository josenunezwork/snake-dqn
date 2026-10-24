"""Outcome-blind complete-unit ledger construction; never launches retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export


def build(input_dir: Path, prior_outputs: list[Path], output: Path) -> dict:
    """Union verified path/size/hash evidence, refusing duplicate/foreign IDs.

    The operator supplies worker receipts with immutable identities; payload
    outcomes are neither deserialized nor used to choose missing units.
    """
    control = json.loads((input_dir / "control.json").read_text())
    namespace, allocation = control["namespace"], digest(input_dir / "control.json")
    units = []
    for root in prior_outputs:
        receipt_path = root / "allocation.json"
        verify_export(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema") != "r30-unit-identity/v1"
            or receipt.get("index") != 0
            or receipt.get("namespace") != namespace
            or receipt.get("allocation") != allocation
            or receipt.get("source_commit") != control["commit"]
        ):
            raise ValueError("foreign prior receipt")
        expected_stage = control.get("stage", "smoke")
        if (
            receipt.get("synthetic") != control["synthetic"]
            or receipt.get("stage") != expected_stage
        ):
            raise ValueError("canary/foreign stage cannot complete scientific allocation")
        payload = root / "unit-0000.pth"
        if payload.exists() and Path(str(payload) + ".complete").exists():
            sha = verify_export(payload)
            if units:
                raise ValueError("duplicate completed unit ID across attempts")
            units.append(
                {
                    "unit_id": f"{namespace}/{allocation}/0",
                    "index": 0,
                    "name": payload.name,
                    "sha256": sha,
                    "size": payload.stat().st_size,
                    "identity_sha256": digest(receipt_path),
                }
            )
            export_bytes(output.parent / payload.name, payload.read_bytes())
            export_bytes(output.parent / "allocation.json", receipt_path.read_bytes())
    value = {
        "schema": "r30-job-ledger/v1",
        "namespace": namespace,
        "allocation": allocation,
        "units": units,
    }
    export_json(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input_dir, args.prior, args.output)))


if __name__ == "__main__":
    main()
