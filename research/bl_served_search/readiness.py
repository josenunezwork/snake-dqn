"""Fail closed on an over-cap forecast, or changed uploaded L2 authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_served_search.exports import digest, verify_export


def check(plan: dict, inputs: Path) -> None:
    """Offline preparation is not dispatch admission; no environment bypass."""
    if plan.get("schema") != "bl-served-search-plan/v1":
        raise ValueError("wrong L2 plan schema")
    for dest, checksum in plan.get("input_hashes", {}).items():
        relative = Path(dest)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe input destination")
        path = inputs / relative
        if digest(path) != checksum:
            raise ValueError(f"changed input: {dest}")
    for dest in ("bank.json", "audit.json"):
        verify_export(inputs / dest)
    bank = json.loads((inputs / "bank.json").read_text())
    audit = json.loads((inputs / "audit.json").read_text())
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != digest(inputs / "bank.json"):
        raise ValueError("bank/audit identity mismatch")
    if bank.get("counts") != {"frozen": 64, "mixed": 64, "scripted": 32}:
        raise ValueError("wrong L2 counts")
    if plan.get("runtime_parity_admitted") is not True:
        raise ValueError("NOT_DISPATCHABLE: L4 runtime parity has not been admitted")
    forecast = plan.get("forecast", {})
    estimate = forecast.get("conservative_allocation_minutes")
    timeout = forecast.get("timeout_minutes")
    if not isinstance(estimate, (int, float)) or not isinstance(timeout, (int, float)):
        raise ValueError("missing conservative forecast")
    if plan.get("dispatchable") is not True or forecast.get("status") != "FITS":
        raise ValueError("NOT_DISPATCHABLE: L4 forecast/preflight is not admitted")
    if not 0 < 2 * estimate <= timeout <= 384:
        raise ValueError("NOT_DISPATCHABLE: conservative forecast exceeds frozen envelope")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    check(json.loads(args.plan.read_text()), args.inputs)


if __name__ == "__main__":
    main()
