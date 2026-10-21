"""Fail closed on an over-cap forecast, or changed uploaded L2 authority."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from research.bl_served_search.exports import digest, verify_export


def fixture_bank(inputs: Path, hashes: dict[str, str]) -> dict:
    """Read only hash-bound fixtures, including the single-upload transport."""
    from research.bl_served_search.fixture_bundle import BUNDLE_NAME, unpack

    if BUNDLE_NAME in hashes:
        with tempfile.TemporaryDirectory(prefix="l2-fixture-check-") as directory:
            manifest = unpack(
                inputs / BUNDLE_NAME, Path(directory) / "fixtures", hashes[BUNDLE_NAME]
            )
            return json.loads(manifest.read_text())
    fixture_dest = "ruler-fixtures/ruler-fixtures.json"
    if fixture_dest not in hashes:
        raise ValueError("NOT_DISPATCHABLE: qualified fixtures must be hash-bound")
    manifest = inputs / fixture_dest
    if digest(manifest) != hashes[fixture_dest]:
        raise ValueError("changed input: " + fixture_dest)
    return json.loads(manifest.read_text())


def check(plan: dict, inputs: Path, parity: Path | None = None) -> None:
    """Recompute every sizing/admission claim; no manifest flag bypasses evidence."""
    from research.bl_served_search.ceiling import (
        source_manifest,
        validate_bank,
        validate_runtime_parity,
    )
    from research.bl_served_search.chunks import CHUNK_COUNT, chunk_rows, world_ids_sha256
    from research.bl_served_search.pricing_v2 import fallback_open, forecast

    if plan.get("schema") != "bl-served-search-plan/v3":
        raise ValueError("wrong L2 plan schema")
    from research.bl_served_search.provenance_v2 import FROZEN_BANK_SHA256

    hashes = plan.get("input_hashes", {})
    for dest, checksum in hashes.items():
        relative = Path(dest)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe input destination")
        if digest(inputs / relative) != checksum:
            raise ValueError("changed input: " + dest)
    for dest in ("bank.json", "audit.json", "ruler-mac.json", "parity-mac.json"):
        if dest not in hashes:
            raise ValueError("NOT_DISPATCHABLE: evidence must be hash-bound: " + dest)
        verify_export(inputs / dest)
    if digest(inputs / "bank.json") != FROZEN_BANK_SHA256:
        raise ValueError("bank differs from unchanged BANK_L2.json")
    bank = json.loads((inputs / "bank.json").read_text())
    audit = json.loads((inputs / "audit.json").read_text())
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != digest(inputs / "bank.json"):
        raise ValueError("bank/audit identity mismatch")
    if plan.get("chunk_count") != CHUNK_COUNT or type(plan.get("chunk_index")) is not int:
        raise ValueError("eight immutable chunks required")
    selected = chunk_rows(validate_bank(bank), plan["chunk_index"])
    if plan.get("world_ids") != [row["world_id"] for row in selected] or plan.get(
        "world_ids_sha256"
    ) != world_ids_sha256(selected):
        raise ValueError("chunk order/hash disagrees with frozen bank")
    l4 = None
    if "ruler-l4.json" in hashes:
        verify_export(inputs / "ruler-l4.json")
        l4 = json.loads((inputs / "ruler-l4.json").read_text())
    expected = forecast(
        json.loads((inputs / "ruler-mac.json").read_text()), fixture_bank(inputs, hashes), l4
    )
    if plan.get("forecast") != expected:
        raise ValueError("NOT_DISPATCHABLE: forecast differs from receipt-derived price")
    if expected["status"] != "FITS" or plan.get("dispatchable") is not True:
        raise ValueError("NOT_DISPATCHABLE: forecast exceeds 12h/200u envelope")
    if not fallback_open(expected):
        raise ValueError("NOT_DISPATCHABLE: Mac sizing fallback opens Thu 12:00 PT only")
    if plan.get("analyzer_sha256") != digest(Path(__file__).with_name("analyze.py")):
        raise ValueError("pinned analyzer changed")
    if parity is None:
        raise ValueError("NOT_DISPATCHABLE: per-assignment runtime parity required")
    verify_export(parity)
    validate_runtime_parity(
        json.loads(parity.read_text()),
        bank_sha256=digest(inputs / "bank.json"),
        source=source_manifest(),
        require_local_runtime=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--parity", type=Path)
    args = parser.parse_args()
    check(json.loads(args.plan.read_text()), args.inputs, args.parity)


if __name__ == "__main__":
    main()
