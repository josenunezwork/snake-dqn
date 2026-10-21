"""Fail closed on an over-cap forecast, or changed uploaded L2 authority."""

from __future__ import annotations

import argparse
import json
import math
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


def check(plan: dict, inputs: Path) -> None:
    """Offline preparation is not dispatch admission; no environment bypass."""
    if plan.get("schema") != "bl-served-search-plan/v2":
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
    from research.bl_served_search.ceiling import validate_bank
    from research.bl_served_search.chunks import chunk_rows

    if plan.get("chunk_count") != 4 or type(plan.get("chunk_index")) is not int:
        raise ValueError("four immutable chunks required")
    selected = chunk_rows(validate_bank(bank), plan["chunk_index"])
    if plan.get("world_ids") != [row["world_id"] for row in selected]:
        raise ValueError("chunk world order disagrees with frozen bank")
    if plan.get("runtime_parity_admitted") is not True:
        raise ValueError("NOT_DISPATCHABLE: L4 runtime parity has not been admitted")
    forecast = plan.get("forecast", {})
    estimate = forecast.get("conservative_allocation_minutes")
    timeout = forecast.get("timeout_minutes")
    if not isinstance(estimate, (int, float)) or not isinstance(timeout, (int, float)):
        raise ValueError("missing conservative forecast")
    if plan.get("dispatchable") is not True or forecast.get("status") != "FITS":
        raise ValueError("NOT_DISPATCHABLE: L4 forecast/preflight is not admitted")
    if not (
        math.isfinite(estimate)
        and math.isfinite(timeout)
        and 0 < 2 * estimate <= timeout <= 384
        and timeout <= 480
    ):
        raise ValueError("NOT_DISPATCHABLE: conservative forecast exceeds frozen envelope")
    rate = forecast.get("measured_l4_aggregate_batches_per_second")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate < 2.18
    ):
        raise ValueError("NOT_DISPATCHABLE: measured L4 R must be >=2.18")
    expected_minutes = (23120 / rate + 900) / 60
    if not math.isclose(estimate, expected_minutes, rel_tol=0, abs_tol=1e-9):
        raise ValueError("NOT_DISPATCHABLE: forecast does not price measured R")
    if forecast.get("total_two_times_minutes") != 8 * estimate or 8 * estimate > 1536:
        raise ValueError("NOT_DISPATCHABLE: aggregate timeout exceeds usable 64u envelope")
    if plan.get("ruler_admitted") is not True:
        raise ValueError("NOT_DISPATCHABLE: separate Mac and L4 ruler receipts required")
    for name in ("ruler-mac.json", "ruler-l4.json"):
        if name not in plan.get("input_hashes", {}):
            raise ValueError("NOT_DISPATCHABLE: ruler receipt must be hash-bound")
        verify_export(inputs / name)
    from research.bl_served_search.ruler import validate_qualification

    l4 = json.loads((inputs / "ruler-l4.json").read_text())
    validate_qualification(json.loads((inputs / "ruler-mac.json").read_text()), l4)
    from research.bl_served_search.ruler import canonical, sha256_bytes

    fixtures = fixture_bank(inputs, plan["input_hashes"])
    if l4["fixture_sha256"] != sha256_bytes(canonical(fixtures).encode()):
        raise ValueError("NOT_DISPATCHABLE: ruler differs from uploaded fixtures")
    if not math.isclose(rate, l4["aggregate_batches_per_second"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("NOT_DISPATCHABLE: forecast R differs from qualified L4 receipt")
    if "runtime-parity.json" not in plan.get("input_hashes", {}):
        raise ValueError("NOT_DISPATCHABLE: runtime parity must be hash-bound")
    verify_export(inputs / "runtime-parity.json")
    from research.bl_served_search.ceiling import source_manifest, validate_runtime_parity

    validate_runtime_parity(
        json.loads((inputs / "runtime-parity.json").read_text()),
        bank_sha256=digest(inputs / "bank.json"),
        source=source_manifest(),
        require_local_runtime=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    check(json.loads(args.plan.read_text()), args.inputs)


if __name__ == "__main__":
    main()
