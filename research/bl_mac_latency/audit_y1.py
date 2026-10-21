"""Reconcile Mac Y1 receipts with every immutable completed batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_mac_latency.ruler_run import distribution
from research.bl_served_search import ruler
from research.bl_served_search.exports import export_json, verify_export


def audit(directory: Path, fixtures_path: Path) -> dict[str, Any]:
    """Validate actual inventory and semantics without running another dense batch."""
    fixture_hash = verify_export(fixtures_path)
    fixtures = json.loads(fixtures_path.read_text())
    rows = ruler.validate_fixtures(fixtures)
    receipt_path = directory / "mac-ruler.json"
    receipt_hash = verify_export(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("platform", {}).get("system") != "Darwin":
        raise ValueError("not a Mac ruler")
    if receipt.get("source_identity") != fixtures["source_identity"]:
        raise ValueError("frozen source identity mismatch")
    if receipt.get("source_identity") != ruler.source_identity():
        raise ValueError("current source bytes differ")
    if receipt.get("fixture_sha256") != ruler.sha256_bytes(ruler.canonical(fixtures).encode()):
        raise ValueError("fixture payload mismatch")
    if receipt.get("completed_batches") != 270 or receipt.get("workers") != 1:
        raise ValueError("expected exactly270 batches on one CPU process")
    batches = []
    for path in sorted((directory / "batches").glob("batch-*.json")):
        verify_export(path)
        batches.append(json.loads(path.read_text()))
    if len(batches) != 270:
        raise ValueError("raw immutable batch inventory incomplete")
    metrics: dict[str, Any] = {}
    aggregate = {row["fixture_id"]: row["batches"] for row in receipt["records"]}
    for fixture in rows:
        name = fixture["fixture_id"]
        raw = [row for row in batches if row["fixture_id"] == name]
        if len(raw) != 30 or len(aggregate.get(name, [])) != 30:
            raise ValueError(f"{name}: incomplete fixture")
        for left, right in zip(raw, aggregate[name]):
            if left["output"]["semantic"] != right["semantic"]:
                raise ValueError(f"{name}: aggregate differs from raw batch")
            if left["output"]["semantic"].get("fired") is not True:
                raise ValueError(f"{name}: not a dense batch")
            if left["guard"]["reasons"]:
                raise ValueError("unsafe guard reading")
            seconds = left["operator_wall_seconds"]
            if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid timer")
        semantics = {ruler.canonical(row["output"]["semantic"]) for row in raw}
        if len(semantics) != 1:
            raise ValueError(f"{name}: repeated frozen root changed semantics")
        all30 = distribution([row["operator_wall_seconds"] for row in raw])
        warm29 = distribution([row["operator_wall_seconds"] for row in raw][1:])
        if receipt["latency_seconds"][name]["all30"] != all30:
            raise ValueError(f"{name}: aggregate quantiles differ")
        metrics[name] = {"all30": all30, "warm29": warm29, "semantics_stable": True}
    return {
        "schema": "bl-mac-y1-inventory-audit/v1",
        "status": "MAC_INVENTORY_PASS",
        "completed_batches": len(batches),
        "receipt_sha256": receipt_hash,
        "fixtures_file_sha256": fixture_hash,
        "operator_source_identity": receipt["source_identity"],
        "latency_seconds": metrics,
        "timing_platform": "this Mac, CPU W1",
        "cross_platform_semantic_acceptance": "NOT_EVALUATED_NO_L4_RECEIPT",
        "l1_qualification": "NOT_GRANTED",
        "serving_or_release": "NOT_GRANTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.fixtures)
    export_json(args.out, result)
    print(json.dumps({"status": result["status"], "completed": result["completed_batches"]}))


if __name__ == "__main__":
    main()
