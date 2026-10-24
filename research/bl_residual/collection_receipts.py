"""Outcome-blind D256/D1024 accounting and D4096 projection receipts.

Only sealed ledger metadata and terminal accounting JSON are read. In particular,
the ledger's unit paths, state directory and sealed rate paths are never opened.
The legacy selected_rescue_lives field records dose-defined increments, not an
observed rescue census: 256 + 768 = 1,024 selected lives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json, verify_export
from research.bl_residual.protocol import (
    F4_REGISTRATION_SHA,
    QUALIFIED_TREE,
    allocation_hash,
    canonical_sha,
)

START_JUDGMENT_SHA = "ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb"
COLLECTION_SCHEMA = "bl-residual-collection-receipt/v1"
PROJECTION_SCHEMA = "bl-residual-d4096-projection/v1"
# I1 owns the writer. This explicit interface is exercised by a sealed fixture
# until that writer is published; no production B2 receipt is manufactured here.
MAC_PROJECTION_SCHEMA = "bl-labels-f0-d4096-mac-projection/v1"
L4_ACCOUNTING_SCHEMA = "bl-residual-d4096-l4-accounting/v1"
SELECTED_INCREMENT = {256: 256, 1024: 768}
LEGACY_D256_DRIVER_SHA = "0694dcfb2dffd5229acb51b3b3164a7949a36a140e64023c99fc719cb0bfefc5"
REGISTERED_RATE_STOP = "sealed rolling six-hour rescue rate below 50 percent Canary-B baseline"


def finite_nonnegative(value: Any) -> bool:
    """Reject booleans, missing, non-finite and negative accounting values."""
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def read_metadata(path: Path) -> tuple[dict, str]:
    """Read a sealed JSON metadata object, without resolving any embedded path."""
    sha = verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("metadata object required")
    return value, sha


def write_collection_receipt(
    ledger: Path,
    ledger_sha256: str,
    stage_fit: Path,
    output: Path,
    *,
    fit_control: Path,
    dose: int,
    series_id: str,
    reserve_allocation_sha256: str,
) -> dict:
    """Seal Mac collection accounting; never load a ledger unit or fit payload."""
    if dose not in SELECTED_INCREMENT or not series_id or not sha256(reserve_allocation_sha256):
        raise ValueError("valid dose, series and reserve identity required")
    source, ledger_sha = read_metadata(ledger)
    fit, fit_receipt_sha = read_metadata(stage_fit)
    control, control_sha = read_metadata(fit_control)
    attempts = source.get("attempt_exports")
    allocation = source.get("allocation")
    namespace = source.get("namespace")
    if (
        ledger_sha != ledger_sha256
        or source.get("schema") != "bl-labels-f0-verified-ledger/v1"
        or not isinstance(namespace, str)
        or not namespace
        or not isinstance(allocation, list)
        or not allocation
        or not isinstance(source.get("units"), list)
        or not source["units"]
        or not isinstance(attempts, list)
        or not attempts
        or any(not isinstance(p, str) or not Path(p).is_absolute() for p in attempts)
        or len(set(attempts)) != len(attempts)
        or fit.get("schema") != "bl-residual-stage-fit/v1"
        or fit.get("dose") != dose
        or fit.get("complete") is not True
        or fit.get("fit_fault_free") is not True
        or not sha256(fit.get("fit_sha256"))
        or not sha256(fit.get("recipe_sha256"))
        or control.get("schema") != "bl-residual-control/v1"
        or control.get("synthetic") is not False
        or control.get("doses") != [dose]
        or control.get("series_id") != series_id
        or control.get("reserve_allocation_sha256") != reserve_allocation_sha256
        or control.get("registration_sha256") != F4_REGISTRATION_SHA
        or control.get("factory_union_ledger_sha256") != ledger_sha
        or control.get("recipe_sha256") != fit["recipe_sha256"]
        or allocation_hash(control) != control.get("allocation")
        or fit.get("allocation") != control["allocation"]
        or fit.get("namespace") != control.get("namespace")
        or fit.get("source_control_sha256") != control_sha
    ):
        raise ValueError("foreign collection ledger or unpassed stage fit")
    parents = set()
    indexes = set()
    for row in allocation:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("allocation_id"), str)
            or not row["allocation_id"]
            or "/" in row["allocation_id"]
            or row["allocation_id"] in parents
            or type(row.get("allocation_index")) is not int
            or row["allocation_index"] < 0
            or row["allocation_index"] in indexes
        ):
            raise ValueError("malformed or duplicate collection allocation metadata")
        parents.add(row["allocation_id"])
        indexes.add(row["allocation_index"])
    units = set()
    for row in source["units"]:
        name = row.get("unit_id") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or "/" in name
            or name in units
            or name.split("--life-", 1)[0].removesuffix("--session") not in parents
            or row.get("path") != f"lives/{name}.json"
            or not sha256(row.get("sha256"))
            or type(row.get("size")) is not int
            or row["size"] <= 0
        ):
            raise ValueError("foreign or duplicate collection unit metadata")
        units.add(name)
    elapsed = 0.0
    receipts = []
    terminal_names = []
    for index, attempt in enumerate(attempts):
        path = Path(attempt) / "runtime-result.json"
        terminal, terminal_sha = read_metadata(path)
        seconds = terminal.get("elapsed_seconds")
        name = terminal.get("start_receipt")
        if (
            terminal.get("schema") != "bl-labels-f0-mac-driver/v1"
            or terminal.get("namespace") != namespace
            or terminal.get("output") != attempt
            or terminal.get("status") != ("COMPLETE" if index == len(attempts) - 1 else "PREEMPTED")
            or terminal.get("invalid_reason") is not None
            or terminal.get("registered_stop_reason") is not None
            or any(
                terminal.get(k) == REGISTERED_RATE_STOP
                for k in ("stop_reason", "post_slowdown_stop_reason")
            )
            or terminal.get("platform_attestation", {}).get("platform") != "darwin-arm64"
            or type(terminal.get("workers")) is not int
            or terminal["workers"] not in ((6, 12) if dose == 256 else (6,))
            or not finite_nonnegative(seconds)
            or seconds == 0
            or terminal.get("prior_elapsed_seconds") != elapsed
            or not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("attempt-")
            or not name.endswith(".start.json")
            or (index == len(attempts) - 1 and terminal.get("pending_parents") != [])
        ):
            raise ValueError("nonterminal, foreign or invalid Mac collection accounting")
        # Mac is a zero-provider-unit placement, never an inferred unknown rate.
        if "charged_units" in terminal and (
            not finite_nonnegative(terminal["charged_units"]) or terminal["charged_units"] != 0
        ):
            raise ValueError("Mac collection has a nonzero or unknown unit charge")
        elapsed += seconds
        terminal_names.append(name.replace(".start.json", ".terminal.json"))
        cumulative = terminal.get("cumulative_elapsed_seconds")
        # The exact original D256 driver sampled the two clocks independently.
        # Preserve its charges under the resume tool's existing 1 ms bound.
        legacy_clock = (
            dose == 256
            and index == 0
            and terminal["workers"] == 6
            and terminal.get("platform_attestation", {}).get("driver_sha256")
            == LEGACY_D256_DRIVER_SHA
            and finite_nonnegative(cumulative)
            and abs(cumulative - elapsed) <= 0.001
        )
        if (cumulative != elapsed and not legacy_clock) or terminal.get(
            "attempts_ledger"
        ) != terminal_names:
            raise ValueError("omitted, duplicate or inconsistent cumulative collection accounting")
        receipts.append({"sha256": terminal_sha, "elapsed_seconds": seconds})
    value = {
        "schema": COLLECTION_SCHEMA,
        "series_id": series_id,
        "registration_sha256": F4_REGISTRATION_SHA,
        "reserve_allocation_sha256": reserve_allocation_sha256,
        "dose": dose,
        "fit_sha256": fit["fit_sha256"],
        "recipe_sha256": fit["recipe_sha256"],
        "stage_fit_receipt_sha256": fit_receipt_sha,
        "source_control_sha256": control_sha,
        "qualified_tree": QUALIFIED_TREE,
        "placement": "mac",
        "charged_units": 0,
        "selected_rescue_lives": SELECTED_INCREMENT[dose],
        "selected_count_basis": "dose-defined-increment-not-observed-rescues",
        "collection_ledger_sha256": ledger_sha,
        "collection_namespace": namespace,
        "collection_allocation_sha256": canonical_sha(allocation),
        "terminal_receipts": receipts,
        "cumulative_wall_seconds": elapsed,
        "metadata_only": True,
    }
    export_json(output, value)
    return value


def validate_mac_projection(value: dict, collections: list[dict]) -> None:
    """Verify I1's B2 interface against the exact two source collection ledgers."""
    walls = [c.get("cumulative_wall_seconds") for c in collections]
    if len(walls) != 2 or any(not finite_nonnegative(w) or w == 0 for w in walls):
        raise ValueError("B2 requires measured cumulative Mac wall time")
    projected = sum(walls) / 3600 / 1024 * 3072 * 1.2
    if (
        value.get("schema") != MAC_PROJECTION_SCHEMA
        or value.get("judgment_sha256") != START_JUDGMENT_SHA
        or value.get("placement") != "mac"
        or value.get("verdict") != "PASS"
        or value.get("selected_lives") != 1024
        or value.get("increment_lives") != 3072
        or value.get("margin") != 1.2
        or value.get("wall_cap_hours") != 130
        or value.get("collection_ledger_sha256")
        != {str(c["dose"]): c.get("collection_ledger_sha256") for c in collections}
        or any(not sha256(c.get("collection_ledger_sha256")) for c in collections)
        or value.get("cumulative_wall_seconds")
        != {str(c["dose"]): w for c, w in zip(collections, walls)}
        or not finite_nonnegative(value.get("projected_mac_hours"))
        or not math.isclose(value["projected_mac_hours"], projected, rel_tol=0, abs_tol=1e-9)
        or projected > 130
    ):
        raise ValueError("B2 Mac-h projection missing, foreign, inconsistent or HOLD")


def l4_accounting(value: dict | None, series_id: str) -> dict:
    """Keep L4 tail charges separate from the Mac R6.4 projection (R7)."""
    rows = [] if value is None else value.get("attempts")
    if value is not None and (
        value.get("schema") != L4_ACCOUNTING_SCHEMA
        or value.get("series_id") != series_id
        or value.get("registration_sha256") != F4_REGISTRATION_SHA
        or value.get("dose") != 4096
    ):
        raise ValueError("foreign D4096 L4 accounting")
    if not isinstance(rows, list):
        raise ValueError("L4 attempts required")
    seen = set()
    total = 0.0
    for row in rows:
        if (
            not isinstance(row, dict)
            or not sha256(row.get("receipt_sha256"))
            or row["receipt_sha256"] in seen
            or not finite_nonnegative(row.get("charged_units"))
        ):
            raise ValueError("unknown, duplicate or invalid L4 charge")
        seen.add(row["receipt_sha256"])
        total += row["charged_units"]
    if not finite_nonnegative(total) or total > 960:
        raise ValueError("D4096 L4 R7 effective cap exceeded")
    return {
        "base_cap_units": 480,
        "effective_cap_units": 960,
        "charged_units": total,
        "attempts": rows,
    }


def write_projection_receipt(
    d256: Path, d1024: Path, mac_projection: Path, output: Path, *, l4_ledger: Path | None = None
) -> dict:
    """Produce the measured-unit projection with an independently bound B2 receipt."""
    collections, hashes = zip(*(read_metadata(p) for p in (d256, d1024)))
    collections = list(collections)
    first = collections[0]
    for c, dose in zip(collections, (256, 1024)):
        if (
            c.get("schema") != COLLECTION_SCHEMA
            or c.get("dose") != dose
            or c.get("series_id") != first.get("series_id")
            or not isinstance(c.get("series_id"), str)
            or not c["series_id"]
            or c.get("registration_sha256") != F4_REGISTRATION_SHA
            or c.get("reserve_allocation_sha256") != first.get("reserve_allocation_sha256")
            or not sha256(c.get("reserve_allocation_sha256"))
            or c.get("recipe_sha256") != first.get("recipe_sha256")
            or not sha256(c.get("recipe_sha256"))
            or not sha256(c.get("fit_sha256"))
            or c.get("qualified_tree") != QUALIFIED_TREE
            or c.get("placement") != "mac"
            or not finite_nonnegative(c.get("charged_units"))
            or c["charged_units"] != 0
            or type(c.get("selected_rescue_lives")) is not int
            or c["selected_rescue_lives"] != SELECTED_INCREMENT[dose]
        ):
            raise ValueError("foreign, unmeasured or non-Mac collection receipt")
    b2, b2_sha = read_metadata(mac_projection)
    validate_mac_projection(b2, collections)
    tail, tail_sha = read_metadata(l4_ledger) if l4_ledger is not None else (None, None)
    value = {
        "schema": PROJECTION_SCHEMA,
        "series_id": first["series_id"],
        "registration_sha256": F4_REGISTRATION_SHA,
        "reserve_allocation_sha256": first["reserve_allocation_sha256"],
        "recipe_sha256": first["recipe_sha256"],
        "d256_fit_sha256": first["fit_sha256"],
        "d1024_fit_sha256": collections[1]["fit_sha256"],
        "qualified_placement_tree": QUALIFIED_TREE,
        "qualified_placement_receipts": [b2_sha],
        "mac_projection_receipt_sha256": b2_sha,
        "collection_receipt_sha256": list(hashes),
        "measurements": [
            {k: c[k] for k in ("dose", "fit_sha256", "charged_units", "selected_rescue_lives")}
            for c in collections
        ],
        "projected_units": 0,
        "verdict": "PASS",
        "l4_tail_accounting": l4_accounting(tail, first["series_id"]),
        "l4_tail_accounting_sha256": tail_sha,
        "metadata_only": True,
    }
    export_json(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collection = sub.add_parser("collection")
    collection.add_argument("--ledger", type=Path, required=True)
    collection.add_argument("--ledger-sha256", required=True)
    collection.add_argument("--stage-fit", type=Path, required=True)
    collection.add_argument("--fit-control", type=Path, required=True)
    collection.add_argument("--dose", type=int, choices=(256, 1024), required=True)
    collection.add_argument("--series-id", required=True)
    collection.add_argument("--reserve-allocation-sha256", required=True)
    projection = sub.add_parser("projection")
    projection.add_argument("--d256", type=Path, required=True)
    projection.add_argument("--d1024", type=Path, required=True)
    projection.add_argument("--mac-projection", type=Path, required=True)
    projection.add_argument("--l4-ledger", type=Path)
    for p in (collection, projection):
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collection":
        value = write_collection_receipt(
            args.ledger,
            args.ledger_sha256,
            args.stage_fit,
            args.out,
            fit_control=args.fit_control,
            dose=args.dose,
            series_id=args.series_id,
            reserve_allocation_sha256=args.reserve_allocation_sha256,
        )
    else:
        value = write_projection_receipt(
            args.d256, args.d1024, args.mac_projection, args.out, l4_ledger=args.l4_ledger
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
