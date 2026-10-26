"""Mac-only timing union and path/size/hash verified resume-ledger preparation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.sandbox_a_resident.units import (
    PHASES,
    allocation,
    digest,
    unit_id,
    verified_descriptor,
)


def allowed_path(path: Path) -> Path:
    resolved = path.resolve()
    if any(token in str(resolved).lower() for token in ("f5", "d4096", "t34-round0-study-01")):
        raise ValueError("protected input path refused before read")
    return resolved


def ledger(control: dict, directories: list[Path]) -> dict:
    """Read completion/hash metadata only, not checkpoint or timing contents."""
    found = {}
    for directory in directories:
        directory = allowed_path(directory)
        for worlds in control["worlds"]:
            for phase in PHASES:
                path = directory / f"e{worlds}-{phase}.tar"
                if not Path(str(path) + ".complete").exists():
                    continue
                identity = unit_id(worlds, phase)
                if identity in found:
                    raise ValueError("duplicate completed unit")
                found[identity] = verified_descriptor(path, identity)
                timing = verified_descriptor(path.with_suffix(".json"), identity)
                found[identity]["timing"] = timing
    return {
        "schema": "sandbox-a-ledger/v1",
        "namespace": "a100-dev/sandbox-a",
        "allocation": allocation(control),
        "units": list(found.values()),
    }


def aggregate(control: dict, receipts: list[Path]) -> dict:
    """The sole timing look: require verified terminal receipts and complete union."""
    expected = {unit_id(e, p) for e in control["worlds"] for p in PHASES}
    found, seen_sessions = {}, set()
    elapsed = 0.0
    for receipt_path in receipts:
        receipt_path = allowed_path(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") != "PASS"
            or receipt.get("stop_verified") is not True
            or receipt.get("error")
            or receipt.get("cleanup_error")
        ):
            raise ValueError("terminal PASS with independently verified release required")
        session = receipt.get("endpoint")
        if not session or session in seen_sessions:
            raise ValueError("missing/duplicate allocation identity")
        seen_sessions.add(session)
        seconds = receipt.get("elapsed_seconds")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("receipt elapsed time invalid")
        elapsed += seconds
        manifest = json.loads((receipt_path.parent / "manifest.json").read_text())
        if digest(receipt_path.parent / "manifest.json") != receipt.get("manifest_sha256"):
            raise ValueError("receipt manifest hash differs")
        exports = {row["path"]: row for row in receipt["exports"]}
        root = receipt_path.parent / "exports" / "assay"
        for name in ("summary.json", "parity.json"):
            row = exports.get(f"assay/{name}")
            path = root / name
            if row is None or digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
                raise ValueError("unverified summary/parity export")
        summary = json.loads((root / "summary.json").read_text())
        if (
            summary.get("allocation") != allocation(control)
            or summary.get("parity", {}).get("status") != "PASS"
            or summary.get("source_commit") != manifest["commit"]
        ):
            raise ValueError("foreign allocation/source or missing resume proof")
        for result in summary["results"]:
            identity = result["id"]
            if identity not in expected or identity in found:
                raise ValueError("duplicate/foreign result")
            stem = f"e{result['recipe']['worlds']}-{result['phase']}"
            for suffix in ("tar", "json"):
                path = root / f"{stem}.{suffix}"
                row = exports.get(f"assay/{stem}.{suffix}")
                if (
                    row is None
                    or digest(path) != row["sha256"]
                    or path.stat().st_size != row["size"]
                ):
                    raise ValueError("unverified result/checkpoint")
            found[identity] = result
    if set(found) != expected:
        raise ValueError("INCOMPLETE: missing complete fixed-cycle units")
    integrated_rows = sum(r["hero_rows"] for r in found.values() if r["phase"] == "integrated")
    return {
        "status": "PASS",
        "label": "simulator-D",
        "results": list(found.values()),
        "allocation_seconds_including_bootstrap_transfer_release": elapsed,
        "integrated_hero_rows_per_allocation_second": integrated_rows / elapsed,
        "decision_scope": (
            "engineering feasibility only; no efficacy/candidate selection; "
            "compare an independently qualified matched L4 path before chartering PPO"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ledger", "aggregate"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    control = json.loads(allowed_path(args.control).read_text())
    result = (
        ledger(control, args.inputs) if args.mode == "ledger" else aggregate(control, args.inputs)
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS_{args.mode.upper()} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
