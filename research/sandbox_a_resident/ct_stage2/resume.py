"""Build a Mac-verified downstream ledger after an outcome-blind mechanical stop.

This prepares inputs only. A future completion packet still needs separately
sealed provenance, current-pin canary, full admission and Argon review; this
command neither dispatches nor makes the initial empty-ledger packet mutable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.ct_resume import (
    _check_poll_fault_evidence,
    _check_retained_receipt,
    _check_terminal_fault_evidence,
)
from research.sandbox_a_resident.ct_stage2.units import build_ledger, load_ledger
from research.sandbox_a_resident.units import digest


def verify_prior(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Retain qualified mechanical/fault precedence, checking every retained poll."""
    receipt_path, manifest_path = output / "receipt.json", output / "manifest.json"
    receipt, manifest = (json.loads(p.read_text()) for p in (receipt_path, manifest_path))
    _check_retained_receipt(receipt, label=str(receipt_path))
    if (
        receipt.get("commit") != config["source_commit"]
        or receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("manifest_sha256") != digest(manifest_path)
        or json.loads(manifest["env"]["SANDBOX_CT_CONTROL_JSON"]) != config
    ):
        raise ValueError("prior mechanical receipt/manifest/config identity differs")
    exports = output / "exports/ct"
    if any(exports.rglob("fault.json*")) or (exports / "summary.json.complete").exists():
        raise ValueError("completed scientific result or scientific fault forbids completion")
    for poll in sorted(output.glob("poll-*.json")):
        _check_poll_fault_evidence(json.loads(poll.read_text()), label=str(poll))
    terminal = output / "terminal.json"
    if terminal.exists():
        _check_terminal_fault_evidence(json.loads(terminal.read_text()), label=str(terminal))
    return {
        "receipt_sha256": digest(receipt_path),
        "manifest_sha256": digest(manifest_path),
        "output": str(output.resolve()),
        "job_id": receipt["job_id"],
    }


def materialize_inputs(config: dict[str, Any], roots: list[Path], target: Path) -> dict[str, Any]:
    """Verify union before create-only copying all complete stage sidecars for upload."""
    ledger = build_ledger(config, roots)
    with tempfile.TemporaryDirectory(prefix="ct-stage2-ledger-") as temp:
        staging = Path(temp)
        (staging / "resume").mkdir()
        for row in ledger["units"]:
            for role in ("payload", "header"):
                desc = row[role]
                name = Path(desc["path"]).name
                owner = next(
                    r for r in roots if (r / name).is_file() and digest(r / name) == desc["sha256"]
                )
                for suffix in ("", ".sha256", ".complete"):
                    shutil.copyfile(owner / (name + suffix), staging / "resume" / (name + suffix))
        path = staging / "verified-ledger.json"
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        load_ledger(path, config, staging)
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(staging, target)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.prior_output) <= 2:
        raise ValueError("one original plus at most two manually prepared completion attempts")
    config = json.loads(args.config.read_text())
    provenance = [verify_prior(p, config) for p in args.prior_output]
    ledger = materialize_inputs(config, [p / "exports/ct" for p in args.prior_output], args.output)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"VERIFIED_DOWNSTREAM_LEDGER units={len(ledger['units'])} allocations=0")


if __name__ == "__main__":
    main()
