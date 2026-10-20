"""Read-only local qualification of the draft package and excluded A6 receipt."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from research.t34_exit_b8.qualification import require_qualified_transport
from research.t34_round0.common import (
    B8_SHA,
    PACKAGE,
    ROOT,
    head,
    read_json,
    sha256,
    source_hashes,
    write_once,
)
from research.t34_round0.registration import check_audit


def validate(preflight_dir: Path | None = None, a5_receipt: Path | None = None) -> dict[str, Any]:
    """Check preparation without producing worlds, changing approval, or allocating."""
    bank_path, audit_path = PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json"
    bank = check_audit(bank_path, audit_path)
    prereg = (PACKAGE / "PREREGISTRATION_ROUND0.md").read_text()
    if "DRAFT_PENDING_OPERATOR_RATIFICATION" not in prereg:
        raise ValueError("draft preregistration status is absent")
    qualified = require_qualified_transport(ROOT, head())
    oc = read_json(PACKAGE / "OC_ROUND0.json")
    if (
        oc.get("no_gameplay") is not True
        or oc.get("reps", 0) < 20000
        or oc.get("simulator_sha256") != sha256(PACKAGE / "oc.py")
        or oc.get("source_pool_sha256")
        != sha256(ROOT / "research/survival_band_v2_20261006/survival_pools_20261006.json")
        or oc.get("protected_regression_check_pass") is not True
    ):
        raise ValueError("operating characteristic receipt is stale, incomplete, or failed")
    receipt = None
    if preflight_dir is not None:
        chunk = read_json(preflight_dir / "chunk.json")
        if (
            chunk.get("status") != "complete"
            or chunk.get("preflight") is not True
            or chunk.get("slice", {}).get("count") != 2
            or chunk.get("operational", {}).get("threads") != 1
            or chunk.get("operational", {}).get("device") != "cpu"
        ):
            raise ValueError("A6 is not a complete two-world CPU/one-thread preflight")
        if chunk.get("run_identity", {}).get("source_hashes") != source_hashes():
            raise ValueError("A6 executable source differs from current source")
        if chunk.get("run_identity", {}).get("checkpoint_sha256") != B8_SHA:
            raise ValueError("A6 did not run released B8 s3")
        pairs = chunk.get("pairs", [])
        if len(pairs) != 2:
            raise ValueError("A6 pair count differs")
        for item in pairs:
            pair_path = preflight_dir / item["path"]
            if sha256(pair_path) != item["sha256"]:
                raise ValueError("A6 pair hash differs")
            pair = read_json(pair_path)
            if pair.get("world", {}).get("horizon") != 500 or not pair.get("preflight"):
                raise ValueError("A6 contains selection data")
            for arm in ("bare", "teacher"):
                meta = pair["arms"][arm]["trace"]
                trace_path = preflight_dir / meta["path"]
                if sha256(trace_path) != meta["sha256"]:
                    raise ValueError("A6 trace hash differs")
                trace = read_json(trace_path)
                if trace.get("labels"):
                    raise ValueError("Round 0 exported fit labels")
        receipt = {
            "path": str(preflight_dir / "chunk.json"),
            "sha256": sha256(preflight_dir / "chunk.json"),
            "elapsed_seconds": chunk["elapsed_seconds"],
            "pairs": 2,
            "horizon": 500,
        }
    sizing = None
    if a5_receipt is not None:
        from research.t34_round0.make_manifest import size_round, validate_a5

        adapter = validate_a5(read_json(a5_receipt))
        from research.t34_round0.sizing import verify_adapter

        verify_adapter(adapter)
        if receipt is None or adapter["a6_measurement_sha256"] != receipt["sha256"]:
            raise ValueError("sizing adapter differs from this A6 receipt")
        sizing = size_round(
            240,
            **{
                k: adapter[k]
                for k in (
                    "teacher_worlds_per_hour",
                    "bare_seconds_per_world",
                    "metric_seconds_per_pair",
                    "cpu_slowdown_factor",
                    "setup_minutes",
                )
            },
        )
    return {
        "schema": "t34-round0-preparation-validation/v1",
        "status": "PASS",
        "commit": head(),
        "selection_worlds": len(bank["rows"]),
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
        "preregistration_sha256": hashlib.sha256(prereg.encode()).hexdigest(),
        "qualified_transport": qualified,
        "preflight": receipt,
        "draft_sizing": sizing,
        "source_hashes": source_hashes(),
        "ratified": False,
        "dispatch_authorized": False,
        "scientific_arms_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a5", type=Path)
    args = parser.parse_args()
    write_once(args.out, validate(args.preflight_dir, args.a5))


if __name__ == "__main__":
    main()
