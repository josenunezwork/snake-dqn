"""Mac-only publication wrapper for completed BL prefix analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json, verify_export
from research.bl_prefix.analysis import e6_summary, e7_summary
from research.published_ledger import bind_published_ledger_check, require_published_ledger


def publish(result_path: Path, out: Path, ledger_proof: dict[str, str]) -> dict[str, Any]:
    """Publish a claim over an immutable compute-only E6/E7/E10 result."""
    verify_export(result_path)
    result = json.loads(result_path.read_text())
    schema = result.get("schema")
    if (
        schema
        not in {
            "bl-prefix-e6-results/v1",
            "bl-prefix-e7-results/v1",
            "bl-prefix-e10-results/v1",
        }
        or result.get("preflight") is True
    ):
        raise ValueError("prefix claim requires a completed non-preflight result")
    if schema == "bl-prefix-e6-results/v1":
        computed = {
            arm: e6_summary(
                result["rows"],
                arm,
                expected_early=result["expected_early"],
                expected_ordinary=result["expected_ordinary"],
            )
            for arm in ("R", "F")
        }
    elif schema == "bl-prefix-e7-results/v1":
        # Selection has one row per ancestor, while the audited bank has one
        # row per world. Preserve the producer's exact bank roster so missing
        # worlds cannot disappear from the completeness check.
        computed = e7_summary(result["confirmations"], result["expected_sources"])
    else:
        computed = result.get("envelope")
    claim = {
        "schema": "bl-prefix-analysis-claim/v1",
        "result_schema": schema,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "status": result.get("status"),
        "analysis": computed,
    }
    claim = bind_published_ledger_check(claim, ledger_proof)
    export_json(out, claim)
    return claim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authoring-repo", type=Path, required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    proof = require_published_ledger(args.authoring_repo, args.ledger)
    print(json.dumps(publish(args.result, args.out, proof), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
