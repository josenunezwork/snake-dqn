"""Mac-only publication wrapper for a completed E3 fit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_info_value.fit import decide_outcome
from research.bl_info_value.io import sha256, verify_export, write_json
from research.published_ledger import bind_published_ledger_check, require_published_ledger


def publish(fit_report: Path, out: Path, ledger_proof: dict[str, str]) -> dict[str, Any]:
    """Bind a compute-only E3 report to the authoring checkout's ledger proof."""
    verify_export(fit_report)
    value = json.loads(fit_report.read_text())
    if value.get("schema") != "bl-e3-fit-v1" or not isinstance(value.get("arms"), dict):
        raise ValueError("E3 fit report has no valid computed inputs")
    claim = {
        "schema": "bl-e3-decision-claim/v1",
        "fit_report_sha256": sha256(fit_report),
        "computed_decision": decide_outcome(value["arms"]),
        "development_only": value.get("development_only") is True,
    }
    claim = bind_published_ledger_check(claim, ledger_proof)
    write_json(out, claim)
    return claim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authoring-repo", type=Path, required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    # Refuse before opening the compute report, which may contain the sole look.
    proof = require_published_ledger(args.authoring_repo, args.ledger)
    print(json.dumps(publish(args.fit_report, args.out, proof), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
