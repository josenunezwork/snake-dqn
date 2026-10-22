"""Read-only, offline clock-evidence admission with create-only diagnostic receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.bl_paced_identity.record import export_json

from .loader import FileRef, clock_problems


def main() -> int:
    """Open a caller-pinned completed report; never run games, dispatch or release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--source-binding-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    problems = []
    try:
        report = FileRef.parse({"path": args.report.name, "sha256": args.report_sha256}).read(
            args.report.parent
        )
        problems = clock_problems(report, args.source_binding_sha256, args.candidate_sha256)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
        problems = [str(exc)]
    receipt = {
        "schema": "bl-evidence-clock-check/v1",
        "verdict": "NOT_READY" if problems else "PASS",
        "candidate_sha256": args.candidate_sha256,
        "source_binding_sha256": args.source_binding_sha256,
        "report_sha256": args.report_sha256,
        "problems": problems,
        "scope": "clock-evidence-only",
        "serving_or_release": "NOT_GRANTED",
    }
    export_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
