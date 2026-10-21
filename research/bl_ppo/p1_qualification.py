"""Seal measured P1 complete-job pricing only after supervisor release proof."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_manifests import QUAL_PAUSE, complete_json, measured_plan


def terminal(path: Path) -> dict[str, Any]:
    """Require success and verified release, not a CLI return code."""
    row = complete_json(path)
    if (
        row.get("status") != "PASS"
        or row.get("stop_verified") is not True
        or row.get("cleanup_error") is not None
        or row.get("error") is not None
        or row.get("terminal", {}).get("status") != "succeeded"
        or row.get("terminal", {}).get("returncode") != 0
    ):
        raise ValueError("P1 pricing requires a successful terminal with verified release")
    seconds = row.get("elapsed_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("supervisor receipt lacks measured whole-session seconds")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid whole-session seconds")
    return row


def finalize(
    *,
    runtime: Path,
    supervisor: Path,
    y3: Path,
    source_supervisor: Path,
    replacement_supervisor: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind runtime timing to actual exports and all three released assignments."""
    run = complete_json(runtime)
    qual_session = terminal(supervisor)
    source = terminal(source_supervisor)
    replacement = terminal(replacement_supervisor)
    durability = complete_json(y3)
    if durability.get("schema") != "bl-p1-y3/v1" or durability.get("status") != "PASS":
        raise ValueError("P1 pricing requires Y3 PASS")
    if not source.get("endpoint") or source.get("endpoint") == replacement.get("endpoint"):
        raise ValueError("Y3 source and replacement must be distinct assignments")
    if len({s.get("commit") for s in (source, replacement, qual_session)}) != 1:
        raise ValueError("P1 qualification source commit differs across assignments")
    exports = qual_session.get("exports", [])
    if not any(
        item.get("sha256") == digest(runtime) and Path(item.get("path", "")).name == runtime.name
        for item in exports
    ):
        raise ValueError("qualification runtime receipt is not bound by supervisor exports")
    if (
        run.get("schema") != "bl-p1-qualification-runtime/v1"
        or run.get("status") != "PAUSED"
        or run.get("critic_qualified") is not True
        or run.get("actor_transitions") != QUAL_PAUSE
        or run.get("target") != 100_000_000
        or run.get("source_commit") != qual_session.get("commit")
    ):
        raise ValueError("qualification runtime is not the fixed P1 1M boundary")
    actor_seconds = run.get("actor_phase_seconds")
    if (
        isinstance(actor_seconds, bool)
        or not isinstance(actor_seconds, (int, float))
        or not math.isfinite(actor_seconds)
        or not 0 < actor_seconds <= qual_session["elapsed_seconds"]
    ):
        raise ValueError("qualification lacks measured collection+update+probe seconds")
    if run.get("timing_includes_collection_updates_probes") is not True:
        raise ValueError("collection-only rate cannot price P1")
    overhead = qual_session["elapsed_seconds"] - actor_seconds
    row = {
        "schema": "bl-p1-complete-job-qualification/v1",
        "status": "PASS",
        "critic_qualified": True,
        "seconds_per_actor_transition": actor_seconds / QUAL_PAUSE,
        "per_job_overhead_seconds": overhead,
        "prefix_seconds": sum(s["elapsed_seconds"] for s in (source, replacement, qual_session)),
        "transfer_and_release_verified": True,
        "source_commit": qual_session["commit"],
        "runtime_sha256": digest(runtime),
        "y3_sha256": digest(y3),
        "supervisor_receipts": [
            {"path": str(p.resolve()), "sha256": digest(p)}
            for p in (source_supervisor, replacement_supervisor, supervisor)
        ],
        "timing_scope": "actor window includes collection, updates and probes; whole-session "
        "minus actor window prices bootstrap/download/export/transfer/release for each chunk",
    }
    # A measured forecast that misses either gate remains NOT_READY evidence.
    try:
        plan = measured_plan(row)
    except ValueError as exc:
        row.update(status="NOT_READY", reason=str(exc))
    else:
        row["forecast_hours"] = plan["complete_trajectory_hours"]
        row["invoice_units"] = plan["invoice_units"]
    export_json(output, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime",
        "supervisor",
        "y3",
        "source-supervisor",
        "replacement-supervisor",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    finalize(
        runtime=args.runtime,
        supervisor=args.supervisor,
        y3=args.y3,
        source_supervisor=args.source_supervisor,
        replacement_supervisor=args.replacement_supervisor,
        output=args.output,
    )


if __name__ == "__main__":
    main()
