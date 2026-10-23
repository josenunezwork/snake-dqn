"""Seal measured P1 complete-job pricing only after supervisor release proof."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_manifests import QUAL_PAUSE, complete_json, measured_plan


def terminal(path: Path) -> dict[str, Any]:
    """Require success and verified release, not a CLI return code."""
    # The unchanged supervisor publishes receipt.json with write_once(),
    # without experiment sidecars. Bind those exact original bytes in our
    # create-only result; never alter the supervisor's receipt.
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or unsafe supervisor receipt")
    row = json.loads(path.read_text())
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


def final_transfer_release_timing(session: dict[str, Any]) -> dict[str, float]:
    """Bind measured verified downloads and the stop/disappearance CLI sequence."""
    transfers = [
        row
        for row in session.get("transfers", [])
        if row.get("direction") == "download" and row.get("status") == "verified"
    ]
    calls = session.get("cli_calls", [])
    stops = [i for i, row in enumerate(calls) if row.get("command") == "stop"]
    if not transfers or not stops:
        raise ValueError("qualification lacks measured final transfer/release timing")
    release = [
        row
        for row in calls[stops[-1] :]
        if row.get("command") == "stop"
        or row.get("command") == "sessions"
        and row.get("status") == "completed"
    ]
    if {row.get("command") for row in release} != {"stop", "sessions"}:
        raise ValueError("qualification lacks measured final release/disappearance timing")
    for row in transfers + release:
        seconds = row.get("elapsed_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("qualification has invalid final transfer/release timing")
    return {
        "verified_download_seconds": sum(row["elapsed_seconds"] for row in transfers),
        "stop_and_disappearance_seconds": sum(row["elapsed_seconds"] for row in release),
    }


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
    if (
        durability.get("schema") != "bl-p1-y3/v1"
        or durability.get("status") != "PASS"
        or durability.get("operational_replay", {}).get("status") != "PASS"
    ):
        raise ValueError("P1 pricing requires full Y3 PASS including operational replay")
    if not source.get("endpoint") or source.get("endpoint") == replacement.get("endpoint"):
        raise ValueError("Y3 source and replacement must be distinct assignments")
    if len({s.get("commit") for s in (source, replacement, qual_session)}) != 1:
        raise ValueError("P1 qualification source commit differs across assignments")
    if (
        durability.get("source_endpoint") != source["endpoint"]
        or durability.get("replacement_endpoint") != replacement["endpoint"]
        or durability.get("source_commit") != source["commit"]
        or not any(item.get("sha256") == digest(y3) for item in replacement.get("exports", []))
        or not any(
            item.get("sha256") == durability.get("source_receipt_sha256")
            for item in source.get("exports", [])
        )
    ):
        raise ValueError("Y3 proof is not bound to its source/replacement terminal exports")
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
        or not run.get("origin_snapshot_sha256")
        or run.get("origin_snapshot_sha256") != durability.get("origin_snapshot_sha256")
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
    final_timing = final_transfer_release_timing(qual_session)
    overhead = qual_session["elapsed_seconds"] - actor_seconds
    row = {
        "schema": "bl-p1-complete-job-qualification/v1",
        "status": "PASS",
        "critic_qualified": True,
        "y3_status": durability["status"],
        "operational_resume_eligible": durability["status"] == "PASS"
        and durability.get("operational_replay", {}).get("status") == "PASS",
        "seconds_per_actor_transition": actor_seconds / QUAL_PAUSE,
        "per_job_overhead_seconds": overhead,
        "prefix_seconds": sum(s["elapsed_seconds"] for s in (source, replacement, qual_session)),
        "transfer_and_release_verified": True,
        "final_transfer_release_timing": final_timing,
        "source_commit": qual_session["commit"],
        "registration_sha256": run.get("registration_sha256"),
        "runtime_sha256": digest(runtime),
        "y3_sha256": digest(y3),
        "snapshot": run.get("snapshot"),
        "origin_snapshot_sha256": run["origin_snapshot_sha256"],
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
