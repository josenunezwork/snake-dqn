"""The separately claimed, once-only F4 descriptive harm export. Never funding."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f4_postmortem_guard as guard


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim_bundle(output: Path, plan: dict[str, Any], lease: dict[str, Any]) -> str:
    """One fixed directory is the attempt fence, including partial plan/claim writes."""
    # mkdir and create-only export each refuse concurrent callers. Never remove
    # this directory, even if the plan or claim write is interrupted.
    output.mkdir(parents=False, exist_ok=False)
    export_json(output / "plan.json", plan)
    export_json(
        output / "claim.json",
        {
            "schema": "bl-residual-f4-postmortem-claim/v1",
            "purpose": "DEVELOPMENT_ONLY",
            "f4_disposition": "STOP_FUNDING",
            "ruling_sha256": guard.RULING_SHA,
            "implementation_commit": plan["implementation_commit"],
            "manifest_sha256": plan["manifest_sha256"],
            "plan_sha256": digest(output / "plan.json"),
            "claimed_at": utc_now(),
            "lease": lease,
            "retry_authorized": False,
        },
    )
    return digest(output / "claim.json")


def compute_doses(
    manifest: dict[str, Any], metadata: dict[str, Any], *, checkpoint: Any
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Use funding.main's D256 then D1024 heldout call sequence, without funding."""
    # Import only inside the runtime thread fence. No fitting/funding functions
    # are invoked, and no heldout result metric except the existing H is read.
    from research.bl_residual.f4_postmortem_counts import AggregateObserver, valid_dose
    from research.bl_residual.heldout import evaluate

    inputs = manifest["allowed_payload_inputs"]
    parents = metadata["parents"]["parent_worlds"]

    def path(option: str) -> Path:
        return Path(inputs[option]["path"])

    doses = []
    checks = {}
    for dose in (256, 1024):
        observer = AggregateObserver(parents)
        checkpoint()
        look = evaluate(
            path(f"--d{dose}-checkpoint"),
            path("--b8"),
            path(f"--d{dose}-table"),
            allocation_path=path(f"--d{dose}-allocation"),
            control_path=path(f"--d{dose}-control"),
            approval_path=path("--approval"),
            archive_receipt=path("--archive-receipt"),
            joint_claim_path=Path(manifest["immutable_joint_claim"]["path"]),
            diagnostics_observer=observer,
        )
        if observer.result is None:
            raise ValueError("no diagnostic masks observed")
        aggregate = {
            "dose": dose,
            "checkpoint_sha256": inputs[f"--d{dose}-checkpoint"]["sha256"],
            **observer.result,
        }
        checks[f"d{dose}_census"] = (
            observer.total_rows == 5936
            and len(look["archive_eligible_life_ids"]) == 71
            and len(look["reserve_life_ids"]) == 64
            and look["checkpoint_sha256"] == aggregate["checkpoint_sha256"]
        )
        checks[f"d{dose}_bins_parents_partitions_denominators"] = valid_dose(aggregate, parents)
        existing_h = look["metrics"]["harmful_override_count_natural"]
        checks[f"d{dose}_same_pass_existing_H"] = aggregate["totals"]["H"] == existing_h
        if dose == 1024:
            checks["retained_d1024_harm_boolean_agrees"] = (existing_h == 0) is metadata[
                "harm_pass"
            ]
        # Do not inspect, persist, log or export any other computed heldout metric.
        del look
        doses.append(aggregate)
        checkpoint()
        print(
            json.dumps({"event": "candidate_complete", "completed": len(doses), "total": 2}),
            flush=True,
        )
    return doses, checks


def run(pin: str) -> None:
    """Publish-before-read and exclusive runtime surround the one spent attempt."""
    from research.bl_residual.f4_postmortem_watchdog import kill_receipt_path, supervise

    publication = guard.publication(pin)
    output = guard.OUTPUT
    if output.exists() or output.is_symlink():
        raise FileExistsError("the sole post-mortem bundle is already reserved/spent")
    supervise(
        lambda arm: _run_worker(pin, publication, arm),
        kill_receipt=kill_receipt_path(output),
        on_kill=lambda record: _record_watchdog_failure(pin, publication, output, record),
    )


def _record_watchdog_failure(
    pin: str, publication: dict[str, Any], output: Path, record: dict[str, object]
) -> None:
    """After the durable kill record, seal INCOMPLETE if the spent claim permits.

    Never inspect an outcome or repair partial immutable writes. A complete
    diagnostic plus a later cleanup stall must still verify as INCOMPLETE.
    """
    claim = output / "claim.json"
    claim_sha = verify_export(claim)
    claimed_at = json.loads(claim.read_bytes())["claimed_at"]
    failure = output / "failure.json"
    if not failure.exists():
        export_json(
            failure,
            {
                "schema": "bl-residual-f4-postmortem-failure/v1",
                "status": "INCOMPLETE",
                "purpose": "DEVELOPMENT_ONLY",
                "f4_disposition": "STOP_FUNDING",
                "ruling_sha256": guard.RULING_SHA,
                "implementation_commit": pin,
                "claim_sha256": claim_sha,
                "started_at": claimed_at,
                "finished_at": utc_now(),
                "wall_seconds": float(record["elapsed_seconds"] or 0),
                "failure_class": "IndependentWatchdogKill",
                "retry_authorized": False,
            },
        )
    if not (output / "development-parents.json").exists():
        export_json(output / "development-parents.json", publication["metadata"]["parents"])


def _run_worker(pin: str, publication: dict[str, Any], arm_watchdog: Callable[[], None]) -> None:
    """The sole inference worker; independent parent remains until this returns."""
    from research.bl_residual.f4_postmortem_runtime import exclusive_runtime

    manifest = publication["manifest"]
    metadata = publication["metadata"]
    output = guard.OUTPUT
    if output.exists() or output.is_symlink():
        raise FileExistsError("the sole post-mortem bundle is already reserved/spent")
    # Publication happens before lease waiting and before any real input is
    # opened. The detached worktree remains pinned and locked by dispatch.sh.
    with exclusive_runtime() as host:
        # Parent ACKs its monotonic ceiling before any claim or payload access.
        # Its SIGKILL remains active through deadline/host context cleanup too.
        arm_watchdog()
        with host.deadline() as deadline:
            started_at = utc_now()
            started = time.monotonic()
            plan = {
                "schema": "bl-residual-f4-postmortem-plan/v1",
                "ruling_sha256": guard.RULING_SHA,
                "implementation_commit": pin,
                "origin_main": publication["origin_main"],
                "source_sha256": publication["source_sha256"],
                "manifest_sha256": digest(guard.ROOT / guard.PACKAGE / "manifest.json"),
                "inputs": manifest["allowed_payload_inputs"],
                "original_joint_claim": manifest["immutable_joint_claim"],
                "output_root": str(output),
                "runtime": manifest["runtime"],
                "purpose": "DEVELOPMENT_ONLY",
                "f4_disposition": "STOP_FUNDING",
            }
            claim_sha = claim_bundle(output, plan, host.lease)
            try:
                verified = guard.verify_inputs(manifest)
                # The original claim is verified, never rewritten or replaced.
                original = json.loads(Path(manifest["immutable_joint_claim"]["path"]).read_bytes())
                guard.validate_original_claim(original, metadata["packet"])
                doses, checks = compute_doses(manifest, metadata, checkpoint=deadline.checkpoint)
                deadline.checkpoint()
                diagnostic = {
                    "schema": "bl-residual-f4-harm-postmortem/v1",
                    "ruling_sha256": guard.RULING_SHA,
                    "implementation_commit": pin,
                    "source_sha256": publication["source_sha256"],
                    "inputs": verified,
                    "claim_sha256": claim_sha,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "platform": platform.platform(),
                    "nice": os.nice(0),
                    "threads": host.threads,
                    "wall_seconds": time.monotonic() - started,
                    "status": "COMPLETE" if all(checks.values()) else "INVALID",
                    "purpose": "DEVELOPMENT_ONLY",
                    "f4_disposition": "STOP_FUNDING",
                    "census": manifest["census"],
                    "doses": doses,
                    "checks": checks,
                }
                export_json(output / "diagnostic.json", diagnostic)
                export_json(output / "development-parents.json", metadata["parents"])
                deadline.checkpoint()
                print(
                    json.dumps(
                        {
                            "event": "sealed_bundle",
                            "path": str(output),
                            "sha256": digest(output / "diagnostic.json"),
                        }
                    ),
                    flush=True,
                )
            except BaseException as exc:
                # No str(exc), traceback, row values or partial counters escape.
                # A spent/partial bundle always refuses every subsequent run.
                export_json(
                    output / "failure.json",
                    {
                        "schema": "bl-residual-f4-postmortem-failure/v1",
                        "status": "INCOMPLETE",
                        "purpose": "DEVELOPMENT_ONLY",
                        "f4_disposition": "STOP_FUNDING",
                        "ruling_sha256": guard.RULING_SHA,
                        "implementation_commit": pin,
                        "claim_sha256": claim_sha,
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "wall_seconds": time.monotonic() - started,
                        "failure_class": type(exc).__name__,
                        "retry_authorized": False,
                    },
                )
                if not (output / "development-parents.json").exists():
                    export_json(output / "development-parents.json", metadata["parents"])
                raise RuntimeError("claimed post-mortem incomplete; no retry authorized") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--publication-commit", required=True)
    args = parser.parse_args()
    if args.action == "check":
        guard.publication(args.publication_commit)
        if guard.OUTPUT.exists():
            raise FileExistsError("post-mortem bundle already reserved/spent")
        print("PUBLICATION_VERIFIED_METADATA_ONLY; no payload opened; no claim created")
    else:
        run(args.publication_commit)


if __name__ == "__main__":
    main()
