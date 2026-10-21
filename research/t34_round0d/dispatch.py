"""Explicit operator dispatch of one Round 0-D chunk through immutable history and the broker.

Importing never allocates. Semantics are Round 0's (``research/t34_round0/dispatch.py``):
create-only remote claim, fixed attempt paths, 48 h dispatch deadline, abort or
unexplained interruption = KILL, first cap = INCONCLUSIVE, one outcome-independent
mechanical rerun, no automatic retry. Generic settlement helpers are imported
unchanged. Round 0-D additions: two concurrent lanes via an in-flight liveness
lock (at most 2 in flight), and the forecast-derived charged-unit ceiling before every
admission (a settled-charge breach seals INCONCLUSIVE).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard
from research.t34_round0.dispatch import (
    _broker_cap_expired,
    _completion_within_timeout,
    _mechanical,
    _runtime_cap_expired,
    _runtime_terminal,
    require_dispatch_deadline,
)
from research.t34_round0d.common import B8_SHA, PACKAGE, ROOT, read_json, sha256, write_once
from research.t34_round0d.history import (
    HistoryError,
    InflightClaim,
    finalize,
    history_lock,
    in_flight,
    in_flight_count,
    phase_name,
    policy,
    require_published_history,
    reserve,
    seal_ceiling,
)
from research.t34_round0d.registration import (
    require_published_registration,
    require_registration,
)

OUTPUT = "round0d"
FAULTS = {"illegal_action", "nonfinite_q", "nonfinite_loss", "action_collapse_100k"}


def _exported_record(receipt: dict[str, Any], job: Path, name: str) -> dict[str, Any] | None:
    """Return one hash-bound Round 0-D terminal export, never a local orphan."""
    path = job / "exports" / OUTPUT / name
    terminal = _runtime_terminal(receipt, job)
    if (
        not path.is_file()
        or not isinstance(terminal, dict)
        or receipt.get("stop_verified") is not True
    ):
        return None
    if terminal.get("job_id") != receipt.get("job_id") or terminal.get("generation") != receipt.get(
        "generation"
    ):
        return None
    entry = next(
        (e for e in terminal.get("exports", []) if e.get("path") == f"{OUTPUT}/{name}"), None
    )
    if not isinstance(entry, dict) or entry.get("sha256") != sha256(path):
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _own_record(
    job: Path, receipt: dict[str, Any], registration: dict[str, Any], name: str, schema: str
) -> dict[str, Any] | None:
    record = _exported_record(receipt, job, name)
    identity = record.get("run_identity", {}) if isinstance(record, dict) else {}
    if (
        record
        and record.get("schema") == schema
        and record.get("preflight") is False
        and identity.get("commit") == registration["commit"]
        and identity.get("runtime_id") == receipt.get("job_id")
    ):
        return record
    return None


def _cap_expired(job: Path, receipt: dict[str, Any], registration: dict[str, Any]) -> bool:
    record = _own_record(job, receipt, registration, "cap_expiry.json", "t34-round0d-cap-expiry/v1")
    return bool(
        record
        and record.get("status") == "INCONCLUSIVE"
        and record.get("no_extension") is True
        and record.get("operational_reason") == "registered chunk wall cap expired"
    )


def _mechanical_incomplete(job: Path, receipt: dict[str, Any], registration: dict) -> bool:
    record = _own_record(job, receipt, registration, "incomplete.json", "t34-round0d-incomplete/v1")
    return bool(
        record
        and record.get("status") == "INCOMPLETE"
        and record.get("mechanical_rerun_eligible") is True
        and record.get("operational_reason") == "export or serialization failure"
    )


def _controller_fault(job: Path, receipt: dict[str, Any], registration: dict[str, Any]) -> bool:
    record = _own_record(job, receipt, registration, "fault.json", "t34-round0d-fault/v1")
    return bool(
        record
        and record.get("unconditional") is True
        and record.get("fault") in FAULTS
        and record.get("arm") in {"bare", "teacher"}
        and record.get("run_identity", {}).get("checkpoint_sha256") == B8_SHA
    )


def _settlement(
    job: Path, receipt: dict[str, Any], registration: dict[str, Any], timeout_minutes: int
) -> tuple[str, str]:
    """Settle an operational receipt without reading any outcome value."""
    if (
        receipt.get("status") == "PASS"
        and receipt.get("stop_verified") is True
        and _completion_within_timeout(receipt, timeout_minutes)
    ):
        return "COMPLETE", "supervisor PASS and verified release"
    if _controller_fault(job, receipt, registration):
        return "COMPLETE", "explained unconditional controller fault; scientific STOP only"
    if (
        _cap_expired(job, receipt, registration)
        or _runtime_cap_expired(receipt, job)
        or _broker_cap_expired(receipt)
    ):
        return "INCONCLUSIVE", "registered first cap expired; INCONCLUSIVE, no extension"
    if _mechanical(receipt, job) or _mechanical_incomplete(job, receipt, registration):
        return "MECHANICAL_FAILURE", "verified release; registered export or serialization failure"
    if _runtime_terminal(receipt, job) and not _completion_within_timeout(receipt, timeout_minutes):
        return "KILL", "chunk exceeded its registered completion timeout"
    return "KILL", "supervisor failed without a registered mechanical terminal cause"


def prior_spend(registration: dict[str, Any]) -> list[dict[str, Any]]:
    """Charge every retained attempt: receipt elapsed time, else (in flight/killed) its cap."""
    root, _, _ = policy(registration)
    charges = []
    for attempt in sorted((root / "chunks").glob("*/attempt-0[12]")):
        manifest_path = attempt / "dispatch_manifest.json"
        cap = (
            read_json(manifest_path)["budget"]["unit_cap"]
            if manifest_path.is_file()
            else float(registration["sizing"]["study_unit_ceiling"])
        )
        receipt_path = attempt / "job" / "receipt.json"
        elapsed = None
        if receipt_path.is_file() and not in_flight(attempt):
            elapsed = read_json(receipt_path).get("elapsed_seconds")
        charges.append({"attempt": str(attempt), "elapsed_seconds": elapsed, "unit_cap": cap})
    return charges


def validate_manifest_closure(
    manifest: dict[str, Any], registration: dict[str, Any], registration_path: Path, a5d: Path
) -> int:
    """The dispatched manifest must equal the one derived from the ratification."""
    from research.t34_round0d.make_manifest import chunk_manifest

    chunk = manifest.get("round0d", {}).get("chunk_id", "")
    if not isinstance(chunk, str) or not chunk.startswith("chunk-"):
        raise HistoryError("manifest lacks a registered chunk id")
    number = int(chunk.removeprefix("chunk-"))
    if number >= len(registration["sizing"]["chunks"]):
        raise HistoryError("manifest names a chunk outside the frozen schedule")
    expected = chunk_manifest(
        registration,
        registration_path,
        a5d,
        number,
        lane=int(manifest["round0d"]["lane"]),
        other_assignments=manifest["budget"]["other_assignments"],
    )
    if manifest != expected:
        raise HistoryError("Round 0-D manifest differs from the ratified recipe")
    return number


def dispatch(
    manifest_path: Path,
    registration_path: Path,
    a5d_path: Path,
    state_dir: Path,
    colab: Path,
    *,
    mechanical_rerun: bool = False,
    outcomes_unseen: bool = False,
) -> dict[str, Any]:
    """Perform one explicit dispatch; never automatically repeat an invocation."""
    from research.colab_cli.supervisor import validate
    from research.t34_round0d.sizing import admission_spend

    registration = read_json(registration_path)
    require_registration(
        registration,
        PACKAGE / "BANK_ROUND0D.json",
        PACKAGE / "SEED_AUDIT_ROUND0D.json",
        mac_side=True,
    )
    require_published_registration(registration, registration_path)
    if sha256(a5d_path) != registration["a5d"]["sha256"]:
        raise HistoryError("A5-D receipt differs from the ratified binding")
    root, _, _ = policy(registration)
    original = read_json(manifest_path)
    number = validate_manifest_closure(original, registration, registration_path, a5d_path)
    chunk = original["round0d"]["chunk_id"]
    size = registration["sizing"]["chunks"][number]
    attempt = root / "chunks" / chunk / ("attempt-02" if mechanical_rerun else "attempt-01")
    phase_name(attempt, registration)
    deadline = original["round0d"]["deadline"]
    require_dispatch_deadline(deadline["dispatch_by_unix_seconds"])
    if not state_dir.is_absolute() or state_dir.exists():
        raise HistoryError("supervisor state directory must be fresh and absolute")
    manifest = copy.deepcopy(original)
    with FrozenWorktreeGuard(ROOT, registration["commit"]), history_lock(registration):
        terminals = require_published_history(registration)
        if any(t["status"] == "KILL" for t in terminals):
            raise HistoryError("KILL closes this study")
        if any(t["status"] == "INCONCLUSIVE" for t in terminals):
            raise HistoryError("registered first cap expiry closes this study with INCONCLUSIVE")
        if any(
            t["reason"].startswith("explained unconditional controller fault") for t in terminals
        ):
            raise HistoryError("unconditional controller fault closes this study with STOP")
        if (root / "replay").exists() or (root / "analysis").exists():
            raise HistoryError("no dispatch or mechanical rerun after a replay/outcome look")
        sizing = registration["sizing"]
        ceiling = float(sizing["study_unit_ceiling"])
        live = in_flight_count(registration)
        if live >= int(sizing["max_in_flight"]):
            raise HistoryError(
                f"{live} attempts already in flight; at most {sizing['max_in_flight']} lanes "
                "(the third account assignment stays free for Path B); refused without a claim"
            )
        spend = admission_spend(
            [
                {"elapsed_seconds": c["elapsed_seconds"], "unit_cap": c["unit_cap"]}
                for c in prior_spend(registration)
            ],
            size,
            ceiling,
        )
        if not spend["allowed"]:
            if live == 0 and spend["infeasible_on_settled_charges"]:
                seal_ceiling(registration, spend)
                raise HistoryError(
                    "registered study ceiling reached on settled charges: sealed INCONCLUSIVE"
                )
            raise HistoryError(
                f"study ceiling would be crossed while attempts are in flight ({spend}); "
                "refused without a claim; retry after they settle"
            )
        if mechanical_rerun:
            prior = read_json(attempt.with_name("attempt-01") / "phase_terminal.json")
            if not outcomes_unseen or prior.get("status") != "MECHANICAL_FAILURE":
                raise HistoryError(
                    "rerun requires measured mechanical failure and outcomes-unseen attestation"
                )
            manifest["job_id"] += "-attempt-02"
            manifest["round0d"]["attempt"] = 2
            manifest["output_dir"] = str(attempt / "job")
            manifest["env"]["T34_ROUND0D_RUNTIME_ID"] = manifest["job_id"]
        validate(manifest, repo=ROOT)
        reserve(attempt, registration, world_ids=list(original["round0d"]["world_ids"]))
        claim = InflightClaim(attempt)
        admission = attempt / "admission.json"
        dispatched_at = read_json(admission).get("dispatched_at_unix_seconds")
        if (
            isinstance(dispatched_at, bool)
            or not isinstance(dispatched_at, (int, float))
            or not math.isfinite(float(dispatched_at))
            or dispatched_at > deadline["dispatch_by_unix_seconds"]
        ):
            claim.release()
            raise HistoryError("admission missed the registered 48-hour dispatch deadline: KILL")
        manifest["inputs"].append(
            {"path": str(admission), "sha256": sha256(admission), "dest": "DISPATCH_ADMISSION.json"}
        )
        manifest["env"]["T34_ROUND0D_DISPATCH_ADMISSION"] = "DISPATCH_ADMISSION.json"
        write_once(attempt / "dispatch_manifest.json", manifest)
        write_once(
            attempt / "dispatch_intent.json",
            {
                "original_manifest_sha256": sha256(manifest_path),
                "mechanical_rerun": mechanical_rerun,
                "outcomes_unseen": outcomes_unseen,
                "spend_admission": spend,
            },
        )
    status, reason = "KILL", "operator abort or unexplained interruption"
    try:
        with FrozenWorktreeGuard(ROOT, registration["commit"]):
            subprocess.run(
                [
                    str(ROOT / "research/colab_cli/launch.sh"),
                    "--state-dir",
                    str(state_dir),
                    "--colab",
                    str(colab),
                    "run",
                    str(attempt / "dispatch_manifest.json"),
                ],
                cwd=ROOT,
                check=True,
            )
            receipt = read_json(attempt / "job/receipt.json")
            status, reason = _settlement(
                attempt / "job", receipt, registration, int(size["timeout_minutes"])
            )
    except subprocess.CalledProcessError:
        receipt_path = attempt / "job/receipt.json"
        if receipt_path.is_file():
            status, reason = _settlement(
                attempt / "job", read_json(receipt_path), registration, int(size["timeout_minutes"])
            )
        else:
            reason = "supervisor failed without a registered mechanical terminal cause"
    except BaseException as exc:
        reason = f"{type(exc).__name__}: operator abort or unexplained interruption"
        raise
    finally:
        try:
            finalize(attempt, registration, status, reason)
        finally:
            claim.release()
    return {"status": status, "phase": str(attempt), "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ratification", required=True, type=Path)
    parser.add_argument("--a5d", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--colab", type=Path, default=Path.home() / ".local/bin/colab")
    parser.add_argument("--mechanical-rerun", action="store_true")
    parser.add_argument("--outcomes-unseen", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            dispatch(
                args.manifest,
                args.ratification,
                args.a5d,
                args.state_dir,
                args.colab,
                mechanical_rerun=args.mechanical_rerun,
                outcomes_unseen=args.outcomes_unseen,
            )
        )
    )


if __name__ == "__main__":
    main()
