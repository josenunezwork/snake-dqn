"""Explicit operator dispatch through immutable Round 0 history and CLI broker.

Importing or validating this module never allocates. Only the `run` command
dispatches, after ratification, a remote claim, and frozen-worktree admission.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard
from research.t34_round0.bank import ordered_rows
from research.t34_round0.common import BINDING_DOCS, PACKAGE, ROOT, read_json, sha256, write_once
from research.t34_round0.history import (
    HistoryError,
    finalize,
    phase_name,
    policy,
    require_published_history,
    reserve,
)
from research.t34_round0.registration import require_registration


def _completion_within_timeout(receipt: dict[str, Any], timeout_minutes: int) -> bool:
    """Require an immutable runtime terminal interval within the registered cap."""
    terminal = receipt.get("terminal")
    if not isinstance(terminal, dict):
        return False
    started, finished = terminal.get("started_at"), terminal.get("finished_at")
    if (
        isinstance(started, bool)
        or isinstance(finished, bool)
        or not isinstance(started, (int, float))
        or not isinstance(finished, (int, float))
        or not math.isfinite(float(started))
        or not math.isfinite(float(finished))
        or float(finished) < float(started)
    ):
        return False
    return float(finished) - float(started) <= timeout_minutes * 60


def require_dispatch_deadline(deadline: float, *, now: float | None = None) -> None:
    """Refuse a chunk admission after the ratified 48-hour dispatch deadline."""
    if (
        not isinstance(deadline, (int, float))
        or isinstance(deadline, bool)
        or not math.isfinite(deadline)
    ):
        raise HistoryError("manifest has an invalid registered dispatch deadline")
    current = time.time() if now is None else now
    if current > deadline:
        raise HistoryError("registered 48-hour Round 0 dispatch deadline has expired: KILL")


def _runtime_terminal(receipt: dict[str, Any], job: Path) -> dict[str, Any] | None:
    """Read the immutable runtime terminal embedded or retained by the broker."""
    terminal = receipt.get("terminal")
    if not isinstance(terminal, dict):
        terminal_path = job / "terminal.json"
        if terminal_path.is_file():
            try:
                terminal = read_json(terminal_path)
            except (OSError, ValueError):
                return None
        else:
            for poll in reversed(sorted(job.glob("poll-*.json"))):
                try:
                    candidate = read_json(poll).get("terminal")
                except (OSError, ValueError):
                    continue
                if isinstance(candidate, dict):
                    terminal = candidate
                    break
    return terminal if isinstance(terminal, dict) else None


def _exported_round0_record(receipt: dict[str, Any], job: Path, name: str) -> dict[str, Any] | None:
    """Return one hash-bound Round 0 terminal export, never a local orphan."""
    path = job / "exports" / "round0" / name
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
        (item for item in terminal.get("exports", []) if item.get("path") == f"round0/{name}"), None
    )
    if not isinstance(entry, dict) or entry.get("sha256") != sha256(path):
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _runtime_cap_expired(receipt: dict[str, Any], job: Path) -> bool:
    """Treat the supervisor's first registered runtime timeout as INCONCLUSIVE."""
    terminal = _runtime_terminal(receipt, job)
    return bool(
        terminal
        and terminal.get("job_id") == receipt.get("job_id")
        and terminal.get("generation") == receipt.get("generation")
        and terminal.get("reason") in {"timeout", "timeout_after_terminal"}
    )


def _broker_cap_expired(receipt: dict[str, Any]) -> bool:
    """Classify only exact registered broker-cap refusals as INCONCLUSIVE."""
    return (
        receipt.get("status") != "PASS"
        and receipt.get("stop_verified") is True
        and receipt.get("cleanup_error") is None
        and receipt.get("error")
        in {
            "Refused('whole-session wall cap reached (including setup/export)')",
            "Refused('priced session unit cap reached')",
            "Refused('priced session unit cap reached during calibration')",
            "Refused('priced session unit cap reached while awaiting account broker')",
            "Refused('deadline reached while awaiting account broker')",
            "Refused('whole-session deadline reached before helper')",
            "Refused('whole-session deadline reached before CLI call')",
            "AccountingError('priced session unit cap reached')",
        }
    )


def _cap_expired(job: Path, receipt: dict[str, Any], registration: dict[str, Any]) -> bool:
    """Recognize the registered first-cap terminal without inspecting outcomes."""
    record = _exported_round0_record(receipt, job, "cap_expiry.json")
    identity = record.get("run_identity", {}) if isinstance(record, dict) else {}
    return bool(
        record
        and record.get("schema") == "t34-round0-cap-expiry/v1"
        and record.get("status") == "INCONCLUSIVE"
        and record.get("no_extension") is True
        and record.get("operational_reason") == "registered chunk wall cap expired"
        and record.get("preflight") is False
        and identity.get("commit") == registration["commit"]
        and identity.get("runtime_id") == receipt.get("job_id")
    )


def _mechanical_incomplete(
    job: Path, receipt: dict[str, Any], registration: dict[str, Any]
) -> bool:
    """Recognize only the registered export/serialization recovery terminal."""
    record = _exported_round0_record(receipt, job, "incomplete.json")
    identity = record.get("run_identity", {}) if isinstance(record, dict) else {}
    return bool(
        record
        and record.get("schema") == "t34-round0-incomplete/v1"
        and record.get("status") == "INCOMPLETE"
        and record.get("mechanical_rerun_eligible") is True
        and record.get("operational_reason") == "export or serialization failure"
        and record.get("preflight") is False
        and identity.get("commit") == registration["commit"]
        and identity.get("runtime_id") == receipt.get("job_id")
    )


def _mechanical(receipt: dict[str, Any], job: Path | None = None) -> bool:
    """Only a successful workload with verified release can retry lost transfer.

    No outcome values are read. Unknown failures, aborts, timeouts and controller
    errors cannot qualify as mechanical recovery.
    """
    terminal = receipt.get("terminal")
    if terminal is None and job is not None:
        # execute() persists the poll before downloading, but a failed transfer
        # prevents its terminal return value reaching the outer broker receipt.
        for path in reversed(sorted(job.glob("poll-*.json"))):
            candidate = read_json(path).get("terminal")
            if candidate:
                terminal = candidate
                break
    if not isinstance(terminal, dict):
        return False
    failed = [t for t in receipt.get("transfers", []) if t.get("status") == "failed"]
    failed_download = any(
        t.get("direction") == "download"
        and any(
            str(t.get("remote", "")).endswith("/output/" + e.get("path", ""))
            for e in terminal.get("exports", [])
        )
        for t in failed
    )
    return (
        receipt.get("status") != "PASS"
        and receipt.get("stop_verified") is True
        and receipt.get("cleanup_error") is None
        and bool(receipt.get("error"))
        and (
            str(receipt.get("error")).startswith("TransferError(")
            or str(receipt.get("error")).startswith(
                "Refused('download hash mismatch or transfer failure:"
            )
            or str(receipt.get("error")).startswith(
                'Refused("download hash mismatch or transfer failure:'
            )
        )
        and failed_download
        and terminal.get("generation") == receipt.get("generation")
        and terminal.get("job_id") == receipt.get("job_id")
        and terminal.get("status") in {"completed", "success", "succeeded", "PASS"}
        and terminal.get("returncode") == 0
        and terminal.get("reason") is None
    )


def _controller_fault(job: Path, receipt: dict[str, Any], registration: dict[str, Any]) -> bool:
    """Retain an explained unconditional controller STOP, without an efficacy look."""
    from research.t34_round0.common import B8_SHA

    path = job / "exports/round0/fault.json"
    if not path.is_file() or receipt.get("stop_verified") is not True:
        return False
    terminal = receipt.get("terminal") or (
        read_json(job / "terminal.json") if (job / "terminal.json").is_file() else {}
    )
    if terminal.get("job_id") != receipt.get("job_id") or terminal.get("generation") != receipt.get(
        "generation"
    ):
        return False
    entry = next(
        (e for e in terminal.get("exports", []) if e.get("path") == "round0/fault.json"), None
    )
    if not entry or entry.get("sha256") != sha256(path):
        return False
    fault = read_json(path)
    identity = fault.get("run_identity", {})
    return (
        fault.get("schema") == "t34-round0-fault/v1"
        and fault.get("preflight") is False
        and fault.get("unconditional") is True
        and fault.get("fault")
        in {"illegal_action", "nonfinite_q", "nonfinite_loss", "action_collapse_100k"}
        and fault.get("arm") in {"bare", "teacher"}
        and identity.get("commit") == registration["commit"]
        and identity.get("checkpoint_sha256") == B8_SHA
        and identity.get("runtime_id") == receipt.get("job_id")
    )


def validate_manifest_closure(
    manifest: dict[str, Any], registration: dict[str, Any], registration_path: Path
) -> None:
    """Refuse a changed entrypoint, hardware, env, input or source before allocation."""
    size = manifest["round0"]["sizing"]
    expected_env = {
        "PYTHONUNBUFFERED": "1",
        "T34_INPUT_MODE": "public",
        "T34_DEVICE": "cpu",
        "T34_ROUND0_START": str(size["start"]),
        "T34_ROUND0_COUNT": str(size["count"]),
        "T34_ROUND0_BANK": "BANK_ROUND0.json",
        "T34_ROUND0_AUDIT": "SEED_AUDIT_ROUND0.json",
        "T34_ROUND0_REGISTRATION": "PREREGISTRATION_ROUND0.md",
        "T34_ROUND0_RATIFICATION": "RATIFICATION.json",
        "T34_ROUND0_A5_RECEIPT": "A5_RECEIPT.json",
        "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
        "T34_TELEMETRY_SECONDS": "120",
        "T34_ROUND0_OUTPUT_DIR": "round0",
        "T34_B8_CHECKPOINT": "B8_s3_final.pth",
        "T34_ROUND0_RUNTIME_ID": manifest["job_id"],
        "T34_ROUND0_COMMIT": registration["commit"],
        "T34_ROUND0_SECONDS": str((int(size["timeout_minutes"]) - 25) * 60),
        "T34_ROUND0_THREADS": "1",
        "T34_ROUND0_PROCESSES": "12",
        "T34_ROUND0_WORLDS_PER_PROCESS": "1",
    }
    expected_inputs = {
        "PUBLIC_INPUTS.json": registration["source_hashes"][
            "research/t34_exit_b8/PUBLIC_INPUTS.json"
        ],
        "BANK_ROUND0.json": registration["bank_sha256"],
        "SEED_AUDIT_ROUND0.json": registration["audit_sha256"],
        "PREREGISTRATION_ROUND0.md": registration["registration_sha256"],
        "RATIFICATION.json": sha256(registration_path),
        "A5_RECEIPT.json": registration["a5"]["sha256"],
    }
    items = manifest.get("inputs", [])
    rows = read_json(PACKAGE / "BANK_ROUND0.json").get("rows")
    if not isinstance(rows, list):
        raise HistoryError("Round 0 bank has no rows for interleaved chunk validation")
    expected_world_ids = [
        row.get("world_id")
        for row in ordered_rows(rows)[size["start"] : size["start"] + size["count"]]
    ]
    from research.t34_round0.make_manifest import dispatch_deadline

    expected_deadline = {
        "dispatch_by_unix_seconds": dispatch_deadline(registration["approved_at"]),
        "completion_within_seconds": int(size["timeout_minutes"]) * 60,
    }
    if (
        manifest.get("script") != "research/t34_round0/run.sh"
        or manifest.get("requirements")
        != {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": size["estimate_minutes"],
        }
        or manifest.get("env") != expected_env
        or manifest.get("source_paths") != sorted({*registration["source_hashes"], *BINDING_DOCS})
        or manifest.get("lease_seconds") != 600
        or len(items) != len(expected_inputs)
        or {item.get("dest"): item.get("sha256") for item in items} != expected_inputs
        or manifest.get("round0", {}).get("layout")
        != {"cpu_inference_processes": 12, "threads_per_process": 1, "worlds_per_process": 1}
        or manifest.get("round0", {}).get("world_ids") != expected_world_ids
        or manifest.get("round0", {}).get("deadline") != expected_deadline
    ):
        raise HistoryError("Round 0 manifest executable closure differs from ratified recipe")


def _settlement(
    job: Path, receipt: dict[str, Any], registration: dict[str, Any], timeout_minutes: int
) -> tuple[str, str]:
    """Settle an operational receipt; a proven first cap forbids mechanical recovery."""
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


def dispatch(
    manifest_path: Path,
    registration_path: Path,
    state_dir: Path,
    colab: Path,
    *,
    mechanical_rerun: bool = False,
    outcomes_unseen: bool = False,
) -> dict[str, Any]:
    """Perform one explicit dispatch; never automatically repeat an invocation."""
    from research.colab_cli.supervisor import validate

    registration = read_json(registration_path)
    require_registration(
        registration, PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json"
    )
    from research.t34_round0.registration import require_published_registration

    require_published_registration(registration, registration_path)
    root, _, _ = policy(registration)
    original = read_json(manifest_path)
    chunk = original.get("round0", {}).get("chunk_id")
    if original.get("commit") != registration["commit"] or not isinstance(chunk, str):
        raise HistoryError("manifest lacks the registered chunk/source identity")
    manifest = copy.deepcopy(original)
    attempt = root / "chunks" / chunk / ("attempt-02" if mechanical_rerun else "attempt-01")
    phase_name(attempt, registration)
    if Path(original["output_dir"]) != root / "chunks" / chunk / "attempt-01/job":
        raise HistoryError("manifest must name the fixed first chunk attempt")
    number = int(chunk.removeprefix("chunk-"))
    sizes = registration.get("sizing", {}).get("chunks", [])
    if number >= len(sizes) or original["round0"].get("sizing") != sizes[number]:
        raise HistoryError("manifest slice or pricing differs from frozen chunk sizing")
    size = sizes[number]
    if (
        original["round0"].get("start") != size["start"]
        or original["round0"].get("count") != size["count"]
        or original.get("timeout_minutes") != size["timeout_minutes"]
        or original.get("max_export_bytes") != size["export_bytes"]
        or original.get("budget", {}).get("unit_cap") != size["unit_cap"]
        or original.get("budget", {}).get("reserve_units") != size["reserve_units"]
        or original.get("budget", {}).get("rate_units_per_hour") != 2.0
        or original.get("env", {}).get("T34_ROUND0_START") != str(size["start"])
        or original.get("env", {}).get("T34_ROUND0_COUNT") != str(size["count"])
    ):
        raise HistoryError("manifest fields differ from registered slice/caps")
    validate_manifest_closure(original, registration, registration_path)
    deadline = original["round0"]["deadline"]
    require_dispatch_deadline(deadline["dispatch_by_unix_seconds"])
    if not state_dir.is_absolute() or state_dir.exists():
        raise HistoryError("supervisor state directory must be fresh and absolute")
    with FrozenWorktreeGuard(ROOT, registration["commit"]):
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
        if mechanical_rerun:
            prior = attempt.with_name("attempt-01")
            terminal = read_json(prior / "phase_terminal.json")
            if not outcomes_unseen or terminal.get("status") != "MECHANICAL_FAILURE":
                raise HistoryError(
                    "rerun requires measured mechanical failure and outcomes-unseen attestation"
                )
            manifest["job_id"] += "-attempt-02"
            manifest["round0"]["attempt"] = 2
            manifest["output_dir"] = str(attempt / "job")
            manifest["env"]["T34_ROUND0_RUNTIME_ID"] = manifest["job_id"]
        # Check all manifest policy/input/source fields before claiming a phase.
        validate(manifest, repo=ROOT)
        world_ids = original.get("round0", {}).get("world_ids")
        if not isinstance(world_ids, list) or not all(
            isinstance(world_id, str) for world_id in world_ids
        ):
            raise HistoryError("manifest lacks a fixed interleaved world-id slice")
        reserve(attempt, registration, world_ids=world_ids)
        admission = attempt / "admission.json"
        admission_value = read_json(admission)
        dispatched_at = admission_value.get("dispatched_at_unix_seconds")
        if (
            isinstance(dispatched_at, bool)
            or not isinstance(dispatched_at, (int, float))
            or not math.isfinite(float(dispatched_at))
        ):
            raise HistoryError("admission lacks an immutable dispatch timestamp")
        if dispatched_at > deadline["dispatch_by_unix_seconds"]:
            raise HistoryError("admission missed the registered 48-hour dispatch deadline: KILL")
        manifest["inputs"].append(
            {"path": str(admission), "sha256": sha256(admission), "dest": "DISPATCH_ADMISSION.json"}
        )
        manifest["env"]["T34_ROUND0_DISPATCH_ADMISSION"] = "DISPATCH_ADMISSION.json"
        write_once(attempt / "dispatch_manifest.json", manifest)
        write_once(
            attempt / "dispatch_intent.json",
            {
                "original_manifest_sha256": sha256(manifest_path),
                "mechanical_rerun": mechanical_rerun,
                "outcomes_unseen": outcomes_unseen,
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
        # No analysis of outcomes here. The complete retained job (including
        # failures) is sealed only after the source monitor has closed.
        finalize(attempt, registration, status, reason)
    return {"status": status, "phase": str(attempt), "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ratification", required=True, type=Path)
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
                args.state_dir,
                args.colab,
                mechanical_rerun=args.mechanical_rerun,
                outcomes_unseen=args.outcomes_unseen,
            )
        )
    )


if __name__ == "__main__":
    main()
