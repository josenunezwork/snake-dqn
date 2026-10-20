"""Explicit operator dispatch through immutable Round 0 history and CLI broker.

Importing or validating this module never allocates. Only the `run` command
dispatches, after ratification, a remote claim, and frozen-worktree admission.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from research.t34_pathb.source_binding_v2 import FrozenWorktreeGuard
from research.t34_round0.common import PACKAGE, ROOT, read_json, sha256, write_once
from research.t34_round0.history import (
    HistoryError,
    finalize,
    phase_name,
    policy,
    require_published_history,
    reserve,
)
from research.t34_round0.registration import require_registration


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
    entry = next(
        (e for e in receipt.get("exports", []) if e.get("path") == "round0/fault.json"), None
    )
    if not entry or entry.get("sha256") != sha256(path):
        return False
    fault = read_json(path)
    identity = fault.get("run_identity", {})
    return (
        fault.get("schema") == "t34-round0-fault/v1"
        and fault.get("preflight") is False
        and fault.get("unconditional") is True
        and fault.get("fault") in {"illegal_action", "nonfinite_q", "action_collapse_100k"}
        and fault.get("arm") in {"bare", "teacher"}
        and identity.get("commit") == registration["commit"]
        and identity.get("checkpoint_sha256") == B8_SHA
        and identity.get("runtime_id") == receipt.get("job_id")
    )


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
    if not state_dir.is_absolute() or state_dir.exists():
        raise HistoryError("supervisor state directory must be fresh and absolute")
    with FrozenWorktreeGuard(ROOT, registration["commit"]):
        terminals = require_published_history(registration)
        if any(t["status"] == "KILL" for t in terminals):
            raise HistoryError("KILL closes this study")
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
        reserve(attempt, registration)
        admission = attempt / "admission.json"
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
            if receipt.get("status") == "PASS" and receipt.get("stop_verified") is True:
                status, reason = "COMPLETE", "supervisor PASS and verified release"
            elif _controller_fault(attempt / "job", receipt, registration):
                status, reason = (
                    "COMPLETE",
                    "explained unconditional controller fault; scientific STOP only",
                )
            elif _mechanical(receipt, attempt / "job"):
                status, reason = (
                    "MECHANICAL_FAILURE",
                    "successful workload; verified release; failed artifact transport",
                )
    except subprocess.CalledProcessError:
        receipt_path = attempt / "job/receipt.json"
        if receipt_path.is_file() and _controller_fault(
            attempt / "job", read_json(receipt_path), registration
        ):
            status, reason = (
                "COMPLETE",
                "explained unconditional controller fault; scientific STOP only",
            )
        elif receipt_path.is_file() and _mechanical(read_json(receipt_path), attempt / "job"):
            status, reason = (
                "MECHANICAL_FAILURE",
                "successful workload; verified release; failed artifact transport",
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
