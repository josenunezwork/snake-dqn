#!/usr/bin/env python3
"""Prepare outcome-blind zero-export completion attempts for the ready queue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object, refusing partial or malformed evidence."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    """Hash evidence without interpreting scientific payloads."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocked_before_allocation(manifest: dict[str, Any], output: Path, log: Path) -> bool:
    """Recognize the exact broker pending-fence refusal before any CLI operation.

    This producer checks the fence before version/sessions/new and writes an
    immutable receipt. It never acquired an assignment, so there is none to stop.
    """
    path = output / "receipt.json"
    if not path.is_file() or not (output / "manifest.json").is_file():
        return False
    receipt = read(path)
    session = receipt.get("session", "")
    account = Path.home() / ".config/colab-cli/snake"
    return (
        re.search(r"^EXIT=[1-9]\d*$", log.read_text(), re.MULTILINE) is not None
        and receipt.get("error")
        == "Refused('unresolved allocation/stop: reconcile pending records first')"
        and receipt.get("status") == "INCOMPLETE"
        and receipt.get("failure_class") == "SUPERVISOR"
        and receipt.get("endpoint") is None
        and receipt.get("accounting") is None
        and receipt.get("usage_before") is None
        and receipt.get("usage_after") is None
        and receipt.get("cleanup_error") is None
        and receipt.get("terminal") is None
        and all(
            receipt.get(k) == []
            for k in (
                "cli_calls",
                "control_operations",
                "exports",
                "transfers",
                "contents_transfers",
            )
        )
        and receipt.get("manifest_sha256") == digest(output / "manifest.json")
        and all(receipt.get(k) == manifest.get(k) for k in ("job_id", "commit"))
        and re.fullmatch(r"snake-[a-z0-9_-]+-[a-f0-9]{12}", session) is not None
        and not (account / f"pending-{session}.json").exists()
        and not (account / f"active-{session}.json").exists()
        and not any((output / name).exists() for name in ("hardware.json", "terminal.json"))
    )


def eligible(
    manifest: dict[str, Any], output: Path, log: Path, unallocated_proof: Path | None = None
) -> dict[str, Any]:
    """Require a bound, exited, stopped mechanical attempt with no outputs."""
    receipt = read(output / "receipt.json")
    executed = output / "manifest.json"
    unallocated = blocked_before_allocation(manifest, output, log)
    if unallocated_proof is not None:
        if __package__:
            from . import dispatch
        else:
            import dispatch
        proof = read(unallocated_proof)
        unallocated = (
            proof.get("schema") == "colab-allocation-unavailable/v1"
            and proof.get("mode") == "LIVE"
            and proof.get("recovery") == {"endpoint": None}
            and proof.get("endpoint_ever_assigned") is False
            and proof.get("vm_compute_started") is False
            and proof.get("receipt", {}).get("path") == str(output / "receipt.json")
            and proof.get("receipt", {}).get("sha256") == digest(output / "receipt.json")
            and dispatch.allocation_unavailable(output, log, manifest["requirements"]["gpu"])
        )
        if not unallocated:
            raise ValueError("retry requires exact live unallocated proof")
    if (
        not re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE)
        or receipt.get("status") != "INCOMPLETE"
        or (not unallocated and receipt.get("stop_verified") is not True)
        or (not unallocated and receipt.get("cleanup_error") is not None)
        or receipt.get("exports") != []
        or receipt.get("terminal") is not None
        or (output / "terminal.json").exists()
        or (output / "exports").is_symlink()
        or ((output / "exports").exists() and any((output / "exports").rglob("*")))
        or receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != digest(executed)
    ):
        raise ValueError("retry requires exited, bound, verified stop and zero outputs")
    actual = read(executed)
    if any(actual.get(key) != manifest.get(key) for key in ("job_id", "commit", "output_dir")):
        raise ValueError("executed identity differs from dispatched manifest")
    if unallocated:
        return receipt
    # Some historical receipts mislabel any TimeoutExpired as RPC_EXEC_POLL_TIMEOUT.
    # Require positive transport evidence, and veto child/science/integrity errors.
    error = str(receipt.get("error", ""))
    if re.search(
        r"nan\b|non.?finite|overflow|integrity|assertion|child.*exit|"
        r"pytest|test.*fail|sha.?256.*mismatch|hash.*mismatch|numerical|out of memory|\boom\b",
        error,
        re.I,
    ):
        raise ValueError("scientific/integrity failure is never retryable")
    category = receipt.get("failure_class")
    patterns = {
        "RPC_EXEC_POLL_TIMEOUT": r"telemetry|kernel|rpc|colab.*exec|transport|poll timed out",
        "TRANSPORT_TIMEOUT": r"transport|rpc|colab.*exec|telemetry",
        "UPLOAD_CAP": r"upload|transfer.*cap",
        "ALLOCATION_REFUSED": r"allocation|assign|capacity|unavailable|503",
        "ALLOCATION_UNAVAILABLE": r"allocation|assign|capacity|unavailable|503",
    }
    if category not in patterns or not re.search(patterns[category], error, re.I):
        raise ValueError("unknown or ambiguous mechanical failure")
    return receipt


def completion_limit(manifest: dict[str, Any], requested: int | None = None) -> tuple[int, int]:
    """Return consumed completions and the narrowest declared package limit."""
    attempt = manifest.get("completion_attempt", 0)
    if type(attempt) is not int or not 0 <= attempt <= 2:
        raise ValueError("invalid completion_attempt")
    limits = [2]
    if manifest.get("timing_calibration", {}).get("no_automatic_retry") is True:
        limits.append(0)
    if "max_completion_attempts" in manifest:
        limits.append(manifest["max_completion_attempts"])
    resume = manifest.get("resume")
    if isinstance(resume, dict):
        limits.append(resume.get("max_outcome_blind_retries", 0))
    elif requested is None:
        raise ValueError("exempt canary requires an explicit package retry limit")
    envelope = manifest.get("technical_envelope", {})
    if "series_max_original_plus_completions" in envelope:
        limits.append(envelope["series_max_original_plus_completions"] - 1)
    if requested is not None:
        limits.append(requested)
    if any(type(limit) is not int or not 0 <= limit <= 2 for limit in limits):
        raise ValueError("invalid package completion limit")
    return attempt, min(limits)


def package_check(manifest: dict[str, Any], broker: Path, python: str) -> None:
    """Use the package's own outcome-blind resume/admission helpers when present.

    No exports means the original uploaded ledger remains valid, including any
    snapshots it already contains. Repacking that ledger would change canary
    input identity unnecessarily. These checks refuse newly completed units.
    """
    script = manifest.get("script", "")
    program = ""
    if script.startswith("research/doom_auc_smoke/"):
        program = (
            "from research.doom_auc_smoke.evidence import build_ledger\n"
            "cfg=json.loads(Path(next(i['path'] for i in m['inputs'] "
            "if i['dest']=='config.json')).read_text())\n"
            "assert build_ledger([Path(m['output_dir'])], cfg['identity'])['units']==[]\n"
        )
    elif script.startswith("research/spatial_student_pilot/"):
        program = (
            "from research.spatial_student_pilot.config import PilotConfig\n"
            "from research.spatial_student_pilot.resume import validate_ledger\n"
            "inputs={i['dest']:Path(i['path']) for i in m['inputs']}\n"
            "cfg=PilotConfig.from_dict(json.loads(inputs['config.json'].read_text()))\n"
            "validate_ledger(json.loads(inputs['verified-ledger.json'].read_text()), cfg)\n"
        )
    elif script.startswith("research/watch6_search_efficacy/"):
        program = "from research.watch6_search_efficacy.admission import admit\nadmit(m)\n"
    if program:
        subprocess.run(
            [
                python,
                "-c",
                "import json,sys\nfrom pathlib import Path\n"
                "sys.path.insert(0,sys.argv[1])\nm=json.loads(sys.argv[2])\n" + program,
                str(broker),
                json.dumps(manifest),
            ],
            cwd=broker,
            check=True,
            timeout=180,
        )


def prepare(
    manifest: Path,
    log: Path,
    destination: Path,
    broker: Path,
    *,
    limit: int | None = None,
    python: str = sys.executable,
    unallocated_proof: Path | None = None,
) -> Path:
    """Create a fresh immutable attempt, preserving source, inputs and all caps."""
    value = read(manifest)
    output = Path(value["output_dir"])
    eligible(value, output, log, unallocated_proof)
    attempt, maximum = completion_limit(value, limit)
    if attempt >= maximum:
        raise ValueError("package completion limit exhausted")
    package_check(value, broker, python)
    # Use the executed packet so supervisor defaults/accounting refresh are retained.
    successor = copy.deepcopy(read(output / "manifest.json"))
    token = uuid.uuid4().hex[:10]
    successor["job_id"] = value["job_id"][:42] + "-auto-" + token
    successor["output_dir"] = str(destination / "output")
    # Keep the scientific canary manifest unchanged apart from operational identity.
    # Main packages that already track completion_attempt advance it explicitly.
    successor["completion_attempt"] = attempt + 1
    if successor.get("requirements", {}).get("gpu") == "A100" and successor["requirements"].get(
        "prefer"
    ) == ["H100", "A100"]:
        # The sibling already consumed H100 preference. Retry its admitted A100
        # envelope directly; variants() correctly rejects an A100 primary.
        successor["requirements"].pop("prefer")
    if not destination.is_absolute():
        raise ValueError("fresh absolute retry destination required")
    destination.mkdir(parents=True, exist_ok=False)
    result = destination / "manifest.json"
    result.write_text(json.dumps(successor, indent=2, sort_keys=True) + "\n")
    proof = {
        "schema": "ops-zero-export-retry/v1",
        "completion_attempt": attempt + 1,
        "max_completion_attempts": maximum,
        "prior_manifest": {"path": str(manifest), "sha256": digest(manifest)},
        "prior_receipt": {
            "path": str(output / "receipt.json"),
            "sha256": digest(output / "receipt.json"),
        },
        "manifest_sha256": digest(result),
        "unallocated_proof": (
            {"path": str(unallocated_proof), "sha256": digest(unallocated_proof)}
            if unallocated_proof is not None
            else None
        ),
        "ledger_action": "preserve original uploaded ledger; zero new exports",
    }
    (destination / "RETRY.json").write_text(json.dumps(proof, indent=2) + "\n")
    for path in (result, destination / "RETRY.json"):
        path.with_name(path.name + ".sha256").write_text(digest(path) + "\n")
        path.with_name(path.name + ".complete").write_text("complete\n")
    return result


def main() -> int:
    """Expose preparation and integrated queue scanning without direct CLI mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        print(prepare(args.manifest, args.log, args.destination, args.broker, limit=args.limit))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
