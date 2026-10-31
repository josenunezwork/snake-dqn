#!/usr/bin/env python3
"""Prepare, but never dispatch, a fresh sibling of a pre-work GPU allocation failure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.colab_cli.cli_guard import KERNEL_CREATION_GUARD_ERROR as KERNEL_ERROR  # noqa: E402
from research.colab_cli.supervisor import validate  # noqa: E402
from tools.ops import qualification  # noqa: E402
from tools.ops.dispatch import BLOCK, require_resume_policy  # noqa: E402
from tools.ops.spend import read_spend  # noqa: E402


class Refused(ValueError):
    """Evidence does not prove the narrow pre-work failure required here."""


def is_kernel_creation_failure(error: Any) -> bool:
    """Match the terminal exception, never a message in a Rich source excerpt."""
    if not isinstance(error, str):
        return False
    try:
        call = ast.parse(error, mode="eval").body
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Refused"
            or len(call.args) != 1
            or call.keywords
        ):
            return False
        detail = ast.literal_eval(call.args[0])
    except (ValueError, SyntaxError):
        return False
    return (
        isinstance(detail, str)
        and detail.startswith("colab exec failed (1): ")
        and re.search(r"(?:\n|: )GuardError: " + re.escape(KERNEL_ERROR) + r"\s*\Z", detail)
        is not None
    )


def is_prework_warmup_failure(warmup: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """Accept only the two documented no-work readiness terminal shapes.

    A single supervisor ``exec`` is the workload boundary.  It can contain a
    transient helper probe before the guarded second kernel-creation POST fails;
    in that shape the supervisor records two *internal* warmup probes, while
    its receipt still binds exactly one failed CLI ``exec`` and no runtime work.
    """
    attempts = warmup.get("attempts")
    if (
        warmup.get("status") != "INCOMPLETE"
        or type(attempts) is not int
        or attempts not in (1, 2)
        or warmup.get("error") != receipt.get("error")
        or warmup.get("elapsed_seconds") != receipt.get("warmup_seconds")
    ):
        return False
    if attempts == 1:
        return "last_probe_error" not in warmup
    # Only the documented helper timeout may precede the terminal guarded
    # creation refusal.  Anything else could conceal useful runtime activity.
    return warmup.get("last_probe_error") == "GuardError('guard helper failed: TimeoutExpired')"


def digest(path: Path) -> str:
    """Hash a local evidence file without loading a large payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict[str, Any]:
    """Require object-shaped JSON evidence."""
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Refused(f"expected JSON object: {path}")
    return value


def _local_output(package: Path, manifest: dict[str, Any]) -> Path:
    value = manifest.get("output_dir")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise Refused("source output_dir must be absolute")
    output = Path(value).resolve(strict=True)
    if output.parent != package:
        raise Refused("source output_dir must be a direct child of the package")
    return output


def select_manifest(package: Path, name: str | None) -> Path:
    """Select exactly one package manifest with an existing terminal receipt."""
    if name is not None:
        path = (package / name).resolve(strict=True)
        if path.parent != package:
            raise Refused("--manifest must name a direct package child")
        return path
    candidates = []
    for path in package.glob("*.json"):
        value = load(path)
        if "job_id" not in value or "output_dir" not in value:
            continue
        output = Path(value["output_dir"])
        if (output / "receipt.json").is_file():
            candidates.append(path)
    if len(candidates) != 1:
        raise Refused("need exactly one attempted manifest; select it with --manifest BASENAME")
    return candidates[0]


def verify_failure(package: Path, source: Path) -> dict[str, Any]:
    """Prove first readiness failure, no workload execution, and verified release.

    Older supervisor receipts have no GPU-seconds counter. For those, zero useful
    GPU work is derived from the failed first readiness exec and absence of any
    upload, terminal, hardware admission, or export. Allocation elapsed-rate spend
    is retained separately; it is never reported as zero provider billing.
    """
    original = load(source)
    output = _local_output(package, original)
    receipt_path = output / "receipt.json"
    receipt = load(receipt_path)
    saved = output / "manifest.json"
    executed = load(saved)
    if receipt.get("manifest_sha256") != digest(saved):
        raise Refused("receipt does not bind saved manifest bytes")
    for key in ("job_id", "commit"):
        if not original.get(key) or receipt.get(key) != original[key]:
            raise Refused(f"source/receipt {key} differs")
    # Dispatch's live inventory refresh is the only permissible source difference.
    comparable = json.loads(json.dumps(executed))
    comparable.setdefault("budget", {})["other_assignments"] = original.get("budget", {}).get(
        "other_assignments", {}
    )
    if comparable != original:
        raise Refused("source differs from executed manifest beyond live assignment inventory")
    if not re.fullmatch(r"research/gpu_sim_spike/[a-zA-Z0-9_/-]+\.sh", original.get("script", "")):
        raise Refused("only GPU-sprint entrypoints are supported")
    if original.get("requirements", {}).get("gpu") in (None, "cpu"):
        raise Refused("only failed GPU allocations are supported")
    if original.get("backup_dir"):
        raise Refused("backup_dir packages need explicit operator preparation")
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("failure_class") != "SUPERVISOR"
        or not is_kernel_creation_failure(receipt.get("error"))
        or receipt.get("stop_verified") is not True
        or "cleanup_error" not in receipt
        or receipt["cleanup_error"] is not None
        or "terminal" not in receipt
        or receipt["terminal"] is not None
        or not receipt.get("endpoint")
    ):
        raise Refused("not a released first-readiness kernel-creation failure")
    for key in ("exports", "transfers", "contents_transfers"):
        if receipt.get(key) != []:
            raise Refused(f"receipt {key} must be explicitly empty")
    for key in ("gpu_seconds", "gpu_time_seconds", "workload_gpu_seconds"):
        if key in receipt and (type(receipt[key]) not in (int, float) or receipt[key] != 0):
            raise Refused(f"receipt {key} must be zero")
    if receipt.get("accounting", {}).get("allocated_endpoint", "missing") is not None:
        raise Refused("hardware accounting was admitted; workload state is uncertain")
    calls = receipt.get("cli_calls", [])
    expected = [
        ("version", "completed"),
        ("sessions", "completed"),
        ("usage", "completed"),
        ("new", "completed"),
        ("exec", "failed"),
        ("stop", "completed"),
        ("sessions", "completed"),
        ("usage", "completed"),
    ]
    if not isinstance(calls, list) or any(not isinstance(row, dict) for row in calls):
        raise Refused("malformed CLI call evidence")
    if [(row.get("command"), row.get("status")) for row in calls] != expected or any(
        type(row.get("attempt")) is not int or row["attempt"] != 1 for row in calls
    ):
        raise Refused("CLI calls do not prove exactly one failed pre-work readiness exec")
    warmup_path = output / "warmup-measurement.json"
    warmup = load(warmup_path)
    if not is_prework_warmup_failure(warmup, receipt):
        raise Refused("warmup evidence does not prove first readiness failed")
    allowed = {
        "accounting-baseline.json",
        "manifest.json",
        "receipt.json",
        "staging",
        "warmup-measurement.json",
    }
    if any(path.name not in allowed or path.is_symlink() for path in output.iterdir()):
        raise Refused("unexpected output evidence (exports, hardware, or runtime may exist)")
    spend = read_spend(receipt_path)
    return {
        "source_manifest": {"path": str(source), "sha256": digest(source)},
        "executed_manifest": {"path": str(saved), "sha256": digest(saved)},
        "receipt": {"path": str(receipt_path), "sha256": spend.sha256},
        "warmup": {"path": str(warmup_path), "sha256": digest(warmup_path)},
        "workload_gpu_seconds": 0,
        "gpu_seconds_basis": "failed first readiness exec; no workload admission or transfers",
        "prior_allocation_elapsed_seconds": spend.elapsed_seconds,
        "prior_elapsed_rate_units": spend.units,
    }


def publish(path: Path, payload: bytes) -> None:
    """Publish create-only preparation files with hash sidecar and marker last."""
    with path.open("xb") as stream:
        stream.write(payload)
    with Path(str(path) + ".sha256").open("x") as stream:
        stream.write(f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n")
    with Path(str(path) + ".complete").open("x") as stream:
        stream.write("complete\n")


def prepare(
    package: Path, *, broker: Path, manifest_name: str | None = None, suffix: str | None = None
) -> tuple[Path, str]:
    """Validate an unchanged workload and publish only a fresh dispatch namespace."""
    package, broker = package.resolve(strict=True), broker.resolve(strict=True)
    source = select_manifest(package, manifest_name)
    proof = verify_failure(package, source)
    original = load(source)
    instructions = package / "DISPATCH.txt"
    if instructions.exists() and any(
        BLOCK.search(line) and "live inventory" not in line.lower()
        for line in instructions.read_text().splitlines()
    ):
        raise Refused("source DISPATCH.txt contains blocking instructions")
    suffix = suffix or datetime.now(timezone.utc).strftime("fresh-%Y%m%dT%H%M%S").lower() + (
        "-" + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", suffix):
        raise Refused("suffix must be a lowercase identifier of at most 40 characters")
    sibling = package.with_name(package.name + "-" + suffix)
    name = original["job_id"][: 63 - len(suffix)].rstrip("-_") + "-" + suffix
    ops = Path(os.environ.get("OPS_DIR", qualification.OPS)).resolve()
    for path in (sibling, ops / "dispatch" / name, ops / "logs" / f"{name}.log"):
        if path.exists() or path.is_symlink():
            raise Refused(f"fresh namespace already exists: {path}")
    new = json.loads(json.dumps(original))
    new["job_id"] = name
    new["output_dir"] = str(sibling / Path(original["output_dir"]).name)
    state = sibling / "state"
    target = sibling / source.name
    require_resume_policy(new)
    registry = ops / "colab-cli-qualified.json"
    qualification.check(broker, registry)
    qualification.check(broker, registry, ref=new["commit"])
    validate(new, repo=broker)  # Offline: no Broker, CLI, auth, or provider calls.
    command = shlex.join(
        [
            str(broker / "tools/ops/dispatch.sh"),
            name,
            str(target),
            "--broker",
            str(broker),
            "--cwd",
            str(broker),
            "--state-dir",
            str(state),
            "--dispatch-txt",
            str(sibling / "DISPATCH.txt"),
        ]
    )
    if "OPS_DIR" in os.environ:
        command = shlex.join(["env", f"OPS_DIR={ops}"]) + " " + command
    sibling.mkdir()  # Exclusive; a concurrent preparation cannot overwrite it.
    publish(target, (json.dumps(new, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    proof.update(
        {
            "schema": "gpu-sprint-fresh-redispatch/v1",
            "prepared_only": True,
            "manifest_sha256": digest(target),
            "job_id": name,
            "state_dir": str(state),
            "output_dir": new["output_dir"],
            "dispatch_command": command,
            "changed_manifest_fields": ["job_id", "output_dir"],
            "qualification_registry_sha256": digest(registry),
            "helper_sha256": digest(Path(__file__)),
            "offline_validation": "PASS",
            "inputs_reused_unchanged": True,
        }
    )
    publish(
        sibling / "PREPARATION.json", (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    )
    publish(
        sibling / "DISPATCH.txt",
        ("# Fresh allocation; operator dispatch only.\n" + command + "\n").encode(),
    )
    return sibling, command


def main(argv: Sequence[str] | None = None) -> int:
    """Expose explicit, offline, one-at-a-time preparation to the operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--broker", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest", help="Manifest basename when the package has several receipts"
    )
    parser.add_argument(
        "--suffix", help="Fresh identifier; default UTC timestamp plus random suffix"
    )
    args = parser.parse_args(argv)
    try:
        sibling, command = prepare(
            args.package, broker=args.broker, manifest_name=args.manifest, suffix=args.suffix
        )
        print(f"PREPARED {sibling}\n{command}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
