"""Create the frozen S1 admission plan without allocating any remote resource.

``prepare`` is deliberately local and create-only.  ``run`` lives in :mod:`run`;
keeping allocation out of this module makes a reviewable plan possible before an
operator has authority to spend a single Colab unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from research.s1_mature_20261018 import bank, ledger, manifests

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "s1-mature-admission-plan/v1"
APPROVAL = manifests.APPROVAL
ACCEPTANCE_UNITS = 4.0


class AdmissionRefused(RuntimeError):
    """The proposed S1 run is not yet admissible."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise AdmissionRefused(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def published_commit(repo: Path) -> str:
    """Require the plan source to be exactly the fetched public main head."""
    head, main = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "origin/main")
    if head != main:
        raise AdmissionRefused("prepare requires HEAD to equal the current published origin/main")
    if git(repo, "status", "--porcelain"):
        raise AdmissionRefused("prepare requires a clean published worktree")
    return head


def source_closure(repo: Path, commit: str) -> dict[str, str]:
    """Hash every tracked source byte visible at the pinned public commit."""
    paths = git(repo, "ls-files", "-z").split("\0")
    result: dict[str, str] = {}
    for relative in paths:
        if relative:
            target = repo / relative
            if not target.is_file():
                raise AdmissionRefused(f"tracked source is absent: {relative}")
            result[relative] = sha256(target)
    if not result or git(repo, "rev-parse", "HEAD") != commit:
        raise AdmissionRefused("source closure cannot be bound to the published commit")
    return result


def require_approved_preregistration(path: Path) -> dict[str, str]:
    text = path.read_text()
    # The original DRAFT header remains part of the immutable historical bytes
    # after ratification, so only a prospective approval line can admit it.
    approved = any(line.strip() == f"**Status: {APPROVAL}.**" for line in text.splitlines()[:12])
    if not approved:
        raise AdmissionRefused("preregistration lacks the prospective Claude approval status line")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def liabilities(manifest_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return registered worst-case reservations including one five-minute cleanup tail."""
    output: dict[str, dict[str, float]] = {}
    for row in manifest_rows:
        job, requirements = row.get("job_id"), row.get("requirements", {})
        arm = str(row.get("env", {}).get("S1_ARM", ""))
        est, timeout = requirements.get("est_minutes"), row.get("timeout_minutes")
        expected = 30 if arm == "n3" else 68 if arm == "n1" else None
        expected_timeout = 60 if arm == "n3" else 136 if arm == "n1" else None
        if not isinstance(job, str) or est != expected or timeout != expected_timeout:
            raise AdmissionRefused(f"manifest {job!r} violates registered {arm} timing")
        hours = (float(timeout) + 5.0) / 60.0
        output[job] = {
            "units": hours * 2.0,
            "gpu_allocation_hours": hours,
            "mac_worker_hours": 0.0,
            "elapsed_hours": hours,
        }
    if len(output) != 10:
        raise AdmissionRefused("S1 requires exactly ten registered training jobs")
    return output


def _input(value: str) -> manifests.ExternalInput:
    name, path, dest = value.split(":", 2)
    return manifests.ExternalInput(name, Path(path), dest)


def priced_mac_bound(path: Path) -> dict[str, float]:
    """Read the completed real-entrypoint pricing receipt, never a hand estimate."""
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise AdmissionRefused("Mac preflight/evaluation pricing receipt is not PASS")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        raise AdmissionRefused("pricing receipt lacks bound training and evaluation evidence")
    training = evidence.get("training")
    evaluation = evidence.get("evaluation")
    for name, item in (("training", training), ("evaluation", evaluation)):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise AdmissionRefused(f"pricing {name} evidence is unbound")
        target = Path(item["path"])
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise AdmissionRefused(f"pricing {name} evidence changed or is absent")
    training_value, evaluation_value = json.loads(Path(training["path"]).read_text()), json.loads(
        Path(evaluation["path"]).read_text()
    )
    if training_value.get("status") != "PASS" or training_value.get("counter_admission") != "PASS":
        raise AdmissionRefused("real M3-C training preflight has not passed counter admission")
    if (
        evaluation_value.get("schema") != "s1-eval-timing/v1"
        or evaluation_value.get("status") != "completed"
        or int(evaluation_value.get("episode_count", 0)) < 10
    ):
        raise AdmissionRefused("disjoint >=10 episode evaluation timing fixture has not passed")
    for relative, digest in evaluation_value["identity"]["source_manifest"].items():
        if sha256(ROOT / relative) != digest:
            raise AdmissionRefused("evaluation timing source changed")
    bounds = value.get("complete_upper_bound")
    if not isinstance(bounds, dict):
        raise AdmissionRefused("pricing receipt lacks a complete_upper_bound")
    try:
        mac = float(bounds["mac_worker_hours"])
        elapsed = float(bounds["elapsed_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdmissionRefused("pricing receipt lacks finite Mac and elapsed bounds") from exc
    if not math.isfinite(mac) or not math.isfinite(elapsed) or mac < 0 or elapsed < 0:
        raise AdmissionRefused("pricing upper bounds must be nonnegative")
    # A small timing fixture does not price the new multi-GiB full-state recovery
    # path.  The transport qualification must measure the actual archive/chunk
    # protocol and bind its source and >=10 disjoint timing episodes.
    transport = value.get("full_replay_transport")
    if not isinstance(transport, dict) or transport.get("status") != "PASS":
        raise AdmissionRefused("full-replay transport is unqualified; S1 admission is refused")
    required = (
        "source_sha256",
        "timing_receipt_sha256",
        "snapshot_bytes",
        "per_run_gpu_hours_upper_bound",
    )
    if (
        any(not transport.get(key) for key in required)
        or int(transport["snapshot_bytes"]) < 4 * 1024**3
    ):
        raise AdmissionRefused("full-replay transport receipt lacks a qualified full-size bound")
    if (
        transport["source_sha256"] != sha256(ROOT / "research/s1_mature_20261018/train.py")
        or transport["timing_receipt_sha256"] != evaluation["sha256"]
    ):
        raise AdmissionRefused("full-replay qualification source or timing binding changed")
    gpu = float(transport["per_run_gpu_hours_upper_bound"])
    if not math.isfinite(gpu) or gpu < 0:
        raise AdmissionRefused("full-replay transport GPU bound must be nonnegative")
    return {"mac_worker_hours": mac, "elapsed_hours": elapsed, "per_run_gpu_hours": gpu}


def prepare(args: argparse.Namespace) -> Path:
    repo, out = args.repo.resolve(), args.out.resolve()
    commit = published_commit(repo)
    prereg = require_approved_preregistration(args.preregistration.resolve())
    seed_value, audit_value = json.loads(args.seed_bank.read_text()), json.loads(
        args.seed_audit.read_text()
    )
    if seed_value.get("seeds") != list(bank.TRAINING_SEEDS):
        raise AdmissionRefused("uploaded seed bank is not the registered S1 seed block")
    world_value = json.loads(args.world_bank.read_text())
    if audit_value.get("pass") is not True or audit_value.get(
        "world_bank_semantic_sha256"
    ) != world_value.get("allocation_semantic_sha256"):
        raise AdmissionRefused("seed audit does not pass for the selected S1 world bank")
    from research.s1_mature_20261018.review_jobs import verified_inputs

    frozen_inputs = verified_inputs()
    parsed_inputs = [_input(item) for item in args.input] if args.input else frozen_inputs
    expected_inputs = {item.dest: item.as_manifest()["sha256"] for item in frozen_inputs}
    if {item.dest: item.as_manifest()["sha256"] for item in parsed_inputs} != expected_inputs:
        raise AdmissionRefused("uploaded inputs differ from the frozen qualified input inventory")
    seed_upload = [item for item in parsed_inputs if item.dest == "seed-bank.json"]
    if len(seed_upload) != 1 or seed_upload[0].path.resolve() != args.seed_bank.resolve():
        raise AdmissionRefused(
            "supervisor seed-bank input must be the exact registered seed-bank bytes"
        )
    rows = manifests.make_manifests(
        seed_bank=args.seed_bank.resolve(),
        published_commit=commit,
        registration_status=APPROVAL,
        owner_release=APPROVAL,
        output_root=args.artifact_root.resolve() / "training",
        inputs=parsed_inputs,
    )
    liability = liabilities(rows)
    mac_bound = priced_mac_bound(args.pricing.resolve())
    liability["mac-final-evaluation"] = {
        "units": 0.0,
        "gpu_allocation_hours": 0.0,
        "mac_worker_hours": mac_bound["mac_worker_hours"],
        "elapsed_hours": mac_bound["elapsed_hours"],
    }
    training = {key: sum(item[key] for item in liability.values()) for key in ledger.DIMENSIONS}
    # The timeout reservation must dominate the measured full-state transport
    # bound for each run; otherwise the supposedly inclusive estimate is false.
    if any(
        item["gpu_allocation_hours"] < mac_bound["per_run_gpu_hours"]
        for job, item in liability.items()
        if job.startswith("s1-mature-")
    ):
        raise AdmissionRefused(
            "registered job timeout does not cover qualified full-replay transport"
        )
    # Acceptance is a conservative two GPU/allocation hours inside its four-unit
    # allowance.  Full-state/export slack is already reserved per job above.
    complete = {
        "units": training["units"] + ACCEPTANCE_UNITS,
        "gpu_allocation_hours": training["gpu_allocation_hours"] + 2.0,
        "mac_worker_hours": mac_bound["mac_worker_hours"],
        "elapsed_hours": training["elapsed_hours"] + 2.0,
    }
    if complete["units"] > 48:
        raise AdmissionRefused("training plus acceptance upper bound exceeds usable units")
    if complete["gpu_allocation_hours"] > 20:
        raise AdmissionRefused("training upper bound exceeds usable GPU allocation hours")
    if complete["mac_worker_hours"] > 19.2 or complete["elapsed_hours"] > 38.4:
        raise AdmissionRefused(
            "complete preflight/training/evaluation upper bound exceeds a usable cap"
        )
    if out.exists() or args.manifest_dir.exists():
        raise AdmissionRefused("plan or manifest directory already exists")
    args.manifest_dir.mkdir(parents=True)
    written: list[dict[str, str]] = []
    for row in rows:
        target = args.manifest_dir / f"{row['job_id']}.json"
        write_once(target, row)
        written.append({"job_id": row["job_id"], "path": str(target), "sha256": sha256(target)})
    mac_manifest = args.manifest_dir / "mac-final-evaluation.json"
    write_once(
        mac_manifest,
        {
            "job_id": "mac-final-evaluation",
            "kind": "mac",
            "workers": 2,
            "timeout_minutes": mac_bound["elapsed_hours"] * 60,
        },
    )
    plan = {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_DISPATCHED",
        "commit": commit,
        "preregistration": prereg,
        "source_closure": source_closure(repo, commit),
        "world_bank": {"path": str(args.world_bank.resolve()), "sha256": sha256(args.world_bank)},
        "seed_audit": {"path": str(args.seed_audit.resolve()), "sha256": sha256(args.seed_audit)},
        "manifests": written,
        "liabilities": liability,
        "mac_manifest": {"path": str(mac_manifest), "sha256": sha256(mac_manifest)},
        "acceptance": {
            "units_included_maximum": ACCEPTANCE_UNITS,
            "review_required_schema": ledger.REVIEW_SCHEMA,
        },
        "pricing_receipt": {"path": str(args.pricing.resolve()), "sha256": sha256(args.pricing)},
        "complete_upper_bound": complete,
        "run_rules": {
            "serialized": True,
            "snapshot_before_every_job": True,
            "no_retries": True,
            "no_outcome_analysis_before_full_bank_and_independent_audit": True,
            "mac_workers_max": 2,
            "single_thread_per_worker": True,
        },
    }
    write_once(out, plan)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="bind approved source and render create-only manifests")
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "research/s1_mature_20261018/PREREGISTRATION.md",
    )
    p.add_argument("--seed-bank", type=Path, required=True)
    p.add_argument("--world-bank", type=Path, required=True)
    p.add_argument("--seed-audit", type=Path, required=True)
    p.add_argument("--pricing", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", action="append", default=[], metavar="NAME:PATH:DEST")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
