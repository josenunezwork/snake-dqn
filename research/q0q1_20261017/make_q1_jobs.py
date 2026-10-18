#!/usr/bin/env python3
"""Materialize the approved Q1 queue packet from completed, immutable evidence.

This Mac-side tool only writes JSON review artifacts.  It never starts a queue
worker, mounts Drive, or replaces an existing job file.  Its deliberately strict
input contract prevents a synthetically filled budget or a stale parity probe
from making the approved registration runnable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017 import q1_bank, q1_evaluate, q1_parity, queue_jobs
from research.q0q1_20261017.qualification import PACKET_SCHEMA, validate_packet

OWNER_RELEASE = "APPROVED_Q0Q1_20261017"
AMENDMENT = "A1"
OWNER_ADMISSION_SCHEMA = "q0q1-owner-admission/v1"
DRIVE_ROOT = "/content/drive/MyDrive/t16_colab"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _number(value: Any, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite positive number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _qualified_receipt(path: Path, hardware: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _json_object(path, f"Q0b {hardware} receipt")
    packet = receipt.get("packet", receipt)
    if not isinstance(packet, Mapping):
        raise ValueError(f"Q0b {hardware} receipt lacks its packet")
    if packet.get("schema") != PACKET_SCHEMA or packet.get("hardware") != hardware:
        raise ValueError(f"Q0b receipt is not a {hardware} qualification packet")
    outcome = validate_packet(packet)
    if outcome.get("status") != "QUALIFIED":
        raise ValueError(f"Q0b {hardware} receipt is not qualified: {outcome.get('errors')}")
    for field in (
        "observed_units",
        "observed_cpu_unit_rate_per_hour",
    ):
        _number(receipt.get(field), f"Q0b {hardware} receipt.{field}")
    for field in ("observed_gpu_hours", "observed_cpu_hours"):
        _number(receipt.get(field), f"Q0b {hardware} receipt.{field}", positive=False)
    return receipt, dict(packet)


def _validate_mac_reference(path: Path) -> dict[str, Any]:
    reference = _json_object(path, "Mac parity reference")
    bank = q1_bank.load_world_bank()
    expected_bank = hashlib.sha256(canonical(bank).encode()).hexdigest()
    expected_source = hashlib.sha256(canonical(q1_evaluate.source_manifest()).encode()).hexdigest()
    if (
        reference.get("schema") != q1_parity.SCHEMA
        or reference.get("bank_namespace") != q1_bank.NAMESPACE
        or reference.get("bank_sha256") != expected_bank
        or reference.get("b8_sha256") != q1_evaluate.B8_SHA256
        or reference.get("source_sha256") != expected_source
        or reference.get("runtime", {}).get("system") != "Darwin"
    ):
        raise ValueError("Mac parity reference is not bound to the registered Q1 identity")
    if {row.get("world_id") for row in reference.get("worlds", [])} != {
        str(world["world_seed"]) for world in bank["parity_worlds"]
    }:
        raise ValueError("Mac parity reference does not contain exactly the four registered worlds")
    return reference


def _validate_runtime_receipt(path: Path) -> dict[str, Any]:
    receipt = _json_object(path, "runtime parity receipt")
    bank_sha = hashlib.sha256(canonical(q1_bank.load_world_bank()).encode()).hexdigest()
    q1_parity.validate_runtime_receipt(
        receipt,
        bank_sha256=bank_sha,
        b8_sha256=q1_evaluate.B8_SHA256,
        source_manifest=q1_evaluate.source_manifest(),
    )
    return receipt


def _validate_owner_admission(path: Path) -> dict[str, Any]:
    admission = _json_object(path, "owner admission")
    registration = Path(__file__).with_name("PREREGISTRATION.md")
    required = {
        "schema": OWNER_ADMISSION_SCHEMA,
        "status": "OWNER_ADMITTED",
        "owner_release": OWNER_RELEASE,
        "amendment": AMENDMENT,
        "registration_path": "research/q0q1_20261017/PREREGISTRATION.md",
        "registration_sha256": sha256_file(registration),
    }
    if any(admission.get(name) != value for name, value in required.items()):
        raise ValueError("owner admission does not bind the approved A1 registration")
    return admission


def _study_plan(
    l4: Mapping[str, Any], a100: Mapping[str, Any], q0_accounting: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the A1 forecast only from observed Q0b metadata, never defaults."""
    metadata = (l4, a100)
    q0_units = _number(q0_accounting.get("q0_units"), "Q0 accounting q0_units")
    q0_gpu = _number(
        q0_accounting.get("q0_gpu_hours"), "Q0 accounting q0_gpu_hours", positive=False
    )
    q0_cpu = _number(
        q0_accounting.get("q0_cpu_hours"), "Q0 accounting q0_cpu_hours", positive=False
    )
    cpu_rate = _number(
        q0_accounting.get("cpu_unit_rate_per_hour"), "Q0 accounting cpu_unit_rate_per_hour"
    )
    for total, field in (
        (q0_units, "observed_units"),
        (q0_gpu, "observed_gpu_hours"),
        (q0_cpu, "observed_cpu_hours"),
    ):
        if total < sum(_number(row[field], f"Q0b {field}", positive=False) for row in metadata):
            raise ValueError(f"Q0 accounting undercounts completed packets' {field}")
    cpu_rates = {
        _number(row["observed_cpu_unit_rate_per_hour"], "Q0b CPU debit rate") for row in metadata
    }
    if len(cpu_rates) != 1:
        raise ValueError("Q0b packets disagree on the observed CPU debit rate")
    if cpu_rate not in cpu_rates:
        raise ValueError("Q0 accounting CPU debit rate does not match the measured Q0b packets")
    cpu_seconds = _number(l4.get("cpu_seconds_per_episode"), "L4 cpu_seconds_per_episode")
    efficiency = _number(l4.get("cpu_worker_efficiency"), "L4 cpu_worker_efficiency")
    if efficiency > 1:
        raise ValueError("L4 cpu_worker_efficiency cannot exceed 1")
    attempt_overhead = l4.get("q1_attempt_overhead_seconds")
    attempt_overhead = _number(attempt_overhead, "L4 q1_attempt_overhead_seconds", positive=True)
    q1_gpu_overhead_hours = 20 * attempt_overhead / 3600
    eval_job_overhead = _number(
        q0_accounting.get("cpu_eval_job_overhead_seconds"),
        "Q0 accounting cpu_eval_job_overhead_seconds",
    )
    # Evaluation has 100 immutable shards with four-world parity probes. Training
    # CPU parity runs inside its L4 attempt and is charged to GPU elapsed time.
    return {
        "schema": "q0q1-study-budget/v1",
        "owner_review_status": OWNER_RELEASE,
        "owner_release": OWNER_RELEASE,
        "amendment": AMENDMENT,
        "dispatch": True,
        "a1_envelope": {
            "q0_units_cap": 40.0,
            "combined_units_cap": 175.0,
            "reserve_fraction": 0.20,
        },
        "q1_training_hardware": "L4",
        "q0_units": q0_units,
        "q0_gpu_hours": q0_gpu,
        "q0_cpu_hours": q0_cpu,
        "q1_gpu_overhead_hours": q1_gpu_overhead_hours,
        "q1_training_attempt_overhead_seconds": attempt_overhead,
        "q1_cpu_overhead_hours": 100 * (4 * cpu_seconds + eval_job_overhead) / (3600 * efficiency),
        "cpu_unit_rate_per_hour": cpu_rate,
        "cpu_seconds_per_episode": cpu_seconds,
        "cpu_workers": 1,
        "cpu_worker_efficiency": efficiency,
        "evaluation_episodes": 3200,
        "runs_per_arm": 5,
        "decisions_per_run": 8_000_000,
        "evaluation_jobs": 100,
        "parity_worlds_per_evaluation_job": 4,
    }


def _clean_head(commit: str, root: Path) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    if actual.returncode or actual.stdout.strip() != commit:
        raise ValueError("--commit must equal the current immutable HEAD")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root, check=False)
    if dirty.returncode:
        raise ValueError("refusing to bind jobs while the checkout has uncommitted source changes")


def _stage_evidence(
    source: Path, target: Path, *, member_name: str | None = None, member_suffix: str | None = None
) -> Path:
    """Copy direct JSON or safely extract a hash-verified queue archive member."""
    if source.suffixes[-2:] != [".tar", ".gz"]:
        _json_object(source, "evidence")
        target.write_bytes(source.read_bytes())
        return target
    sidecar = source.with_name(source.name + ".sha256")
    fields = sidecar.read_text().split() if sidecar.is_file() else []
    if not fields or fields[0] != sha256_file(source):
        raise ValueError(f"queue archive sidecar does not bind {source}")
    with tarfile.open(source, "r:gz") as archive:
        if member_name is not None:
            members = [member for member in archive.getmembers() if member.name == member_name]
            if not members and member_suffix is not None:
                members = [
                    member for member in archive.getmembers() if member.name.endswith(member_suffix)
                ]
        else:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith(member_suffix or "receipt.json")
            ]
        if (
            len(members) != 1
            or members[0].name.startswith("/")
            or ".." in Path(members[0].name).parts
        ):
            raise ValueError(
                "queue archive lacks one safe "
                f"{member_name or member_suffix or 'receipt.json'} member"
            )
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("queue archive evidence member cannot be read")
        raw = stream.read()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("queue archive evidence member is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("queue archive evidence member is not a JSON object")
    target.write_bytes(raw)
    return target


def _completed_q0_receipt(path: Path, schema: str) -> None:
    receipt = _json_object(path, "Q0 receipt")
    if receipt.get("schema") != schema or receipt.get("status") != "COMPLETE":
        raise ValueError(f"Q1 requires completed Q0 receipt {schema}")


def make_jobs(
    *,
    commit: str,
    l4_receipt_path: Path,
    a100_receipt_path: Path,
    runtime_receipt_path: Path,
    mac_reference_path: Path,
    owner_admission_path: Path,
    q0_accounting_path: Path,
    q0b_l4_job_id: str,
    q0b_a100_job_id: str,
    study_plan_drive_path: str,
    drive_root: str = DRIVE_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    l4, l4_packet = _qualified_receipt(l4_receipt_path, "L4")
    a100, a100_packet = _qualified_receipt(a100_receipt_path, "A100")
    if a100_packet.get("source") != l4_packet.get("source") or a100_packet.get(
        "config"
    ) != l4_packet.get("config"):
        raise ValueError("Q0b hardware comparison must bind identical sources and recipes")
    runtime_receipt = _validate_runtime_receipt(runtime_receipt_path)
    mac_reference = _validate_mac_reference(mac_reference_path)
    if runtime_receipt.get("reference") != mac_reference:
        raise ValueError("runtime parity receipt does not embed the supplied Mac reference")
    if l4.get("runtime_parity_receipt_sha256") != sha256_file(runtime_receipt_path):
        raise ValueError("L4 packet does not bind the supplied runtime parity receipt")
    _validate_owner_admission(owner_admission_path)
    accounting = _json_object(q0_accounting_path, "Q0 observed accounting")
    plan = _study_plan(l4, a100, accounting)
    from research.q0q1_20261017.qualification import validate_admission

    result = validate_admission(l4_packet, plan)
    if (
        result.get("status") != "ADMITTED_FOR_DISPATCH"
        or result.get("dispatch_authorized") is not True
    ):
        raise ValueError(f"A1 forecast is not admitted for dispatch: {result.get('errors')}")
    measured = validate_packet(l4_packet).get("measured_cells", {})
    for cell in ("restart/n3", "mature/n3", "restart/n1", "mature/n1"):
        speed = measured.get(cell, {}).get("decisions_per_second")
        if not isinstance(speed, (int, float)) or speed <= 0 or 8_000_000 / speed > 5100:
            raise ValueError(
                f"L4 qualification cannot complete Q1 {cell} within the 5,100-second cap"
            )
    bindings = queue_jobs.Bindings(
        q0b_l4_receipt=str(l4_receipt_path),
        q0b_l4_receipt_sha256=sha256_file(l4_receipt_path),
        study_plan=study_plan_drive_path,
        study_plan_sha256=hashlib.sha256(pretty_json(plan)).hexdigest(),
        runtime_parity_receipt=str(runtime_receipt_path),
        runtime_parity_receipt_sha256=sha256_file(runtime_receipt_path),
        mac_parity_reference=str(mac_reference_path),
        mac_parity_reference_sha256=sha256_file(mac_reference_path),
        parity_owner_admission=str(owner_admission_path),
        parity_owner_admission_sha256=sha256_file(owner_admission_path),
    )
    jobs = [job for job in queue_jobs.build_jobs(commit, bindings) if job["id"].startswith("q1-")]
    for job in jobs:
        job["env"]["Q0Q1_REGISTRATION_STATUS"] = OWNER_RELEASE
        job["env"]["Q0Q1_OWNER_RELEASE"] = OWNER_RELEASE
        job["env"]["Q0Q1_Q0B_A100_RECEIPT"] = str(a100_receipt_path)
        job["env"]["Q0Q1_Q0B_A100_RECEIPT_SHA256"] = sha256_file(a100_receipt_path)
        job["env"]["Q0Q1_MASS_ARCHIVE_DIR"] = "q0q1-20261017/q0/audit-t23-mass-a100r"
        job["env"]["Q0Q1_INIT_ARCHIVE_DIR"] = "q0q1-20261017/q0/audit-b8-init-a100r2"
        preserved = [item for item in job["depends_on"] if not item.startswith("q0-audit-")]
        job["depends_on"] = list(
            dict.fromkeys(
                [
                    "q0-audit-t23-mass-a100r",
                    "q0-audit-b8-init-a100r2",
                    *preserved,
                    q0b_l4_job_id,
                    q0b_a100_job_id,
                ]
            )
        )
        if job["id"].startswith("q1-train-"):
            job["timeout_minutes"] = math.ceil(
                (5100 + plan["q1_training_attempt_overhead_seconds"] + 60) / 60
            )
    runtime = l4_packet.get("runtime", {})
    cores = runtime.get("cpu_cores") if isinstance(runtime, Mapping) else None
    if (
        isinstance(cores, bool)
        or not isinstance(cores, (int, float))
        or int(cores) != cores
        or cores < 1
    ):
        raise ValueError("L4 packet lacks an integer observed CPU-core capacity")
    cores = int(cores)
    for job in jobs:
        if job["id"].startswith("q1-train-"):
            collector = "n1" if job["env"]["Q1_ARM"] in ("C", "D") else "n3"
            actors = {
                cell.get("actor_count")
                for cell in l4_packet.get("cells", [])
                if cell.get("collector") == collector
            }
            if (
                len(actors) != 1
                or not isinstance(next(iter(actors)), int)
                or cores < next(iter(actors))
            ):
                raise ValueError("L4 packet cannot hold the registered actor configuration")
            job["requirements"]["min_vcpu"] = cores
            # Verify the exact normalized source/config identity before an arm
            # can inherit the timing result.  This is import-local to avoid the
            # queue adapter becoming a generator dependency at module import.
            from research.q0q1_20261017 import run_queue_job

            with tempfile.TemporaryDirectory(prefix="q1-binding-") as temporary:
                env = dict(job["env"])
                env["Q1_ACTORS"] = str(next(iter(actors)))
                run_queue_job.qualified_recipe_binding(
                    l4_packet, run_queue_job.command(env, Path(temporary))
                )
    if len(jobs) != 120 or sum(job["id"].startswith("q1-train-") for job in jobs) != 20:
        raise AssertionError(
            "Q1 generator did not make exactly 20 training and 100 evaluation jobs"
        )
    return jobs, plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--q0b-l4-receipt", type=Path, required=True)
    parser.add_argument("--q0b-a100-receipt", type=Path, required=True)
    parser.add_argument("--q0-mass-archive", type=Path, required=True)
    parser.add_argument("--q0-init-archive", type=Path, required=True)
    parser.add_argument("--runtime-parity-receipt", type=Path)
    parser.add_argument("--mac-parity-reference", type=Path, required=True)
    parser.add_argument(
        "--q0-accounting",
        type=Path,
        required=True,
        help="observed Q0 units/hours and nonzero CPU debit rate; never a forecast",
    )
    parser.add_argument(
        "--owner-admission",
        type=Path,
        default=Path(__file__).with_name("OWNER_ADMISSION_20261018.json"),
    )
    parser.add_argument("--q0b-l4-job-id", required=True)
    parser.add_argument("--q0b-a100-job-id", required=True)
    parser.add_argument("--drive-root", default=DRIVE_ROOT)
    parser.add_argument(
        "--drive-output-dir",
        default=f"{DRIVE_ROOT}/q0q1-20261017/q1/jobs_approved",
        help="Colab-visible directory where --out-dir is synchronized before enqueue",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    _clean_head(args.commit, root)
    if args.out_dir.exists():
        parser.error(f"refusing to replace existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    evidence = args.out_dir / "evidence"
    evidence.mkdir()
    mass = _stage_evidence(
        args.q0_mass_archive, evidence / "q0-mass-receipt.json", member_name="attempt/receipt.json"
    )
    init = _stage_evidence(
        args.q0_init_archive, evidence / "q0-init-receipt.json", member_name="attempt/receipt.json"
    )
    _completed_q0_receipt(mass, "q0-t23-mass-audit/v1")
    _completed_q0_receipt(init, "q0-b8-init-fidelity/v1")
    l4 = _stage_evidence(
        args.q0b_l4_receipt, evidence / "q0b-l4-receipt.json", member_name="attempt/receipt.json"
    )
    a100 = _stage_evidence(
        args.q0b_a100_receipt,
        evidence / "q0b-a100-receipt.json",
        member_name="attempt/receipt.json",
    )
    runtime = _stage_evidence(
        args.runtime_parity_receipt or args.q0b_l4_receipt,
        evidence / "runtime-parity.json",
        member_name="attempt/measurement/runtime-parity.json",
        member_suffix="runtime-parity.json",
    )
    mac = _stage_evidence(args.mac_parity_reference, evidence / "mac-parity-reference.json")
    owner = _stage_evidence(args.owner_admission, evidence / "owner-admission.json")
    jobs, plan = make_jobs(
        commit=args.commit,
        l4_receipt_path=l4,
        a100_receipt_path=a100,
        runtime_receipt_path=runtime,
        mac_reference_path=mac,
        owner_admission_path=owner,
        q0_accounting_path=args.q0_accounting,
        q0b_l4_job_id=args.q0b_l4_job_id,
        q0b_a100_job_id=args.q0b_a100_job_id,
        study_plan_drive_path=f"{args.drive_output_dir.rstrip('/')}/study-plan.json",
        drive_root=args.drive_root,
    )
    drive_evidence = f"{args.drive_output_dir.rstrip('/')}/evidence"
    replacement = {
        str(l4): f"{drive_evidence}/q0b-l4-receipt.json",
        str(a100): f"{drive_evidence}/q0b-a100-receipt.json",
        str(runtime): f"{drive_evidence}/runtime-parity.json",
        str(mac): f"{drive_evidence}/mac-parity-reference.json",
        str(owner): f"{drive_evidence}/owner-admission.json",
    }
    for job in jobs:
        for key, value in tuple(job["env"].items()):
            if value in replacement:
                job["env"][key] = replacement[value]
    (args.out_dir / "study-plan.json").write_bytes(pretty_json(plan))
    for job in jobs:
        (args.out_dir / f"{job['id']}.json").write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
