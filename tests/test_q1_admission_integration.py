"""End-to-end admission proof for the approved Q0b/Q1 packet generator.

The evidence is synthetic so this test never starts a learner or an evaluator,
but the generator, packet validator, parity validator, queue adapter and all
file digest checks are real.  The temporary directory is laid out like Drive,
including renamed Q0 audit output directories.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from research.q0q1_20261017 import inventory_receipt
from research.q0q1_20261017 import make_q1_jobs as generator
from research.q0q1_20261017 import q0b_measure, queue_jobs, run_queue_job
from tests.test_make_q1_jobs import _a3_accounting
from tests.test_q1_parity import _receipt as parity_receipt
from tests.test_q1_qualification import _packet as qualification_packet

COMMIT = "a" * 40
APPROVED = "APPROVED_Q0Q1_20261017"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive(directory: Path, receipt: dict) -> Path:
    directory.mkdir(parents=True)
    archive = directory / "result.tar.gz"
    raw = json.dumps(receipt, sort_keys=True).encode()
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("attempt/receipt.json")
        member.size = len(raw)
        stream.addfile(member, io.BytesIO(raw))
    (directory / "result.tar.gz.sha256").write_text(f"{_sha(archive)}  result.tar.gz\n")
    return archive


def _q0b_packet(tmp_path: Path, hardware: str, runtime_hash: str) -> dict:
    """Create a qualified packet with the exact registered command hashes."""
    packet = copy.deepcopy(qualification_packet())
    packet["hardware"] = hardware
    packet["runtime"]["cpu_cores"] = 12
    packet["runtime_parity_receipt_sha256"] = runtime_hash

    # Newer Q0b packets bind the complete measurement source manifest.  The
    # fallback keeps this fixture compatible with the pre-manifest validator.
    source_manifest = getattr(q0b_measure, "measurement_source_manifest", None)
    if callable(source_manifest):
        packet["source"] = dict(source_manifest())
    else:
        packet["source"] = {
            "q0b_measure": _sha(Path(q0b_measure.__file__)),
            "train_m3c": _sha(Path("research/redesign_m3c_20261008/train_m3c.py")),
        }
    environment_identity = getattr(q0b_measure, "_environment_identity", None)
    if callable(environment_identity):
        packet["environment"] = dict(environment_identity())
    return packet


def _config_bindings(packet: dict, tmp_path: Path) -> None:
    """Bind Q0b's four measured cells to actual normalized Q1 argv hashes."""
    jobs = queue_jobs.build_jobs(COMMIT, queue_jobs.Bindings())
    representatives = {
        "restart/n3": "q1-train-a-261701",
        "mature/n3": "q1-train-b-261701",
        "restart/n1": "q1-train-c-261701",
        "mature/n1": "q1-train-d-261701",
    }
    by_id = {job["id"]: job for job in jobs}
    bindings = {}
    for cell, job_id in representatives.items():
        env = dict(by_id[job_id]["env"])
        collector = cell.rsplit("/", 1)[1]
        env["Q1_ACTORS"] = str(3 if collector == "n1" else 6)
        bindings[cell] = run_queue_job.normalized_command_sha(
            run_queue_job.command(env, tmp_path / cell.replace("/", "-"))
        )
    packet["config"] = bindings


def _evidence(tmp_path: Path) -> dict[str, Path | dict]:
    drive = tmp_path / "drive"
    mass = _write_archive(
        drive / "q0q1-20261017/q0/audit-t23-mass-a100r5",
        {"schema": "q0-t23-mass-audit/v1", "status": "COMPLETE"},
    )
    init = _write_archive(
        drive / "q0q1-20261017/q0/audit-b8-init-a100r6",
        {"schema": "q0-b8-init-fidelity/v1", "status": "COMPLETE"},
    )

    runtime = tmp_path / "runtime-parity.json"
    runtime_value = parity_receipt()
    runtime.write_text(json.dumps(runtime_value, sort_keys=True) + "\n")
    mac = tmp_path / "mac-parity-reference.json"
    mac.write_text(json.dumps(runtime_value["reference"], sort_keys=True) + "\n")

    b8 = tmp_path / "B8_s3_final.pth"
    t22 = tmp_path / "t22.json"
    t24 = tmp_path / "t24.json"
    b8.write_bytes(b"synthetic B8 carrier")
    t22.write_text("{}\n")
    t24.write_text("{}\n")

    l4_packet = _q0b_packet(tmp_path, "L4", _sha(runtime))
    l4_packet["hashes"] = {
        "b8_initializer": _sha(b8),
        "t22_receipt": _sha(t22),
        "t24_receipt": _sha(t24),
    }
    _config_bindings(l4_packet, tmp_path)
    a100_packet = copy.deepcopy(l4_packet)
    a100_packet["hardware"] = "A100"
    l4_receipt = {
        **l4_packet,
        "observed_units": 2.0,
        "observed_gpu_hours": 1.0,
        "observed_cpu_hours": 0.5,
        "observed_cpu_unit_rate_per_hour": 0.5,
        "cpu_seconds_per_episode": 2.0,
        "cpu_worker_efficiency": 0.8,
        "q1_attempt_overhead_seconds": 45.0,
        "stages": {"setup_seconds": 120.0, "cleanup_seconds": 60.0},
    }
    l4 = _write_archive(
        tmp_path / "q0b-l4-result",
        l4_receipt,
    )
    a100_receipt = {
        **a100_packet,
        "observed_units": 2.0,
        "observed_gpu_hours": 1.0,
        "observed_cpu_hours": 0.5,
        "observed_cpu_unit_rate_per_hour": 0.5,
        "cpu_seconds_per_episode": 2.0,
        "cpu_worker_efficiency": 0.8,
        "q1_attempt_overhead_seconds": 45.0,
        "stages": {"setup_seconds": 120.0, "cleanup_seconds": 60.0},
    }
    a100 = _write_archive(
        tmp_path / "q0b-a100-result",
        a100_receipt,
    )
    accounting = tmp_path / "q0-accounting.json"
    accounting.write_text(
        json.dumps(
            {
                **_a3_accounting("q0b-measure-l4", "q0b-measure-a100"),
                "q0_gpu_hours": 2.0,
                "q0_cpu_hours": 2.0,
                "cpu_unit_rate_per_hour": 0.5,
                "cpu_eval_job_overhead_seconds": 2.0,
            }
        )
    )
    # Use the immutable production A3 admission, including its real binding.
    owner = generator.owner_admission.DEFAULT_ADMISSION.resolve()
    # Retain the real producer/consumer binding logic while substituting only
    # the Mac-only inventory scans with passing synthetic audit outcomes.
    q0_bank = inventory_receipt.audit_q0_bank.build_world_bank()
    q0_audit = {
        "schema": "q0-audit-seed-audit/v1",
        "world_bank_semantic_sha256": q0_bank["allocation_semantic_sha256"],
        "candidate_count": 160,
        "registered_collision_seeds": [],
        "pass": True,
    }
    q1_audit = json.loads(Path("research/q0q1_20261017/Q1_SEED_AUDIT.json").read_text())
    with (
        patch.object(inventory_receipt.audit_q0_bank, "audit_world_bank", return_value=q0_audit),
        patch.object(inventory_receipt.q1_bank, "audit_world_bank", return_value=q1_audit),
    ):
        inventory_value = inventory_receipt.create_receipt()
    inventory = tmp_path / "inventory-receipt.json"
    inventory.write_text(json.dumps(inventory_value, sort_keys=True) + "\n")
    return {
        "drive": drive,
        "mass": mass,
        "init": init,
        "l4": l4,
        "a100": a100,
        "runtime": runtime,
        "mac": mac,
        "accounting": accounting,
        "owner": owner,
        "inventory": inventory,
    }


def _cli_args(evidence: dict[str, Path | dict], out: Path, drive_output_dir: Path) -> list[str]:
    return [
        "--commit",
        COMMIT,
        "--q0b-l4-receipt",
        str(evidence["l4"]),
        "--q0b-a100-receipt",
        str(evidence["a100"]),
        "--q0-mass-archive",
        str(evidence["mass"]),
        "--q0-init-archive",
        str(evidence["init"]),
        "--runtime-parity-receipt",
        str(evidence["runtime"]),
        "--mac-parity-reference",
        str(evidence["mac"]),
        "--inventory-receipt",
        str(evidence["inventory"]),
        "--q0-accounting",
        str(evidence["accounting"]),
        "--q0b-l4-job-id",
        "q0b-measure-l4",
        "--q0b-a100-job-id",
        "q0b-measure-a100",
        "--drive-root",
        str(evidence["drive"]),
        "--drive-output-dir",
        str(drive_output_dir),
        "--out-dir",
        str(out),
    ]


def _job_env(job_path: Path, drive: Path) -> dict[str, str]:
    job = json.loads(job_path.read_text())
    env = dict(job["env"])
    # The queue manifest is written with the Colab-visible drive paths.  For a
    # local admission proof, map that root to the temporary Drive fixture.
    env["QUEUE_DRIVE_ROOT"] = str(drive)
    return env


def test_cli_materializes_120_bound_jobs_and_real_admission_checks(monkeypatch, tmp_path):
    evidence = _evidence(tmp_path)
    out = tmp_path / "jobs"
    drive_output = evidence["drive"] / "q0q1-20261017/q1/jobs_approved"
    monkeypatch.setattr(generator, "_clean_head", lambda commit, root: None)
    assert generator.main(_cli_args(evidence, out, drive_output)) == 0

    jobs = sorted(out.glob("q1-*.json"))
    assert len(jobs) == 120
    assert len(list(out.glob("q1-train-*.json"))) == 20
    assert len(list(out.glob("q1-eval-*.json"))) == 100
    assert {json.loads(path.read_text())["env"]["Q0Q1_OWNER_RELEASE"] for path in jobs} == {
        APPROVED
    }

    # Materialize the generator's Drive-visible evidence and plan at the paths
    # bound into each job.  Every SHA is checked against the actual bytes.
    evidence_dir = drive_output / "evidence"
    evidence_dir.mkdir(parents=True)
    for name in (
        "q0b-l4-receipt.json",
        "q0b-a100-receipt.json",
        "runtime-parity.json",
        "mac-parity-reference.json",
        "owner-admission.json",
        "inventory-receipt.json",
    ):
        source = out / "evidence" / name
        (evidence_dir / name).write_bytes(source.read_bytes())
    (drive_output / "study-plan.json").write_bytes((out / "study-plan.json").read_bytes())

    train = next(path for path in jobs if path.name.startswith("q1-train-a-"))
    evaluate = next(path for path in jobs if path.name.startswith("q1-eval-a-"))
    for path in (train, evaluate):
        env = _job_env(path, evidence["drive"])
        for name in (
            "Q0Q1_BANK_INVENTORY_RECEIPT",
            "Q0Q1_Q0B_L4_RECEIPT",
            "Q0Q1_STUDY_PLAN",
            "Q1_RUNTIME_PARITY_RECEIPT",
            "Q1_MAC_PARITY_REFERENCE",
            "Q0Q1_PARITY_OWNER_ADMISSION",
            "Q0Q1_Q0B_A100_RECEIPT",
        ):
            bound = Path(env[name])
            digest_name = f"{name}_SHA256"
            assert bound.is_file(), (name, bound)
            assert env[digest_name] == _sha(bound), name
        run_queue_job.admission_checks(env)


@pytest.mark.parametrize(
    "mutation",
    [
        "study-plan",
        "mac",
        "source",
        "packet",
        "q0",
        "inventory",
        "owner",
        "amendment",
        "plan-amendment",
    ],
)
def test_admission_rejects_mutated_bound_evidence(monkeypatch, tmp_path, mutation):
    evidence = _evidence(tmp_path)
    out = tmp_path / "jobs"
    drive_output = evidence["drive"] / "q0q1-20261017/q1/jobs_approved"
    monkeypatch.setattr(generator, "_clean_head", lambda commit, root: None)
    assert generator.main(_cli_args(evidence, out, drive_output)) == 0
    evidence_dir = drive_output / "evidence"
    evidence_dir.mkdir(parents=True)
    for source in (out / "evidence").glob("*.json"):
        (evidence_dir / source.name).write_bytes(source.read_bytes())
    (drive_output / "study-plan.json").write_bytes((out / "study-plan.json").read_bytes())
    train_path = next(out.glob("q1-train-a-*.json"))
    env = _job_env(train_path, evidence["drive"])

    targets = {
        "owner": Path(env["Q0Q1_PARITY_OWNER_ADMISSION"]),
        "amendment": Path(env["Q0Q1_PARITY_OWNER_ADMISSION"]),
        "plan-amendment": Path(env["Q0Q1_STUDY_PLAN"]),
        "inventory": Path(env["Q0Q1_BANK_INVENTORY_RECEIPT"]),
        "study-plan": Path(env["Q0Q1_STUDY_PLAN"]),
        "mac": Path(env["Q1_MAC_PARITY_REFERENCE"]),
        "source": Path(env["Q0Q1_Q0B_L4_RECEIPT"]),
        "packet": Path(env["Q0Q1_Q0B_L4_RECEIPT"]),
        "q0": evidence["drive"] / "q0q1-20261017/q0/audit-t23-mass-a100r5/result.tar.gz",
    }
    target = targets[mutation]
    if mutation == "q0":
        target.write_bytes(b"changed archive")
    elif mutation == "plan-amendment":
        plan = json.loads(target.read_text())
        plan["amendment"] = "A1"
        target.write_text(json.dumps(plan))
        env["Q0Q1_STUDY_PLAN_SHA256"] = _sha(target)
    elif mutation == "amendment":
        owner = json.loads(target.read_text())
        owner["amendment"] = "A1"
        target.write_text(json.dumps(owner))
        env["Q0Q1_PARITY_OWNER_ADMISSION_SHA256"] = _sha(target)
    elif mutation == "source":
        source_packet = json.loads(target.read_text())
        source_packet["source"]["train_m3c"] = "0" * 64
        target.write_text(json.dumps(source_packet, sort_keys=True) + "\n")
        env["Q0Q1_Q0B_L4_RECEIPT_SHA256"] = _sha(target)
    else:
        target.write_bytes(target.read_bytes() + b"changed")
    if mutation == "source":
        packet = run_queue_job._bound_json_bytes(
            env["Q0Q1_Q0B_L4_RECEIPT"], env["Q0Q1_Q0B_L4_RECEIPT_SHA256"]
        )
        env["Q1_ACTORS"] = "6"
        with pytest.raises((RuntimeError, ValueError)):
            run_queue_job.qualified_recipe_binding(
                packet, run_queue_job.command(env, tmp_path / "source-check")
            )
    else:
        with pytest.raises((RuntimeError, ValueError)):
            run_queue_job.admission_checks(env)
