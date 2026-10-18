"""Build (but never publish) the Q0/Q1 Colab queue manifests.

The caller must supply the final immutable repository SHA plus the future Q0b
qualification and owner-admitted parity receipts.  Keeping those bindings as
required arguments prevents a draft registration from becoming runnable by
accidentally substituting a placeholder digest.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research.q0q1_20261017 import q1_bank

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REGISTRATION_STATUS = "DRAFT_PENDING_OWNER_REVIEW"
ARMS = ("A", "B", "C", "D")
INPUTS = "/content/t16-colab-extracted/t16-colab-inputs"
DRIVE_ROOT = "/content/drive/MyDrive/t16_colab"


@dataclass(frozen=True)
class Bindings:
    """Future evidence pins; intentionally empty fields describe an honest draft."""

    q0b_l4_receipt: str = ""
    q0b_l4_receipt_sha256: str = ""
    study_plan: str = ""
    study_plan_sha256: str = ""
    parity_owner_admission: str = ""
    parity_owner_admission_sha256: str = ""

    def validate(self) -> None:
        for path_name, hash_name in (("q0b_l4_receipt", "q0b_l4_receipt_sha256"),
                                     ("study_plan", "study_plan_sha256"),
                                     ("parity_owner_admission", "parity_owner_admission_sha256")):
            path, digest = getattr(self, path_name), getattr(self, hash_name)
            if bool(path) != bool(digest):
                raise ValueError(f"{path_name} and {hash_name} must be supplied together")
            if path and (not path.startswith("/") or not SHA256_RE.fullmatch(digest) or set(digest) == {"0"}):
                raise ValueError(f"{path_name} must have a non-placeholder absolute-path SHA-256 binding")


def _validate_commit(commit: str) -> None:
    if not COMMIT_RE.fullmatch(commit) or set(commit) == {"0"}:
        raise ValueError("commit must be a non-placeholder, full immutable 40-character SHA")


def _job(
    *,
    job_id: str,
    commit: str,
    kind: str,
    gpu: str,
    min_vcpu: int,
    est_minutes: int,
    depends_on: Iterable[str],
    output_dir: str,
    env: dict[str, str],
    priority: int,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "commit": commit,
        "script": "research/q0q1_20261017/run_queue_job.sh",
        "env": {
            "Q0Q1_REGISTRATION_STATUS": REGISTRATION_STATUS,
            "Q0Q1_JOB_KIND": kind,
            "Q0Q1_JOB_ID": job_id,
            "Q0Q1_INPUT_ROOT": INPUTS,
            "QUEUE_DRIVE_ROOT": DRIVE_ROOT,
            **env,
        },
        "requirements": {"gpu": gpu, "min_vcpu": min_vcpu, "est_minutes": est_minutes},
        "depends_on": list(depends_on),
        "output_dir": output_dir,
        "priority": priority,
        "timeout_minutes": est_minutes * 2,
    }


def build_jobs(commit: str, bindings: Bindings, *, eval_shards: int = 4) -> list[dict[str, Any]]:
    """Return the complete Q0/Q1 dependency graph without writing JSON files."""
    _validate_commit(commit)
    bindings.validate()
    if eval_shards < 1:
        raise ValueError("eval_shards must be positive")
    common = {}
    for env_path, env_hash, value_path, value_hash in (
        ("Q0Q1_Q0B_L4_RECEIPT", "Q0Q1_Q0B_L4_RECEIPT_SHA256", bindings.q0b_l4_receipt, bindings.q0b_l4_receipt_sha256),
        ("Q0Q1_STUDY_PLAN", "Q0Q1_STUDY_PLAN_SHA256", bindings.study_plan, bindings.study_plan_sha256),
        ("Q0Q1_PARITY_OWNER_ADMISSION", "Q0Q1_PARITY_OWNER_ADMISSION_SHA256", bindings.parity_owner_admission, bindings.parity_owner_admission_sha256),
    ):
        if value_path:
            common[env_path], common[env_hash] = value_path, value_hash
    jobs: list[dict[str, Any]] = []
    q0_mass = "q0-audit-t23-mass"
    q0_init = "q0-audit-b8-init"
    jobs.extend((
        _job(job_id=q0_mass, commit=commit, kind="q0_t23_mass", gpu="cpu", min_vcpu=2,
             est_minutes=240, depends_on=(), output_dir="q0q1-20261017/q0/audit-t23-mass",
             env={"Q0Q1_T23_ROOT": f"{DRIVE_ROOT}/t23-r0", **common}, priority=100),
        _job(job_id=q0_init, commit=commit, kind="q0_b8_init", gpu="cpu", min_vcpu=2,
             est_minutes=120, depends_on=(), output_dir="q0q1-20261017/q0/audit-b8-init",
             env={"Q0Q1_T23_ROOT": f"{DRIVE_ROOT}/t23-r0", **common}, priority=100),
    ))
    q0_dependencies = (q0_mass, q0_init)
    train_ids: dict[tuple[str, int], str] = {}
    for arm in ARMS:
        for seed in q1_bank.TRAINING_SEEDS:
            job_id = f"q1-train-{arm.lower()}-{seed}"
            train_ids[arm, seed] = job_id
            jobs.append(_job(
                job_id=job_id, commit=commit, kind="q1_train", gpu="L4", min_vcpu=12,
                est_minutes=180, depends_on=q0_dependencies,
                output_dir=f"q0q1-20261017/q1/training/arm-{arm}/seed-{seed}", priority=90,
                env={"Q1_ARM": arm, "Q1_SEED": str(seed), "Q1_TRANSITIONS": "8000000", **common},
            ))
    # B8 is evaluated once for each fresh seed bank.  Candidate shards consume the
    # matching B8 shard, never recomputing it for each arm.
    for seed in q1_bank.TRAINING_SEEDS:
        for shard_index in range(eval_shards):
            selector = f"{shard_index}/{eval_shards}"
            b8_id = f"q1-eval-b8-{seed}-s{shard_index}of{eval_shards}"
            base_env = {"Q1_SEED": str(seed), "Q1_SHARD": selector, **common}
            jobs.append(_job(
                job_id=b8_id, commit=commit, kind="q1_eval", gpu="cpu", min_vcpu=2,
                est_minutes=120, depends_on=q0_dependencies,
                output_dir=f"q0q1-20261017/q1/evaluation/seed-{seed}/B8/shard-{shard_index}-of-{eval_shards}",
                env={"Q1_ARM": "B8", **base_env}, priority=70,
            ))
            for arm in ARMS:
                candidate_id = f"q1-eval-{arm.lower()}-{seed}-s{shard_index}of{eval_shards}"
                train_output = f"{DRIVE_ROOT}/q0q1-20261017/q1/training/arm-{arm}/seed-{seed}"
                jobs.append(_job(
                    job_id=candidate_id, commit=commit, kind="q1_eval", gpu="cpu", min_vcpu=2,
                    est_minutes=120, depends_on=(*q0_dependencies, train_ids[arm, seed], b8_id),
                    output_dir=f"q0q1-20261017/q1/evaluation/seed-{seed}/{arm}/shard-{shard_index}-of-{eval_shards}",
                    env={"Q1_ARM": arm, "Q1_TRAINING_ARCHIVE": f"{train_output}/result.tar.gz",
                         "Q1_TRAINING_ARCHIVE_SHA256": f"{train_output}/result.tar.gz.sha256", **base_env}, priority=60,
                ))
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("Q0/Q1 generator made duplicate IDs")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--q0b-l4-receipt")
    parser.add_argument("--q0b-l4-receipt-sha256")
    parser.add_argument("--study-plan")
    parser.add_argument("--study-plan-sha256")
    parser.add_argument("--parity-owner-admission")
    parser.add_argument("--parity-owner-admission-sha256")
    parser.add_argument("--eval-shards", type=int, default=4)
    parser.add_argument("--out", type=Path, help="optional one JSON array; never enqueues")
    args = parser.parse_args(argv)
    jobs = build_jobs(args.commit, Bindings(args.q0b_l4_receipt or "", args.q0b_l4_receipt_sha256 or "",
                                           args.study_plan or "", args.study_plan_sha256 or "",
                                           args.parity_owner_admission or "", args.parity_owner_admission_sha256 or ""),
                      eval_shards=args.eval_shards)
    text = json.dumps(jobs, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
