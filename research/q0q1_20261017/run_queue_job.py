"""Fail-closed Q0/Q1 queue adapter.

Every job calculates on local Colab storage and publishes a tarball plus hash
only after the scientific receipt says ``scientificdone``.  This adapter does
not retry or overwrite an output directory; queue requeue generations must use
a different job-private output location and receive explicit operator review.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017.queue_jobs import REGISTRATION_STATUS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_owner_release(env: Mapping[str, str]) -> None:
    """Keep DRAFT registrations non-executable before any filesystem work."""
    if env.get("Q0Q1_REGISTRATION_STATUS", REGISTRATION_STATUS) == REGISTRATION_STATUS:
        raise RuntimeError("DRAFT_PENDING_OWNER_REVIEW: Q0/Q1 queue dispatch is hard-blocked")
    if env.get("Q0Q1_OWNER_RELEASE") != "APPROVED_Q0Q1_20261017":
        raise RuntimeError("missing explicit Q0/Q1 owner release")


def _bound_json_bytes(path_text: str, expected: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"unbound or changed receipt: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid receipt JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"receipt is not a JSON object: {path}")
    return value


def _bound_json(path_text: str, expected: str, *, required_status: str) -> dict[str, Any]:
    value = _bound_json_bytes(path_text, expected)
    if value.get("status") != required_status:
        raise RuntimeError(f"receipt is not {required_status}: {path_text}")
    return value


def admission_checks(env: Mapping[str, str]) -> None:
    kind = env.get("Q0Q1_JOB_KIND")
    if kind == "q1_train":
        packet = _bound_json_bytes(env["Q0Q1_Q0B_L4_RECEIPT"], env["Q0Q1_Q0B_L4_RECEIPT_SHA256"])
        from research.q0q1_20261017.qualification import validate_packet

        qualification = validate_packet(packet)
        if qualification.get("status") != "QUALIFIED" or packet.get("hardware") != "L4":
            raise RuntimeError("Q1 training requires a hash-bound qualified L4 Q0b packet")
    elif kind == "q1_eval":
        _bound_json(env["Q0Q1_PARITY_OWNER_ADMISSION"], env["Q0Q1_PARITY_OWNER_ADMISSION_SHA256"], required_status="OWNER_ADMITTED")


def stage_training_final(env: Mapping[str, str], local_root: Path) -> str:
    """Copy a dependency's immutable archive locally and return its sole final."""
    if env.get("Q0Q1_JOB_KIND") != "q1_eval" or env.get("Q1_ARM") == "B8":
        return str(Path(env["Q0Q1_INPUT_ROOT"]) / "init/B8_s3_final.pth")
    archive, sidecar = Path(env["Q1_TRAINING_ARCHIVE"]), Path(env["Q1_TRAINING_ARCHIVE_SHA256"])
    if not archive.is_file() or not sidecar.is_file():
        raise RuntimeError("candidate training dependency archive or its sidecar is missing")
    expected = sidecar.read_text().split(maxsplit=1)[0] if sidecar.read_text().split() else ""
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(archive) != expected:
        raise RuntimeError("candidate training archive SHA-256 does not match its sidecar")
    staged = local_root / "training.tar.gz"
    shutil.copy2(archive, staged)
    with tarfile.open(staged, "r:gz") as tar:
        members = tar.getmembers()
        if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
            raise RuntimeError("candidate training archive has an unsafe member name")
        tar.extractall(local_root / "training", filter="data")
    finals = list((local_root / "training").rglob("final.pth"))
    if len(finals) != 1:
        raise RuntimeError("candidate training archive must contain exactly one final.pth")
    return str(finals[0])


def command(env: Mapping[str, str], local_root: Path) -> list[str]:
    py = env.get("Q0Q1_PYTHON", "/content/t16-colab-venv/bin/python")
    inputs = Path(env["Q0Q1_INPUT_ROOT"])
    kind = env["Q0Q1_JOB_KIND"]
    if kind == "q0_t23_mass":
        return [py, "-m", "research.q0q1_20261017.audit_t23_mass", "--artifact-root", env["Q0Q1_T23_ROOT"],
                "--b8", str(inputs / "init/B8_s3_final.pth"), "--out", str(local_root / "receipt.json"), "--run-development"]
    if kind == "q0_b8_init":
        return [py, "-m", "research.q0q1_20261017.audit_b8_init", "--b8", str(inputs / "init/B8_s3_final.pth"),
                "--t23-root", env["Q0Q1_T23_ROOT"], "--out", str(local_root / "receipt.json"), "--run-observations"]
    if kind == "q1_train":
        return [py, "-m", "research.redesign_m3c_20261008.train_m3c", "--seed", env["Q1_SEED"],
                "--init", str(inputs / "init/B8_s3_final.pth"), "--demo-data", str(inputs / "demo"),
                "--out", str(local_root / "run"), "--transitions", env["Q1_TRANSITIONS"], "--device", "cuda",
                "--actor-device", "cuda", "--schedule-package", "mature" if env["Q1_ARM"] in ("B", "D") else "restart",
                "--collection-profile", "q1-task-aligned" if env["Q1_ARM"] in ("C", "D") else "t23-n3",
                "--namespace", f"q1-{env['Q1_ARM'].lower()}-{env['Q1_SEED']}/v1"]
    if kind == "q1_eval":
        checkpoint = env["Q1_TRAINING_FINAL"]
        return [py, "-m", "research.q0q1_20261017.q1_evaluate", "--root", str(local_root),
                "--bank", "research/q0q1_20261017/Q1_WORLD_BANK.json", "--b8-checkpoint", str(inputs / "init/B8_s3_final.pth"),
                "--checkpoint", checkpoint, "--arm", env["Q1_ARM"], "--seed", env["Q1_SEED"], "--shard", env["Q1_SHARD"]]
    raise RuntimeError(f"unknown Q0/Q1 job kind: {kind!r}")


def _scientific_done(kind: str, local_root: Path) -> None:
    receipt = local_root / "receipt.json"
    if kind == "q1_eval":
        candidates = list((local_root / "evaluation").rglob("*.json"))
        if not candidates:
            raise RuntimeError("evaluation did not produce a shard receipt")
        value = json.loads(candidates[0].read_text())
        if value.get("status") != "completed" or value.get("clean_exit") is not True:
            raise RuntimeError("evaluation receipt is not scientifically complete")
        return
    if kind.startswith("q0_"):
        value = json.loads(receipt.read_text())
        # Q0 reports descriptive INCOMPLETE if a required fresh adapter is absent;
        # that is a valid completed audit attempt, never a scientific efficacy result.
        if not value.get("schema", "").startswith("q0-"):
            raise RuntimeError("Q0 receipt schema missing")
        return
    summary = local_root / "run" / "summary.json"
    value = json.loads(summary.read_text())
    if value.get("stopped_for_time") is not False or value.get("halt") not in (None, False, "", "none", "None"):
        raise RuntimeError("training did not reach scientificdone")
    if int(value.get("transitions", value.get("learner_transitions", 0))) < 8_000_000:
        raise RuntimeError("training did not reach the registered 8M learner decisions")


def run(env: Mapping[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    require_owner_release(env)
    admission_checks(env)
    output = Path(env["QUEUE_OUTPUT_DIR"])
    if not output.is_absolute() or not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("QUEUE_OUTPUT_DIR must be an existing empty job-private directory")
    local_parent = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix="q0q1-", dir=local_parent) as temporary:
        local = Path(temporary)
        child_env = {**env, "PYTHONPATH": f"/content/t16-snake-dqn:{env.get('PYTHONPATH', '')}"}
        run_env = dict(env)
        if env["Q0Q1_JOB_KIND"] == "q1_eval":
            run_env["Q1_TRAINING_FINAL"] = stage_training_final(env, local)
        completed = subprocess.run(command(run_env, local), cwd="/content/t16-snake-dqn", env=child_env)
        if completed.returncode != 0:
            raise RuntimeError(f"child exited {completed.returncode}; no result was published")
        _scientific_done(env["Q0Q1_JOB_KIND"], local)
        archive = output / "result.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(local, arcname="attempt")
        (output / "result.tar.gz.sha256").write_text(f"{sha256_file(archive)}  result.tar.gz\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (KeyError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Q0/Q1 queue job refused: {exc}", file=sys.stderr)
        raise SystemExit(64)
