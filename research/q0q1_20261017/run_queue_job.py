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
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017.queue_jobs import REGISTRATION_STATUS
from research.q0q1_20261017 import q1_bank


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


def _receipt_from_archive(archive: Path, sidecar: Path) -> dict[str, Any]:
    """Read a queue-produced receipt without trusting an unverified Drive copy."""
    if not archive.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing queue dependency archive or sidecar: {archive}")
    fields = sidecar.read_text().split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]) or sha256_file(archive) != fields[0]:
        raise RuntimeError(f"queue dependency archive digest mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("attempt/receipt.json")
        stream = tar.extractfile(member)
        if stream is None:
            raise RuntimeError(f"queue dependency receipt missing: {archive}")
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"queue dependency receipt is not an object: {archive}")
    return value


def q0_admission_checks(env: Mapping[str, str]) -> None:
    root = Path(env["QUEUE_DRIVE_ROOT"]) / "q0q1-20261017/q0"
    required = {
        "q0-t23-mass-audit/v1": root / "audit-t23-mass/result.tar.gz",
        "q0-b8-init-fidelity/v1": root / "audit-b8-init/result.tar.gz",
    }
    for schema, archive in required.items():
        receipt = _receipt_from_archive(archive, archive.with_name("result.tar.gz.sha256"))
        if receipt.get("schema") != schema or receipt.get("status") != "COMPLETE":
            raise RuntimeError(f"Q1 training requires a completed passing Q0 receipt: {schema}")


def stage_q0_mass_receipt(env: Mapping[str, str], local_root: Path) -> Path:
    """Stage the verified Q0 mass receipt for B8 fidelity without new worlds."""
    archive = Path(env["QUEUE_DRIVE_ROOT"]) / "q0q1-20261017/q0/audit-t23-mass/result.tar.gz"
    _receipt_from_archive(archive, archive.with_name("result.tar.gz.sha256"))
    with tarfile.open(archive, "r:gz") as tar:
        stream = tar.extractfile("attempt/receipt.json")
        if stream is None:
            raise RuntimeError("Q0 mass archive lacks its receipt")
        raw = stream.read()
    receipt = local_root / "q0-mass-receipt.json"
    receipt.write_bytes(raw)
    value = json.loads(raw)
    if value.get("schema") != "q0-t23-mass-audit/v1" or value.get("status") != "COMPLETE":
        raise RuntimeError("Q0 mass receipt is not a completed corpus producer")
    return receipt


def qualified_recipe_binding(packet: Mapping[str, Any], train_command: list[str]) -> None:
    """Require Q0b's source/config hashes to bind the actual queued recipe.

    Volatile seed, output and namespace values are deliberately normalized: the
    qualification packet qualifies each package, not one replica directory.
    """
    source_digest = sha256_file(Path("research/redesign_m3c_20261008/train_m3c.py"))
    normalized = list(train_command)
    for flag in ("--seed", "--out", "--namespace"):
        index = normalized.index(flag)
        normalized[index + 1] = f"<{flag[2:]}>"
    config_digest = hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest()
    source = packet.get("source", {})
    config = packet.get("config", {})
    if not isinstance(source, Mapping) or source_digest not in source.values():
        raise RuntimeError("Q0b source binding does not match the queued trainer bytes")
    if not isinstance(config, Mapping) or config_digest not in config.values():
        raise RuntimeError("Q0b config binding does not match the queued trainer arguments")


def admission_checks(env: Mapping[str, str]) -> None:
    kind = env.get("Q0Q1_JOB_KIND")
    if kind == "q1_train":
        packet = _bound_json_bytes(env["Q0Q1_Q0B_L4_RECEIPT"], env["Q0Q1_Q0B_L4_RECEIPT_SHA256"])
        plan = _bound_json_bytes(env["Q0Q1_STUDY_PLAN"], env["Q0Q1_STUDY_PLAN_SHA256"])
        from research.q0q1_20261017.qualification import validate_admission

        admission = validate_admission(packet, plan)
        if admission.get("status") != "ADMITTED_FOR_OWNER_REVIEW" or packet.get("hardware") != "L4":
            raise RuntimeError("Q1 training requires full hash-bound L4 budget admission")
        q0_admission_checks(env)
        _bound_json_bytes(env["Q1_RUNTIME_PARITY_RECEIPT"], env["Q1_RUNTIME_PARITY_RECEIPT_SHA256"])
        collector = "n1" if env["Q1_ARM"] in ("C", "D") else "n3"
        cells = packet.get("cells", [])
        matches = [cell for cell in cells if cell.get("collector") == collector]
        actor_counts = {cell.get("actor_count") for cell in matches}
        if len(actor_counts) != 1 or not isinstance(next(iter(actor_counts)), int):
            raise RuntimeError("qualified packet lacks one explicit actor count for this Q1 collector")
    elif kind == "q1_eval":
        _bound_json(env["Q0Q1_PARITY_OWNER_ADMISSION"], env["Q0Q1_PARITY_OWNER_ADMISSION_SHA256"], required_status="OWNER_ADMITTED")
        _bound_json_bytes(env["Q1_MAC_PARITY_REFERENCE"], env["Q1_MAC_PARITY_REFERENCE_SHA256"])


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
    provenance = list((local_root / "training").rglob("training-provenance.json"))
    if len(finals) != 1 or len(provenance) != 1:
        raise RuntimeError("candidate archive must contain exactly one final.pth and provenance receipt")
    if not isinstance(env, dict):
        raise RuntimeError("evaluation environment must be mutable while staging provenance")
    env["Q1_TRAINING_PROVENANCE"] = str(provenance[0])
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
                "--t23-root", env["Q0Q1_T23_ROOT"], "--out", str(local_root / "receipt.json"), "--run-observations",
                "--observation-receipt", env["Q0Q1_MASS_OBSERVATION_RECEIPT"]]
    if kind == "q1_train":
        actors = env["Q1_ACTORS"]
        return [py, "-m", "research.redesign_m3c_20261008.train_m3c", "--seed", env["Q1_SEED"],
                "--init", str(inputs / "init/B8_s3_final.pth"), "--demo-data", str(inputs / "demo"),
                "--out", str(local_root / "run"), "--transitions", env["Q1_TRANSITIONS"], "--device", "cuda",
                "--max-wall-seconds", "5100",
                "--actor-device", "cuda", "--worlds", "128", "--actors", actors,
                "--checkpoint-every", "500000", "--reward", "mass", "--mass-kappa", "1.0",
                "--kill-scale", "0", "--gamma-end", "0.998", "--gamma-anneal-end", "8000000",
                "--mixed-frac", "0.25", "--anchor-mode", "v8margin", "--onpolicy-labels",
                "--label-frac", "1", "--obs-version", "bv8", "--v61-backend", "numba",
                "--learner-fast", "--learner-precision", "fp32", "--batch-scale", "2",
                "--ring-dir", "/dev/shm", "--compile-learner", "reduce-overhead",
                "--league-b8-checkpoint", str(inputs / "init/B8_s3_final.pth"), "--gather-threads", "3",
                "--actor-queue-frames", "2", "--schedule-package", "mature" if env["Q1_ARM"] in ("B", "D") else "restart",
                "--collection-profile", "task-aligned-n1" if env["Q1_ARM"] in ("C", "D") else "legacy",
                "--learner-seats", "1" if env["Q1_ARM"] in ("C", "D") else "3",
                "--namespace", q1_bank.training_namespace(env["Q1_ARM"])]
    if kind == "q1_eval":
        checkpoint = env["Q1_TRAINING_FINAL"]
        return [py, "-m", "research.q0q1_20261017.q1_evaluate", "--root", str(local_root),
                "--bank", "research/q0q1_20261017/Q1_WORLD_BANK.json", "--b8-checkpoint", str(inputs / "init/B8_s3_final.pth"),
                "--checkpoint", checkpoint, "--arm", env["Q1_ARM"], "--seed", env["Q1_SEED"], "--shard", env["Q1_SHARD"],
                "--parity-receipt", env["Q1_EVAL_PARITY_RECEIPT"],
                *([] if env["Q1_ARM"] == "B8" else ["--provenance", env["Q1_TRAINING_PROVENANCE"]])]
    raise RuntimeError(f"unknown Q0/Q1 job kind: {kind!r}")


def classify_training_outcome(summary: Mapping[str, Any]) -> str:
    """Classify only the registered online tripwires as scientific KILLs."""
    halt = str(summary.get("halt") or "").lower()
    registered = ("non-finite loss", "nonfinite loss", "non-finite q", "nonfinite q",
                  "non-finite gradient", "nonfinite gradient", "action collapse",
                  "greedy boost rate")
    return "KILL" if any(marker in halt for marker in registered) else "INCOMPLETE"


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_training_outcome(env: Mapping[str, str], local_root: Path, reason: str) -> None:
    """Retain an arm-level KILL/INCOMPLETE classification for failed attempts."""
    path = local_root / "run/summary.json"
    if not path.is_file():
        return
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(summary, dict):
        return
    outcome = {
        "schema": "q1-training-outcome/v1", "arm": env["Q1_ARM"],
        "training_seed": int(env["Q1_SEED"]), "status": classify_training_outcome(summary),
        "reason": str(summary.get("halt") or reason), "summary": summary,
        "summary_sha256": _canonical_sha(summary),
        "source_sha256": sha256_file(Path("research/redesign_m3c_20261008/train_m3c.py")),
    }
    (local_root / "run/training-outcome.json").write_text(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _scientific_done(kind: str, local_root: Path, env: Mapping[str, str] | None = None) -> None:
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
    if not isinstance(value.get("actor_exitcodes"), list) or any(value["actor_exitcodes"]):
        raise RuntimeError("training actors did not all exit cleanly")
    if value.get("ring_overrun_rows") != 0:
        raise RuntimeError("training recorded ring overruns")
    def count(name: str) -> int:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"training summary lacks nonnegative integer {name}")
        return item
    rows, updates, debt = count("emitted_replay_rows"), count("updates"), count("pending_update_debt")
    decisions, accounted = count("learner_decisions"), count("learner_decisions_accounted")
    dropped, pending = count("truncation_rows_dropped"), count("pending_nstep_rows")
    by_actor = value.get("pending_nstep_rows_by_actor")
    if (not isinstance(by_actor, dict) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in by_actor.values())
            or value.get("nstep_accounting_reconciled") is not True):
        raise RuntimeError("training n-step accounting receipt is malformed")
    delta = value.get("decision_replay_delta")
    if (updates == 0 or rows != 256 * updates + debt or delta != accounted - rows
            or accounted - rows != dropped + pending or sum(by_actor.values()) != pending):
        raise RuntimeError("training replay/update/n-step accounting does not reconcile")
    if int(value.get("transitions", value.get("learner_transitions", 0))) < 8_000_000:
        raise RuntimeError("training did not reach the registered 8M learner decisions")
    actors = int(value.get("actors", 0)); seats = int(value.get("m3c", {}).get("learner_seats", 0))
    overshoot = int(value.get("transition_budget_overshoot", -1))
    bound = actors * int(value.get("args", {}).get("worlds", 0)) * seats - 1
    if actors < 1 or seats < 1 or overshoot < 0 or overshoot > bound:
        raise RuntimeError("training exceeded the registered actor-packet overshoot bound")
    final = local_root / "run/final.pth"
    if not final.is_file():
        raise RuntimeError("training did not save a final checkpoint")
    if env is not None:
        import torch

        meta = torch.load(final, map_location="cpu", weights_only=False).get("meta", {})
        expected_init = sha256_file(Path(env["Q0Q1_INPUT_ROOT"]) / "init/B8_s3_final.pth")
        if (meta.get("seed") != int(env["Q1_SEED"]) or int(meta.get("transitions", 0)) < 8_000_000
                or meta.get("init_sha256") != expected_init):
            raise RuntimeError("final checkpoint metadata lacks the registered seed/init/8M binding")


def _write_training_provenance(env: Mapping[str, str], local_root: Path) -> None:
    """Bind evaluator-required provenance to the saved final bytes, never to a filename."""
    runtime_receipt = Path(env["Q1_RUNTIME_PARITY_RECEIPT"])
    expected = env["Q1_RUNTIME_PARITY_RECEIPT_SHA256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(runtime_receipt) != expected:
        raise RuntimeError("training runtime parity receipt is absent or changed")
    summary = json.loads((local_root / "run/summary.json").read_text())
    final = local_root / "run/final.pth"
    if not final.is_file():
        raise RuntimeError("scientifically complete training lacks final.pth")
    import numpy as np
    import torch

    runtime = {
        "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    actors = int(summary["actors"])
    seats = int(summary["m3c"]["learner_seats"])
    worlds = int(summary["args"]["worlds"])
    decisions = int(summary["learner_decisions"])
    overshoot = int(summary["transition_budget_overshoot"])
    bound = actors * worlds * seats - 1
    counters = {
        "actual_learner_decisions": decisions,
        "transition_budget_overshoot": overshoot,
        "registered_overshoot_bound": bound,
        "actors": actors, "worlds": worlds, "learner_seats": seats,
        "emitted_replay_rows": int(summary["emitted_replay_rows"]),
        "completed_updates": int(summary["updates"]),
        "pending_update_debt": int(summary["pending_update_debt"]),
        "truncation_rows_dropped": int(summary["truncation_rows_dropped"]),
        "pending_nstep_rows": int(summary["pending_nstep_rows"]),
    }
    args = summary.get("args", {})
    source = Path("research/redesign_m3c_20261008/train_m3c.py")
    payload = {
        "schema": "q1-training-provenance/v1", "arm": env["Q1_ARM"],
        "training_seed": int(env["Q1_SEED"]), "endpoint": 8_000_000, "clean_exit": True,
        "checkpoint_sha256": sha256_file(final), "parity_pass": True, "tripwire_triggered": False,
        "environment": runtime,
        "environment_sha256": hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_sha256": sha256_file(source),
        "config": args,
        "config_sha256": hashlib.sha256(json.dumps(args, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "counters": counters,
        "runtime_parity_receipt_sha256": expected,
    }
    (local_root / "run/training-provenance.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def _publish_attempt(output: Path, local: Path, *, job_id: str | None, status: str, error: str | None) -> None:
    receipt = {"schema": "q0q1-queue-attempt/v1", "status": status, "job_id": job_id, "error": error}
    (local / "queue-attempt-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    archive = output / "result.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(local, arcname="attempt")
    (output / "result.tar.gz.sha256").write_text(f"{sha256_file(archive)}  result.tar.gz\n")


def run(env: Mapping[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    require_owner_release(env)
    admission_checks(env)
    output = Path(env["QUEUE_OUTPUT_DIR"])
    if not output.is_absolute() or not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("QUEUE_OUTPUT_DIR must be an existing empty job-private directory")
    local_parent = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    local = Path(tempfile.mkdtemp(prefix="q0q1-", dir=local_parent))
    failure: Exception | None = None
    run_env = dict(env)
    try:
        child_env = {**env, "PYTHONPATH": f"/content/t16-snake-dqn:{env.get('PYTHONPATH', '')}"}
        if env["Q0Q1_JOB_KIND"] == "q0_b8_init":
            run_env["Q0Q1_MASS_OBSERVATION_RECEIPT"] = str(stage_q0_mass_receipt(env, local))
        if env["Q0Q1_JOB_KIND"] == "q1_train":
            packet = _bound_json_bytes(env["Q0Q1_Q0B_L4_RECEIPT"], env["Q0Q1_Q0B_L4_RECEIPT_SHA256"])
            collector = "n1" if env["Q1_ARM"] in ("C", "D") else "n3"
            run_env["Q1_ACTORS"] = str(next(cell["actor_count"] for cell in packet["cells"] if cell["collector"] == collector))
            qualified_recipe_binding(packet, command(run_env, local))
        if env["Q0Q1_JOB_KIND"] == "q1_eval":
            run_env["Q1_TRAINING_FINAL"] = stage_training_final(run_env, local)
            parity = subprocess.run([
                run_env.get("Q0Q1_PYTHON", "/content/t16-colab-venv/bin/python"), "-m",
                "research.q0q1_20261017.q1_parity", "--verify-reference",
                run_env["Q1_MAC_PARITY_REFERENCE"], "--bank", "research/q0q1_20261017/Q1_WORLD_BANK.json",
                "--b8", str(Path(run_env["Q0Q1_INPUT_ROOT"]) / "init/B8_s3_final.pth"),
                "--out", str(local / "runtime-parity.json"),
            ], cwd="/content/t16-snake-dqn", env=child_env)
            if parity.returncode != 0:
                raise RuntimeError(f"per-runtime parity verification exited {parity.returncode}")
            run_env["Q1_EVAL_PARITY_RECEIPT"] = str(local / "runtime-parity.json")
        completed = subprocess.run(command(run_env, local), cwd="/content/t16-snake-dqn", env=child_env)
        if completed.returncode != 0:
            raise RuntimeError(f"child exited {completed.returncode}")
        _scientific_done(env["Q0Q1_JOB_KIND"], local, run_env)
        if env["Q0Q1_JOB_KIND"] == "q1_train":
            _write_training_provenance(run_env, local)
    except Exception as exc:  # retain every failed scientific attempt for audit
        failure = exc
    finally:
        if env.get("Q0Q1_JOB_KIND") == "q1_train" and failure is not None:
            _write_training_outcome(run_env, local, str(failure))
        _publish_attempt(output, local, job_id=env.get("Q0Q1_JOB_ID"), status="COMPLETED" if failure is None else "INCOMPLETE", error=None if failure is None else str(failure))
        shutil.rmtree(local, ignore_errors=True)
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (KeyError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Q0/Q1 queue job refused: {exc}", file=sys.stderr)
        raise SystemExit(64)
