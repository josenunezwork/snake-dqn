"""Separate, fail-closed CPU preview alongside an unchanged primary learner.

This supervisor owns only its preview lock, receipts, and child process groups.
Higher reserve thresholds reduce interference risk; independent sampled guards
cannot guarantee that the preview always stops before the primary controller.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

GIB = 1024**3
LIMITS = {
    "admission_available_gib": 24,
    "abort_available_gib": 20,
    "sidecar_rss_gib": 3,
    "combined_rss_gib": 8,
    "cpu_percent": 80,
    "cpu_sustained_seconds": 10,
    "primary_heartbeat_seconds": 120,
    "child_heartbeat_seconds": 120,
    "poll_seconds": 0.2,
    "term_grace_seconds": 5,
    "cpu_threads": 2,
    "interop_threads": 1,
    "nice": 10,
}


def require(condition: bool, message: str) -> None:
    """Enforce checks even under optimized Python."""
    if not condition:
        raise RuntimeError(message)


def utc() -> str:
    """Return an explicit UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def stamp(value: str) -> datetime:
    """Reject naive frozen clocks."""
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timezone required")
    return parsed


def read(path: Path) -> dict[str, Any]:
    """Read an explicit metadata object, rejecting nonfinite constants."""

    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite metadata: {value}")

    result = json.loads(path.read_text(), parse_constant=invalid)
    require(isinstance(result, dict), f"metadata object required: {path}")
    return result


def sha(path: Path) -> str:
    """Authenticate source/metadata only; checkpoint payloads are forbidden here."""
    path = path.resolve()
    require(
        path.suffix.lower() not in {".pth", ".pt", ".ckpt", ".bin", ".safetensors"},
        "supervisor may not hash checkpoint payloads",
    )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable(path: Path, value: dict[str, Any]) -> None:
    """Create and fsync immutable receipt content and its directory."""
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def authenticate(intent: dict[str, Any], intent_path: Path, identity: str) -> None:
    """Check the frozen preview and original authority without editing either."""
    require(sha(intent_path) == identity, "preview intent drift")
    original = intent["original_intent"]
    require(sha(Path(original["path"])) == original["sha256"], "original intent drift")
    for mapping in (intent["source_files"], intent["authority_files"]):
        for path, expected in mapping.items():
            require(sha(Path(path)) == expected, f"frozen input drift: {path}")


def remaining(intent: dict[str, Any], index: int) -> float:
    """Reserve all remaining preview caps and the final handoff."""
    required = sum(stage["cap_seconds"] for stage in intent["stages"][index:]) + 120
    require(
        (stamp(intent["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds() >= required,
        "remaining preview caps and handoff do not fit",
    )
    original = read(Path(intent["original_intent"]["path"]))
    original_required = (
        intent["original_remaining_cap_seconds"] + intent["original_handoff_seconds"]
    )
    require(original_required == 58320, "original remaining budget differs from reviewed exception")
    require(
        (stamp(original["deadline"]) - datetime.now(timezone.utc)).total_seconds()
        >= original_required,
        "original full remaining caps and handoff no longer fit",
    )
    return required


def primary_process(pid: int, creation: float) -> psutil.Process:
    """Bind a live process by both PID and creation time, never name alone."""
    process = psutil.Process(pid)
    require(abs(process.create_time() - creation) < 0.001, "primary PID identity changed")
    require(
        process.is_running() and process.status() != psutil.STATUS_ZOMBIE,
        "primary process is no longer alive",
    )
    return process


def process_tree(process: psutil.Process) -> dict[int, dict[str, Any]]:
    """Measure every live member; ambiguous access is a preview stop."""
    result = {}
    for item in [process, *process.children(recursive=True)]:
        try:
            result[item.pid] = {
                "pid": item.pid,
                "create_time": item.create_time(),
                "rss_bytes": int(item.memory_info().rss),
            }
        except psutil.NoSuchProcess:
            # Natural child exit can race its sample. Popen is reconciled below;
            # the bound primary child must still appear in its live tree.
            if item.pid == process.pid:
                raise
    return result


def snapshot(intent: dict[str, Any], admission: bool = False) -> dict[str, Any]:
    """Observe primary identity/health and the combined memory envelope."""
    primary = intent["primary"]
    controller = primary_process(primary["controller_pid"], primary["controller_create_time"])
    learner = primary_process(primary["child_pid"], primary["child_create_time"])
    main_tree = process_tree(controller)
    require(learner.pid in main_tree, "primary child is outside the bound controller tree")
    directory = Path(primary["stage_dir"])
    require(
        read(directory / "started.json")["stage_id"] == primary["stage"],
        "primary stage identity changed",
    )
    for failure in (
        directory / "completed.json",
        directory / "output/failure.json",
        directory.parent / "failure.json",
        directory.parent / "completed.json",
    ):
        require(not failure.exists(), f"primary stage ended or failed: {failure}")
    heartbeat_path = Path(primary["heartbeat_path"])
    require(
        heartbeat_path.resolve() == (directory / "output/heartbeat.json").resolve(),
        "incorrect primary heartbeat path",
    )
    beat = read(heartbeat_path)
    monotonic = beat["monotonic"]
    require(
        type(monotonic) in (int, float) and math.isfinite(monotonic),
        "invalid primary heartbeat clock",
    )
    age = time.monotonic() - monotonic
    require(-1 <= age <= LIMITS["primary_heartbeat_seconds"], "primary heartbeat stale")
    own_tree = process_tree(psutil.Process(os.getpid()))
    require(not (set(main_tree) & set(own_tree)), "preview and primary process trees overlap")
    available = int(psutil.virtual_memory().available)
    own_rss = sum(row["rss_bytes"] for row in own_tree.values())
    combined_rss = own_rss + sum(row["rss_bytes"] for row in main_tree.values())
    reserve = LIMITS["admission_available_gib" if admission else "abort_available_gib"] * GIB
    require(available >= reserve, "available-memory preview reserve breached")
    require(own_rss <= LIMITS["sidecar_rss_gib"] * GIB, "preview subtree RSS breached")
    require(combined_rss <= LIMITS["combined_rss_gib"] * GIB, "combined subtree RSS breached")
    return {
        "utc": utc(),
        "monotonic": time.monotonic(),
        "available_bytes": available,
        "preview_rss_bytes": own_rss,
        "combined_rss_bytes": combined_rss,
        "primary_tree": list(main_tree.values()),
        "preview_tree": list(own_tree.values()),
        "primary_heartbeat_age_seconds": age,
        "primary_heartbeat_mtime": heartbeat_path.stat().st_mtime,
        "primary_heartbeat_monotonic": monotonic,
        "global_cpu_percent": float(psutil.cpu_percent(interval=None)),
    }


def stop_preview(child: subprocess.Popen[Any]) -> dict[str, Any]:
    """Terminate only the new child's private process group and reap its leader."""
    diagnostics = []
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=LIMITS["term_grace_seconds"])
    except subprocess.TimeoutExpired:
        diagnostics.append("TERM grace exhausted")
    # The leader may exit while a descendant remains. Its group identity is
    # the private session created for this exact child, never a primary PID.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=LIMITS["term_grace_seconds"])
    except subprocess.TimeoutExpired:
        diagnostics.append("leader reap unconfirmed")
    return {
        "confirmed_exit": child.poll() is not None,
        "returncode": child.poll(),
        "diagnostics": diagnostics,
        "signalled_preview_group": child.pid,
    }


def child_priority() -> None:
    """Apply lower priority inside the new child only, before exec."""
    os.nice(LIMITS["nice"])


def run_stage(
    intent: dict[str, Any], intent_path: Path, identity: str, root: Path, index: int
) -> None:
    """Run one new CPU child, preserving failure and refusing any retry."""
    stage = intent["stages"][index]
    authenticate(intent, intent_path, identity)
    required = remaining(intent, index)
    directory = root / "stages" / stage["id"]
    directory.mkdir()
    output = directory / "output"
    output.mkdir()
    values = {
        "run": str(root),
        "stage_out": str(output),
        "heartbeat": str(output / "heartbeat.json"),
        "intent": str(intent_path),
    }
    command = [part.format_map(values) for part in stage["command"]]
    require(len(command) >= 2 and command[1] in intent["source_files"], "unfrozen child entrypoint")
    durable(
        directory / "started.json",
        {
            "schema_version": 1,
            "stage_id": stage["id"],
            "utc": utc(),
            "intent_sha256": identity,
            "command": command,
            "cap_seconds": stage["cap_seconds"],
            "remaining_required_seconds": required,
        },
    )
    admitted = snapshot(intent, admission=True)
    remaining(intent, index)
    durable(directory / "admitted.json", admitted)
    environment = os.environ.copy()
    environment.update(
        {
            name: "2"
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "TORCH_NUM_THREADS",
            )
        }
    )
    environment.update(SNAKE_DQN_DEVICE="cpu", TORCH_NUM_INTEROP_THREADS="1", PYTHONUNBUFFERED="1")
    child = None
    begun = time.monotonic()
    cpu_since = None
    previous_input_check = begun
    summary = {
        "samples": 0,
        "minimum_available_bytes": admitted["available_bytes"],
        "maximum_preview_rss_bytes": admitted["preview_rss_bytes"],
        "maximum_combined_rss_bytes": admitted["combined_rss_bytes"],
        "maximum_global_cpu_percent": admitted["global_cpu_percent"],
    }

    def record(observation: dict[str, Any], stream: Any) -> None:
        stream.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        summary["samples"] += 1
        summary["minimum_available_bytes"] = min(
            summary["minimum_available_bytes"], observation["available_bytes"]
        )
        for output_key, sample_key in (
            ("maximum_preview_rss_bytes", "preview_rss_bytes"),
            ("maximum_combined_rss_bytes", "combined_rss_bytes"),
            ("maximum_global_cpu_percent", "global_cpu_percent"),
        ):
            summary[output_key] = max(summary[output_key], observation[sample_key])

    try:
        with (
            (directory / "child.log").open("x") as log,
            (directory / "samples.jsonl").open("x") as samples,
        ):
            child = subprocess.Popen(
                command,
                cwd=intent["repo"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=child_priority,
            )
            durable(
                directory / "child.json",
                {
                    "pid": child.pid,
                    "create_time": psutil.Process(child.pid).create_time(),
                    "utc": utc(),
                },
            )
            while child.poll() is None:
                observation = snapshot(intent)
                current = time.monotonic()
                record(observation, samples)
                if observation["global_cpu_percent"] > LIMITS["cpu_percent"]:
                    cpu_since = current if cpu_since is None else cpu_since
                    require(
                        current - cpu_since < LIMITS["cpu_sustained_seconds"],
                        "sustained global CPU pressure",
                    )
                else:
                    cpu_since = None
                require(current - begun <= stage["cap_seconds"], "preview stage wall cap")
                require(
                    (stamp(intent["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
                    >= 120,
                    "preview handoff reserve exhausted",
                )
                heartbeat = output / "heartbeat.json"
                reference = (
                    heartbeat.stat().st_mtime
                    if heartbeat.exists()
                    else time.time() - (current - begun)
                )
                require(
                    time.time() - reference <= LIMITS["child_heartbeat_seconds"],
                    "preview child heartbeat stale",
                )
                if current - previous_input_check >= 5:
                    authenticate(intent, intent_path, identity)
                    previous_input_check = current
                time.sleep(LIMITS["poll_seconds"])
            child.wait(timeout=LIMITS["term_grace_seconds"])
            require(child.returncode == 0, "preview child returned nonzero")
            require(time.monotonic() - begun <= stage["cap_seconds"], "preview stage exceeded cap")
            # No leaf is allowed to leave a background process behind.
            leftovers = psutil.Process(os.getpid()).children(recursive=True)
            require(not leftovers, "preview child left descendants")
            authenticate(intent, intent_path, identity)
            final = snapshot(intent)
            record(final, samples)
            os.fsync(samples.fileno())
        report = output / "report.json"
        payload = read(report)
        require(payload["status"] == "PASS", "missing/failed preview stage report")
        durable(
            directory / "completed.json",
            {
                "schema_version": 1,
                "stage_id": stage["id"],
                "status": "COMPLETE",
                "utc": utc(),
                "intent_sha256": identity,
                "observed_returncode": 0,
                "confirmed_exit": True,
                "termination": "natural_exit",
                "elapsed_seconds": time.monotonic() - begun,
                "started_sha256": sha(directory / "started.json"),
                "admitted_sha256": sha(directory / "admitted.json"),
                "report_sha256": sha(report),
                "source_files_sha256": intent["source_files"],
                "authority_files_sha256": intent["authority_files"],
                "resource_summary": summary,
                "counters": payload.get("counters", {}),
                "samples_sha256": sha(directory / "samples.jsonl"),
            },
        )
    except BaseException as exc:
        termination = stop_preview(child) if child is not None else {"launched": False}
        durable(
            directory / "failure.json",
            {
                "utc": utc(),
                "status": "INVALID_STOP",
                "error": f"{type(exc).__name__}: {exc}",
                "termination": termination,
                "retry_authorized": False,
                "resource_summary": summary,
            },
        )
        raise


def run(intent_path: Path, output: Path) -> None:
    """Admit the separately authorized preview without touching primary locks."""
    intent_path, output = intent_path.resolve(), output.resolve()
    intent = read(intent_path)
    require(intent["schema_version"] == 1, "intent schema")
    require(intent["admitted"] is True, "preview is not admitted")
    require(intent["concurrency_exception"]["authorized"] is True, "no concurrency exception")
    require(output == Path(intent["output_root"]).resolve(), "output root differs from intent")
    require(intent["resource_limits"] == LIMITS, "resource limits differ from reviewed values")
    require(
        [(s["id"], s["cap_seconds"]) for s in intent["stages"]]
        == [("parent", 900), ("candidate", 900), ("audit", 180)],
        "fixed stage plan differs",
    )
    require(
        0 < (stamp(intent["deadline_utc"]) - stamp(intent["frozen_utc"])).total_seconds() <= 2400,
        "prospective window must be at most 2400 seconds",
    )
    require(stamp(intent["frozen_utc"]) <= datetime.now(timezone.utc), "future freeze")
    require(str(Path(__file__).resolve()) in intent["source_files"], "supervisor not frozen")
    review_binding = intent["implementation_review"]
    review_path = Path(review_binding["path"])
    require(sha(review_path) == review_binding["sha256"], "implementation review drift")
    require(
        intent["authority_files"].get(str(review_path)) == review_binding["sha256"],
        "implementation review not bound in authority files",
    )
    review = read(review_path)
    require(review["status"] == "PASS_FOR_ADMISSION", "implementation review not passed")
    for name in ("supervise.py", "evaluate.py", "audit.py"):
        path = str(Path(__file__).resolve().parent / name)
        require(
            review["reviewed_source_sha256"].get(path) == intent["source_files"].get(path)
            and path in intent["source_files"],
            "reviewed preview source identity",
        )
    identity = sha(intent_path)
    authenticate(intent, intent_path, identity)
    remaining(intent, 0)
    output.mkdir(exist_ok=False)
    durable(
        output / "started.json",
        {
            "schema_version": 1,
            "utc": utc(),
            "pid": os.getpid(),
            "intent_sha256": identity,
            "no_retry": True,
        },
    )
    (output / "stages").mkdir()
    lock = None
    try:
        # This new dedicated file has no relation to either primary CPU slot.
        lock = (output.parent / "preview-sidecar.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        remaining(intent, 0)
        psutil.cpu_percent(interval=None)
        durable(output / "admission.json", snapshot(intent, admission=True))
        completed_inputs = {}
        for index in range(len(intent["stages"])):
            for path, expected in completed_inputs.items():
                require(sha(Path(path)) == expected, "completed preview metadata drift")
            run_stage(intent, intent_path, identity, output, index)
            directory = output / "stages" / intent["stages"][index]["id"]
            for name in ("started.json", "admitted.json", "completed.json", "output/report.json"):
                path = directory / name
                completed_inputs[str(path)] = sha(path)
        authenticate(intent, intent_path, identity)
        for path, expected in completed_inputs.items():
            require(sha(Path(path)) == expected, "final preview metadata drift")
        require(
            (stamp(intent["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds() >= 120,
            "final preview handoff reserve exhausted",
        )
        durable(
            output / "completed.json",
            {
                "utc": utc(),
                "status": "AUDITED_PREVIEW_COMPLETE",
                "intent_sha256": identity,
                "promotion_eligible": False,
                "original_study_modified": False,
                "audit_report_sha256": sha(output / "stages/audit/output/report.json"),
            },
        )
    except BaseException as exc:
        durable(
            output / "failure.json",
            {
                "utc": utc(),
                "status": "INVALID_STOP",
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
            },
        )
        raise
    finally:
        if lock is not None:
            lock.close()


def main() -> None:
    """Parse a new preview intent and its create-only evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.intent, args.out)


if __name__ == "__main__":
    main()
