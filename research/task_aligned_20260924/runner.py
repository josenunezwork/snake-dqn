"""Create-only admission of the frozen task-aligned study; never resume a stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COUNTERS = (
    "native_frames",
    "hero_transitions",
    "model_forwards",
    "checkpoint_loads",
    "optimizer_updates",
)


def digest(path: Path) -> str:
    """Hash bytes without deserializing model payloads."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting nonfinite JSON constants."""

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON: {value}")

    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def durable(path: Path, value: dict[str, Any]) -> None:
    """Create a receipt exactly once, fsync its content and directory."""
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


def now() -> str:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str) -> datetime:
    """Require an explicit timezone in authority timestamps."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("naive authority timestamp")
    return parsed


def require(condition: bool, message: str) -> None:
    """Do not let Python optimization disable admission checks."""
    if not condition:
        raise RuntimeError(message)


def check_sources(intent: dict[str, Any]) -> None:
    """Authenticate the explicit source closure, without discovering old data."""
    for path, expected in intent["source_sha256"].items():
        require(digest(Path(path)) == expected, f"source drift: {path}")


def validate_intent(intent: dict[str, Any]) -> None:
    """Reject malformed or excessive plans before admitting any child."""
    require(intent["schema_version"] == 1, "intent schema")
    require(intent["admitted"] is True, "intent not admitted")
    require(intent["device"] in {"cpu", "mps"}, "device")
    stages = intent["stages"]
    require(bool(stages) and stages[0]["kind"] == "qualification", "qualification first")
    require(stages[-1]["kind"] == "audit", "audit last")
    require(sum(s["kind"] == "audit" for s in stages) == 1, "one audit")
    identifiers = [s["id"] for s in stages]
    require(len(set(identifiers)) == len(identifiers), "duplicate stage")
    for stage in stages:
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", stage["id"]) is not None,
            "unsafe stage id",
        )
        require(
            stage["kind"] in {"qualification", "training", "evaluation", "analysis", "audit"},
            "unknown stage kind",
        )
        require(stage.get("device", intent["device"]) in {"cpu", "mps"}, "stage device")
        cap = stage["cap_seconds"]
        require(
            type(cap) in (int, float) and math.isfinite(cap) and 0 < cap <= 86400,
            "invalid stage cap",
        )
        require(
            bool(stage["argv"]) and all(isinstance(x, str) for x in stage["argv"]),
            "argv must be an argument array",
        )
        require(set(COUNTERS) <= set(stage["counter_caps"]), "missing physical caps")
        for cap in stage["counter_caps"].values():
            require(type(cap) is int and cap >= 0, "invalid counter cap")
        if stage["kind"] == "training":
            require(stage["counter_caps"]["hero_transitions"] <= 10_000_000, "per-lineage ceiling")
    training = [s for s in stages if s["kind"] == "training"]
    require(
        len(training) == 3 and len({s["lineage"] for s in training}) == 3,
        "exactly three fixed training lineages",
    )
    require(
        sum(s["counter_caps"]["hero_transitions"] for s in training) <= 30_000_000,
        "total scientific hero-transition ceiling",
    )
    require(sum(s["cap_seconds"] for s in stages) <= 86400, "24-hour numerical ceiling")
    require(intent["handoff_seconds"] == 120, "handoff reserve must be 120 seconds")
    require(timestamp(intent["frozen_utc"]) <= datetime.now(timezone.utc), "future freeze")
    require(timestamp(intent["deadline"]) > timestamp(intent["frozen_utc"]), "deadline order")
    require(
        (timestamp(intent["deadline"]) - timestamp(intent["frozen_utc"])).total_seconds() <= 86400,
        "prospective elapsed window exceeds 24 hours",
    )
    closure = intent["source_sha256"]
    require(str(Path(__file__).resolve()) in closure, "runner not frozen")
    require(intent["supervisor_source"] in closure, "supervisor not frozen")
    require(
        str(Path(intent["repo"]) / "src/scripts/pqn_correctness_diagnostic.py") in closure,
        "monitor not frozen",
    )


def check_remaining(intent: dict[str, Any], index: int) -> float:
    """Reserve every remaining stage, including audit, plus handoff."""
    needed = sum(s["cap_seconds"] for s in intent["stages"][index:]) + 120
    require(
        (timestamp(intent["deadline"]) - datetime.now(timezone.utc)).total_seconds() >= needed,
        "remaining caps and handoff do not fit",
    )
    return needed


def report_ok(path: Path, intent: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    """Validate bounded physical counters and fixed output identity."""
    report = read(path)
    require(
        report["schema_version"] == 1
        and report["study_id"] == intent["study_id"]
        and report["stage_id"] == stage["id"]
        and report["status"] == "PASS",
        "stage report identity/status",
    )
    for key, cap in stage["counter_caps"].items():
        count = report["counters"][key]
        require(type(count) is int and 0 <= count <= cap, f"counter violation: {key}")
    for key, value in stage.get("expected_report_fields", {}).items():
        require(report.get(key) == value, f"required report field: {key}")
    if stage["kind"] == "qualification":
        require(report["fixed_science_fits"] is True, "fixed science not qualified")
        require(report["runtime_contracts_pass"] is True, "runtime contracts not qualified")
        estimate = report["estimated_science_seconds"]
        science_caps = sum(s["cap_seconds"] for s in intent["stages"] if s["kind"] == "training")
        require(
            type(estimate) in (int, float)
            and math.isfinite(estimate)
            and 0 < estimate <= science_caps,
            "fixed science throughput budget",
        )
    return report


def qualify_dose(report: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    """Seal only the predeclared conservative hardware-derived rollout dose."""
    training = [s for s in intent["stages"] if s["kind"] == "training"]
    cycle = report["conservative_cycle_seconds"]
    overhead = report["fixed_overhead_seconds"]
    require(
        type(cycle) in (int, float) and math.isfinite(cycle) and cycle > 0,
        "invalid measured cycle cost",
    )
    require(
        type(overhead) in (int, float) and math.isfinite(overhead) and overhead >= 0,
        "invalid measured fixed overhead",
    )
    require(report["shared_probe_lineage"] == 2026096101, "fixed timing probe lineage")
    cap = min(s["cap_seconds"] for s in training)
    selected = min(39060, int((cap / 2 - overhead) / cycle) * 5)
    require(
        selected >= 7815 and report["selected_rollouts"] == selected,
        "qualification dose differs from hardware-only rule",
    )
    forecast = 2 * (overhead + selected / 5 * cycle)
    require(
        report["forecast_per_lineage_seconds"] == forecast
        and report["estimated_science_seconds"] == 3 * forecast,
        "qualification forecast",
    )
    payload = {
        "selected_rollouts": selected,
        "conservative_cycle_seconds": cycle,
        "fixed_overhead_seconds": overhead,
        "shared_probe_lineage": 2026096101,
        "profile_rollouts": {
            "solo": selected // 5,
            "S2": selected * 2 // 5,
            "S6": selected * 2 // 5,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["dose_sha256"] = hashlib.sha256(encoded).hexdigest()
    require(report["dose_sha256"] == payload["dose_sha256"], "dose binding")
    return payload


def load_supervisor(path: Path) -> Any:
    """Load reviewed helpers only; never invoke an old supervisor's main."""
    spec = importlib.util.spec_from_file_location("task_aligned_supervisor_helpers", path)
    require(spec is not None and spec.loader is not None, "supervisor import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(intent_path: Path, root: Path) -> None:
    """Admit the ordered plan once and stop on the first uncertainty."""
    intent_path = intent_path.resolve()
    intent = read(intent_path)
    validate_intent(intent)
    require(root == Path(intent["output_root"]).resolve(), "root differs from frozen intent")
    check_sources(intent)
    root.mkdir(parents=False, exist_ok=False)
    intent_hash = digest(intent_path)
    durable(
        root / "started.json",
        {
            "study_id": intent["study_id"],
            "utc": now(),
            "intent_sha256": intent_hash,
            "pid": os.getpid(),
        },
    )
    try:
        supervisor = load_supervisor(Path(intent["supervisor_source"]))
        environment = os.environ.copy()
        environment.update(supervisor.capacity_environment(2))
        environment["SNAKE_DQN_DEVICE"] = intent["device"]
        os.environ.update(supervisor.capacity_environment(2))
        supervisor.capacity_preflight(9.6)
        monitor, _ = supervisor.load_monitor(Path(intent["repo"]))
        limits = {
            "rss_bytes": 8 * 1024**3,
            "available_bytes": math.ceil(9.6 * 1024**3),
            "mps_driver_bytes": 24 * 1024**3,
            "no_heartbeat_seconds": 20.0,
            "term_grace_seconds": 5.0,
            "poll_seconds": 0.2,
        }
        totals = {key: 0 for key in COUNTERS}
        prior = {}
        dose = None
        for index, stage in enumerate(intent["stages"]):
            check_sources(intent)
            require(digest(intent_path) == intent_hash, "intent drift")
            for path, expected in prior.items():
                require(digest(Path(path)) == expected, f"completed output drift: {path}")
            needed = check_remaining(intent, index)
            directory = root / stage["id"]
            directory.mkdir(exist_ok=False)
            out = directory / "output"
            out.mkdir(exist_ok=False)
            values = {
                "root": str(root),
                "out": str(out),
                "intent": str(intent_path),
                "heartbeat": str(out / "heartbeat.json"),
                "stage_id": stage["id"],
            }
            command = [item.format_map(values) for item in stage["argv"]]
            child_environment = environment.copy()
            child_environment["SNAKE_DQN_DEVICE"] = stage.get("device", intent["device"])
            require(
                len(command) >= 2
                and command[0] == intent["python"]
                and command[1] in intent["source_sha256"],
                "unbound stage entrypoint",
            )
            durable(
                directory / "started.json",
                {
                    "schema_version": 1,
                    "stage_id": stage["id"],
                    "utc": now(),
                    "intent_sha256": intent_hash,
                    "argv": command,
                    "cap_seconds": stage["cap_seconds"],
                    "remaining_required_seconds": needed,
                    "device": child_environment["SNAKE_DQN_DEVICE"],
                },
            )
            handles = []
            child = None
            result = None
            hardware = None
            try:
                handles = supervisor.acquire_all_slots()
                check_remaining(intent, index)
                check_sources(intent)
                require(digest(intent_path) == intent_hash, "intent drift after lock")
                for path, expected in prior.items():
                    require(digest(Path(path)) == expected, f"post-lock output drift: {path}")
                hardware = supervisor.capacity_preflight(9.6)
                check_remaining(intent, index)
                durable(directory / "admitted.json", {"utc": now(), "hardware": hardware})
                with (directory / "child.log").open("x") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=intent["repo"],
                        env=child_environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    result = monitor._monitor_process(
                        child,
                        limits,
                        wall_seconds=stage["cap_seconds"],
                        heartbeat_path=out / "heartbeat.json",
                        now=time.monotonic,
                        sleep=time.sleep,
                        process_factory=monitor.psutil.Process,
                    )
                    result["observed_returncode"] = child.poll()
            except BaseException as exc:
                result = {
                    "cause": "harness_error",
                    "confirmed_exit": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if child is not None and child.poll() is None:
                    termination = monitor.stop_process(child, 5.0, diagnostics=[])
                    result = result or {"cause": "unconfirmed_exit", "confirmed_exit": False}
                    result["finally_termination"] = termination
                if handles:
                    supervisor.close_slots(handles)
            durable(
                directory / "supervisor.json",
                {"result": result, "limits": limits, "hardware": hardware, "utc": now()},
            )
            require(
                result is not None
                and result.get("cause") is None
                and result.get("confirmed_exit") is True
                and result.get("termination") == "natural_exit"
                and result.get("observed_returncode") == 0,
                "failed or ambiguous child",
            )
            require(result["elapsed_seconds"] <= stage["cap_seconds"], "stage cap exceeded")
            check_sources(intent)
            require(digest(intent_path) == intent_hash, "intent drift after child")
            for path, expected in prior.items():
                require(digest(Path(path)) == expected, f"prior output drift after child: {path}")
            report_path = out / "report.json"
            report = report_ok(report_path, intent, stage)
            if stage["kind"] == "qualification":
                dose = qualify_dose(report, intent)
                durable(root / "qualified-dose.json", dose)
                prior[str(root / "qualified-dose.json")] = digest(root / "qualified-dose.json")
            if stage["kind"] == "training":
                require(dose is not None, "training without dose qualification")
                require(
                    report["lineage"] == stage["lineage"]
                    and report["selected_rollouts"] == dose["selected_rollouts"]
                    and report["completed_rollouts"] == dose["selected_rollouts"]
                    and report["profile_rollouts"] == dose["profile_rollouts"]
                    and report["dose_sha256"] == dose["dose_sha256"],
                    "training dose identity",
                )
            artifacts = {}
            for artifact in report["artifacts"]:
                path = (out / artifact["path"]).resolve()
                require(path.is_relative_to(out.resolve()), "artifact outside stage output")
                require(path.is_file() and digest(path) == artifact["sha256"], "artifact drift")
                artifacts[str(path)] = artifact["sha256"]
            for key in COUNTERS:
                totals[key] += report["counters"][key]
            complete = {
                "schema_version": 1,
                "stage_id": stage["id"],
                "utc": now(),
                "intent_sha256": intent_hash,
                "report_sha256": digest(report_path),
                "supervisor_sha256": digest(directory / "supervisor.json"),
                "artifacts": artifacts,
                "counters": report["counters"],
            }
            durable(directory / "completed.json", complete)
            prior[str(report_path)] = complete["report_sha256"]
            prior.update(artifacts)
            for name in ("started.json", "admitted.json", "supervisor.json", "completed.json"):
                prior[str(directory / name)] = digest(directory / name)
        for path, expected in prior.items():
            require(digest(Path(path)) == expected, f"final output drift: {path}")
        require(
            (timestamp(intent["deadline"]) - datetime.now(timezone.utc)).total_seconds() >= 120,
            "handoff reserve exhausted",
        )
        durable(
            root / "completed.json",
            {
                "status": "AUDITED_PENDING_ROOT_CLOSEOUT",
                "utc": now(),
                "intent_sha256": intent_hash,
                "counters": totals,
                "audit_report_sha256": digest(
                    root / intent["stages"][-1]["id"] / "output/report.json"
                ),
            },
        )
    except BaseException as exc:
        durable(
            root / "failure.json",
            {
                "status": "INVALID_STOP",
                "utc": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "retry_authorized": False,
            },
        )
        raise


def main() -> None:
    """Parse the explicit new intent and new evidence root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.intent, args.root.resolve())


if __name__ == "__main__":
    main()
