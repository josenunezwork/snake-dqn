"""Composite offline proof for the prospective supervision policy.

This binds independently verified identity/operation evidence to the exact CPU
ledger and ordinary host-load guards. It cannot supply the missing live observer:
OFFLINE_VALID is a replay result, never live panel admission or qualification.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

from research.bl_paced_identity.events import canonical_bytes
from research.bl_served_search import host_load, supervision_cpu, supervision_identity
from research.bl_served_search.background_report import _samples

POLICY = "classified-supervised-v2"
LIVE_CAPTURE_CAPABILITY = "NOT_ADMISSIBLE"


def digest(value: Any) -> str:
    """Bind exactly the finite canonical bytes used by immutable exports."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Validate the prospective schema without mistaking a seal for observation."""
    return supervision_identity.validate_registry(registry)


def _crosscheck(identity: dict, cpu: dict, registry: dict, errors: list[str]) -> None:
    roots = {r["id"]: supervision_identity.process_id(r) for r in registry["roots"]}
    lifecycles = identity["lifecycles"]
    processes = {row["id"]: row for row in cpu["processes"]}
    operations = {row["id"]: row for row in cpu["operations"]}
    if set(cpu["roots"]) != set(roots.values()) or set(processes) != set(lifecycles):
        errors.append("identity/CPU process inventory differs")
    if set(operations) != set(identity["verified_operations"]):
        errors.append("identity/CPU operation inventory differs")
    if (
        registry["capture_start_ns"] != cpu["samples"][0]["time_ns"]
        or registry["capture_end_ns"] != cpu["samples"][-1]["time_ns"]
    ):
        errors.append("identity/CPU capture coverage differs")
    for key, verified in lifecycles.items():
        actual = processes.get(key, {})
        expected = {
            "root_id": roots[verified["root_id"]],
            "parent_id": verified.get("parent_id"),
            "operation_id": verified.get("operation_id"),
            "exit_ns": verified.get("exit_ns"),
        }
        if key not in roots.values():
            expected["launch_ns"] = verified["launch_ns"]
            exit_record = verified.get("exit_record", {})
            for field in ("final_cpu_ns", "final_scope", "descendants_complete"):
                if field not in exit_record or actual.get(field) != exit_record[field]:
                    errors.append("identity/CPU final counter evidence differs: " + key)
            if actual.get("final_scope") == "subtree" and (
                "final_inclusions" not in exit_record
                or exit_record["final_inclusions"] != actual.get("final_inclusions")
            ):
                errors.append("identity/CPU final inclusion ledger differs: " + key)
        if any(actual.get(field) != value for field, value in expected.items()):
            errors.append("identity/CPU lifecycle differs: " + key)
    for key, verified in identity["verified_operations"].items():
        expected = {
            "root_id": roots[verified["root_id"]],
            "start_ns": verified["start_ns"],
            "end_ns": verified["end_ns"],
        }
        if any(operations.get(key, {}).get(field) != value for field, value in expected.items()):
            errors.append("identity/CPU operation span differs: " + key)


def _permitted(row: dict, at: int, lifecycles: dict) -> bool:
    key = supervision_identity.process_id(row)
    verified = lifecycles.get(key)
    return bool(
        verified is not None
        and row.get("command") == verified["identity"]["interpreter"]
        and verified["launch_ns"] <= at
        and (verified["exit_ns"] is None or at <= verified["exit_ns"])
    )


def _permitted_interval(row: dict, start: int, end: int, lifecycles: dict) -> bool:
    verified = lifecycles.get(supervision_identity.process_id(row))
    return bool(
        verified is not None
        and row.get("command") == verified["identity"]["interpreter"]
        and verified["launch_ns"] < end
        and (verified["exit_ns"] is None or verified["exit_ns"] > start)
    )


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def replay_ordinary(
    points: list,
    intervals: list,
    lifecycles: dict,
    ledger: dict,
    *,
    desktop_background: set[str] | None = None,
) -> dict:
    """Replay both original ordinary abort paths and sustained service ceiling.

    Complete raw point rows retain historical CPU. CPU interval process rows
    retain their measured deltas. Only exact witnessed birth identities are
    removed from ordinary work; their CPU is accounted by the separate ledger.
    Raw evidence is never mutated. Churn/coverage remains conservative.
    """
    errors: list[str] = []
    projected = []
    reconciliation = []
    if not points or not intervals:
        errors.append("missing raw point or ordinary interval evidence")
    try:
        previous_point = None
        for screen in points:
            at = screen["time_ns"]
            if type(at) is not int or at < 0 or not isinstance(screen["processes"], list):
                raise ValueError("malformed point screen")
            if (
                at < ledger.get("start_ns", at)
                or at > ledger.get("end_ns", at)
                or (previous_point is not None and at <= previous_point)
            ):
                raise ValueError("point screen outside accounting stream or out of order")
            previous_point = at
            for row in screen["processes"]:
                if (
                    not _number(row.get("cpu_percent"))
                    or not isinstance(row.get("command"), str)
                    or type(row.get("pid")) is not int
                    or row["pid"] <= 0
                    or type(row.get("ppid")) is not int
                    or row["ppid"] < 0
                ):
                    raise ValueError("malformed point CPU/executable")
                kind = host_load.process_kind(row["command"])
                if digest([at, at, row]) in (desktop_background or ()):
                    kind = "desktop_renderer"
                if (
                    not _permitted(row, at, lifecycles)
                    and kind not in host_load.BACKGROUND_KINDS
                    and row["cpu_percent"] >= 20
                ):
                    errors.append("ordinary workload/unknown point >=20%")
        guard = host_load.LoadGuard()
        previous_end = None
        for raw in intervals:
            sample = deepcopy(raw)
            a, b = sample["start_ns"], sample["end_ns"]
            if (
                type(a) is not int
                or type(b) is not int
                or b <= a
                or (previous_end is not None and a != previous_end)
            ):
                raise ValueError("ordinary CPU stream missing/overlapping interval")
            previous_end = b
            if type(sample.get("coverage_complete")) is not bool or not isinstance(
                sample.get("process_churn"), list
            ):
                raise ValueError("ordinary CPU coverage/churn evidence missing")
            rows = []
            totals = {"background": 0.0, "workload": 0.0, "unknown": 0.0}
            seen_permitted = set()
            for row in sample["processes"]:
                if (
                    not _number(row.get("cpu_percent"))
                    or not isinstance(row.get("command"), str)
                    or type(row.get("pid")) is not int
                    or row["pid"] <= 0
                    or type(row.get("ppid")) is not int
                    or row["ppid"] < 0
                ):
                    raise ValueError("malformed interval CPU/executable")
                if _permitted_interval(row, a, b, lifecycles):
                    key = supervision_identity.process_id(row)
                    if key in seen_permitted:
                        errors.append("duplicate permitted process in ordinary interval")
                    seen_permitted.add(key)
                    numerator = row.get("cpu_delta_numerator")
                    denominator = row.get("counter_denominator")
                    expected = sum(
                        charge["cpu_numerator"]
                        for charge in ledger.get("charges", [])
                        if charge["process_id"] == key and a < charge["end_ns"] <= b
                    )
                    if (
                        type(numerator) is not int
                        or numerator < 0
                        or type(denominator) is not int
                        or denominator <= 0
                        or denominator != ledger.get("counter_denominator")
                        or numerator != expected
                        or not math.isclose(
                            row["cpu_percent"],
                            100 * numerator / (denominator * (b - a)),
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    ):
                        errors.append("independent supervision sampler/CPU ledger mismatch")
                    reconciliation.append(
                        {
                            "process_id": key,
                            "start_ns": a,
                            "end_ns": b,
                            "cpu_delta_numerator": numerator,
                            "ledger_delta_numerator": expected,
                        }
                    )
                    continue
                desktop_verified = digest([a, b, row]) in (desktop_background or ())
                row["kind"] = host_load.process_kind(row["command"])
                if desktop_verified:
                    # The combined reducer independently replays desktop identity
                    # first. Retain the CPU in the SAME shared background pool.
                    row["kind"] = "desktop_renderer"
                group = (
                    "background"
                    if row["kind"] in host_load.BACKGROUND_KINDS
                    else "workload" if row["kind"] == "agent_workload" else "unknown"
                )
                totals[group] += row["cpu_percent"]
                rows.append(row)
            required = {
                key
                for key, proc in lifecycles.items()
                if proc["launch_ns"] < b and (proc["exit_ns"] is None or proc["exit_ns"] > a)
            } | {c["process_id"] for c in ledger.get("charges", []) if a < c["end_ns"] <= b}
            if required != seen_permitted:
                errors.append("missing permitted process reconciliation in ordinary interval")
            sample.update(processes=rows, **{k + "_cpu_percent": v for k, v in totals.items()})
            projected.append(sample)
            try:
                guard.check(sample)
            except RuntimeError as exc:
                errors.append(str(exc))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append(str(exc))
    try:
        # Validate only the unchanged wire shape on our derived v2 projection;
        # never read a historical v1 artifact through this policy path.
        _samples({"load_policy": "classified-v1", "cpu_samples": projected})
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append("ordinary CPU wire evidence: " + str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "ordinary_cpu_samples": projected,
        "sampler_reconciliation": reconciliation,
    }


def replay(
    registry: dict,
    journal: list,
    cpu_evidence: dict,
    *,
    observations: list,
    point_screens: list,
    ordinary_intervals: list,
    final: bool = True,
) -> dict:
    """Independently bind identity, lifecycle, payload, CPU and ordinary guards."""
    identity = supervision_identity.verify_identity(registry, journal, observations, final=final)
    cpu = supervision_cpu.replay_cpu(cpu_evidence, final=final)
    errors = [*identity["errors"], *cpu["errors"]]
    if final and not cpu.get("admission_ready"):
        errors.append("60-second host-only admission unavailable before exposure")
    if identity["valid"] and cpu["valid"]:
        try:
            _crosscheck(identity, cpu_evidence, registry, errors)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            errors.append("identity/CPU binding malformed: " + str(exc))
    ordinary = replay_ordinary(point_screens, ordinary_intervals, identity["lifecycles"], cpu)
    errors.extend(ordinary["errors"])
    if cpu.get("intervals") and (
        [(r["start_ns"], r["end_ns"]) for r in ordinary["ordinary_cpu_samples"]]
        != [(r["start_ns"], r["end_ns"]) for r in cpu["intervals"]]
    ):
        errors.append("ordinary/supervision coverage boundaries differ")
    return {
        **cpu,
        "valid": not errors,
        "status": "OFFLINE_VALID" if not errors else "INVALID",
        "errors": errors,
        "identity": identity,
        "registry_digest": digest(registry),
        "ordinary_cpu_samples": ordinary["ordinary_cpu_samples"],
        "sampler_reconciliation": ordinary["sampler_reconciliation"],
        "coverage_complete": not errors and cpu.get("complete", False),
        "live_capture_capability": LIVE_CAPTURE_CAPABILITY,
    }
