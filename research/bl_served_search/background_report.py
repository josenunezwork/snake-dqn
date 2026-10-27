"""Background-load covariates for the served-tick timing diagnostic.

This module deliberately does not decide the panel verdict.  It aligns sealed,
off-tick host samples to individual trace rows so a report can describe the
environment in which timing was observed without treating routine macOS work as
an automatic contamination reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from research.bl_served_search.served_tick_probe import _distribution

POLICY = "classified-v1"
DESKTOP_POLICY = "classified-desktop-v1"
SUPPORTED_POLICIES = {POLICY, DESKTOP_POLICY}
LOW_BACKGROUND_PERCENT = 20.0
MAX_SAMPLE_NS = 5_000_000_000
PROCESS_KINDS = {"macos_service", "desktop_renderer", "agent_workload", "unknown"}
DESKTOP_PROCESS_KINDS = PROCESS_KINDS | {"desktop_app"}
BACKGROUND_KINDS = {"macos_service", "desktop_renderer", "desktop_app"}
BACKGROUND_CATEGORY_KINDS = {
    "service": "macos_service",
    "desktop_renderer": "desktop_renderer",
    "desktop_app": "desktop_app",
}
RAW_CATEGORY_FIELDS = {
    "service": "macos_service_cpu_percent",
    "desktop_renderer": "desktop_renderer_cpu_percent",
    "desktop_app": "desktop_app_cpu_percent",
}


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"malformed {name}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"malformed {name}")
    return float(value)


def _replay_kind(row: Mapping[str, Any], registry_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply workload and system precedence before sealed desktop admission."""
    from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
    from research.bl_served_search.host_load import _workload_kind

    command = row.get("command")
    if not isinstance(command, str):
        return {"kind": "unknown", "reason": "missing_command"}
    # A declared substantive workload always wins, even when hosted by an
    # enrolled GUI executable.  Bare executable names alone cannot decide
    # this: Claude/Codex desktop mains are eligible only through their sealed
    # PID/birth/UID binding, while their CLI names remain workloads below.
    if row.get("known_workload") or row.get("workload"):
        return {"kind": "agent_workload", "reason": "known_workload"}
    if any(
        item.get("exact_executable") == command
        for item in registry_evidence.get("registry", {}).get("workload_exclusions", [])
    ):
        return {"kind": "agent_workload", "reason": "workload_exclusion"}
    workload = _workload_kind(command)
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(command).parts:
        if workload is not None:
            return {"kind": workload, "reason": "workload_precedence"}
        return {"kind": "macos_service", "reason": "system_executable"}
    from research.bl_served_search.desktop_apps import classify_exported

    enrolled = classify_exported(dict(row), dict(registry_evidence))
    if enrolled.get("kind") in {"desktop_app", "desktop_renderer"}:
        return enrolled
    if workload is not None:
        return {"kind": workload, "reason": "workload_precedence"}
    return enrolled


def _samples(host_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = host_evidence.get("load_policy")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError("unsupported host load policy")
    if policy == DESKTOP_POLICY:
        registry = host_evidence.get("desktop_registry")
        registry_sha256 = host_evidence.get("desktop_registry_sha256")
        registry_evidence = host_evidence.get("desktop_registry_evidence")
        if (
            not isinstance(registry, Mapping)
            or registry.get("policy") != DESKTOP_POLICY
            or not isinstance(registry_sha256, str)
            or len(registry_sha256) != 64
            or not isinstance(registry_evidence, Mapping)
            or registry_evidence.get("policy") != DESKTOP_POLICY
            or registry_evidence.get("registry") != registry
            or registry_evidence.get("registry_sha256") != registry_sha256
        ):
            raise ValueError("desktop policy registry evidence differs")
    raw = host_evidence.get("cpu_samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing CPU samples")
    result: list[dict[str, Any]] = []
    previous_end = -1
    for index, sample in enumerate(raw):
        if not isinstance(sample, Mapping):
            raise ValueError("malformed CPU sample")
        start = _integer(sample.get("start_ns"), "sample start")
        end = _integer(sample.get("end_ns"), "sample end")
        if end <= start or start < previous_end:
            raise ValueError("out-of-order or overlapping CPU samples")
        previous_end = end
        if type(sample.get("coverage_complete")) is not bool:
            raise ValueError("malformed sample coverage")
        if not isinstance(sample.get("process_churn"), list):
            raise ValueError("malformed sample process churn")
        if sample["coverage_complete"] and (sample["process_churn"] or end - start > MAX_SAMPLE_NS):
            raise ValueError("sample coverage conflicts with churn or interval length")
        processes = sample.get("processes")
        if not isinstance(processes, list):
            raise ValueError("malformed sample processes")
        checked_processes = []
        for process in processes:
            if not isinstance(process, Mapping):
                raise ValueError("malformed sampled process")
            if type(process.get("pid")) is not int or type(process.get("ppid")) is not int:
                raise ValueError("malformed sampled process identity")
            if not isinstance(process.get("command"), str) or not isinstance(
                process.get("kind"), str
            ):
                raise ValueError("malformed sampled process classification")
            if process["kind"] not in (
                DESKTOP_PROCESS_KINDS if policy == DESKTOP_POLICY else PROCESS_KINDS
            ):
                raise ValueError("unknown sampled process kind")
            classification = process.get("classification")
            if policy == DESKTOP_POLICY and (
                not isinstance(classification, Mapping)
                or classification.get("kind") != process["kind"]
            ):
                raise ValueError("desktop process classification evidence differs")
            if policy == DESKTOP_POLICY:
                replay = _replay_kind(process, registry_evidence)
                if replay.get("kind") != process["kind"]:
                    raise ValueError("desktop registry replay differs")
                if process["kind"] in {"desktop_app", "desktop_renderer"} and replay != dict(
                    classification
                ):
                    raise ValueError("desktop identity replay differs")
            if process["kind"] == "desktop_app":
                # The collector binds the process to its sealed enrollment.  Do
                # not turn a friendly command name into an admission here.
                identity = process.get("desktop_identity", process)
                if not isinstance(identity, Mapping):
                    raise ValueError("desktop app missing identity evidence")
                # Keep both the collector's compact process fields and the
                # nested export form so raw exports remain independently useful.
                for key, aliases in {
                    "entry": ("entry", "registry_id"),
                    "role": ("role",),
                    "reason": ("reason", "admission_reason"),
                    "identity": ("identity",),
                }.items():
                    value = next(
                        (identity.get(alias) for alias in aliases if identity.get(alias)), None
                    )
                    if not isinstance(value, (str, Mapping)) or not value:
                        raise ValueError("malformed desktop app identity evidence")
                if any(
                    classification.get(key) != identity.get(key)
                    for key in ("entry", "application", "role", "identity")
                ):
                    raise ValueError("desktop app identity/classification differs")
            checked_processes.append(
                {
                    "pid": process["pid"],
                    "ppid": process["ppid"],
                    "command": process["command"],
                    **({"birth": process["birth"]} if "birth" in process else {}),
                    **({"uid": process["uid"]} if "uid" in process else {}),
                    "kind": process["kind"],
                    "cpu_percent": _number(process.get("cpu_percent"), "process CPU"),
                    **({"classification": dict(classification)} if classification else {}),
                    **(
                        {"desktop_identity": dict(process.get("desktop_identity", process))}
                        if process["kind"] == "desktop_app"
                        else {}
                    ),
                }
            )
        category_totals = {
            category: sum(row["cpu_percent"] for row in checked_processes if row["kind"] == kind)
            for category, kind in BACKGROUND_CATEGORY_KINDS.items()
        }
        totals = {
            "background": sum(category_totals.values()),
            "workload": sum(
                row["cpu_percent"] for row in checked_processes if row["kind"] == "agent_workload"
            ),
            "unknown": sum(
                row["cpu_percent"] for row in checked_processes if row["kind"] == "unknown"
            ),
        }
        reported = {
            "background": _number(sample.get("background_cpu_percent"), "background CPU"),
            "workload": _number(sample.get("workload_cpu_percent"), "workload CPU"),
            "unknown": _number(sample.get("unknown_cpu_percent"), "unknown CPU"),
        }
        if any(not math.isclose(totals[key], reported[key], abs_tol=1e-6) for key in totals):
            raise ValueError("sample CPU totals do not reconcile with process rows")
        # The desktop policy exports independently auditable categories.  They
        # are not separate allowances: their sum is the sole shared pool.
        if policy == DESKTOP_POLICY:
            for category, total in category_totals.items():
                field = RAW_CATEGORY_FIELDS[category]
                if not math.isclose(_number(sample.get(field), field), total, abs_tol=1e-6):
                    raise ValueError("sample desktop CPU totals do not reconcile with process rows")
        result.append(
            {
                "index": index,
                "start_ns": start,
                "end_ns": end,
                "background_cpu_percent": reported["background"],
                **{f"{category}_cpu_percent": total for category, total in category_totals.items()},
                "workload_cpu_percent": reported["workload"],
                "unknown_cpu_percent": reported["unknown"],
                "coverage_complete": sample["coverage_complete"],
                "processes": checked_processes,
            }
        )
    return result


def _background_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconcile the shared pool without converting it into separate quotas."""
    high_since: int | None = None
    last_end: int | None = None
    max_sustained = 0
    for sample in samples:
        duration = sample["end_ns"] - sample["start_ns"]
        high = sample["background_cpu_percent"] > 150.0 and duration <= MAX_SAMPLE_NS
        if high:
            if high_since is None or last_end != sample["start_ns"]:
                high_since = sample["start_ns"]
            max_sustained = max(max_sustained, sample["end_ns"] - high_since)
        else:
            high_since = None
        last_end = sample["end_ns"]
    maxima = {
        category: max((sample[f"{category}_cpu_percent"] for sample in samples), default=None)
        for category in ("service", "desktop_renderer", "desktop_app")
    }
    maxima["combined_background"] = max(
        (sample["background_cpu_percent"] for sample in samples), default=None
    )
    return {
        "cpu_percent_max": maxima,
        "shared_background_pool": {
            "strict_percent": 150.0,
            "strict_duration_ns": 60_000_000_000,
            "max_observed_sustained_ns": max_sustained,
            "exceeded": max_sustained > 60_000_000_000,
        },
    }


def _process_observations(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep raw role/identity decisions visible without deduplicating CPU evidence."""
    rows = []
    for sample in samples:
        for process in sample["processes"]:
            row = {
                "sample_index": sample["index"],
                "coverage_complete": sample["coverage_complete"],
                **process,
            }
            rows.append(row)
    return rows


def _classification_changes(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    changes = []
    for sample in samples:
        for process in sample["processes"]:
            key = (process["pid"], process.get("birth"), process.get("uid"))
            current = {
                **process.get("classification", {"kind": process["kind"]}),
                "command": process["command"],
            }
            old = previous.get(key)
            if old is not None and old != current:
                changes.append(
                    {
                        "sample_index": sample["index"],
                        "pid": process["pid"],
                        "previous_command": old.get("command"),
                        "command": process["command"],
                        "previous": dict(old),
                        "current": dict(current),
                    }
                )
            previous[key] = current
    return changes


def _application_coverage(
    samples: Sequence[Mapping[str, Any]], ticks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        interval_cpu: dict[tuple[str, str], float] = {}
        for process in sample["processes"]:
            if process["kind"] not in {"desktop_app", "desktop_renderer"}:
                continue
            identity = process.get("classification", {})
            key = (str(identity.get("entry", "unknown")), str(identity.get("role", "unknown")))
            row = by_key.setdefault(
                key,
                {
                    "entry": key[0],
                    "application": identity.get("application"),
                    "role": key[1],
                    "kind": process["kind"],
                    "cpu_percent_max": 0.0,
                    "sample_indexes": [],
                    "coverage_complete_samples": 0,
                },
            )
            interval_cpu[key] = interval_cpu.get(key, 0.0) + process["cpu_percent"]
        for key, cpu in interval_cpu.items():
            row = by_key[key]
            row["cpu_percent_max"] = max(row["cpu_percent_max"], cpu)
            row["sample_indexes"].append(sample["index"])
            row["coverage_complete_samples"] += int(sample["coverage_complete"])
    for row in by_key.values():
        indexes = set(row["sample_indexes"])
        row["affected_tick_count"] = sum(
            bool(indexes & set(tick["sample_indexes"])) for tick in ticks
        )
    return list(by_key.values())


def desktop_overall(host_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconcile the full desktop-policy host observation.

    This deliberately consumes every interval, including incomplete/churned
    ones.  The reducer uses it alongside per-cell alignment so the shared pool
    cannot be made quiet by retaining only favorable overlap windows.
    """
    if host_evidence.get("load_policy") != DESKTOP_POLICY:
        raise ValueError("desktop overall requires classified-desktop-v1")
    samples = _samples(host_evidence)
    _validate_desktop_guard(samples)
    applications = _application_coverage(samples, [])
    for application in applications:
        application.pop("affected_tick_count")
        application["observed_interval_count"] = len(application["sample_indexes"])
    return {
        "load_policy": DESKTOP_POLICY,
        "sample_count": len(samples),
        "samples": samples,
        "process_observations": _process_observations(samples),
        "classification_changes": _classification_changes(samples),
        "background_pool": _background_summary(samples),
        "applications": applications,
        "coverage_complete_count": sum(sample["coverage_complete"] for sample in samples),
    }


def _validate_desktop_guard(samples: Sequence[Mapping[str, Any]]) -> None:
    """Replay the interval guard; report generation cannot bless bad evidence."""
    for sample in samples:
        for process in sample["processes"]:
            classification = process.get("classification", {})
            nested = classification.get("classifier_decision", {})
            if (
                classification.get("reason") == "known_workload"
                or classification.get("known_workload")
                or (
                    isinstance(nested, Mapping)
                    and (nested.get("reason") == "known_workload" or nested.get("known_workload"))
                )
            ):
                raise ValueError("prohibited workload under desktop role in interval evidence")
            if classification.get("identity_binding_changed") or (
                isinstance(nested, Mapping) and nested.get("identity_binding_changed")
            ):
                raise ValueError("desktop identity binding changed in interval evidence")
            if process["kind"] not in BACKGROUND_KINDS and process["cpu_percent"] >= 20.0:
                raise ValueError("ordinary workload/unknown >=20% in desktop CPU evidence")
    shared = _background_summary(samples)["shared_background_pool"]
    if shared["exceeded"]:
        raise ValueError("desktop shared background >150% for >60s")


def validate_desktop_host(host_evidence: Mapping[str, Any]) -> None:
    """Fail closed before panel reduction if desktop interval evidence violates its guard."""
    if host_evidence.get("load_policy") != DESKTOP_POLICY:
        raise ValueError("desktop validation requires classified-desktop-v1")
    samples = _samples(host_evidence)
    _validate_desktop_guard(samples)
    registry_evidence = host_evidence["desktop_registry_evidence"]
    observations = host_evidence.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("desktop point observations missing")
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ValueError("malformed desktop point observation")
        screen = observation.get("process_screen", observation)
        if not isinstance(screen, Mapping):
            raise ValueError("malformed desktop point screen")
        processes = screen.get("classified_processes")
        exempt = screen.get("probe_exempt_pids")
        if (
            not isinstance(processes, list)
            or not isinstance(exempt, list)
            or not all(type(pid) is int for pid in exempt)
        ):
            raise ValueError("desktop point screen evidence missing")
        exempt_pids = set(exempt)
        for process in processes:
            if not isinstance(process, Mapping) or type(process.get("pid")) is not int:
                raise ValueError("malformed desktop point process")
            if process["pid"] in exempt_pids:
                continue
            classification = process.get("classification")
            replay = _replay_kind(process, registry_evidence)
            if (
                not isinstance(classification, Mapping)
                or classification.get("kind") != process.get("kind")
                or replay.get("kind") != process.get("kind")
            ):
                raise ValueError("desktop point classification replay differs")
            if (
                classification.get("identity_binding_changed")
                or classification.get("reason") == "known_workload"
            ):
                raise ValueError("desktop point prohibited workload evidence")
            cpu = _number(process.get("cpu_percent"), "desktop point CPU")
            if process["kind"] not in BACKGROUND_KINDS and cpu >= 20.0:
                raise ValueError("ordinary workload/unknown >=20% in desktop point evidence")


def _measured_rows(result: Mapping[str, Any], production: bool) -> list[Mapping[str, Any]]:
    rows = result.get("trace")
    if not isinstance(rows, list):
        raise ValueError("missing trace")
    _integer(result.get("warmup_ticks", 0), "warmup_ticks")
    # Each arm exports its trace after warmup.  Keep the asserted warmup field
    # as provenance, but never slice trace again here.
    if not rows:
        raise ValueError("no measured trace rows")
    return rows


def _production_delivery(
    result: Mapping[str, Any], count: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    warmup = _integer(result.get("warmup_ticks", 0), "warmup_ticks")
    receipts = result.get("client_receive")
    sends = result.get("frame_send")
    if not isinstance(receipts, list) or not isinstance(sends, list):
        raise ValueError("missing production delivery evidence")
    receipts = [
        row for row in receipts if isinstance(row, Mapping) and row.get("source") == "hub_broadcast"
    ]
    if len(receipts) != warmup + count or len(sends) != warmup + count:
        raise ValueError("incomplete production delivery evidence")
    measured_receipts, measured_sends = list(receipts[warmup:]), list(sends[warmup:])
    for receipt, send in zip(measured_receipts, measured_sends):
        if receipt.get("frame") != send.get("frame") or send.get("status") != "ok":
            raise ValueError("misaligned production delivery evidence")
        if _integer(receipt.get("receive_ns"), "client receive") < _integer(
            send.get("start_ns"), "send start"
        ):
            raise ValueError("client receipt precedes send")
    return measured_receipts, measured_sends


def _row_window(
    row: Mapping[str, Any], index: int, production: bool, receipt: Mapping[str, Any] | None
) -> tuple[int, int]:
    target = _integer(row.get("target_ns"), "trace target")
    wake = _integer(row.get("wake_ns"), "trace wake")
    start = _integer(row.get("start_ns"), "trace start")
    end = _integer(row.get("end_ns"), "trace end")
    if end < start:
        raise ValueError("trace end precedes start")
    window_start = min(target, wake, start)
    window_end = end
    if production:
        snapshot_end = _integer(row.get("snapshot_end_ns"), "snapshot end")
        if snapshot_end < start:
            raise ValueError("snapshot end precedes trace start")
        if receipt is None:
            raise ValueError("missing production receipt")
        receive = _integer(receipt.get("receive_ns"), "client receive")
        window_end = max(window_end, snapshot_end, receive)
    if window_end <= window_start:
        raise ValueError(f"empty trace window at tick {index}")
    return window_start, window_end


def _covariate(samples: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    overlaps = []
    covered = start
    complete = True
    observed_weight = 0
    reliable_weight = 0
    weighted = {
        "background": 0.0,
        "service": 0.0,
        "desktop_renderer": 0.0,
        "desktop_app": 0.0,
        "workload": 0.0,
        "unknown": 0.0,
    }
    maximum = {category: None for category in weighted}
    for sample in samples:
        if sample["end_ns"] <= start or sample["start_ns"] >= end:
            continue
        overlap_start, overlap_end = max(start, sample["start_ns"]), min(end, sample["end_ns"])
        if overlap_start > covered:
            complete = False
        covered = max(covered, overlap_end)
        duration = overlap_end - overlap_start
        usable = (
            sample["coverage_complete"] and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_NS
        )
        overlaps.append(sample["index"])
        observed_weight += duration
        for category in weighted:
            value = sample[f"{category}_cpu_percent"]
            weighted[category] += value * duration
            maximum[category] = max(maximum[category] or 0.0, value)
        if usable:
            reliable_weight += duration
        else:
            complete = False
    if covered < end:
        complete = False
    coverage_fraction = reliable_weight / (end - start)
    means = {
        category: weighted[category] / observed_weight if observed_weight else None
        for category in weighted
    }
    low = bool(
        complete
        and coverage_fraction == 1.0
        and means["background"] is not None
        and maximum["background"] is not None
        and maximum["background"] <= LOW_BACKGROUND_PERCENT
        and all(
            sample["background_cpu_percent"] <= LOW_BACKGROUND_PERCENT
            and sample["workload_cpu_percent"] < LOW_BACKGROUND_PERCENT
            and sample["unknown_cpu_percent"] < LOW_BACKGROUND_PERCENT
            for sample in samples
            if sample["index"] in overlaps
            and sample["coverage_complete"]
            and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_NS
        )
    )
    return {
        "sample_indexes": overlaps,
        "coverage_complete": complete and coverage_fraction == 1.0,
        "coverage_fraction": coverage_fraction,
        **{f"{category}_cpu_percent_time_weighted_mean": means[category] for category in means},
        **{f"{category}_cpu_percent_max": maximum[category] for category in maximum},
        "low_background": low,
    }


def _metric_rows(
    rows: list[Mapping[str, Any]],
    production: bool,
    receipts: list[Mapping[str, Any]],
    sends: list[Mapping[str, Any]],
    fps: int = 24,
) -> list[dict[str, int]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    values: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        item: dict[str, int] = {}
        for name, start_key, end_key in (
            ("tick_wall_ns", "start_ns", "end_ns"),
            ("tick_thread_cpu_ns", "start_cpu_ns", "end_cpu_ns"),
        ):
            start, end = _integer(row.get(start_key), start_key), _integer(
                row.get(end_key), end_key
            )
            if end < start:
                raise ValueError(f"negative {name}")
            item[name] = end - start
        target, wake = _integer(row.get("target_ns"), "target_ns"), _integer(
            row.get("wake_ns"), "wake_ns"
        )
        item["wake_lateness_ns"] = max(0, wake - target)
        update = (
            row.get("spans", {}).get("game_update")
            if isinstance(row.get("spans"), Mapping)
            else None
        )
        if isinstance(update, Mapping):
            item["update_ns"] = _integer(update.get("wall_ns"), "game update wall")
        if not production:
            epoch = _integer(rows[0].get("target_ns"), "epoch target")
            deadline = epoch + (_integer(row.get("tick"), "trace tick") + 1) * 1_000_000_000 // fps
            item["absolute_overrun_severity_ns"] = max(
                0, _integer(row["end_ns"], "end_ns") - deadline
            )
            item["absolute_overrun"] = int(_integer(row["end_ns"], "end_ns") >= deadline)
        else:
            snapshot_start = _integer(row.get("snapshot_start_cpu_ns"), "snapshot start CPU")
            snapshot_end = _integer(row.get("snapshot_end_cpu_ns"), "snapshot end CPU")
            if snapshot_end < snapshot_start:
                raise ValueError("negative snapshot CPU")
            item["snapshot_thread_cpu_ns"] = snapshot_end - snapshot_start
            receive = _integer(receipts[index].get("receive_ns"), "client receive")
            item["step_to_client_ns"] = receive - _integer(row.get("start_ns"), "start_ns")
            if item["step_to_client_ns"] < 0:
                raise ValueError("client receipt precedes step")
            item["send_to_client_ns"] = receive - _integer(
                sends[index].get("start_ns"), "send start"
            )
        values.append(item)
    return values


def _timing(
    metrics: list[dict[str, int]],
    indexes: list[int],
    metric_keys: list[str],
    receipts: list[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    require_low_gap: bool,
) -> dict[str, Any]:
    selected = [metrics[index] for index in indexes]
    distributions = {
        key: _distribution([item[key] for item in selected if key in item])
        for key in metric_keys
        if key != "absolute_overrun"
    }
    # Delivery gaps are only valid for consecutive retained trace rows.  Do
    # not bridge a high-background or coverage-excluded tick in the low subset.
    receipt_indexes = indexes if receipts else []
    gaps = [
        _integer(receipts[current].get("receive_ns"), "client receive")
        - _integer(receipts[previous].get("receive_ns"), "client receive")
        for previous, current in zip(receipt_indexes, receipt_indexes[1:])
        if current == previous + 1
        and (
            not require_low_gap
            or receipts[current]["receive_ns"] == receipts[previous]["receive_ns"]
            or (
                receipts[current]["receive_ns"] > receipts[previous]["receive_ns"]
                and _covariate(
                    samples,
                    _integer(receipts[previous].get("receive_ns"), "client receive"),
                    _integer(receipts[current].get("receive_ns"), "client receive"),
                )["low_background"]
            )
        )
    ]
    if any(gap < 0 for gap in gaps):
        raise ValueError("client receipt ordering")
    if receipts:
        distributions["client_frame_gap_ns"] = _distribution(gaps)
    return {
        "count": len(selected),
        "absolute_overrun_count": sum(item.get("absolute_overrun", 0) for item in selected),
        "metrics": distributions,
    }


def summarize_background(
    result: Mapping[str, Any],
    host_evidence: Mapping[str, Any],
    *,
    production: bool = False,
    fps: int = 24,
) -> dict[str, Any]:
    """Align host CPU samples and publish overall plus low-background timings.

    This is a covariate report only: it neither filters the diagnostic reducer nor
    changes its verdict. Missing coverage remains explicit and cannot become a
    low-background observation by interpolation.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not isinstance(result, Mapping) or not isinstance(host_evidence, Mapping):
        raise ValueError("result and host evidence must be mappings")
    samples = _samples(host_evidence)
    policy = host_evidence["load_policy"]
    rows = _measured_rows(result, production)
    receipts, sends = _production_delivery(result, len(rows)) if production else ([], [])
    ticks = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("malformed trace row")
        start, end = _row_window(row, index, production, receipts[index] if production else None)
        tick = {
            "index": index,
            "tick": _integer(row.get("tick"), "trace tick"),
            "window_start_ns": start,
            "window_end_ns": end,
        }
        tick.update(_covariate(samples, start, end))
        ticks.append(tick)
    metrics = _metric_rows(rows, production, receipts, sends, fps=fps)
    metric_keys = sorted({key for item in metrics for key in item})
    all_indexes = list(range(len(rows)))
    low_indexes = [tick["index"] for tick in ticks if tick["low_background"]]
    scoped_indexes = {sample_index for tick in ticks for sample_index in tick["sample_indexes"]}
    scoped_samples = [sample for sample in samples if sample["index"] in scoped_indexes]
    return {
        "schema": "served-tick-background-report/v1",
        "load_policy": policy,
        "production": production,
        "low_background_definition": {
            "background_cpu_percent_max": LOW_BACKGROUND_PERCENT,
            "workload_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "unknown_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "requires_complete_coverage": True,
            "max_sample_interval_ns": MAX_SAMPLE_NS,
        },
        "ticks": ticks,
        "process_observations": _process_observations(scoped_samples),
        "classification_changes": _classification_changes(scoped_samples),
        "background_pool": _background_summary(scoped_samples),
        "applications": _application_coverage(scoped_samples, ticks),
        "counts": {
            "ticks": len(ticks),
            "coverage_eligible": sum(tick["coverage_complete"] for tick in ticks),
            "low_background": len(low_indexes),
            "high_background": sum(
                (tick["background_cpu_percent_max"] or 0) > LOW_BACKGROUND_PERCENT for tick in ticks
            ),
            "high_workload": sum(
                (tick["workload_cpu_percent_max"] or 0) >= LOW_BACKGROUND_PERCENT for tick in ticks
            ),
            "high_unknown": sum(
                (tick["unknown_cpu_percent_max"] or 0) >= LOW_BACKGROUND_PERCENT for tick in ticks
            ),
        },
        "timing": {
            "overall": _timing(
                metrics, all_indexes, metric_keys, receipts, samples, require_low_gap=False
            ),
            "low_background": _timing(
                metrics, low_indexes, metric_keys, receipts, samples, require_low_gap=True
            ),
        },
        "interpretation": (
            "Background load is an observational covariate only; these summaries establish neither "
            "causation nor a timing gate."
        ),
    }
