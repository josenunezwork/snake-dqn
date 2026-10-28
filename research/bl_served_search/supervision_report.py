"""Prospective v2 covariates for panels with admitted remote supervision.

This module is deliberately separate from ``background_report``.  In particular,
it never reads or reinterprets a classified-v1 export.  It keeps every timing row
in ``overall`` and makes the operation-free, low-contention subset explicit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Callable

from research.bl_served_search.background_report import (
    LOW_BACKGROUND_PERCENT,
    MAX_SAMPLE_NS,
    _metric_rows,
    _production_delivery,
    _row_window,
    _samples,
)
from research.bl_served_search.served_tick_probe import _distribution

POLICY = "classified-supervised-v2"
MAX_SUPERVISION_PERCENT = 20.0
MIN_INTERVAL_NS = 500_000_000
MAX_INTERVAL_NS = 5_000_000_000
OPERATION_PADDING_NS = 1_000_000_000


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"malformed {name}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"malformed {name}")
    return float(value)


def _intervals(replay_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = replay_result.get("intervals")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing supervision intervals")
    result: list[dict[str, Any]] = []
    previous = -1
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError("malformed supervision interval")
        start, end = _integer(row.get("start_ns"), "supervision start"), _integer(
            row.get("end_ns"), "supervision end"
        )
        if end <= start or start < previous:
            raise ValueError("out-of-order or overlapping supervision intervals")
        previous = end
        duration = end - start
        complete = row.get("coverage_complete", True)
        if type(complete) is not bool:
            raise ValueError("malformed supervision coverage")
        per_root = row.get("roots")
        if not isinstance(per_root, Mapping) or any(
            not isinstance(root_id, str) or not root_id for root_id in per_root
        ):
            raise ValueError("malformed per-root supervision CPU")
        per_root_values = {
            root_id: _number(value.get("observed_cpu_s"), "per-root observed CPU")
            for root_id, value in per_root.items()
            if isinstance(value, Mapping)
        }
        per_root_conservative = {
            root_id: _number(value.get("conservative_cpu_s"), "per-root conservative CPU")
            for root_id, value in per_root.items()
            if isinstance(value, Mapping)
        }
        if len(per_root_values) != len(per_root) or len(per_root_conservative) != len(per_root):
            raise ValueError("malformed per-root supervision CPU")
        cpu_s = _number(row.get("observed_cpu_s"), "observed supervision CPU seconds")
        conservative_cpu_s = _number(
            row.get("conservative_cpu_s"), "conservative supervision CPU seconds"
        )
        if not math.isclose(sum(per_root_values.values()), cpu_s, abs_tol=1e-9) or not math.isclose(
            sum(per_root_conservative.values()), conservative_cpu_s, abs_tol=1e-9
        ):
            raise ValueError("per-root supervision CPU does not reconcile")
        result.append(
            {
                "index": index,
                "start_ns": start,
                "end_ns": end,
                "cpu_percent": _number(row.get("observed_cpu_pct"), "observed supervision CPU"),
                "conservative_cpu_percent": _number(
                    row.get("cpu_pct"), "conservative supervision CPU"
                ),
                "cpu_s": cpu_s,
                "conservative_cpu_s": conservative_cpu_s,
                "conservative_cpu_numerator": _integer(
                    row.get("conservative_cpu_numerator"), "conservative CPU numerator"
                ),
                "counter_denominator": _integer(
                    row.get("counter_denominator"), "counter denominator", positive=True
                ),
                "per_root_cpu_s": per_root_values,
                "per_root_conservative_cpu_s": per_root_conservative,
                "coverage_complete": complete,
                "interval_eligible": complete and MIN_INTERVAL_NS <= duration <= MAX_INTERVAL_NS,
            }
        )
    return result


def _operations(replay_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = replay_result.get("operations", {})
    if not isinstance(raw, Mapping):
        raise ValueError("malformed supervision operations")
    result = []
    for index, (operation_id, row) in enumerate(raw.items()):
        if not isinstance(row, Mapping):
            raise ValueError("malformed supervision operation")
        start, end = _integer(row.get("start_ns"), "operation start"), _integer(
            row.get("end_ns"), "operation end"
        )
        if end < start:
            raise ValueError("operation ends before start")
        root_id = row.get("root_id")
        if not isinstance(root_id, str) or not root_id:
            raise ValueError("operation missing root identity")
        result.append(
            {
                "index": index,
                "operation_id": str(operation_id),
                "root_id": root_id,
                "start_ns": start,
                "end_ns": end,
                "cpu_s": _number(row.get("observed_cpu_s"), "operation CPU"),
                "conservative_cpu_s": _number(
                    row.get("conservative_cpu_s"), "operation conservative CPU"
                ),
                "accounting_complete": row.get("complete") is True,
            }
        )
    return result


def _overlap(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start))


def _coverage(intervals: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    covered, reliable, weighted, conservative_weighted, observed = start, 0, 0.0, 0.0, 0
    maximum: float | None = None
    conservative_maximum: float | None = None
    indexes = []
    for interval in intervals:
        if interval["end_ns"] <= start or interval["start_ns"] >= end:
            continue
        overlap_start, overlap_end = max(start, interval["start_ns"]), min(end, interval["end_ns"])
        duration = overlap_end - overlap_start
        indexes.append(interval["index"])
        observed += duration
        weighted += interval["cpu_percent"] * duration
        conservative_weighted += interval["conservative_cpu_percent"] * duration
        maximum = max(maximum or 0.0, interval["cpu_percent"])
        conservative_maximum = max(
            conservative_maximum or 0.0, interval["conservative_cpu_percent"]
        )
        if overlap_start > covered:
            covered = overlap_start
        covered = max(covered, overlap_end)
        if interval["interval_eligible"]:
            reliable += duration
    complete = covered >= end and reliable == end - start
    return {
        "supervision_interval_indexes": indexes,
        "supervision_coverage_complete": complete,
        "supervision_coverage_fraction": reliable / (end - start),
        "supervision_cpu_percent_time_weighted_mean": weighted / observed if observed else None,
        "supervision_cpu_percent_max": maximum,
        "supervision_conservative_cpu_percent_time_weighted_mean": (
            conservative_weighted / observed if observed else None
        ),
        "supervision_conservative_cpu_percent_max": conservative_maximum,
    }


def _host_covariate(samples: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    covered, reliable, observed = start, 0, 0
    weighted = {kind: 0.0 for kind in ("background", "workload", "unknown")}
    maximum: dict[str, float | None] = {kind: None for kind in weighted}
    indexes = []
    for sample in samples:
        if sample["end_ns"] <= start or sample["start_ns"] >= end:
            continue
        left, right = max(start, sample["start_ns"]), min(end, sample["end_ns"])
        duration = right - left
        indexes.append(sample["index"])
        observed += duration
        if left > covered:
            covered = left
        covered = max(covered, right)
        usable = (
            sample["coverage_complete"] and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_NS
        )
        if usable:
            reliable += duration
        for kind in weighted:
            value = sample[f"{kind}_cpu_percent"]
            weighted[kind] += value * duration
            maximum[kind] = max(maximum[kind] or 0.0, value)
    complete = covered >= end and reliable == end - start
    return {
        "sample_indexes": indexes,
        "coverage_complete": complete,
        "coverage_fraction": reliable / (end - start),
        **{
            f"{kind}_cpu_percent_time_weighted_mean": (
                weighted[kind] / observed if observed else None
            )
            for kind in weighted
        },
        **{f"{kind}_cpu_percent_max": maximum[kind] for kind in weighted},
    }


def _operation_overlap(operations: Sequence[Mapping[str, Any]], start: int, end: int) -> list[str]:
    return [
        str(operation["operation_id"])
        for operation in operations
        if _overlap(
            start,
            end,
            operation["start_ns"] - OPERATION_PADDING_NS,
            operation["end_ns"] + OPERATION_PADDING_NS,
        )
    ]


def _raw_operation_overlap(
    operations: Sequence[Mapping[str, Any]], start: int, end: int
) -> list[str]:
    return [
        str(operation["operation_id"])
        for operation in operations
        if _overlap(start, end, operation["start_ns"], operation["end_ns"])
    ]


def _combined_low(
    samples: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> bool:
    """Check every atomic overlap; never pair a host sample with an arbitrary poll interval."""
    for sample in samples:
        if _overlap(start, end, sample["start_ns"], sample["end_ns"]) == 0:
            continue
        if (
            sample["workload_cpu_percent"] >= LOW_BACKGROUND_PERCENT
            or sample["unknown_cpu_percent"] >= LOW_BACKGROUND_PERCENT
        ):
            return False
        for interval in intervals:
            if (
                _overlap(
                    start,
                    end,
                    max(sample["start_ns"], interval["start_ns"]),
                    min(sample["end_ns"], interval["end_ns"]),
                )
                == 0
            ):
                continue
            background = Fraction(str(sample["background_cpu_percent"]))
            supervision = Fraction(
                interval["conservative_cpu_numerator"] * 100,
                interval["counter_denominator"] * (interval["end_ns"] - interval["start_ns"]),
            )
            if background + supervision > MAX_SUPERVISION_PERCENT:
                return False
    return True


def _charges(replay_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read the unique CPU placement brackets used for conservative rolling limits."""
    raw = replay_result.get("charges")
    if not isinstance(raw, list):
        raise ValueError("malformed supervision charges")
    result, identifiers = [], set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError("malformed supervision charge")
        explicit_id = row.get("id")
        # Ledger charges deliberately share operation_id (and root bookkeeping
        # has no operation).  Only a ledger-issued explicit id is globally
        # unique; otherwise the sealed list position is the stable identity.
        identifier = str(explicit_id) if explicit_id is not None else f"charge-{index}"
        start, end = _integer(
            row.get("start_ns", row.get("start_monotonic_ns")), "charge start"
        ), _integer(row.get("end_ns", row.get("end_monotonic_ns")), "charge end")
        if not identifier or identifier in identifiers or end < start:
            raise ValueError("malformed supervision charge identity")
        identifiers.add(identifier)
        result.append(
            {
                "id": identifier,
                "root_id": row.get("root_id"),
                "start_ns": start,
                "end_ns": end,
                "cpu_s": _number(row.get("cpu_s", row.get("measured_cpu_seconds")), "charge CPU"),
                "conservative_cpu_s": _number(
                    row.get("conservative_cpu_s", row.get("conservative_cpu_seconds")),
                    "conservative charge CPU",
                ),
            }
        )
    return result


def _rolling_cpu_max(
    charges: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    window_ns: int,
    field: str,
    *,
    conservative: bool,
) -> float | None:
    """Recompute a cell maximum, carrying pre-cell debt without prorating bounds."""
    if end < start:
        raise ValueError("invalid rolling scope")
    # Sweep open active intervals on the real time axis. Integer midpoints
    # cannot represent a peak inside an open interval only one nanosecond wide.
    edges = {}
    total = 0.0
    for charge in charges:
        left = charge["start_ns"] if conservative else charge["end_ns"]
        right = charge["end_ns"] + window_ns
        amount = charge[field]
        if left < start < right:
            total += amount
        if start <= left <= end:
            edges.setdefault(left, [0.0, 0.0])[0] += amount
        if start < right <= end:
            edges.setdefault(right, [0.0, 0.0])[1] += amount
    maximum = total
    for point in sorted(edges):
        added, removed = edges[point]
        total -= removed
        maximum = max(maximum, total)
        # Conservative starts are open; there must be scope to their right.
        if point < end or not conservative:
            total += added
            maximum = max(maximum, total)
    return maximum


def summarize_scope(start_ns: int, end_ns: int, replay_result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe supervision CPU and operation accounting for one complete scope."""
    start, end = _integer(start_ns, "scope start"), _integer(end_ns, "scope end")
    if end <= start or not isinstance(replay_result, Mapping):
        raise ValueError("invalid supervision scope")
    intervals, operations = _intervals(replay_result), _operations(replay_result)
    charges = _charges(replay_result)
    covariate = _coverage(intervals, start, end)
    overlapping = [op for op in operations if _overlap(start, end, op["start_ns"], op["end_ns"])]
    declared_roots = replay_result.get("roots", {})
    if not isinstance(declared_roots, Mapping):
        raise ValueError("malformed supervision roots")
    roots: dict[str, dict[str, Any]] = {}
    for root_id, root in declared_roots.items():
        if not isinstance(root_id, str) or not root_id or not isinstance(root, Mapping):
            raise ValueError("malformed supervision root")
        roots[root_id] = {
            "root_observed_cpu_s": 0.0,
            "root_conservative_cpu_s": 0.0,
            "operation_count": 0,
            "operation_cpu_s": 0.0,
            "operation_duration_ns": 0,
            "accounting_complete": True,
        }
    for operation in overlapping:
        root = roots.setdefault(
            operation["root_id"],
            {
                "root_observed_cpu_s": 0.0,
                "root_conservative_cpu_s": 0.0,
                "operation_count": 0,
                "operation_cpu_s": 0.0,
                "operation_duration_ns": 0,
                "accounting_complete": True,
            },
        )
        root["operation_count"] += 1
        root["operation_cpu_s"] += operation["cpu_s"]
        root["operation_duration_ns"] += _overlap(
            start, end, operation["start_ns"], operation["end_ns"]
        )
        root["accounting_complete"] = (
            root["accounting_complete"] and operation["accounting_complete"]
        )
    for charge in charges:
        root_id = charge["root_id"]
        if not isinstance(root_id, str) or not root_id:
            raise ValueError("charge missing root identity")
        root = roots.setdefault(
            root_id,
            {
                "root_observed_cpu_s": 0.0,
                "root_conservative_cpu_s": 0.0,
                "operation_count": 0,
                "operation_cpu_s": 0.0,
                "operation_duration_ns": 0,
                "accounting_complete": True,
            },
        )
        if start < charge["end_ns"] <= end:
            root["root_observed_cpu_s"] += charge["cpu_s"]
        if _overlap(start, end, charge["start_ns"], charge["end_ns"]):
            root["root_conservative_cpu_s"] += charge["conservative_cpu_s"]
    return {
        "scope_start_ns": start,
        "scope_end_ns": end,
        **covariate,
        "operation_count": len(overlapping),
        "operation_duration_ns": sum(
            _overlap(start, end, op["start_ns"], op["end_ns"]) for op in overlapping
        ),
        "operation_cpu_s": sum(op["cpu_s"] for op in overlapping),
        "accounting_complete": replay_result.get("valid") is True
        and replay_result.get("complete") is True
        and covariate["supervision_coverage_complete"]
        and all(op["accounting_complete"] for op in overlapping),
        "roots": roots,
        "observed_max_burst_cpu_percent": max(
            (
                entry["cpu_percent"]
                for entry in intervals
                if _overlap(start, end, entry["start_ns"], entry["end_ns"])
            ),
            default=None,
        ),
        "observed_max_10s_cpu_s": _rolling_cpu_max(
            charges, start, end, 10_000_000_000, "cpu_s", conservative=False
        ),
        "observed_max_60s_cpu_s": _rolling_cpu_max(
            charges, start, end, 60_000_000_000, "cpu_s", conservative=False
        ),
        "conservative_max_10s_cpu_s": _rolling_cpu_max(
            charges, start, end, 10_000_000_000, "conservative_cpu_s", conservative=True
        ),
        "conservative_max_60s_cpu_s": _rolling_cpu_max(
            charges, start, end, 60_000_000_000, "conservative_cpu_s", conservative=True
        ),
        "max_operation_cpu_s": max((op["cpu_s"] for op in overlapping), default=None),
    }


def _timing(
    metrics: list[dict[str, int]],
    indexes: list[int],
    keys: list[str],
    receipts: Sequence[Mapping[str, Any]],
    *,
    allow_gap: Callable[[int, int], bool] | None = None,
) -> dict[str, Any]:
    selected = [metrics[index] for index in indexes]
    report = {
        "count": len(selected),
        "absolute_overrun_count": sum(item.get("absolute_overrun", 0) for item in selected),
        "metrics": {
            key: _distribution([row[key] for row in selected if key in row])
            for key in keys
            if key != "absolute_overrun"
        },
    }
    if receipts:
        gaps = []
        for previous, current in zip(indexes, indexes[1:]):
            if current != previous + 1:
                continue
            start = _integer(receipts[previous].get("receive_ns"), "client receive")
            end = _integer(receipts[current].get("receive_ns"), "client receive")
            if end < start:
                raise ValueError("client receipt ordering")
            if allow_gap is None or allow_gap(start, end):
                gaps.append(end - start)
        report["metrics"]["client_frame_gap_ns"] = _distribution(gaps)
    return report


def summarize_supervision(
    result: Mapping[str, Any],
    host_evidence: Mapping[str, Any],
    replay_result: Mapping[str, Any],
    *,
    production: bool = False,
    fps: int = 24,
    desktop_host: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report all timing rows plus a prospective low-contention sensitivity subset."""
    if fps <= 0 or not all(
        isinstance(value, Mapping) for value in (result, host_evidence, replay_result)
    ):
        raise ValueError("malformed supervision report input")
    if replay_result.get("policy") != POLICY:
        raise ValueError("unsupported supervision policy")
    if host_evidence.get("load_policy") != POLICY:
        raise ValueError("unsupported supervised host load policy")
    # Standalone supervision validates the frozen CPU wire shape. Combined
    # replay supplies its verified desktop view so original app kinds and the
    # registry/history validation survive into the reporting projection.
    core_host = dict(desktop_host or host_evidence)
    if desktop_host is None:
        core_host["load_policy"] = "classified-v1"
    if "ordinary_cpu_samples" in replay_result:
        core_host["cpu_samples"] = replay_result["ordinary_cpu_samples"]
    samples, intervals, operations = (
        _samples(core_host),
        _intervals(replay_result),
        _operations(replay_result),
    )
    rows = result.get("trace")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing trace")
    receipts, sends = _production_delivery(result, len(rows)) if production else ([], [])
    ticks = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("malformed trace row")
        start, end = _row_window(row, index, production, receipts[index] if production else None)
        host, supervision = _host_covariate(samples, start, end), _coverage(intervals, start, end)
        overlap = _operation_overlap(operations, start, end)
        raw_overlap = _raw_operation_overlap(operations, start, end)
        low = bool(
            replay_result.get("valid") is True
            and replay_result.get("complete") is True
            and host["coverage_complete"]
            and supervision["supervision_coverage_complete"]
            and not overlap
        )
        if low:
            low = _combined_low(samples, intervals, start, end)
        ticks.append(
            {
                "index": index,
                "tick": _integer(row.get("tick"), "trace tick"),
                "window_start_ns": start,
                "window_end_ns": end,
                **host,
                **supervision,
                "operation_overlap_ids": overlap,
                "poll_overlap_ids": raw_overlap,
                "low_background": low,
            }
        )
    metrics = _metric_rows(rows, production, receipts, sends, fps=fps)
    keys = sorted({key for metric in metrics for key in metric})
    low_indexes = [tick["index"] for tick in ticks if tick["low_background"]]
    panel_start, panel_end = min(tick["window_start_ns"] for tick in ticks), max(
        tick["window_end_ns"] for tick in ticks
    )

    def _low_gap(start: int, end: int) -> bool:
        if end == start:
            if _operation_overlap(operations, start, start + 1):
                return False
            for sample in samples:
                if sample["start_ns"] <= start < sample["end_ns"]:
                    for interval in intervals:
                        if interval["start_ns"] <= start < interval["end_ns"]:
                            return bool(
                                sample["coverage_complete"]
                                and interval["interval_eligible"]
                                and _combined_low(samples, intervals, start, start + 1)
                            )
            return False
        return bool(
            _host_covariate(samples, start, end)["coverage_complete"]
            and _coverage(intervals, start, end)["supervision_coverage_complete"]
            and not _operation_overlap(operations, start, end)
            and _combined_low(samples, intervals, start, end)
        )

    return {
        "schema": "served-tick-supervision-report/v2",
        "load_policy": POLICY,
        "production": production,
        "ticks": ticks,
        "counts": {
            "ticks": len(ticks),
            "low_background": len(low_indexes),
            "operation_overlap": sum(bool(tick["operation_overlap_ids"]) for tick in ticks),
            "poll_overlap": sum(bool(tick["poll_overlap_ids"]) for tick in ticks),
        },
        "scope": summarize_scope(panel_start, panel_end, replay_result),
        "timing": {
            "overall": _timing(metrics, list(range(len(ticks))), keys, receipts),
            "low_background": _timing(metrics, low_indexes, keys, receipts, allow_gap=_low_gap),
        },
        "interpretation": (
            "Prospective supervision covariates retain all measured ticks; "
            "the low-background subset is descriptive only."
        ),
    }
