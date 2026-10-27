"""Background-load covariates for the served-tick timing diagnostic.

This module deliberately does not decide the panel verdict.  It aligns sealed,
off-tick host samples to individual trace rows so a report can describe the
environment in which timing was observed without treating routine macOS work as
an automatic contamination reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from research.bl_served_search.served_tick_probe import _distribution

POLICY = "classified-v1"
LOW_BACKGROUND_PERCENT = 20.0
MAX_SAMPLE_NS = 5_000_000_000
PROCESS_KINDS = {"macos_service", "desktop_renderer", "agent_workload", "unknown"}


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"malformed {name}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"malformed {name}")
    return float(value)


def _samples(host_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if host_evidence.get("load_policy") != POLICY:
        raise ValueError("unsupported host load policy")
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
            if process["kind"] not in PROCESS_KINDS:
                raise ValueError("unknown sampled process kind")
            checked_processes.append(
                {
                    "pid": process["pid"],
                    "ppid": process["ppid"],
                    "command": process["command"],
                    "kind": process["kind"],
                    "cpu_percent": _number(process.get("cpu_percent"), "process CPU"),
                }
            )
        totals = {
            "background": sum(
                row["cpu_percent"]
                for row in checked_processes
                if row["kind"] in {"macos_service", "desktop_renderer"}
            ),
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
        result.append(
            {
                "index": index,
                "start_ns": start,
                "end_ns": end,
                "background_cpu_percent": reported["background"],
                "workload_cpu_percent": reported["workload"],
                "unknown_cpu_percent": reported["unknown"],
                "coverage_complete": sample["coverage_complete"],
                "processes": checked_processes,
            }
        )
    return result


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
    weighted = {"background": 0.0, "workload": 0.0, "unknown": 0.0}
    maximum = {"background": None, "workload": None, "unknown": None}
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
    return {
        "schema": "served-tick-background-report/v1",
        "load_policy": POLICY,
        "production": production,
        "low_background_definition": {
            "background_cpu_percent_max": LOW_BACKGROUND_PERCENT,
            "workload_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "unknown_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "requires_complete_coverage": True,
            "max_sample_interval_ns": MAX_SAMPLE_NS,
        },
        "ticks": ticks,
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
