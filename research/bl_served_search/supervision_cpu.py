"""Exact, fail-closed CPU ledger for prospective supervised engineering panels.

This is an offline arithmetic boundary, not a process observer or identity proof.
The caller must independently authenticate every identity and lifecycle. The input
schema is ``served-supervision-cpu/v2`` with the v2 policy, exact integer CPU
nanoseconds (exclusive sampled counters), and a complete lifecycle provenance.
An exact rational clock uses ``counter_unit='rational_nanoseconds'`` and positive
``counter_denominator``: every ``cpu_ns`` input is then an integer numerator in
that shared denominator. No conversion rounds Mach absolute ticks to ns. Charge
``cpu_ns`` is retained as a compatibility numerator field; ``cpu_numerator`` and
``counter_denominator`` explicitly express its units.
``launch_ns`` is the verified pre-spawn monotonic lower bound. OS process
creation identity belongs in ``id`` and must never be used as monotonic time.

Samples cover every root and every then-live helper. Roots use their first sample
as baseline; helpers start at zero at verified launch. A helper's final counter
may be exclusive (``self``) or descendant inclusive (``subtree``). The latter is
reconciled recursively using exact observed reaping inclusions, subtracting all
descendant usage exactly once. A declaration of complete ancestry alone cannot
prove what a descendant-inclusive resource API included. No
estimated counter, floating point counter, rounded counter, or inferred exit is
accepted. Every positive increment has an uncertainty bracket: it is charged in
full to every intersecting monitored interval/window for ceiling decisions.
Measured totals and conservative bounds are deliberately separate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

POLICY = "classified-supervised-v2"
SCHEMA = "served-supervision-cpu/v2"
SECOND = 1_000_000_000


class _Invalid(ValueError):
    """An evidence defect prevents CPU certification."""


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise _Invalid(f"{name}: exact nonnegative integer required")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _Invalid(f"{name}: nonempty identity required")
    return value


def _overlaps(a: int, b: int, c: int, d: int) -> bool:
    return a < d and b > c


def _rolling(charges: list[dict[str, Any]], width: int, *, conservative: bool) -> int:
    """Find the exact all-placement maximum, including off-grid/trailing windows.

    A bracket (a,b) intersects (x,x+width) iff x lies in (a-width,b).
    Sweep these open intervals, removing ends before adding starts. With
    nonnegative charges, the intervals between event boundaries contain a
    maximum; boundary ties never require arbitrary floating point epsilon.
    """
    if not charges:
        return 0
    edges: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for charge in charges:
        left = charge["start_ns"] if conservative else charge["end_ns"]
        right = charge["end_ns"]
        edges[left - width][0] += charge["cpu_ns"]
        edges[right][1] += charge["cpu_ns"]
    current = maximum = 0
    for point in sorted(edges):
        added, removed = edges[point]
        current -= removed
        maximum = max(maximum, current)
        current += added
        maximum = max(maximum, current)
    return maximum


def replay_cpu(evidence: dict[str, Any], *, final: bool = True) -> dict[str, Any]:
    """Replay complete exact CPU evidence; return invalid rather than repair it.

    Required top-level fields: schema/policy, counter_unit='nanoseconds',
    counter_scope='exclusive', counter_resolution_ns=1, provenance_complete=True,
    roots=[process IDs], processes=[lifecycle rows], operations=[operation rows],
    samples=[{time_ns,capture_start_ns,capture_end_ns,cpu_ns:{process ID: CPU}}], and
    exposure_start_ns. Root lifecycle rows have null parent_id/operation_id and
    exit_ns; helper rows have root_id,parent_id,operation_id,launch_ns,exit_ns,
    final_cpu_ns,final_scope ('self'|'subtree'), descendants_complete=True.

    Every helper/operation must begin within this continuous sampled stream;
    missing earlier lifecycle/counters cannot be replaced by an admission tag.
    final=False checks an online prefix, permitting outstanding helper finals.
    final=True requires all helpers and operations to have complete exits.
    """
    result: dict[str, Any] = {
        "policy": POLICY,
        "schema": SCHEMA,
        "valid": False,
        "errors": [],
        "complete": False,
        "admission_ready": False,
        "charges": [],
        "intervals": [],
        "roots": {},
        "operations": {},
        "maxima": {},
    }
    try:
        _replay(evidence, result, final=final)
    except (KeyError, TypeError, _Invalid) as exc:
        result["errors"].append(str(exc))
    result["valid"] = not result["errors"]
    result["admission_ready"] = bool(
        result["valid"]
        and result.get("end_ns", 0) >= result.get("exposure_start_ns", 0)
        and result.get("exposure_start_ns", 0) - result.get("start_ns", 0) >= 60 * SECOND
    )
    return result


def _replay(e: dict[str, Any], out: dict[str, Any], *, final: bool) -> None:
    if not isinstance(e, dict) or e.get("schema") != SCHEMA or e.get("policy") != POLICY:
        raise _Invalid("CPU schema/policy mismatch; historical evidence cannot be reinterpreted")
    unit = e.get("counter_unit")
    denominator = _integer(e.get("counter_denominator", 1), "counter denominator", positive=True)
    if unit not in {"nanoseconds", "rational_nanoseconds"} or (
        unit == "nanoseconds" and denominator != 1
    ):
        raise _Invalid("CPU unit/common denominator mismatch")
    cpu_scale = SECOND * denominator
    out.update(counter_unit=unit, counter_denominator=denominator)
    if (
        e.get("counter_scope") != "exclusive"
        or type(e.get("counter_resolution_ns")) is not int
        or e["counter_resolution_ns"] != 1
        or e.get("provenance_complete") is not True
    ):
        raise _Invalid("inexact counter scope/resolution or incomplete provenance")
    samples = e["samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise _Invalid("at least two complete CPU samples required")
    times = [_integer(s["time_ns"], "sample time") for s in samples]
    capture_starts = []
    for sample, timestamp in zip(samples, times):
        lower = _integer(sample["capture_start_ns"], "sample capture start")
        upper = _integer(sample["capture_end_ns"], "sample capture end")
        if upper != timestamp or lower > upper:
            raise _Invalid("sample timestamp must equal capture completion; invalid read bracket")
        capture_starts.append(lower)
    if any(b < a for a, b in zip(capture_starts, capture_starts[1:])):
        raise _Invalid("sample capture starts regress")
    if any(upper - lower > 5 * SECOND for lower, upper in zip(capture_starts, times)):
        raise _Invalid("CPU capture bracket exceeds 5 seconds")
    if any(upper - lower > 5 * SECOND for lower, upper in zip(capture_starts, times[1:])):
        raise _Invalid("uncertain inter-sample CPU coverage exceeds 5 seconds")
    for first, second in zip(times, times[1:]):
        if not SECOND // 2 <= second - first <= 5 * SECOND:
            raise _Invalid("sample interval outside 0.5-5 seconds; no tail/gap repair")
    start, end = times[0], times[-1]
    if e.get("start_ns", start) != start or e.get("end_ns", end) != end:
        raise _Invalid("stream endpoints do not match complete sample coverage")
    exposure = _integer(e["exposure_start_ns"], "exposure start")
    if exposure < start + 60 * SECOND:
        raise _Invalid("60 continuous seconds required before measured exposure")
    out.update(start_ns=start, end_ns=end, exposure_start_ns=exposure)
    roots = e["roots"]
    if not isinstance(roots, list) or not roots or len(set(roots)) != len(roots):
        raise _Invalid("missing/duplicate registered roots")
    for rid in roots:
        _name(rid, "root")
    processes: dict[str, dict[str, Any]] = {}
    for row in e["processes"]:
        pid = _name(row["id"], "process")
        if pid in processes:
            raise _Invalid("duplicate process lifecycle")
        processes[pid] = row
    operations: dict[str, dict[str, Any]] = {}
    for row in e["operations"]:
        oid = _name(row["id"], "operation")
        if oid in operations or row["root_id"] not in roots:
            raise _Invalid("duplicate operation or foreign root")
        a = _integer(row["start_ns"], "operation start")
        b = row.get("end_ns")
        if a < start or a > end:
            raise _Invalid("operation has unavailable earlier lifecycle or future start")
        if b is not None:
            b = _integer(b, "operation end")
            if b < a or b > end:
                raise _Invalid("operation end outside stream")
        elif final:
            raise _Invalid("missing operation end")
        operations[oid] = row
    children: dict[str, list[str]] = defaultdict(list)
    for pid, row in processes.items():
        if row["root_id"] not in roots:
            raise _Invalid("foreign process root")
        if pid in roots:
            if (
                row["root_id"] != pid
                or row.get("parent_id") is not None
                or row.get("operation_id") is not None
                or row.get("exit_ns") is not None
            ):
                raise _Invalid("root must remain continuously alive and separate from operations")
            if "launch_ns" in row and _integer(row["launch_ns"], "root launch") > start:
                raise _Invalid("root launched after stream began")
            continue
        parent = row["parent_id"]
        oid = row["operation_id"]
        if parent not in processes or oid not in operations:
            raise _Invalid("unobserved helper parent/operation")
        if (
            processes[parent]["root_id"] != row["root_id"]
            or operations[oid]["root_id"] != row["root_id"]
        ):
            raise _Invalid("helper crosses registered roots")
        if parent not in roots and processes[parent]["operation_id"] != oid:
            raise _Invalid("helper crosses operation ownership")
        a = _integer(row["launch_ns"], "helper launch")
        b = row.get("exit_ns")
        if a < start or a > end or a < operations[oid]["start_ns"]:
            raise _Invalid("helper missing earlier lifecycle or invalid launch")
        if (
            parent not in roots
            and processes[parent].get("exit_ns") is not None
            and a > processes[parent]["exit_ns"]
        ):
            raise _Invalid("helper launched after parent exit; observed chain unavailable")
        if parent not in roots and a < processes[parent]["launch_ns"]:
            raise _Invalid("helper predates parent launch")
        if b is None:
            if final:
                raise _Invalid("missing helper exit/final usage")
            if operations[oid].get("end_ns") is not None:
                raise _Invalid("ended operation still has a live helper")
            if row.get("final_cpu_ns") is not None:
                raise _Invalid("final CPU without verified exit")
        else:
            b = _integer(b, "helper exit")
            if b < a or b > end:
                raise _Invalid("helper exit outside complete stream")
            op_end = operations[oid].get("end_ns")
            if op_end is not None and b > op_end:
                raise _Invalid("helper outlives recorded operation")
            _integer(row["final_cpu_ns"], "helper final CPU")
            if row.get("final_scope") not in {"self", "subtree"}:
                raise _Invalid("ambiguous final counter scope")
            if row.get("descendants_complete") is not True:
                raise _Invalid("incomplete descendant final provenance")
        children[parent].append(pid)
    if set(roots) - processes.keys():
        raise _Invalid("registered root absent from CPU stream")
    for pid in processes:
        seen: set[str] = set()
        cursor = pid
        while cursor not in roots:
            if cursor in seen:
                raise _Invalid("cyclic helper ancestry")
            seen.add(cursor)
            cursor = processes[cursor]["parent_id"]
        if cursor != processes[pid]["root_id"]:
            raise _Invalid("helper ancestry differs from registered root")
    exclusive: dict[str, int] = {}
    subtree: dict[str, int] = {}

    def own_usage(pid: str) -> int:
        if pid in exclusive:
            return exclusive[pid]
        row = processes[pid]
        if row.get("exit_ns") is None:
            raise _Invalid("inclusive final has outstanding descendant")
        raw = row["final_cpu_ns"]
        if row["final_scope"] == "subtree":
            pending = list(children[pid])
            descendant_ids = set()
            while pending:
                child = pending.pop()
                descendant_ids.add(child)
                pending.extend(children[child])
            inclusions = row.get("final_inclusions")
            if not isinstance(inclusions, list):
                raise _Invalid("inclusive final missing exact reaping inclusion records")
            witnessed = set()
            for inclusion in inclusions:
                child = inclusion["process_id"]
                if child not in descendant_ids or child in witnessed:
                    raise _Invalid("inclusive final duplicate/foreign inclusion")
                witnessed.add(child)
                parent = processes[child]["parent_id"]
                reap = _integer(inclusion["reaped_ns"], "reaping timestamp")
                child_exit = processes[child].get("exit_ns")
                parent_exit = processes[parent].get("exit_ns")
                if child_exit is None or parent_exit is None:
                    raise _Invalid("inclusive final has outstanding descendant")
                if inclusion["parent_id"] != parent or not child_exit <= reap <= parent_exit:
                    raise _Invalid("inclusive final invalid reaping propagation")
            if witnessed != descendant_ids:
                raise _Invalid("inclusive final unresolved descendant inclusion")
            descendants = sum(tree_usage(child) for child in children[pid])
            if any(processes[c]["exit_ns"] > row["exit_ns"] for c in children[pid]):
                raise _Invalid("inclusive final does not cover descendant exits")
            raw -= descendants
        if raw < 0:
            raise _Invalid("inclusive final smaller than descendant usage")
        exclusive[pid] = raw
        return raw

    def tree_usage(pid: str) -> int:
        if pid not in subtree:
            subtree[pid] = own_usage(pid) + sum(tree_usage(child) for child in children[pid])
        return subtree[pid]

    for pid, row in processes.items():
        if pid not in roots and row.get("exit_ns") is not None:
            own_usage(pid)
    last: dict[str, tuple[int, int]] = {}
    for pid, row in processes.items():
        if pid not in roots:
            last[pid] = (row["launch_ns"], 0)
    charges = out["charges"]

    def charge(pid: str, a: int, b: int, amount: int, kind: str) -> None:
        if amount < 0:
            raise _Invalid("CPU counter regression or unresolved final discrepancy")
        if amount and b <= a:
            raise _Invalid("positive CPU in zero/negative timestamp bracket")
        if not amount:
            return
        row = processes[pid]
        charges.append(
            {
                "process_id": pid,
                "root_id": row["root_id"],
                "operation_id": row.get("operation_id"),
                "start_ns": a,
                "end_ns": b,
                "cpu_ns": amount,
                "cpu_numerator": amount,
                "counter_denominator": denominator,
                "cpu_s": amount / cpu_scale,
                "conservative_cpu_s": amount / cpu_scale,
                "kind": kind,
            }
        )

    for sample, time, capture_start in zip(samples, times, capture_starts):
        counters = sample["cpu_ns"]
        if not isinstance(counters, dict) or counters.keys() - processes.keys():
            raise _Invalid("unexpected process in independent sampler")
        for pid, row in processes.items():
            alive = pid in roots or (
                row["launch_ns"] <= time and (row.get("exit_ns") is None or time < row["exit_ns"])
            )
            if alive and pid not in counters:
                raise _Invalid("missing live root/helper sample")
            if pid not in counters:
                continue
            value = _integer(counters[pid], "sample CPU")
            if not alive and row.get("exit_ns") != time:
                raise _Invalid("sample outside process lifecycle")
            if pid in last:
                previous_time, previous_value = last[pid]
                charge(pid, previous_time, time, value - previous_value, "sample")
            last[pid] = (max(capture_start, row.get("launch_ns", capture_start)), value)
    for pid, own in exclusive.items():
        previous_time, previous_value = last[pid]
        charge(pid, previous_time, processes[pid]["exit_ns"], own - previous_value, "final")
    for rid in roots:
        selected = [c for c in charges if c["root_id"] == rid]
        total = sum(c["cpu_ns"] for c in selected) / cpu_scale
        out["roots"][rid] = {
            "observed_cpu_s": total,
            "conservative_cpu_s": total,
            "process_count": sum(p["root_id"] == rid for p in processes.values()),
            "self_cpu_s": sum(c["cpu_ns"] for c in selected if c["process_id"] == rid) / cpu_scale,
            "helper_cpu_s": sum(c["cpu_ns"] for c in selected if c["process_id"] != rid)
            / cpu_scale,
        }
    for a, b in zip(times, times[1:]):
        observed = [c for c in charges if a < c["end_ns"] <= b]
        conservative = [c for c in charges if _overlaps(a, b, c["start_ns"], c["end_ns"])]
        observed_ns = sum(c["cpu_ns"] for c in observed)
        conservative_ns = sum(c["cpu_ns"] for c in conservative)
        interval = {
            "start_ns": a,
            "end_ns": b,
            "observed_cpu_numerator": observed_ns,
            "conservative_cpu_numerator": conservative_ns,
            "counter_denominator": denominator,
            "observed_cpu_s": observed_ns / cpu_scale,
            "conservative_cpu_s": conservative_ns / cpu_scale,
            "observed_cpu_pct": observed_ns * 100 / ((b - a) * denominator),
            "cpu_pct": conservative_ns * 100 / ((b - a) * denominator),
            "roots": {
                rid: {
                    "observed_cpu_s": sum(c["cpu_ns"] for c in observed if c["root_id"] == rid)
                    / cpu_scale,
                    "conservative_cpu_s": sum(
                        c["cpu_ns"] for c in conservative if c["root_id"] == rid
                    )
                    / cpu_scale,
                }
                for rid in roots
            },
        }
        out["intervals"].append(interval)
        if conservative_ns * 2 > (b - a) * denominator:
            out["errors"].append(f"aggregate supervision burst >50% at {a}:{b}")
    for rid in roots:
        selected = [c for c in charges if c["root_id"] == rid]
        root_maxima = {}
        for conservative, prefix in ((False, "observed"), (True, "conservative")):
            field = "conservative_cpu_s" if conservative else "observed_cpu_s"
            root_maxima[f"{prefix}_burst_pct"] = max(
                i["roots"][rid][field] * SECOND * 100 / (i["end_ns"] - i["start_ns"])
                for i in out["intervals"]
            )
            for seconds in (10, 60):
                root_maxima[f"{prefix}_{seconds}s_cpu_s"] = (
                    _rolling(selected, seconds * SECOND, conservative=conservative) / cpu_scale
                )
        out["roots"][rid]["maxima"] = root_maxima
    for oid, row in operations.items():
        stop = row.get("end_ns") if row.get("end_ns") is not None else end
        own = [c for c in charges if c["operation_id"] == oid]
        # Root bookkeeping is in the aggregate once. Its uncertain overlap is
        # conservatively also checked against each operation's individual cap.
        root_cost = [
            c
            for c in charges
            if c["process_id"] == row["root_id"]
            and _overlaps(row["start_ns"], stop, c["start_ns"], c["end_ns"])
        ]
        own_ns = sum(c["cpu_ns"] for c in own)
        bound_ns = own_ns + sum(c["cpu_ns"] for c in root_cost)
        out["operations"][oid] = {
            "root_id": row["root_id"],
            "start_ns": row["start_ns"],
            "end_ns": row.get("end_ns"),
            "duration_s": (stop - row["start_ns"]) / SECOND,
            "observed_cpu_s": own_ns / cpu_scale,
            "conservative_cpu_s": bound_ns / cpu_scale,
            "complete": row.get("end_ns") is not None
            and all(
                p.get("exit_ns") is not None
                for p in processes.values()
                if p.get("operation_id") == oid
            ),
        }
        if bound_ns > cpu_scale:
            out["errors"].append(f"operation {oid} >1.0 CPU-s")
    maxima = out["maxima"]
    for conservative, prefix in ((False, "observed"), (True, "conservative")):
        maxima[f"{prefix}_burst_pct"] = max(
            i["cpu_pct" if conservative else "observed_cpu_pct"] for i in out["intervals"]
        )
        for seconds, ceiling in ((10, 2), (60, 3)):
            amount = _rolling(charges, seconds * SECOND, conservative=conservative)
            maxima[f"{prefix}_{seconds}s_cpu_s"] = amount / cpu_scale
            if conservative and amount > ceiling * cpu_scale:
                out["errors"].append(f"rolling {seconds}s supervision >{ceiling}.0 CPU-s")
    out["observed_cpu_s"] = sum(c["cpu_ns"] for c in charges) / cpu_scale
    out["complete"] = all(
        row.get("exit_ns") is not None for pid, row in processes.items() if pid not in roots
    ) and all(row.get("end_ns") is not None for row in operations.values())
