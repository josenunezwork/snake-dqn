"""Offline cadence measurement of the entire retained C/T fixture execution.

Host times are the supervisor's poll-observation elapsed clock. Bootstrap before
the first C/T line is reported separately, never silently counted as a heartbeat.
No scientific output is opened by this measurement.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_json as json
from research.sandbox_a_resident.ct_progress import (
    capture_inventory,
    validate_heartbeat,
    verify_cadence,
)

PREFIXES = ("CT_PROGRESS ", "CT_HEARTBEAT ", "CT_SIGUSR1_CLEAN_EXIT ")


def measure_liveness(output: Path, supervisor_log: Path) -> dict[str, Any]:
    """Recompute gaps from retained logs, refusing missing or altered records."""
    base = output / "exports/ct-cuda-canary"
    capture = json.loads((base / "capture.json").read_text())
    inventory = capture_inventory()
    if set(capture) != inventory["report"]:
        raise ValueError("unqualified capture report shape")
    probes = []
    child_lines: Counter[str] = Counter()
    for probe in capture["probes"]:
        name = probe["name"]
        if name not in {"complete", "signal"} or set(probe) != inventory[name]:
            raise ValueError("unqualified capture probe shape")
        cadence = probe["progress_cadence"]
        if set(cadence) != inventory["cadence"] or any(
            set(row) != inventory["observation"] for row in cadence["observations"]
        ):
            raise ValueError("unqualified capture cadence shape")
        text = (base / f"{name}-entrypoint.log").read_text()
        child_lines.update(text.splitlines())
        verify_cadence(text, cadence)
        heartbeats = [
            json.loads(line.partition(" ")[2])
            for line in text.splitlines()
            if line.startswith("CT_HEARTBEAT ")
        ]
        for row in heartbeats:
            validate_heartbeat(row)
        child_times = [0.0, *(row["elapsed_sec"] for row in heartbeats)]
        if any(b < a for a, b in zip(child_times, child_times[1:])):
            raise ValueError("non-monotonic child heartbeat clock")
        child_maximum = max(b - a for a, b in zip(child_times, child_times[1:]))
        if child_maximum > 30:
            raise ValueError("child heartbeat exceeds 30-second bound")
        phases = []
        start = heartbeats[0]
        for row in heartbeats[1:]:
            if row["phase"] != start["phase"]:
                phases.append(
                    dict(
                        phase=start["phase"],
                        start_seconds=start["elapsed_sec"],
                        end_seconds=row["elapsed_sec"],
                        duration_seconds=row["elapsed_sec"] - start["elapsed_sec"],
                    )
                )
                start = row
        probes.append(
            dict(
                name=name,
                parent_receive_max_gap_seconds=cadence["max_gap_seconds"],
                parent_receive_heartbeat_max_gap_seconds=cadence["heartbeat_max_gap_seconds"],
                child_reported_heartbeat_max_gap_seconds=child_maximum,
                phases=phases,
            )
        )
    parent_rows = []
    for line in (base / "progress-history.log").read_text().splitlines():
        if child_lines[line]:
            child_lines[line] -= 1
        elif line.startswith("CT_HEARTBEAT "):
            row = json.loads(line.partition(" ")[2])
            validate_heartbeat(row)
            parent_rows.append(row)
        else:
            raise ValueError("unqualified parent history line")
    if any(child_lines.values()) or not parent_rows:
        raise ValueError("parent history lacks exact child lines")
    parent_times = [0.0, *(row["elapsed_sec"] for row in parent_rows)]
    parent_max = max(b - a for a, b in zip(parent_times, parent_times[1:]))
    reported = capture["startup_heartbeat_cadence"]
    if (
        set(reported) != {"line_count", "max_gap_seconds", "limit_seconds"}
        or reported["line_count"] != len(parent_rows)
        or reported["limit_seconds"] != 30
        or type(reported["max_gap_seconds"]) not in {int, float}
        or not parent_max - 1e-6 <= reported["max_gap_seconds"] <= 30
        or any(b < a for a, b in zip(parent_times, parent_times[1:]))
    ):
        raise ValueError("parent heartbeat cadence mismatch or exceeds 30-second bound")
    host = []
    seen: set[str] = set()
    poll_index = 0
    previous = -1.0
    for number, line in enumerate(supervisor_log.read_text().splitlines(), 1):
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "job_log_tail" not in row or "elapsed" not in row:
            continue
        elapsed = row["elapsed"]
        if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError("non-monotonic host poll clock")
        previous = elapsed
        poll = json.loads((output / f"poll-{poll_index:06d}.json").read_text())
        if poll["job_log_tail"] != row["job_log_tail"]:
            raise ValueError("host log differs from retained poll")
        poll_index += 1
        lines = [line for line in row["job_log_tail"].splitlines() if line.startswith(PREFIXES)]
        fresh = [line for line in lines if line not in seen]
        seen.update(lines)
        if fresh:
            host.append(dict(elapsed=elapsed, log_line=number, new_lines=len(fresh)))
    if len(host) < 2:
        raise ValueError("incomplete host poll inventory")
    gap, before, after = max(
        (b["elapsed"] - a["elapsed"], a["elapsed"], b["elapsed"]) for a, b in zip(host, host[1:])
    )
    if gap > 60:
        raise ValueError("host-observed progress gap exceeds 60-second bound")
    return dict(
        schema="ct-cuda-liveness-review/v1",
        status="HOLD_RESOLVED_CT_CONSECUTIVE_STATUS_CADENCE",
        limit_seconds=60,
        host_observed_max_gap_seconds=gap,
        host_max_gap_endpoints_seconds=[before, after],
        host_first_ct_line_seconds=host[0]["elapsed"],
        host_last_ct_line_seconds=host[-1]["elapsed"],
        host_observation_points=host,
        host_poll_count=poll_index,
        parent_reported_heartbeat_max_gap_seconds=reported["max_gap_seconds"],
        probes=probes,
        scope="consecutive C/T status lines; pre-entrypoint provider/bootstrap reported separately",
    )
