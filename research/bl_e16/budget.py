"""Validate a frozen study-wide reservation; no allocation or provider calls."""

from __future__ import annotations

import math
from typing import Any, Mapping


def validate_reservation(value: Mapping[str, Any], arms: list[str]) -> dict[str, dict[str, Any]]:
    """Worst-case reservations across every arm and final evaluation total <=20u."""
    if (
        value.get("schema") != "bl-e16-budget/v1"
        or value.get("unit_cap") != 20
        or value.get("allocation_hour_cap") != 10
        or value.get("reserve_units") != 4
    ):
        raise ValueError("E16 requires frozen 20u/10h/4u total reservation")
    jobs = value.get("jobs", [])
    if not isinstance(jobs, list) or {j.get("lane") for j in jobs} != {*arms, "evaluation"}:
        raise ValueError("reservation must cover exactly every arm and evaluation")
    if len(jobs) != len(arms) + 1 or len({j.get("job_id") for j in jobs}) != len(jobs):
        raise ValueError("duplicate or missing study reservation")
    for job in jobs:
        for key in ("unit_cap", "timeout_minutes", "rate_units_per_hour", "estimate_minutes"):
            number = job.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError("reservation needs positive finite priced caps")
        if job["timeout_minutes"] < 2 * job["estimate_minutes"]:
            raise ValueError("every job must reserve 2x its measured estimate")
        if job["timeout_minutes"] * job["rate_units_per_hour"] / 60 > job["unit_cap"]:
            raise ValueError("reservation underprices maximum timeout")
        if job.get("includes_setup_export") is not True:
            raise ValueError("reservation must charge setup/export/failures")
    if sum(j["unit_cap"] for j in jobs) + 4 > 20 or sum(j["timeout_minutes"] for j in jobs) > 600:
        raise ValueError("aggregate study liability exceeds frozen caps")
    return {job["lane"]: dict(job) for job in jobs}
