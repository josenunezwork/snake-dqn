"""Validate a frozen study-wide reservation; no allocation or provider calls."""

from __future__ import annotations

import math
from typing import Any, Mapping

from research.bl_e16.admission import (
    ALLOCATION_HOUR_CAP,
    L4_UNITS_PER_HOUR,
    MAX_FORECAST_HOURS,
    RESERVE_UNITS,
    UNIT_CAP,
)


def validate_reservation(value: Mapping[str, Any], arms: list[str]) -> dict[str, dict[str, Any]]:
    """Reserve all arms and evaluation within the amended study-wide caps."""
    if (
        value.get("schema") != "bl-e16-budget/v1"
        or value.get("unit_cap") != UNIT_CAP
        or value.get("allocation_hour_cap") != ALLOCATION_HOUR_CAP
        or value.get("reserve_units") != RESERVE_UNITS
    ):
        raise ValueError("E16 requires frozen 40u/20h/4u total reservation")
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
        if job["rate_units_per_hour"] != L4_UNITS_PER_HOUR:
            raise ValueError("E16 reservation must use the registered L4 rate")
        if job["timeout_minutes"] * job["rate_units_per_hour"] / 60 > job["unit_cap"]:
            raise ValueError("reservation underprices maximum timeout")
        if job.get("includes_setup_export") is not True:
            raise ValueError("reservation must charge setup/export/failures")
    if sum(j["estimate_minutes"] for j in jobs) > MAX_FORECAST_HOURS * 60:
        raise ValueError("aggregate forecast exceeds prospective 8h admission ceiling")
    if (
        sum(j["unit_cap"] for j in jobs) + RESERVE_UNITS > UNIT_CAP
        or sum(j["timeout_minutes"] for j in jobs) > ALLOCATION_HOUR_CAP * 60
    ):
        raise ValueError("aggregate study liability exceeds frozen caps")
    return {job["lane"]: dict(job) for job in jobs}
