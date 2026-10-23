"""Shared registered R1 clean-prefix calculation for harness and final reducer."""

from __future__ import annotations

from research.bl_mac_latency import l1


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected non-negative integer timestamp/count")
    return value


def carried_violation(row: dict) -> dict:
    """Carry irreversible events at or before the last clean host observation."""
    clean = []
    for host in row["host"]:
        timestamp = _integer(host["monotonic_ns"])
        if not isinstance(host["foreign_heavy_processes"], list):
            raise ValueError("invalid foreign process list")
        if host["foreign_heavy_processes"] == []:
            clean.append(timestamp)
    t_clean = max(clean) if clean else None  # JSON null represents minus infinity.
    overruns = 0
    for tick in row["ticks"]:
        finished = _integer(tick["finished_ns"])
        if type(tick["overrun"]) is not bool:
            raise ValueError("overrun must be boolean")
        overruns += tick["overrun"] and t_clean is not None and finished <= t_clean
    doom = 0
    for sample in row["samples"]:
        if "error" in sample or "S_seconds" not in sample:
            continue
        seconds = sample["S_seconds"]
        l1.distribution([seconds])  # Preserve the reducer's finite/nonnegative validation.
        visible = _integer(sample["visible_ns"])
        doom += (
            row["workers"] == 4
            and sample["phase"] == "warm"
            and sample["fixture_id"] == "frozen-800"
            and seconds > 5.0
            and t_clean is not None
            and visible <= t_clean
        )
    return {"T_clean_ns": t_clean, "carried_overruns": overruns, "carried_doom": doom}
