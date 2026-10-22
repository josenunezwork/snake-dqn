"""Freeze snapshot retention, download and job aggregate byte allowances before data."""

from __future__ import annotations

from typing import Any

from research.bl_ppo.byte_bounds import (
    derive_snapshot_byte_bounds,
    maximum_refit_raw_transitions,
)

MIB = 1024**2
TRANSPORT_FILE_CAP = 2 * 1024**3
MAX_DOWNLOAD_CLOSURE_CAP = 8 * 1024**3


def freeze_control_budget(
    control: dict[str, Any], *, timeout_minutes: int | float, max_export_bytes: int
) -> dict[str, Any]:
    """Derive all allowances without observing rows, sizes or training outcomes.

    The qualified supervisor's max_export_bytes is a PER FILE limit. The new
    max_total_export_bytes covers all retained job payloads AND sidecars. Three
    invocations and three transported origin closures cover Y3's source and
    replacement paths without changing their cadence or discarding old exports.
    """
    worlds, steps = int(control["worlds"]), int(control["rollout_steps"])
    bounds = derive_snapshot_byte_bounds(
        worlds=worlds,
        processes=int(control["processes"]),
        rollout_steps=steps,
        max_refit_transitions=maximum_refit_raw_transitions(worlds, steps),
        max_actor_transitions=int(control["target"]),
        trace_actions=1000,
    )
    cadence = control.get("checkpoint_every_seconds", 1800)
    if type(cadence) not in (int, float) or cadence <= 0:
        raise ValueError("checkpoint cadence must be positive")
    if type(timeout_minutes) not in (int, float) or timeout_minutes <= 0:
        raise ValueError("timeout must be positive")
    if type(max_export_bytes) is not int or not 0 < max_export_bytes <= TRANSPORT_FILE_CAP:
        raise ValueError("snapshot plan exceeds qualified per-file transport cap")
    components = bounds["components"]
    if any(item["effective_bytes"] > max_export_bytes for item in components.values()):
        raise ValueError("derived snapshot component exceeds manifest max_export_bytes")
    closure = bounds["total_export_bytes"]
    # Source/replacement catalogues can contain original origin, final source
    # continuation and latest operational snapshot, each with distinct banks.
    download_set = 3 * closure["effective_bytes"] + 64 * 1024
    gib = 1024**3
    download_base_cap = ((3 * closure["base_bytes"] + 32 * 1024 + gib - 1) // gib) * gib
    download_cap = 2 * download_base_cap
    if download_set > download_cap or download_cap > MAX_DOWNLOAD_CLOSURE_CAP:
        raise ValueError("derived transitive snapshot set exceeds declared download cap")
    periodic = int(timeout_minutes * 60 // cadence)
    invocations, transported_closures = 3, 3
    retained = periodic + invocations + transported_closures
    # Reserve actual NPZ row payloads, indexes, JSON monitor histories and all
    # flat-name SHA/marker framing. No diagnostic or RNG behavior is changed.
    reports = bounds["retained_rows"]["monitor_reports"]
    ancillary_base = (
        invocations * (reports + 4) * 64 * 1024
        + 2 * 10_000 * (257 * 4 + 256)
        + (retained * 4 + invocations * (reports + 4) + 32) * 4096
    )
    total_base = retained * closure["base_bytes"] + ancillary_base
    budget = {
        "schema": "bl-p1-export-byte-budget/v1",
        "generosity_multiplier": 2,
        "timeout_minutes": timeout_minutes,
        "checkpoint_every_seconds": cadence,
        "max_periodic_snapshots": periodic,
        "max_run_invocations": invocations,
        "max_transported_closures": transported_closures,
        "retained_closure_allowance": retained,
        "ancillary_base_bytes": ancillary_base,
        "base_total_export_bytes": total_base,
        "max_total_export_bytes": 2 * total_base,
        "max_export_bytes": max_export_bytes,
        "download_closure_cap_bytes": download_cap,
        "download_closure_base_cap_bytes": download_base_cap,
        "derived_download_closure_bytes": download_set,
        "drive_backup_cap_bytes": 2 * total_base,
        "upload_bundle_compressed_cap_bytes": 6 * MIB,
        "upload_bundle_unpacked_cap_bytes": 64 * MIB,
        "upload_transport_cap_seconds": 300,
        "upload_normal_forecast_cap_seconds": 150,
        "transport_limit_semantics": (
            "max_export_bytes is per-file; max_total_export_bytes is aggregate"
        ),
    }
    control["snapshot_byte_bounds"] = bounds
    control["max_snapshot_bytes"] = components["primary_snapshot"]["effective_bytes"]
    control["export_byte_budget"] = budget
    return budget


def validate_control_budget(control: dict[str, Any]) -> dict[str, Any]:
    """Recompute rather than trusting a supplied nested cap or row size."""
    budget = control.get("export_byte_budget")
    if not isinstance(budget, dict):
        raise ValueError("P1 control lacks its frozen export byte budget")
    expected_control = dict(control)
    expected = freeze_control_budget(
        expected_control,
        timeout_minutes=budget.get("timeout_minutes"),
        max_export_bytes=budget.get("max_export_bytes"),
    )
    if any(
        expected_control[key] != control.get(key)
        for key in ("snapshot_byte_bounds", "max_snapshot_bytes", "export_byte_budget")
    ):
        raise ValueError("P1 frozen export byte arithmetic differs")
    return expected


def validate_manifest_budget(manifest: dict[str, Any]) -> None:
    """Bind nested runtime allowances to the actual dispatch envelope."""
    control = manifest["bl_p1"]["control"]
    budget = validate_control_budget(control)
    if (
        budget["max_export_bytes"] != manifest["max_export_bytes"]
        or budget["max_total_export_bytes"] != manifest.get("max_total_export_bytes")
        or budget["timeout_minutes"] < manifest["timeout_minutes"]
    ):
        raise ValueError("manifest and runtime export byte allowances differ")
    if manifest.get("backup_dir"):
        raise ValueError("Drive backup requires a separately declared staging byte budget")
