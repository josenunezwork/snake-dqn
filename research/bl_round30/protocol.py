"""Frozen R30 bindings and first-cap-wins accounting, independent of F4."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export

JUDGMENT_SHA = "f69c44863a8331028ade0dff20ee69eb9dc5959a92e8480a7329cb39d2b58d5c"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
NAMESPACE = "t34-r30-20261023-v1"
SALT = "t34-r30-parent-split-20261023-v1"
CAPS = {
    "r30-1": {"base_hours": 1.2, "base_units": 2.4, "hours": 2.4, "units": 4.8},
    "r30-2": {"base_hours": 2.4, "base_units": 4.8, "hours": 4.8, "units": 9.6},
}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_series(value: dict, stage: str) -> dict:
    """Account setup/canaries/failed attempts once, including combined limits."""
    cap = CAPS[stage]
    if value.get("ruling_sha256") != JUDGMENT_SHA or value.get("attempt") not in (0, 1, 2):
        raise ValueError("foreign ruling or exhausted completion attempts")
    fields = ("stage_hours", "stage_units", "combined_hours", "combined_units")
    for key in fields:
        x = value.get(key)
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError("verified cumulative accounting required")
    if value.get("attempt", 0) and not (
        value.get("mechanical_failure_verified") is True
        and value.get("stop_verified") is True
        and value.get("missing_units_verified") is True
        and value.get("failure_kind") in ("transport", "interruption", "export")
    ):
        raise ValueError("completion requires mechanical failure, verified stop and missing units")
    if value.get("unencumbered_reservation_verified") is not True:
        raise ValueError("HOLD: Claude must reserve verified unencumbered credits")
    hours = min(cap["hours"] - value["stage_hours"], 7.2 - value["combined_hours"])
    units = min(cap["units"] - value["stage_units"], 14.4 - value["combined_units"])
    if hours <= 0 or units <= 0:
        raise ValueError("INCOMPLETE: first local/combined cap exhausted")
    return {"remaining_hours": hours, "remaining_units": units}


def bound_json(path: Path, sha: str) -> dict:
    if digest(path) != sha:
        raise ValueError("stale binding")
    verify_export(path)
    return json.loads(path.read_text())


def require_successor_gate(gate: dict, allocation: str, diagnosis: dict | None = None) -> str:
    """Never reopen R9 labels 4–8 or substitute an F4 endpoint."""
    if gate.get("allocation") != allocation or gate.get("ruling_sha256") != JUDGMENT_SHA:
        raise ValueError("foreign successor gate")
    if (
        gate.get("f4_terminal") is not True
        or type(gate.get("f4_r9_label")) is not int
        or gate["f4_r9_label"] not in range(1, 11)
    ):
        raise ValueError("HOLD: F4 terminal disposition required")
    if gate["f4_r9_label"] in range(4, 9):
        raise ValueError("HOLD: R9 closure needs a new explicit Astra ruling")
    if gate.get("f4_continuation_permitted") is not True:
        raise ValueError("HOLD: terminal disposition does not permit continuation")
    for key in ("optimizer_resolved", "conversion_trigger_passed", "archive_admitted"):
        if gate.get(key) is not True:
            raise ValueError(f"HOLD: {key}")
    optimizer = gate.get("optimizer")
    if optimizer not in ("adam", "lbfgs"):
        raise ValueError("one frozen matched optimizer required")
    if optimizer == "lbfgs" and gate.get("convergence_independently_verified") is not True:
        raise ValueError("HOLD: independently verified deterministic convergence required")
    if optimizer == "lbfgs" and (
        gate.get("lbfgs_max_evaluations") != 500
        or gate.get("lbfgs_max_seconds") != 3600
        or gate.get("convergence_tolerance") != 1e-5
    ):
        raise ValueError("HOLD: convergence procedure differs from frozen R30-1 ceiling/tolerance")
    if diagnosis is None:
        raise ValueError("HOLD: immutable R30-1 Mac diagnosis is required")
    if (
        diagnosis.get("ruling_sha256") != JUDGMENT_SHA
        or diagnosis.get("corpus_allocation") != allocation
        or diagnosis.get("development_only") is not True
        or diagnosis.get("conversion", {}).get("label") != "GO"
        or diagnosis.get("diagnostics", {}).get("certificate", {}).get("certified") is not True
    ):
        raise ValueError("HOLD: prior optimizer certificate/conversion trigger not passed")
    required_label = "OPTIMIZER_ADEQUATE" if optimizer == "adam" else "OPTIMIZER_MATERIAL"
    if diagnosis.get("label") != required_label:
        raise ValueError("HOLD: matched optimizer is not the resolved R30-1 procedure")
    if gate.get("convergence_procedure") != optimizer:
        raise ValueError("matched convergence procedure differs")
    if gate.get("allow_successor") is not True:
        raise ValueError("HOLD: explicit derived successor admission missing")
    return optimizer
