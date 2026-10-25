"""Frozen R30 bindings and first-cap-wins accounting, independent of F4."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export

JUDGMENT_SHA = "f69c44863a8331028ade0dff20ee69eb9dc5959a92e8480a7329cb39d2b58d5c"
AMENDMENT_SHA = "df62897bf2624b64f701db60a1eb1725f06bc5b0acd017e57f54db3aae8d109d"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
NAMESPACE = "bl-r30-postf4-20261025-v1"
SALT = "t34-r30-parent-split-20261023-v1"
CAPS = {
    "r30-1": {"base_minutes": 72, "effective_minutes": 144},
    "r30-2": {"base_minutes": 144, "effective_minutes": 288},
}
COMBINED_EFFECTIVE_MINUTES = 432


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_amendment_binding(value: dict) -> None:
    """Require the original R30 authority and its narrow post-F4 amendment."""
    if value.get("ruling_sha256") != JUDGMENT_SHA or value.get("amendment_sha256") != AMENDMENT_SHA:
        raise ValueError("foreign R30 authority/amendment binding")


def require_series(value: dict, stage: str) -> dict:
    """Account all prior use against the amended finite wall envelopes."""
    cap = CAPS[stage]
    try:
        require_amendment_binding(value)
    except ValueError as exc:
        raise ValueError("foreign ruling/amendment or exhausted completion attempts") from exc
    if value.get("attempt") not in (0, 1, 2):
        raise ValueError("foreign ruling or exhausted completion attempts")
    fields = ("stage_minutes", "combined_minutes")
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
    minutes = min(
        cap["effective_minutes"] - value["stage_minutes"],
        COMBINED_EFFECTIVE_MINUTES - value["combined_minutes"],
    )
    if minutes <= 0:
        raise ValueError("INCOMPLETE: first local/combined cap exhausted")
    return {"remaining_minutes": minutes}


def bound_json(path: Path, sha: str) -> dict:
    if digest(path) != sha:
        raise ValueError("stale binding")
    verify_export(path)
    return json.loads(path.read_text())


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"HOLD: invalid {name}")
    return value


def require_successor_gate(gate: dict, allocation: str, diagnosis: dict | None = None) -> str:
    """Admit only the amendment's label-5 rescue-ranking successor route."""
    try:
        require_amendment_binding(gate)
    except ValueError as exc:
        raise ValueError("foreign successor gate") from exc
    if gate.get("allocation") != allocation:
        raise ValueError("foreign successor gate")
    if (
        gate.get("f4_terminal") is not True
        or type(gate.get("f4_r9_label")) is not int
        or gate["f4_r9_label"] not in range(1, 11)
    ):
        raise ValueError("HOLD: F4 terminal disposition required")
    if gate["f4_r9_label"] != 5:
        raise ValueError("HOLD: post-F4 amendment permits only original R9 label 5")
    exception = gate.get("label5_exception")
    if exception != {
        "amendment_sha256": AMENDMENT_SHA,
        "f4_r9_label": 5,
        "scope": "r30-2-rescue-survivor-mass-ranking",
        "terminal_label": "STOP_FUNDING",
    }:
        raise ValueError("HOLD: label-5 exception is absent, forged, or out of scope")
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
    try:
        require_amendment_binding(diagnosis)
    except ValueError as exc:
        raise ValueError("HOLD: diagnosis authority/amendment differs") from exc
    if (
        diagnosis.get("corpus_allocation") != allocation
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
    conversion = diagnosis.get("conversion")
    if not isinstance(conversion, dict):
        raise ValueError("HOLD: resolved conversion record is missing")
    control = conversion.get("control")
    if not isinstance(control, dict):
        raise ValueError("HOLD: resolved conversion control is missing")
    expected_name = "adam" if optimizer == "adam" else "lbfgs"
    if (
        control.get("name") != expected_name
        or control.get("procedure") != optimizer
        or gate.get("control_name") != expected_name
        or gate.get("control_procedure") != optimizer
    ):
        raise ValueError("HOLD: conversion control differs from resolved optimizer")
    endpoint_sha = _sha256(control.get("endpoint_sha256"), "control endpoint SHA")
    if gate.get("control_endpoint_sha256") != endpoint_sha:
        raise ValueError("HOLD: successor gate control endpoint differs from diagnosis")
    parameters_sha = _sha256(control.get("parameters_sha256"), "control parameters SHA")
    if gate.get("control_parameters_sha256") != parameters_sha:
        raise ValueError("HOLD: successor gate control parameters differ from diagnosis")
    binding = control.get("convergence_binding")
    if not isinstance(binding, dict) or gate.get("convergence_binding") != binding:
        raise ValueError("HOLD: successor convergence binding differs from diagnosis")
    if binding.get("procedure") != optimizer:
        raise ValueError("HOLD: convergence binding procedure differs")
    for key in (
        "initializer_parameters_sha256",
        "recipe_sha256",
        "batch_plan_sha256",
        "cache_sha256",
    ):
        _sha256(binding.get(key), f"convergence binding {key}")
    diagnosis_sha = _sha256(gate.get("diagnosis_sha256"), "diagnosis SHA")
    if gate.get("diagnosis_sha256") != diagnosis_sha:
        raise ValueError("HOLD: diagnosis binding differs")
    if gate.get("allow_successor") is not True:
        raise ValueError("HOLD: explicit derived successor admission missing")
    return optimizer
