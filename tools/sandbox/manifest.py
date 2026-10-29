"""Strict, intentionally small manifest contract for sandbox development jobs.

The manifest is a human-reviewable ten-line JSON document.  This module has no
transport or provider dependency: it validates the job packet before an operator
uses the separately-qualified dispatch path.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA_NAME = "standing-a100-development-sandbox/v1"
REQUIRED_KEYS = frozenset(
    {
        "schema",
        "study",
        "inputs",
        "salt",
        "endpoint",
        "timeout",
        "excluded_studies",
        "owner_rule",
        "outputs",
        "crossing",
    }
)
PROTECTED_TOKENS = ("f5", "big-lever/exp/f5-", "d4096", "t34-round0-study-01")


def protected_name(value: str) -> bool:
    """Match study names at component boundaries, never random digest substrings."""
    return bool(
        re.search(
            r"(?:^|[/_.-])(?:f5|d4096|t34-round0-study-01)(?=$|[/_.-])",
            value.lower().replace("\\", "/"),
        )
    )


class ManifestError(ValueError):
    """A manifest cannot establish its required sandbox boundary."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ManifestError(f"{name} must be a positive finite number")
    return result


def _forbidden(value: Any, location: str = "manifest") -> None:
    if isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        if protected_name(normalized):
            raise ManifestError(f"{location} names a protected study/path")
    elif isinstance(value, dict):
        for key, nested in value.items():
            if key == "sha256":
                continue
            _forbidden(key, location)
            _forbidden(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _forbidden(nested, f"{location}[{index}]")


def _sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ManifestError(f"{name} must be a lowercase SHA-256 hex digest")


def validate_manifest(payload: Any) -> dict[str, Any]:
    """Validate an in-memory sandbox manifest and return its typed mapping."""
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    if set(payload) != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - set(payload))
        extra = sorted(set(payload) - REQUIRED_KEYS)
        raise ManifestError(f"manifest keys must be exact; missing={missing}, extra={extra}")
    if payload["schema"] != SCHEMA_NAME:
        raise ManifestError(f"schema must be {SCHEMA_NAME!r}")
    if not isinstance(payload["study"], str) or not payload["study"].strip():
        raise ManifestError("study must be a nonempty string")
    _forbidden(payload["study"], "study")
    if not isinstance(payload["salt"], str) or not payload["salt"].startswith("a100-dev/"):
        raise ManifestError("salt must be a domain-separated a100-dev/ string")
    _forbidden(payload["salt"], "salt")

    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ManifestError("inputs must be a nonempty array")
    for index, record in enumerate(inputs):
        _forbidden(record, f"inputs[{index}]")
        if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
            raise ManifestError(f"inputs[{index}] must be an input record with kind")
        kind = record["kind"]
        if kind in {
            "closed-f0",
            "frozen-checkpoint",
            "frozen-opponent",
            "prepared-critic-snapshot",
            "critic-rescore-models",
            "training-export",
            "training-export-sidecar",
            "training-export-derivative",
            "verified-ledger",
            "verified-unit",
            "registry",
        }:
            if not isinstance(record.get("path"), str) or not record["path"]:
                raise ManifestError(f"inputs[{index}].path is required for {kind}")
            _sha256(record.get("sha256"), f"inputs[{index}].sha256")
        elif kind == "worlds":
            if record.get("study") != payload["study"]:
                raise ManifestError("world input study must equal manifest study")
            indices, seeds = record.get("indices"), record.get("seeds")
            if (
                not isinstance(indices, list)
                or not isinstance(seeds, list)
                or len(indices) != len(seeds)
            ):
                raise ManifestError("world inputs require equal nonempty indices and seeds")
            if not indices:
                raise ManifestError("world inputs must not be empty")
            for world_index, seed in zip(indices, seeds):
                if (
                    isinstance(world_index, bool)
                    or not isinstance(world_index, int)
                    or world_index < 0
                ):
                    raise ManifestError("world index must be a nonnegative integer")
                if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
                    raise ManifestError("world seed must be an unsigned 64-bit integer")
        else:
            raise ManifestError(f"inputs[{index}].kind is not admitted: {kind!r}")
        if (
            kind
            in {
                "frozen-opponent",
                "prepared-critic-snapshot",
                "critic-rescore-models",
                "training-export",
                "training-export-sidecar",
                "training-export-derivative",
            }
            and payload["study"] != "sandbox-a-train01"
        ):
            raise ManifestError("training inputs are exclusive to sandbox-a-train01")

    endpoint = payload["endpoint"]
    _forbidden(endpoint, "endpoint")
    if not isinstance(endpoint, dict) or endpoint.get("kind") not in {"updates", "steps", "hours"}:
        raise ManifestError("endpoint must declare kind updates, steps, or hours")
    _number(endpoint.get("value"), "endpoint.value")
    timeout = payload["timeout"]
    if not isinstance(timeout, dict):
        raise ManifestError("timeout must be an object")
    if set(timeout) == {"base_seconds", "seconds", "purpose"}:
        if timeout["purpose"] != "runaway-guard":
            raise ManifestError("timeout purpose must be runaway-guard")
        estimate = _number(timeout["base_seconds"], "timeout.base_seconds")
    elif set(timeout) == {"measured_estimate_seconds", "seconds"}:
        # Retain the original schema for existing Job A packets.
        estimate = _number(
            timeout["measured_estimate_seconds"], "timeout.measured_estimate_seconds"
        )
    else:
        raise ManifestError("timeout must declare a base and doubled runaway guard")
    actual = _number(timeout.get("seconds"), "timeout.seconds")
    if not math.isclose(actual, estimate * 2, rel_tol=0.0, abs_tol=1e-9):
        raise ManifestError("timeout.seconds must equal 2x base seconds")
    excluded = payload["excluded_studies"]
    if not isinstance(excluded, list) or not all(
        isinstance(item, str) and item for item in excluded
    ):
        raise ManifestError("excluded_studies must be a nonempty string array")
    required_exclusions = {"F5", "F4-fold-0", "R30", "strict", "probe", "serving"}
    if not required_exclusions.issubset(set(excluded)):
        raise ManifestError("excluded_studies omits a mandatory protected set")
    if not isinstance(payload["owner_rule"], str) or "ADOPTED.md" not in payload["owner_rule"]:
        raise ManifestError("owner_rule must cite the adopted sandbox rule")
    outputs = payload["outputs"]
    _forbidden(outputs, "outputs")
    if not isinstance(outputs, dict) or outputs.get("label") != "simulator-D":
        raise ManifestError("outputs must be tagged simulator-D")
    if not isinstance(outputs.get("paths"), list) or not outputs["paths"]:
        raise ManifestError("outputs.paths must be a nonempty array")
    broker = outputs.get("broker_manifest")
    if not isinstance(broker, dict) or set(broker) != {"path", "sha256"}:
        raise ManifestError("outputs.broker_manifest must have path and sha256")
    if not isinstance(broker["path"], str) or not broker["path"]:
        raise ManifestError("outputs.broker_manifest.path must be a nonempty string")
    _sha256(broker["sha256"], "outputs.broker_manifest.sha256")
    crossing = payload["crossing"]
    if not isinstance(crossing, dict) or crossing.get("max_frozen_checkpoint_shas") != 1:
        raise ManifestError("crossing must cap frozen checkpoint SHA crossings at one")
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and enforce the committed ten-physical-line JSON representation."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest: {path}") from exc
    lines = text.splitlines()
    if (
        len(lines) != 10
        or not lines[0].lstrip().startswith("{")
        or not lines[-1].rstrip().endswith("}")
    ):
        raise ManifestError(
            "manifest must be exactly ten physical JSON lines with braces on first/last"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is invalid JSON: {exc.msg}") from exc
    return validate_manifest(payload)
