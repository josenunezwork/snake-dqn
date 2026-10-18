"""Q1 parity receipt comparison, retaining T23's exact fail-closed comparator."""

from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping
from research.t23_r0_screen_20261017 import parity as t23
from research.q0q1_20261017 import q1_bank

SCHEMA = "q1-parity/v1"
HORIZON = 5000
MIX_COUNTS = {"scripted": 2, "frozen": 1, "mixed": 1}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_runtime_receipt(
    value: Mapping[str, Any],
    *,
    bank_sha256: str,
    b8_sha256: str,
    source_manifest: Mapping[str, str],
) -> None:
    """Validate a completed Mac/CPU parity comparison before Q1 scoring."""
    required = {
        "schema": "q1-runtime-parity-receipt/v1",
        "pass": True,
        "bank_sha256": bank_sha256,
        "b8_sha256": b8_sha256,
        "source_manifest": dict(source_manifest),
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("runtime parity receipt has wrong identity or failed comparison")
    environment = value.get("environment")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("sha256"), str):
        raise ValueError("runtime parity receipt lacks evaluation-runtime environment identity")


def compare_artifacts(mac: Mapping[str, Any], colab: Mapping[str, Any]) -> dict[str, Any]:
    """Require Q1 bindings, then apply the historical action/frame comparator."""
    errors = []
    for name, value in (("mac", mac), ("colab", colab)):
        if value.get("schema") != SCHEMA:
            errors.append(f"{name}: wrong Q1 parity schema")
        if value.get("bank_namespace") != q1_bank.NAMESPACE:
            errors.append(f"{name}: wrong Q1 bank namespace")
    # Translate only the envelope; T23's comparator is the fixed low-level oracle.
    left = {**mac, "schema": t23.SCHEMA}
    right = {**colab, "schema": t23.SCHEMA}
    result = t23.compare_artifacts(left, right)
    result["schema"] = SCHEMA
    result["pass"] = not errors and bool(result.get("pass"))
    result["errors"] = errors + list(result.get("errors", []))
    return result
