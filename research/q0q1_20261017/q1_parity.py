"""Q1 parity receipt comparison, retaining T23's exact fail-closed comparator."""

from __future__ import annotations
from typing import Any, Mapping
from research.t23_r0_screen_20261017 import parity as t23
from research.q0q1_20261017 import q1_bank

SCHEMA = "q1-parity/v1"
HORIZON = 5000
MIX_COUNTS = {"scripted": 2, "frozen": 1, "mixed": 1}


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
