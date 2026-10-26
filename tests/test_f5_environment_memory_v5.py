"""Prospective v5 admission against the complete refused attempt-03 L4 evidence."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from research.bl_residual import f5_environment_v5 as environment
from tests.test_bl_residual_f5_environment import _receipt

RAW_SHA256 = "464c3be2185d21fb4576ae09f2129904eb16234703b334816f9b2adfa1a6ccbe"
LEDGER = Path(__file__).resolve().parents[1] / (
    "research/bl_residual/ledger/F5_W_CAPTURE_AFTER_22e86986_REPAIR_03/output-03/exports"
)


def _real_raw() -> dict:
    """Decode the byte-exact historical capture; never a synthetic package projection."""
    parts = sorted(LEDGER.glob("environment-raw.json.gz.part-*"))
    assert parts, "The complete real attempt-03 evidence must be published with this regression"
    # Companion files are not compressed stream parts.
    parts = [path for path in parts if not path.name.endswith((".sha256", ".complete"))]
    compressed = b"".join(path.read_bytes() for path in parts)
    payload = gzip.decompress(compressed)
    assert hashlib.sha256(payload).hexdigest() == RAW_SHA256
    return json.loads(payload)


def _prospective_v5(raw: dict) -> dict:
    """Migrate envelope bookkeeping only; retain every acquired observation unchanged.

    This is a counterfactual regression, never admission of the refused capture.
    The only validator semantics changed from v4 are the memory relation. A new
    successful acquisition is still mandatory on the next real assignment.
    """
    receipt = copy.deepcopy(raw)
    assert receipt.pop("_admission_failures") == ["F5 environment parsed GPU memory differs"]
    assert receipt["capture_status"] == "REFUSED"
    receipt.update(
        schema=environment.SCHEMA,
        schema_sha256=environment.schema_sha256(),
        capture_status="SUCCESS",
    )
    assert {key for key in raw if raw[key] != receipt.get(key)} == {
        "_admission_failures",
        "schema",
        "schema_sha256",
        "capture_status",
    }
    return receipt


def test_full_v5_validation_passes_all_real_attempt03_observations() -> None:
    raw = _real_raw()
    with pytest.raises(environment.F5EnvironmentRefusal):
        environment.validate_environment(raw)
    receipt = _prospective_v5(raw)
    assert receipt["gpu_memory"] == 23_659_151_360
    assert receipt["gpu_driver_query"]["memory_total"] == "23034"
    assert 23034 * environment.MIB_BYTES == 24_152_899_584
    assert len(receipt["locked_wheels"]) == 69
    environment.validate_environment(receipt)
    assert environment.collect_environment_failures(receipt) == []
    classified = environment.classify_receipt(receipt)
    assert classified[json.dumps(("gpu_memory",))] == "invariant"
    assert classified[json.dumps(("gpu_driver_query", "memory_total"))] == "invariant"
    environment.compare_environment(receipt, copy.deepcopy(receipt), mode="same-assignment")


@pytest.mark.parametrize("field", ["torch", "smi"])
@pytest.mark.parametrize("mode", ["same-assignment", "cross-assignment"])
def test_real_memory_totals_remain_separate_exact_invariants(field: str, mode: str) -> None:
    baseline = _receipt()
    baseline.pop("libc")
    baseline["schema"] = environment.SCHEMA
    baseline["schema_sha256"] = environment.schema_sha256()
    baseline["gpu_memory"] = 23_659_151_360
    changed = copy.deepcopy(baseline)
    if field == "torch":
        changed["gpu_memory"] -= 1
    else:
        changed["gpu_driver_query"]["memory_total"] = "23035"
        changed["gpu_driver_query_raw"] = changed["gpu_driver_query_raw"].replace("23034", "23035")
    # Both receipts satisfy physical consistency; identity still rejects a one-unit drift.
    environment.validate_environment(baseline)
    environment.validate_environment(changed)
    with pytest.raises(environment.F5EnvironmentRefusal, match="differs"):
        environment.compare_environment(baseline, changed, mode=mode)
