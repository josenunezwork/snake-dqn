"""Pinned capture-05 v5 reference adapter for the prospective v6 policy.

The active environment module deliberately accepts native v6 receipts only.
This module is the sole, comparison-only bridge for the pre-existing capture-05
SUCCESS evidence.  It validates the original bytes with the frozen v5 validator
before making an in-memory v6 comparison projection; it never writes or labels
the old receipt as a native v6 capture.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from research.bl_residual import f5_environment, f5_environment_v5

CAPTURE05_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/"
    "F5_W_TIMING_OPERATIONAL_20261026_ROUND08_02_REQUAL/capture05-environment.json"
)
CAPTURE05_SHA256 = "2b5e0713cc907a08688e1d27bcd171669f345a2f25478a4e6e07be9ce2a29e6d"
CAPTURE05_SIZE = 13_419_976
SCHEMA = "bl-residual-f5-legacy-reference-adapter/v1"


class LegacyReferenceRefusal(ValueError):
    """The sole permitted legacy reference cannot establish a v6 comparison."""


def _refuse(reason: str) -> None:
    raise LegacyReferenceRefusal("F5 legacy capture-05 reference refused: " + reason)


def _read_capture05() -> tuple[dict[str, Any], bytes]:
    """Read exactly the one pinned legacy receipt without path substitution."""
    if CAPTURE05_PATH.is_symlink() or not CAPTURE05_PATH.is_file():
        _refuse("pinned capture-05 file is absent or unsafe")
    raw = CAPTURE05_PATH.read_bytes()
    if len(raw) != CAPTURE05_SIZE or hashlib.sha256(raw).hexdigest() != CAPTURE05_SHA256:
        _refuse("pinned capture-05 bytes or size differ")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LegacyReferenceRefusal(
            "F5 legacy capture-05 reference refused: malformed JSON"
        ) from exc
    if not isinstance(receipt, dict):
        _refuse("pinned capture-05 receipt is not an object")
    if (
        receipt.get("schema") != f5_environment_v5.SCHEMA
        or receipt.get("capture_status") != "SUCCESS"
    ):
        _refuse("pinned capture-05 is not a v5 SUCCESS receipt")
    if "_admission_failures" in receipt or "raw_receipt" in receipt:
        _refuse("pinned capture-05 contains refused or nested diagnostic evidence")
    try:
        f5_environment_v5.validate_environment(receipt)
    except f5_environment_v5.F5EnvironmentRefusal as exc:
        raise LegacyReferenceRefusal(str(exc)) from exc
    return receipt, raw


def capture05_reference() -> dict[str, Any]:
    """Return one non-native, comparison-only wrapper around exact v5 bytes."""
    receipt, raw = _read_capture05()
    return {
        "schema": SCHEMA,
        "kind": "PINNED_V5_SUCCESS_COMPARISON_ONLY",
        "fit_usable": False,
        "scientific_use_authorized": False,
        "source": {
            "path": str(CAPTURE05_PATH),
            "sha256": CAPTURE05_SHA256,
            "size": CAPTURE05_SIZE,
            "schema": f5_environment_v5.SCHEMA,
            "policy": f5_environment_v5.policy_binding(),
        },
        # The original bytes occur exactly once in a sealed control.  Runtime
        # workers validate these embedded bytes and never need the Mac artifact
        # path.  It is retained as source evidence, never re-stamped as v6.
        "raw_v5_base64": base64.b64encode(raw).decode("ascii"),
        "comparison_projection": {
            "kind": "IN_MEMORY_V6_KERNEL_PROVENANCE_PROJECTION",
            "allowed_platform_profile": "Linux-{release}-x86_64-with-glibc2.39",
            "legacy_libc": {"implementation": "glibc", "version": "2.39"},
        },
    }


def validate_capture05_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse any caller-selected legacy input or mutable adapter envelope."""
    required = {
        "schema",
        "kind",
        "fit_usable",
        "scientific_use_authorized",
        "source",
        "raw_v5_base64",
        "comparison_projection",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _refuse("legacy reference wrapper fields differ")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "PINNED_V5_SUCCESS_COMPARISON_ONLY"
        or value.get("fit_usable") is not False
        or value.get("scientific_use_authorized") is not False
        or value.get("source")
        != {
            "path": str(CAPTURE05_PATH),
            "sha256": CAPTURE05_SHA256,
            "size": CAPTURE05_SIZE,
            "schema": f5_environment_v5.SCHEMA,
            "policy": f5_environment_v5.policy_binding(),
        }
        or value.get("comparison_projection")
        != {
            "kind": "IN_MEMORY_V6_KERNEL_PROVENANCE_PROJECTION",
            "allowed_platform_profile": "Linux-{release}-x86_64-with-glibc2.39",
            "legacy_libc": {"implementation": "glibc", "version": "2.39"},
        }
    ):
        _refuse("legacy reference wrapper differs from the sole pinned capture-05 binding")
    encoded = value.get("raw_v5_base64")
    if not isinstance(encoded, str):
        _refuse("legacy reference payload is malformed")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        receipt = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyReferenceRefusal(
            "F5 legacy capture-05 reference refused: malformed payload"
        ) from exc
    if len(raw) != CAPTURE05_SIZE or hashlib.sha256(raw).hexdigest() != CAPTURE05_SHA256:
        _refuse("embedded legacy payload bytes or size differ from capture-05")
    # The embedded original bytes are independently validated as a frozen v5
    # receipt because they must remain usable after upload to a runtime with no
    # Mac path.
    if not isinstance(receipt, dict) or receipt.get("schema") != f5_environment_v5.SCHEMA:
        _refuse("embedded legacy payload schema differs")
    if (
        receipt.get("capture_status") != "SUCCESS"
        or "_admission_failures" in receipt
        or "raw_receipt" in receipt
    ):
        _refuse("embedded legacy payload is refused or diagnostic evidence")
    try:
        f5_environment_v5.validate_environment(receipt)
    except f5_environment_v5.F5EnvironmentRefusal as exc:
        raise LegacyReferenceRefusal(str(exc)) from exc
    return dict(value)


def _v6_projection(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Construct an ephemeral comparison projection, never a captured receipt."""
    wrapper = validate_capture05_reference(reference)
    try:
        legacy = json.loads(base64.b64decode(wrapper["raw_v5_base64"], validate=True))
    except (
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:  # defensive after validation
        raise LegacyReferenceRefusal(
            "F5 legacy capture-05 reference refused: malformed payload"
        ) from exc
    projected = copy.deepcopy(legacy)
    projected["schema"] = f5_environment.SCHEMA
    projected["schema_sha256"] = f5_environment.schema_sha256()
    projected["libc"] = dict(wrapper["comparison_projection"]["legacy_libc"])
    return projected


def compare_capture05_reference(
    reference: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare capture-05 through v6 only; native v5 runtime receipts refuse."""
    wrapper = validate_capture05_reference(reference)
    projected = _v6_projection(wrapper)
    try:
        comparison = f5_environment.compare_environment(projected, actual, mode="cross-assignment")
    except f5_environment.F5CrossAssignmentRefusal:
        raise
    except f5_environment.F5EnvironmentRefusal as exc:
        raise LegacyReferenceRefusal(str(exc)) from exc
    return {
        "reference_kind": "PINNED_V5_SUCCESS_COMPARISON_ONLY",
        "source": dict(wrapper["source"]),
        "comparison_projection": dict(wrapper["comparison_projection"]),
        "comparison": comparison,
    }
