"""A4 bridge for preserved A3 Q1 parity evidence.

The A3 runtime packet and its Mac reference are evidence of the evaluator that
existed at commit 17a927c9.  A4 adds a process pool, so those immutable bytes
cannot be relabelled as current evidence.  This module accepts them only after a
new Mac reference and a measured timing receipt demonstrate byte-identical rows
across the historical serial evaluator and the current serial/pool evaluators.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from research.q0q1_20261017 import q1_evaluate, q1_parity

HISTORICAL_COMMIT = "17a927c9"
HISTORICAL_EVALUATOR_PATH = "research/q0q1_20261017/q1_evaluate.py"
HISTORICAL_PARITY_PATH = "research/q0q1_20261017/q1_parity.py"
HISTORICAL_EVALUATOR_SHA256 = "29de3cb47dd4ec05760dcd47bb281d7cec7e22a3310662261670a8605d4af835"
HISTORICAL_PARITY_SHA256 = "54d83013d3c51b2f565ffddcfbc8cd92ca79fad1e390a26e30b841d231010c73"
SCHEMA = "q0q1-a4-historical-parity-bridge/v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def historical_source_manifest(current: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the A3 source closure, differing only in its frozen Q1 adapters."""
    value = dict(q1_evaluate.source_manifest() if current is None else current)
    value[HISTORICAL_EVALUATOR_PATH] = HISTORICAL_EVALUATOR_SHA256
    value[HISTORICAL_PARITY_PATH] = HISTORICAL_PARITY_SHA256
    return value


def load_historical_evaluator(root: Path | None = None) -> ModuleType:
    """Load the exact A3 serial evaluator from git, never from the working tree."""
    repo = root or Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "show", f"{HISTORICAL_COMMIT}:{HISTORICAL_EVALUATOR_PATH}"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("cannot load the pinned A3 evaluator from git")
    raw = completed.stdout
    if hashlib.sha256(raw).hexdigest() != HISTORICAL_EVALUATOR_SHA256:
        raise RuntimeError("pinned A3 evaluator bytes do not match their registered SHA-256")
    with tempfile.TemporaryDirectory(prefix="q1-a4-historical-") as directory:
        path = Path(directory) / "q1_evaluate_a3.py"
        path.write_bytes(raw)
        spec = importlib.util.spec_from_file_location("q1_evaluate_a3_historical", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot construct the pinned A3 evaluator module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _require_artifact(
    value: Mapping[str, Any],
    *,
    source: Mapping[str, str],
    bank_sha256: str,
    b8_sha256: str,
    name: str,
) -> list[Mapping[str, Any]]:
    if (
        value.get("schema") != q1_parity.SCHEMA
        or value.get("bank_sha256") != bank_sha256
        or value.get("b8_sha256") != b8_sha256
        or value.get("source_sha256") != digest(dict(source))
        or value.get("runtime", {}).get("system") != "Darwin"
    ):
        raise ValueError(f"{name} is not a Mac artifact for its claimed Q1 source identity")
    worlds = value.get("worlds")
    if not isinstance(worlds, list) or len(worlds) < 3:
        raise ValueError(f"{name} must contain at least three reserved-bank world observations")
    if len({row.get("world_id") for row in worlds if isinstance(row, Mapping)}) != len(worlds):
        raise ValueError(f"{name} has duplicate or malformed world observations")
    return worlds


def _row_hashes(value: Any, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"timing receipt lacks at least three {name} observations")
    if any(not isinstance(item, str) or len(item) != 64 for item in value):
        raise ValueError(f"timing receipt has malformed {name}")
    return list(value)


def validate_historical_parity(
    runtime: Mapping[str, Any],
    old_mac: Mapping[str, Any],
    current_mac: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one-way A3-to-A4 bridge and return hash-bound metadata.

    ``runtime`` remains immutable A3 evidence and is checked using the exact old
    source closure.  ``current_mac`` and ``timing`` are newly measured A4
    evidence.  No old receipt is modified or interpreted as current-parallel.
    """
    current_source = q1_evaluate.source_manifest()
    historical_source = historical_source_manifest(current_source)
    bank_sha256 = timing.get("bank_sha256")
    b8_sha256 = timing.get("b8_sha256")
    if not isinstance(bank_sha256, str) or not isinstance(b8_sha256, str):
        raise ValueError("timing receipt lacks Q1 bank or B8 identity")
    if runtime.get("source_manifest") != historical_source:
        raise ValueError("historical runtime receipt source is not the exact A3 adapter closure")
    q1_parity.validate_runtime_receipt(
        runtime,
        bank_sha256=bank_sha256,
        b8_sha256=b8_sha256,
        source_manifest=historical_source,
    )
    old_worlds = _require_artifact(
        old_mac,
        source=historical_source,
        bank_sha256=bank_sha256,
        b8_sha256=b8_sha256,
        name="historical Mac reference",
    )
    current_worlds = _require_artifact(
        current_mac,
        source=current_source,
        bank_sha256=bank_sha256,
        b8_sha256=b8_sha256,
        name="fresh current Mac reference",
    )
    if runtime.get("reference") != old_mac:
        raise ValueError(
            "historical runtime receipt does not embed the supplied historical Mac reference"
        )
    if {row["world_id"] for row in old_worlds} != {row["world_id"] for row in current_worlds}:
        raise ValueError("historical and current Mac references use different reserved worlds")
    if (
        timing.get("schema") != "q1-eval-timing-probe/v1"
        or timing.get("status") != "completed"
        or timing.get("source_manifest") != current_source
        or timing.get("historical_source_manifest") != historical_source
        or timing.get("purpose") != "capacity measurement only; not Q1 decision evidence"
    ):
        raise ValueError(
            "timing receipt is not a current, explicitly historical-bridged capacity probe"
        )
    rows = _row_hashes(timing.get("row_sha256"), "parallel row hashes")
    serial = _row_hashes(timing.get("serial_row_sha256"), "serial row hashes")
    historical = _row_hashes(timing.get("historical_row_sha256"), "historical serial row hashes")
    if rows != serial or rows != historical:
        raise ValueError(
            "current pool, current serial, and historical serial numerical rows differ"
        )
    repeats = timing.get("repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("timing receipt lacks a positive whole-number repeat count")
    if (
        timing.get("parity_world_count") != len(old_worlds)
        or timing.get("world_count") != len(rows)
        or len(rows) != len(old_worlds) * repeats
    ):
        raise ValueError("timing rows do not exactly cover every parity world in every repeat")
    if (
        not isinstance(timing.get("elapsed_seconds"), (int, float))
        or timing["elapsed_seconds"] <= 0
    ):
        raise ValueError("timing receipt lacks a positive measured elapsed duration")
    return {
        "schema": SCHEMA,
        "historical_commit": HISTORICAL_COMMIT,
        "historical_source_manifest": historical_source,
        "current_source_manifest": current_source,
        "historical_runtime_sha256": digest(runtime),
        "historical_mac_sha256": digest(old_mac),
        "current_mac_sha256": digest(current_mac),
        "timing_sha256": digest(timing),
        "bank_sha256": bank_sha256,
        "b8_sha256": b8_sha256,
        "row_sha256": rows,
    }
