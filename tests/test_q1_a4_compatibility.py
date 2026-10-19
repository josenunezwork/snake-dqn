"""Focused guards for the A4 historical evaluator bridge."""

from __future__ import annotations

import hashlib

import pytest

from research.q0q1_20261017 import a4_compatibility as subject

HASH = "a" * 64


def _artifact(source: dict[str, str], *, worlds: int = 3) -> dict:
    return {
        "schema": "q1-parity/v1",
        "bank_sha256": HASH,
        "b8_sha256": HASH,
        "source_sha256": subject.digest(source),
        "runtime": {"system": "Darwin"},
        "worlds": [{"world_id": str(index)} for index in range(worlds)],
    }


@pytest.fixture
def bridge(monkeypatch):
    current = {
        "research/q0q1_20261017/q1_evaluate.py": "c" * 64,
        "research/q0q1_20261017/q1_parity.py": subject.HISTORICAL_PARITY_SHA256,
        "other.py": "d" * 64,
    }
    monkeypatch.setattr(subject.q1_evaluate, "source_manifest", lambda: current)
    calls = []
    monkeypatch.setattr(
        subject.q1_parity,
        "validate_runtime_receipt",
        lambda value, **kwargs: calls.append((value, kwargs)),
    )
    historical = subject.historical_source_manifest(current)
    old_mac = _artifact(historical)
    current_mac = _artifact(current)
    runtime = {"source_manifest": historical, "reference": old_mac}
    rows = [hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(6)]
    timing = {
        "schema": "q1-eval-timing-probe/v1",
        "status": "completed",
        "purpose": "capacity measurement only; not Q1 decision evidence",
        "bank_sha256": HASH,
        "b8_sha256": HASH,
        "source_manifest": current,
        "historical_source_manifest": historical,
        "row_sha256": rows,
        "serial_row_sha256": list(rows),
        "historical_row_sha256": list(rows),
        "parity_world_count": 3,
        "world_count": 6,
        "repeats": 2,
        "elapsed_seconds": 1.0,
    }
    return runtime, old_mac, current_mac, timing, calls


def test_bridge_accepts_only_bound_historical_and_current_evidence(bridge):
    runtime, old_mac, current_mac, timing, calls = bridge
    result = subject.validate_historical_parity(runtime, old_mac, current_mac, timing)

    assert result["historical_commit"] == subject.HISTORICAL_COMMIT
    assert result["row_sha256"] == timing["row_sha256"]
    assert calls[0][1]["source_manifest"] == timing["historical_source_manifest"]


def test_bridge_rejects_wrong_old_evaluator_adapter(bridge):
    runtime, old_mac, current_mac, timing, _ = bridge
    runtime["source_manifest"][subject.HISTORICAL_EVALUATOR_PATH] = "f" * 64

    with pytest.raises(ValueError, match="exact A3 adapter closure"):
        subject.validate_historical_parity(runtime, old_mac, current_mac, timing)


def test_bridge_rejects_missing_historical_runtime_proof(bridge):
    runtime, old_mac, current_mac, timing, _ = bridge
    runtime.pop("source_manifest")

    with pytest.raises(ValueError, match="exact A3 adapter closure"):
        subject.validate_historical_parity(runtime, old_mac, current_mac, timing)


def test_bridge_rejects_tampered_historical_numerical_rows(bridge):
    runtime, old_mac, current_mac, timing, _ = bridge
    timing["historical_row_sha256"][1] = "e" * 64

    with pytest.raises(ValueError, match="numerical rows differ"):
        subject.validate_historical_parity(runtime, old_mac, current_mac, timing)


def test_bridge_rejects_incomplete_repeat_coverage(bridge):
    runtime, old_mac, current_mac, timing, _ = bridge
    timing["world_count"] = 3

    with pytest.raises(ValueError, match="exactly cover every parity world"):
        subject.validate_historical_parity(runtime, old_mac, current_mac, timing)
