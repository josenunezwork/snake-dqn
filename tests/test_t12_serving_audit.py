"""Evidence-integrity coverage for the independent T12 serving audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research.t12_b8_serving_20261017 import serving_audit as audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, rel: str, value: object) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _result(intent_sha: str, kind: str, data: dict, index: int = 0) -> dict:
    return {
        "schema": audit.SCHEMA,
        "intent_sha256": intent_sha,
        "kind": kind,
        "index": index,
        "seed_or_seeds": [index],
        "checkpoint_sha256": audit.CHECKPOINT_SHA256,
        "started_utc": "2026-10-17T00:00:00Z",
        "wall_seconds": 0.0,
        "status": "complete",
        "error": None,
        "data": data,
    }


def _minimal_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    intent = {
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "checkpoint_sha256": audit.CHECKPOINT_SHA256,
    }
    _write(root, "intent.json", intent)
    intent_sha = _sha(root / "intent.json")
    for rel in audit._expected_files() - {"intent.json"}:
        _write(root, rel, _result(intent_sha, rel.split("/")[-1].split("-")[0], {}))
    files = {rel: _sha(root / rel) for rel in audit._expected_files()}
    _write(root, "receipt.json", {"intent_sha256": intent_sha, "files": files})
    return root


def test_partial_evidence_is_invalid(tmp_path):
    root = _minimal_run(tmp_path)
    (root / "records/watch-003.json").unlink()
    result = audit.audit(root)
    assert result["verdict"] == "INVALID"
    assert "receipt hash mismatch" in result["error"]


def test_receipt_tamper_is_invalid_even_when_record_still_exists(tmp_path):
    root = _minimal_run(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"].pop("records/play-024.json")
    _write(root, "receipt.json", receipt)
    result = audit.audit(root)
    assert result["verdict"] == "INVALID"
    assert "omissions" in result["error"]


def test_rehashed_record_with_wrong_intent_binding_is_invalid(tmp_path):
    root = _minimal_run(tmp_path)
    target = root / "records/watch-000.json"
    item = json.loads(target.read_text())
    item["intent_sha256"] = "0" * 64
    _write(root, "records/watch-000.json", item)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"]["records/watch-000.json"] = _sha(target)
    _write(root, "receipt.json", receipt)
    assert audit.audit(root)["verdict"] == "INVALID"


def test_malformed_but_valid_json_is_invalid_not_an_audit_crash(tmp_path):
    root = _minimal_run(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"] = []
    _write(root, "receipt.json", receipt)
    result = audit.audit(root)
    assert result["verdict"] == "INVALID"
    assert result["serving_path_qualified"] is False


def test_pacing_raw_initial_attempt_passes_at_derived_bound():
    row = {
        "ui_speed_bound_fps": 60,
        "pacing_attempts": [
            {"requested_fps": 60, "ticks": 120, "raw_tick_ms": [10.0] * 120, "overruns": 0}
        ],
    }
    assert audit._pacing_failures(row, [10.0] * 1500) == []


def test_pacing_one_lowered_retry_is_valid_but_forged_count_fails():
    first = {"requested_fps": 60, "ticks": 120, "raw_tick_ms": [20.0] * 120, "overruns": 120}
    second = {"requested_fps": 49, "ticks": 120, "raw_tick_ms": [10.0] * 120, "overruns": 0}
    row = {"ui_speed_bound_fps": 49, "pacing_attempts": [first, second]}
    assert audit._pacing_failures(row, [10.0] * 1500) == []
    first["overruns"] = 0
    assert "overrun count" in audit._pacing_failures(row, [10.0] * 1500)[0]
