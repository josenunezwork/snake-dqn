"""Synthetic sealed handoff contracts between F4 producer, admission, and Watch6."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import analyze, gameplay


def _fixture_payload() -> dict:
    """Reuse the analyzer's synthetic-only fixture builder without artifacts."""
    path = Path(__file__).with_name("test_bl_residual_f4_analyzer.py")
    spec = importlib.util.spec_from_file_location("f4_analyzer_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._payload()


def test_actual_shaped_pass_predicates_produce_sealed_watch_admission(tmp_path: Path) -> None:
    payload = _fixture_payload()
    assert payload["gameplay"]["trajectory_tripwires"] == {"D1024": False, "D4096": False}
    assert all(payload["probe"]["tripwires"].values())
    admission = analyze.h5000_admission(payload)
    assert admission["schema"] == analyze.H5000_ADMISSION_SCHEMA
    assert admission["status"] == "PASS_WATCH6"
    assert admission["scientific_result"] is False
    receipt = tmp_path / "h5000-go.json"
    sealed = export_json(receipt, admission)
    assert receipt.with_suffix(".json.complete").read_text() == "complete\n"
    assert receipt.with_suffix(".json.sha256").exists()
    assert receipt.with_suffix(".json.sha256").read_text().startswith(sealed["sha256"])


def test_watch_preflight_accepts_only_the_sealed_actual_admission_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _fixture_payload()
    receipt_value = analyze.h5000_admission(payload)
    receipt = tmp_path / "h5000-go.json"
    seal = export_json(receipt, receipt_value)
    b8, d4096 = tmp_path / "b8.pth", tmp_path / "d4096.pth"
    b8.write_bytes(b"b8")
    d4096.write_bytes(b"top")
    monkeypatch.setattr(gameplay, "_require_mac", lambda: None)
    monkeypatch.setattr(gameplay, "_approved_successor", lambda _: {})
    monkeypatch.setattr(gameplay, "_validate_candidates", lambda *_: None)
    monkeypatch.setattr(
        gameplay,
        "sha256_file",
        lambda path: (
            gameplay.B8_SHA256 if Path(path) == b8 else receipt_value["controller_sha256"]["D4096"]
        ),
    )
    handoff = {**receipt_value, "receipt_path": str(receipt), "receipt_sha256": seal["sha256"]}
    # A malformed bank fails *after* the sealed admission preflight.  If the
    # admission envelope drifts, the earlier RuntimeError is the assertion.
    with pytest.raises(ValueError, match="audited bank"):
        gameplay.run_watch6_sessions(
            {},
            b8_path=b8,
            d4096_path=d4096,
            review_evidence={},
            candidate_metadata={},
            h5000_go_receipt=handoff,
        )
