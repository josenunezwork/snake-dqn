"""Admission-only contracts for the retained A100 wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ops import dispatch


def main(job_id: str, output: str, *, acceptance: bool = False) -> dict:
    result = {
        "job_id": job_id,
        "output_dir": output,
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"]},
        "timeout_minutes": 120,
        "canary": {"receipt": {"path": "/tmp/receipt", "sha256": "a" * 64}},
        "retention_chain": {"qualification": {"path": "/tmp/qualification", "sha256": "b" * 64}},
        "_broker": str(Path.cwd()),
    }
    if acceptance:
        result["resume_exemption"] = {"kind": "acceptance"}
    else:
        result["resume"] = {"unit_kind": "episode", "ledger_path": "inputs/ledger.json"}
    return result


def test_retained_main_allows_resumable_packet_longer_than_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.colab_cli import canary, retention

    monkeypatch.setattr(canary, "check", lambda *_: None)
    monkeypatch.setattr(retention, "validate_a100_chain", lambda *_: ({}, "key"))
    monkeypatch.setattr(retention, "runtime_cache", lambda *_: None)
    monkeypatch.setattr(retention, "require_native_qualification", lambda *_: None)
    dispatch._chain_values([main("one", "/tmp/one"), main("two", "/tmp/two")], 240, 40)


def test_retained_acceptance_packet_must_remain_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.colab_cli import canary

    monkeypatch.setattr(canary, "check", lambda *_: None)
    with pytest.raises(dispatch.Refused, match="<=60"):
        dispatch._chain_values(
            [main("one", "/tmp/one", acceptance=True), main("two", "/tmp/two")], 240, 40
        )
